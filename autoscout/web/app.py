"""Dashboard + API (FastAPI). Run with `autoscout serve`."""
from __future__ import annotations

import json
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

import cv2
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from .. import config, db, games
from ..aggregate import team_table
from ..field import Calibration, FieldModel

app = FastAPI(title="FRC Auto-Scout")
STATIC = Path(__file__).parent / "static"
JOBS: dict[str, subprocess.Popen] = {}


def con():
    return db.connect()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/events")
def events():
    c = con()
    rows = c.execute("SELECT event_key, year, COUNT(*) n, SUM(status='done') done FROM matches GROUP BY event_key").fetchall()
    out = {r["event_key"]: {"event_key": r["event_key"], "year": r["year"], "matches": r["n"], "done": r["done"] or 0}
           for r in rows}
    for p in config.EVENTS_DIR.glob("*.yaml"):
        if p.stem.endswith("_overrides"):
            continue
        d = yaml.safe_load(p.read_text())
        out.setdefault(p.stem, {"event_key": p.stem, "year": d.get("year"), "matches": 0, "done": 0})
        out[p.stem]["calibrated"] = len(d.get("points", {})) >= 4
    return sorted(out.values(), key=lambda e: (-(e["year"] or 0), e["event_key"]))


@app.get("/api/event/{event_key}")
def event(event_key: str):
    _event_key(event_key)
    c = con()
    ms = [dict(r) for r in c.execute(
        "SELECT match_key,label,comp_level,set_number,match_number,red,blue,red_score,blue_score,status,error "
        "FROM matches WHERE event_key=? ORDER BY comp_level='f', comp_level='sf', set_number, match_number", (event_key,))]
    for m in ms:
        m["red"] = json.loads(m["red"]) if m["red"] else []
        m["blue"] = json.loads(m["blue"]) if m["blue"] else []
    year = int(event_key[:4])
    try:
        defs = games.get(year).metric_defs
    except KeyError:
        defs = []
    return {"event_key": event_key, "year": year, "matches": ms, "teams": team_table(c, event_key), "metric_defs": defs,
            "job": _job_status(event_key)}


@app.get("/api/match/{match_key}")
def match(match_key: str):
    r = db.load_result(con(), match_key)
    if r is None:
        raise HTTPException(404, "not processed")
    r.pop("raw_tracks", None)
    for v in r.get("launches", {}).values():
        for l in v:
            l.pop("pts", None)
    try:
        r["metric_defs"] = games.get(r["year"]).metric_defs
    except KeyError:
        r["metric_defs"] = []
    kind, _ = _annotated_video(match_key)
    clock = r.get("clock") or {}
    # demo videos (scripts/render_demo.py) run in real time from 1 s before auto, so video time maps to demo time
    offset = clock["auto_end_vt"] - clock["auto_len"] - 1.0 if kind == "demo" and "auto_end_vt" in clock else None
    r["annotated_video"] = {"kind": kind, "offset": offset} if kind else None
    return r


def _annotated_video(match_key: str):
    for kind, p in (("demo", config.ROOT / "demos" / f"{match_key}_annotated.mp4"),
                    ("demo", config.ROOT / "demo" / f"{match_key}.mp4"),
                    ("debug", config.DATA_DIR / "results" / f"{match_key}_debug.mp4")):
        if p.exists():
            return kind, p
    return None, None


@app.get("/api/match/{match_key}/robots")
def robots(match_key: str, n: int = 16):
    """Robots (tracklets) worth a person's label, plus the ones already labelled."""
    from ..pipeline import RESULTS_DIR, load_labels
    from ..vision.tracker import RobotTracker

    tp = RESULTS_DIR / f"{match_key}_tracklets.json"
    if not tp.exists():
        raise HTTPException(404, "no saved robot tracklets: process the match again")
    tracker = RobotTracker.from_dict(json.loads(tp.read_text()))
    labels = load_labels(match_key)
    for t in tracker.done:
        t.manual = labels.get(t.id)
    ask = set(tracker.to_ask(n))
    thumbs = RESULTS_DIR / f"{match_key}_robots"

    def row(t):
        votes = [o[4] for o in t.obs if o[4]]
        return {"id": t.id, "t0": round(t.t0, 1), "t1": round(t.t1, 1), "team": t.team, "manual": t.manual,
                "read": max(t.reads, key=t.reads.get) if t.reads else None,
                "alliance": max(set(votes), key=votes.count) if votes else None}
    items = [row(t) for t in tracker.done if (t.id in ask or t.id in labels) and (thumbs / f"{t.id}.jpg").exists()]
    return {"teams": tracker.teams, "robots": sorted(items, key=lambda r: r["t0"]), "n_labelled": len(labels)}


@app.get("/api/match/{match_key}/robot/{tid}.jpg")
def robot_thumb(match_key: str, tid: int):
    from ..pipeline import RESULTS_DIR

    p = RESULTS_DIR / f"{match_key}_robots" / f"{tid}.jpg"
    if not p.exists():
        raise HTTPException(404, "no image")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/match/{match_key}/robots")
def label_robots(match_key: str, body: dict):
    """{tracklet id: team | 0 (not a robot) | null (clear)} -> relabel the match and recompute stats."""
    from ..pipeline import RESULTS_DIR, save_labels

    r = save_labels(match_key, {int(k): v for k, v in body.items()})
    db.save_result(con(), r, str(RESULTS_DIR / f"{match_key}.json"))
    return robots(match_key)


@app.get("/api/match/{match_key}/video.mp4")
def match_video(match_key: str):
    _, p = _annotated_video(match_key)
    if p is None:
        raise HTTPException(404, "no annotated video (run with --debug-video or scripts/render_demo.py)")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/team/{event_key}/{team}")
def team(event_key: str, team: int):
    rows = con().execute("SELECT match_key, metrics FROM team_matches WHERE event_key=? AND team=?", (event_key, team))
    return [{"match_key": r["match_key"], **json.loads(r["metrics"])} for r in rows]


@app.post("/api/override/{match_key}")
def override(match_key: str, body: dict):
    from ..pipeline import RESULTS_DIR, reanalyze

    c = con()
    for slot, t in body.items():
        db.set_override(c, match_key, slot, int(t))
    r = reanalyze(match_key)
    db.save_result(c, r, str(RESULTS_DIR / f"{match_key}.json"))
    return {"ok": True, "identity": r["identity"]}


# ---- processing jobs -------------------------------------------------------------------
def _job_status(key):
    p = JOBS.get(key)
    if p is None:
        return None
    return {"running": p.poll() is None, "returncode": p.returncode}


@app.post("/api/run/{event_key}")
def run_event(event_key: str, limit: int = 0):
    _event_key(event_key)
    cal = Calibration.load(event_key)
    if not cal or len(cal.points) < 4:
        raise HTTPException(400, "Calibrate this event before processing matches")
    if _job_status(event_key) and _job_status(event_key)["running"]:
        return {"started": False, "reason": "already running"}
    log = open(config.DATA_DIR / f"run_{event_key}.log", "w")
    JOBS[event_key] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "run-event", event_key,
                                        "--limit", str(limit)], stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.get("/api/progress")
def progress():
    from ..progress import all_progress

    return all_progress()


@app.get("/api/run/{event_key}/log")
def run_log(event_key: str):
    p = config.DATA_DIR / f"run_{event_key}.log"
    return Response(p.read_text(encoding="utf-8", errors="replace")[-6000:] if p.exists() else "", media_type="text/plain")


# ---- calibration -------------------------------------------------------------------------
@app.get("/api/calibration/{event_key}")
def get_cal(event_key: str):
    cal = Calibration.load(event_key)
    year = int(event_key[:4])
    fm = FieldModel.load(year)
    return {"event_key": event_key, "year": year, "landmarks": fm.landmarks,
            "view": list(cal.view) if cal else None, "points": cal.points if cal else {},
            "has_reference": bool(cal and cal.reference and (config.ROOT / cal.reference).exists())}


@app.get("/api/calibration/{event_key}/ref.jpg")
def cal_ref(event_key: str):
    cal = Calibration.load(event_key)
    if not cal or not cal.reference:
        raise HTTPException(404, "no reference frame; run `autoscout calibrate <match_key>`")
    return FileResponse(config.ROOT / cal.reference)


@app.post("/api/calibration/{event_key}")
def save_cal(event_key: str, body: dict):
    cal = Calibration.load(event_key)
    year = int(event_key[:4])
    if cal is None:
        cal = Calibration(event_key=event_key, year=year, view=tuple(body["view"]), points={})
    cal.view = tuple(int(v) for v in body["view"])
    cal.points = {k: tuple(v) for k, v in body["points"].items()}
    err = None
    if len(cal.points) >= 4:
        err = cal.fit()
    cal.save()
    return {"ok": True, "reprojection_error_m": err}


@app.get("/api/calibration/{event_key}/preview.jpg")
def cal_preview(event_key: str):
    import numpy as np

    cal = Calibration.load(event_key)
    if not cal or not cal.reference:
        raise HTTPException(404)
    fm = FieldModel.load(cal.year)
    im = cv2.imread(str(config.ROOT / cal.reference))
    if len(cal.points) >= 4:
        cal.fit(fm)
        for x in np.arange(0, fm.length + 0.01, 1.0):
            a, b = cal.to_image([(x, 0), (x, fm.width)]).astype(int)
            cv2.line(im, tuple(a), tuple(b), (120, 120, 120), 1)
        for name, (x0, y0, x1, y1) in fm.zones.items():
            poly = cal.to_image([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).astype(np.int32)
            col = (255, 80, 0) if "blue" in name else (0, 0, 255) if "red" in name else (0, 220, 0)
            cv2.polylines(im, [poly], True, col, 2)
        for name, e in fm.elements.items():
            c = cal.to_image([e["center"]])[0].astype(int)
            cv2.circle(im, tuple(c), 6, (0, 255, 255), -1)
            cv2.putText(im, name, (c[0] + 6, c[1]), 0, 0.5, (0, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", im)
    return Response(buf.tobytes(), media_type="image/jpeg")


# ---- labeling (seed boxes for SAM2 propagation) ---------------------------------------------
@app.get("/api/videos")
def videos():
    out = []
    for d in (config.DATA_DIR / "videos", config.DATA_DIR / "videos1080"):
        if d.exists():
            out += [str(p.relative_to(config.ROOT)) for p in sorted(d.glob("*.mp4"))]
    return out


@app.get("/api/frame.jpg")
def frame(video: str, t: float):
    from ..video import Video

    p = (config.ROOT / video).resolve()
    if config.DATA_DIR.resolve() not in p.parents:
        raise HTTPException(400, "video must be under data/")
    im = Video(p).frame_at(t)
    if im is None:
        raise HTTPException(404)
    ok, buf = cv2.imencode(".jpg", im)
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/api/label/seeds")
def add_seeds(body: dict, background: BackgroundTasks):
    path = config.CONFIG_DIR / "seeds" / "dashboard.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    segs = yaml.safe_load(path.read_text()) if path.exists() else []
    seg = {"video": body["video"], "event": body["event"], "t": float(body["t"]),
           "duration": int(body.get("duration", 30)), "fps": int(body.get("fps", 4)),
           "robots": [[r[0], *[int(v) for v in r[1:]]] for r in body["robots"]]}
    segs.append(seg)
    path.write_text(yaml.safe_dump(segs, sort_keys=False))
    if body.get("propagate", True):
        one = config.DATA_DIR / "tmp" / "seed_once.yaml"
        one.parent.mkdir(parents=True, exist_ok=True)
        one.write_text(yaml.safe_dump([seg]))
        log = open(config.DATA_DIR / "label_jobs.log", "a")
        JOBS["label"] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "label-seeds", str(one)],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"ok": True, "segments": len(segs)}


@app.get("/api/label/suggest")
def suggest(video: str, t: float, event: str = ""):
    """Detector boxes for a frame (full-frame pixels) to pre-fill the labeling canvas."""
    from ..video import Video
    from ..vision.detect import RobotDetector, alliance_from_pixels

    p = (config.ROOT / video).resolve()
    im = Video(p).frame_at(t)
    cal = Calibration.load(event) if event else None
    x, y, w, h = cal.view if cal else (0, 0, im.shape[1], im.shape[0])
    det = _detector()
    out = []
    for d in det(im[y: y + h, x: x + w], conf=0.3):
        d.x0 += x; d.x1 += x; d.y0 += y; d.y1 += y
        a = d.alliance or alliance_from_pixels(im, d)[0] or "red"
        out.append([a, round(d.x0), round(d.y0), round(d.x1), round(d.y1)])
    return out


_DET = {}


def _detector():
    from ..vision.detect import default_detector, find_weights

    w = find_weights()
    if _DET.get("w") != w:
        _DET.update(w=w, det=default_detector())
    return _DET["det"]


@app.post("/api/label/save")
def save_labels(body: dict):
    """Save hand-corrected boxes for one frame directly as a training example."""
    from ..train.sam_label import write_label
    from ..video import Video

    p = (config.ROOT / body["video"]).resolve()
    im = Video(p).frame_at(float(body["t"]))
    cal = Calibration.load(body.get("event", "")) if body.get("event") else None
    x, y, w, h = cal.view if cal else (0, 0, im.shape[1], im.shape[0])
    boxes = [(r[0], r[1] - x, r[2] - y, r[3] - x, r[4] - y) for r in body["robots"]]
    stem = f"{p.stem}_u{float(body['t']):.1f}".replace(".", "_")
    write_label(im[y: y + h, x: x + w], boxes, stem)
    return {"ok": True, "stem": stem}


@app.post("/api/train")
def train():
    if _job_status("train") and _job_status("train")["running"]:
        raise HTTPException(409, "Detector training is already running")
    log = open(config.DATA_DIR / "train.log", "w")
    JOBS["train"] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "train-robots"],
                                     stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.get("/api/dataset")
def dataset():
    from ..train.sam_label import DATASET

    n = len(list((DATASET / "images").glob("*.jpg"))) if (DATASET / "images").exists() else 0
    return {"images": n, "label_job": _job_status("label"), "train_job": _job_status("train")}


# Dashboard workflow endpoints reuse the same commands as the CLI.
def _event_key(key: str):
    if not re.fullmatch(r"20\d{2}[a-z0-9]+", key):
        raise HTTPException(400, "Enter an event key such as 2026txhou")
    if int(key[:4]) not in (2025, 2026):
        raise HTTPException(400, "Supported seasons are 2025 and 2026")


def _start_job(key: str, args: list[str]):
    if _job_status(key) and _job_status(key)["running"]:
        raise HTTPException(409, "This job is already running")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / f"run_{key}.log", "w") as log:
        JOBS[key] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", *args],
                                     stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.post("/api/event/{event_key}/reanalyze")
def dashboard_reanalyze(event_key: str):
    _event_key(event_key)
    return _start_job(event_key, ["reanalyze", event_key])


@app.post("/api/calibration/{event_key}/prepare")
def prepare_calibration(event_key: str, body: dict):
    _event_key(event_key)
    mk = str(body.get("match_key", ""))
    if not re.fullmatch(re.escape(event_key) + r"_(?:qm\d+|(?:ef|qf|sf|f)\d+m\d+)", mk):
        raise HTTPException(400, "Enter a match key from this event, such as " + event_key + "_qm1")
    return _start_job(event_key, ["calibrate", mk])


@app.get("/api/event/{event_key}/report", response_class=Response)
def dashboard_report(event_key: str):
    from ..validate import event_report
    _event_key(event_key)
    with con() as c:
        return Response(event_report(c, event_key), media_type="text/plain")


@app.get("/api/event/{event_key}/export")
def dashboard_export(event_key: str):
    _event_key(event_key)
    with con() as c:
        rows = [{"match_key": r["match_key"], "team": r["team"], **json.loads(r["metrics"])}
                for r in c.execute("SELECT match_key, team, metrics FROM team_matches WHERE event_key=? ORDER BY team, match_key", (event_key,))]
    out = io.StringIO()
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["match_key", "team"]
    writer = csv.DictWriter(out, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{event_key}.csv"'})


@app.get("/api/jobs")
def dashboard_jobs():
    return [{"key": key, **_job_status(key)} for key in JOBS]


# ---- runs overview, annotated videos, accuracy ----------------------------------------------
DEMOS = config.ROOT / "demos"
_ACC_CACHE: dict[str, tuple[float, dict]] = {}


def _results():
    from ..pipeline import RESULTS_DIR

    return RESULTS_DIR


@app.get("/api/runs")
def runs():
    """Everything the Runs page shows that changes while matches process."""
    from ..progress import all_progress

    per_event: dict[str, dict] = {}
    recent = []
    for p in _results().glob("*_*.json"):
        if p.stem.count("_") != 1 or p.stem.endswith("_tracklets"):
            continue
        ev = p.stem.split("_")[0]
        e = per_event.setdefault(ev, {"event_key": ev, "processed": 0, "last": 0.0, "secs": []})
        e["processed"] += 1
        mt = p.stat().st_mtime
        e["last"] = max(e["last"], mt)
        recent.append((mt, p))
    out_recent = []
    for mt, p in sorted(recent, reverse=True)[:12]:
        r = _summary(p, mt)
        if r is None:
            continue
        e = per_event[p.stem.split("_")[0]]
        if r.get("processing_s"):
            e["secs"].append(r["processing_s"])
        out_recent.append({"match_key": p.stem, "label": r.get("label"), "finished": mt,
                           "processing_s": r.get("processing_s"), "red_score": r.get("red_score"),
                           "blue_score": r.get("blue_score"), "video": _annotated_name(p.stem)})
    for e in per_event.values():
        e["avg_s"] = round(sum(e["secs"]) / len(e["secs"])) if e["secs"] else None
        del e["secs"]
    jobs = [{"key": k, **_job_status(k)} for k in JOBS]
    return {"progress": all_progress(), "jobs": jobs, "recent": out_recent,
            "events": sorted(per_event.values(), key=lambda e: -e["last"])}


_SUMMARIES: dict[str, tuple[float, dict]] = {}


def _summary(p: Path, mtime: float) -> dict | None:
    """The few fields of a (multi-MB) result file the Runs page needs, parsed once per change."""
    hit = _SUMMARIES.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        r = json.loads(p.read_text())
    except Exception:  # being written right now
        return None
    keep = {k: r.get(k) for k in ("label", "processing_s", "red_score", "blue_score", "red", "blue", "year")}
    _SUMMARIES[str(p)] = (mtime, keep)
    return keep


def _annotated_name(match_key: str):
    p = DEMOS / f"{match_key}_annotated.mp4"
    return p.name if p.exists() else None


@app.get("/api/annotated")
def annotated_list():
    out = []
    for p in sorted(DEMOS.glob("*_annotated.mp4"), key=lambda p: -p.stat().st_mtime) if DEMOS.exists() else []:
        mk = p.name[: -len("_annotated.mp4")]
        info = {}
        rp = _results() / f"{mk}.json"
        r = _summary(rp, rp.stat().st_mtime) if rp.exists() else None
        if r:
            info = {k: r.get(k) for k in ("label", "red", "blue", "red_score", "blue_score", "year")}
        out.append({"name": p.name, "match_key": mk, "event_key": mk.split("_")[0],
                    "size_mb": round(p.stat().st_size / 1e6, 1), "updated": p.stat().st_mtime, **info})
    return out


@app.get("/api/annotated/{name}")
def annotated_file(name: str):
    if not re.fullmatch(r"[\w-]+_annotated\.mp4", name) or not (DEMOS / name).exists():
        raise HTTPException(404, "no such video")
    return FileResponse(DEMOS / name, media_type="video/mp4")


@app.post("/api/annotate/{match_key}")
def annotate(match_key: str):
    """Render the annotated video of a processed match in the background."""
    if not re.fullmatch(r"20\d\d[a-z0-9]+_(qm|sf|f)\d+(m\d+)?", match_key):
        raise HTTPException(400, "bad match key")
    rp = _results() / f"{match_key}.json"
    if not rp.exists():
        raise HTTPException(404, "process the match first")
    from ..video import VIDEO_DIR

    vid = json.loads(rp.read_text()).get("video")
    src = next((p for p in VIDEO_DIR.glob(f"{vid}.*")), None)
    if src is None:
        raise HTTPException(404, "match video is not downloaded")
    key = f"render:{match_key}"
    if _job_status(key) and _job_status(key)["running"]:
        return {"started": False, "reason": "already rendering"}
    DEMOS.mkdir(exist_ok=True)
    log = open(config.DATA_DIR / f"render_{match_key}.log", "w")
    JOBS[key] = subprocess.Popen([sys.executable, "scripts/render_demo.py", match_key, str(src),
                                  str(DEMOS / f"{match_key}_annotated.mp4")], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=config.ROOT)
    return {"started": True}


@app.get("/api/accuracy/{event_key}")
def accuracy(event_key: str):
    """Per-team attribution vs OPR for an event (cached until a result changes)."""
    from ..evaluation import attribution_accuracy, load_results

    _event_key(event_key)
    files = [p for p in _results().glob(f"{event_key}_qm*.json") if p.stem.count("_") == 1]
    stamp = max((p.stat().st_mtime for p in files), default=0.0)
    hit = _ACC_CACHE.get(event_key)
    if hit and hit[0] == stamp:
        return hit[1]
    res = attribution_accuracy(event_key, load_results(event_key, _results()))
    _ACC_CACHE[event_key] = (stamp, res)
    return res


@app.get("/api/detector-benchmark")
def detector_benchmark():
    p = config.DATA_DIR / "det_eval" / "results.json"
    return json.loads(p.read_text()) if p.exists() else []

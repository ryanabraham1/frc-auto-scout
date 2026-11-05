"""Dashboard + API (FastAPI). Run with `autoscout serve`."""
from __future__ import annotations

import json
import csv
import io
import math
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
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out.setdefault(p.stem, {"event_key": p.stem, "year": d.get("year"), "matches": 0, "done": 0})
        out[p.stem]["calibrated"] = len(d.get("points", {})) >= 4
    return sorted(out.values(), key=lambda e: (-(e["year"] or 0), e["event_key"]))


@app.get("/api/event/{event_key}")
def event(event_key: str):
    _event_key(event_key)
    c = con()
    ms = [dict(r) for r in c.execute(
        "SELECT match_key,label,comp_level,set_number,match_number,red,blue,red_score,blue_score,status,error "
        "FROM matches WHERE event_key=? ORDER BY CASE comp_level WHEN 'qm' THEN 0 WHEN 'ef' THEN 1 WHEN 'qf' THEN 2 "
        "WHEN 'sf' THEN 3 WHEN 'f' THEN 4 ELSE 5 END, set_number, match_number", (event_key,))]
    for m in ms:
        m["red"] = json.loads(m["red"]) if m["red"] else []
        m["blue"] = json.loads(m["blue"]) if m["blue"] else []
    year = int(event_key[:4])
    try:
        defs = games.get(year).metric_defs
    except KeyError:
        defs = []
    return {"event_key": event_key, "year": year, "matches": ms, "teams": team_table(c, event_key), "metric_defs": defs,
            "job": _event_job(event_key)}


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
    kind, vpath = _annotated_video(match_key)
    clock = r.get("clock") or {}
    # demo videos (scripts/render_demo.py) run in real time from 1 s before auto, so video time maps to demo time
    offset = clock["auto_end_vt"] - clock["auto_len"] - 1.0 if kind == "demo" and "auto_end_vt" in clock else None
    r["annotated_video"] = ({"kind": kind, "offset": offset, "downloadable": vpath.name == f"{match_key}_annotated.mp4" and vpath.parent == DEMOS}
                            if kind else None)
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
    _event_key(event_key)
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


def _orphan_runs(event_key: str):
    """`run-event` processes for this event that this dashboard did not start (e.g. before a restart)."""
    import psutil

    own = {p.pid for p in JOBS.values() if p.poll() is None}
    out = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except psutil.Error:
            continue
        if "autoscout.cli" in cmd and "run-event" in cmd and event_key in cmd:
            out.append(p)
    # the venv launcher starts a second python with the same command line: keep only the outermost
    pids = {p.pid for p in out} | own
    return [p for p in out if p.pid not in own and p.info["ppid"] not in pids]


def _event_job(event_key: str):
    st = _job_status(event_key)
    if st and st["running"]:
        return st
    if _orphan_runs(event_key):
        return {"running": True, "returncode": None, "external": True}
    return st


def _kill_tree(pid: int):
    import psutil

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = root.children(recursive=True) + [root]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


@app.post("/api/run/{event_key}")
def run_event(event_key: str, limit: int = 0, body: dict | None = None):
    """Body (all optional): matches [keys], debug_video bool, workers 1-4."""
    _event_key(event_key)
    cal = Calibration.load(event_key)
    if not cal or len(cal.points) < 4:
        raise HTTPException(400, "Calibrate this event before processing matches")
    if (_event_job(event_key) or {}).get("running"):
        return {"started": False, "reason": "already running"}
    body = body or {}
    try:
        workers = max(1, min(int(body.get("workers", 1)), 4))
    except (TypeError, ValueError):
        raise HTTPException(400, "workers must be a number from 1 to 4")
    args = ["run-event", event_key, "--limit", str(limit), "--workers", str(workers)]
    matches = [str(k) for k in body.get("matches") or []]
    for k in matches:
        if not re.fullmatch(re.escape(event_key) + r"_(?:qm\d+|(?:ef|qf|sf|f)\d+m\d+)", k):
            raise HTTPException(400, f"{k} is not a match of {event_key}")
    if matches:
        args += ["--matches", ",".join(matches)]
    if body.get("debug_video"):
        args.append("--debug-video")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / f"run_{event_key}.log", "w") as log:
        JOBS[event_key] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", *args],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.post("/api/run/{event_key}/cancel")
def cancel_run(event_key: str):
    """Stop the event's processing job (and its workers, downloads and encoders)."""
    from ..progress import DIR, all_progress

    _event_key(event_key)
    pids = [p.pid for p in _orphan_runs(event_key)]
    p = JOBS.get(event_key)
    if p is not None and p.poll() is None:
        pids.append(p.pid)
    for pid in pids:
        _kill_tree(pid)
    # matches it was in the middle of would otherwise look stuck on the Overview page
    cleared = 0
    for d in all_progress():
        if d["match_key"].startswith(event_key + "_") and d["stage"] not in ("done", "error"):
            d.update(stage="error", error="cancelled", detail="cancelled", eta_s=None)
            (DIR / f"{d['match_key']}.json").write_text(json.dumps({k: v for k, v in d.items() if k != "stale"}))
            cleared += 1
    if not pids and not cleared:
        return {"cancelled": False, "reason": "nothing is running for this event"}
    with open(config.DATA_DIR / f"run_{event_key}.log", "a") as log:
        log.write("\n--- cancelled from the dashboard ---\n")
    return {"cancelled": True}


@app.get("/api/run/{event_key}/status")
def run_status(event_key: str):
    _event_key(event_key)
    return _event_job(event_key) or {"running": False}


@app.get("/api/event/{event_key}/schedule")
def schedule(event_key: str):
    """Every match of the event on TBA, with whether it has video and its processing status."""
    from ..progress import all_progress
    from ..tba import TBA, MatchInfo, sort_key

    _event_key(event_key)
    try:
        ms = sorted((MatchInfo.from_tba(m) for m in TBA(max_age_s=300).event_matches(event_key)), key=sort_key)
    except Exception as e:
        raise HTTPException(502, f"Could not load the schedule from The Blue Alliance: {e}") from e
    status = {r["match_key"]: (r["status"], r["error"]) for r in con().execute(
        "SELECT match_key, status, error FROM matches WHERE event_key=?", (event_key,))}
    live = {d["match_key"]: d for d in all_progress()}
    out = []
    for m in ms:
        st, err = status.get(m.key, (None, None))
        pr = live.get(m.key)
        if pr and pr["stage"] not in ("done", "error") and not pr.get("stale"):
            st = "processing"
        out.append({"match_key": m.key, "label": m.label, "comp_level": m.comp_level, "red": m.red, "blue": m.blue,
                    "has_video": bool(m.videos or m.local_video), "status": st or "new", "error": err})
    return out


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
    _event_key(event_key)
    cal = Calibration.load(event_key)
    year = int(event_key[:4])
    fm = FieldModel.load(year)
    return {"event_key": event_key, "year": year, "landmarks": fm.landmarks,
            "view": list(cal.view) if cal else None, "points": cal.points if cal else {},
            "aux_views": cal.aux_views or {} if cal else {},
            "aux_watch": cal.aux_watch or {} if cal else {},
            "overlay": cal.overlay or {} if cal else {},
            "elements": list(fm.elements),
            "frame_size": list(cal.frame_size) if cal else [1920, 1080],
            "has_reference": bool(cal and cal.reference and (config.ROOT / cal.reference).exists())}


@app.get("/api/calibration/{event_key}/ref.jpg")
def cal_ref(event_key: str):
    _event_key(event_key)
    cal = Calibration.load(event_key)
    if not cal or not cal.reference:
        raise HTTPException(404, "no reference frame; run `autoscout calibrate <match_key>`")
    return FileResponse(config.ROOT / cal.reference)


@app.post("/api/calibration/{event_key}")
def save_cal(event_key: str, body: dict):
    _event_key(event_key)
    cal = Calibration.load(event_key)
    year = int(event_key[:4])
    if cal is None:
        cal = Calibration(event_key=event_key, year=year, view=tuple(body["view"]), points={})
    original_aux_views = cal.aux_views or {}
    # Validate all optional rectangles before changing or writing the calibration.
    rectangles = {}
    for field in ("aux_views", "aux_watch", "overlay"):
        if field not in body:
            continue
        values = body[field]
        if not isinstance(values, dict):
            raise HTTPException(400, f"{field} must contain named rectangles")
        rectangles[field] = {}
        for name, rect in values.items():
            if field == "overlay" and name not in ("timer", "blue_score", "red_score"):
                raise HTTPException(400, "Unknown scoreboard rectangle")
            if field != "overlay" and name not in FieldModel.load(year).elements:
                raise HTTPException(400, "Choose a field element for each secondary camera")
            if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rect)):
                raise HTTPException(400, "Rectangles need four finite numbers")
            x, y, a, b = map(int, rect)
            w, h = (a - x, b - y) if field == "aux_watch" else (a, b)
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > cal.frame_size[0] or y + h > cal.frame_size[1]:
                raise HTTPException(400, "Draw rectangles inside the reference image")
            rectangles[field][name] = [x, y, a, b]
    for field, values in rectangles.items():
        if field == "overlay":
            if values and not all(k in values or k in (cal.overlay or {}) for k in ("timer", "blue_score", "red_score")):
                raise HTTPException(400, "Mark all three scoreboard rectangles, or leave them automatic")
            if values:
                cal.overlay = {**(cal.overlay or {}), **values, "frame_size": list(cal.frame_size)}
        else:
            setattr(cal, field, values)
    cal.view = tuple(int(v) for v in body["view"])
    cal.points = {k: tuple(v) for k, v in body["points"].items()}
    err = None
    if len(cal.points) >= 4:
        err = cal.fit()
    cal.save()
    if "aux_views" in rectangles:
        # Camera fits use full-frame pixels and must not survive moved or removed panes.
        cams_path = config.EVENTS_DIR / f"{event_key}_cams.json"
        if cams_path.exists():
            cams = json.loads(cams_path.read_text())
            cams = {k: v for k, v in cams.items() if k in cal.aux_views
                    and original_aux_views.get(k) == cal.aux_views[k]}
            cams_path.write_text(json.dumps(cams, indent=1))
    return {"ok": True, "reprojection_error_m": err}


@app.get("/api/calibration/{event_key}/cameras")
def camera_status(event_key: str):
    _event_key(event_key)
    from ..pipeline import RESULTS_DIR, closeup_calibration
    from ..vision.fusion import MIN_PAIRS, calibration_pairs
    cal = Calibration.load(event_key)
    results = [json.loads(p.read_text()) for p in RESULTS_DIR.glob(f"{event_key}_*.json")
               if p.stem.count("_") == 1]
    pairs = calibration_pairs(results)
    fitted = closeup_calibration(event_key)
    return {"matches": len(results), "minimum_pairs": MIN_PAIRS,
            "views": {name: {"pairs": len(pairs.get(name, [])), "fit": fitted.get(name)}
                      for name in (cal.aux_views or {} if cal else {})}}


@app.post("/api/calibration/{event_key}/cameras")
def fit_cameras(event_key: str):
    _event_key(event_key)
    cal = Calibration.load(event_key)
    if not cal or not cal.aux_views:
        raise HTTPException(400, "Save secondary camera rectangles first")
    return _start_job(event_key, ["calibrate-cams", event_key])


@app.get("/api/calibration/{event_key}/preview.jpg")
def cal_preview(event_key: str):
    _event_key(event_key)
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


@app.get("/api/label/videos")
def label_videos():
    from ..train.train_robots import source_aliases

    aliases = source_aliases()
    out = []
    for video in videos():
        stem = Path(video).stem
        match = aliases.get(stem, "")
        event = match.split("_")[0] if re.match(r"20\d{2}[a-z0-9]+_", match) else ""
        quality = "1080p" if "videos1080" in Path(video).parts else "local"
        out.append(dict(video=video, event=event, match=match,
                        label=f"{match or stem} ({quality})"))
    return sorted(out, key=lambda v: (v["match"] or "~", v["video"]))


@app.get("/api/frame.jpg")
def frame(video: str, t: float):
    _, im, _ = _label_frame(video, t)
    ok, buf = cv2.imencode(".jpg", im)
    return Response(buf.tobytes(), media_type="image/jpeg")


def _label_frame(video: str, t: float, event: str = ""):
    from ..video import Video

    p = (config.ROOT / video).resolve()
    if config.DATA_DIR.resolve() not in p.parents or not p.is_file():
        raise HTTPException(400, "Choose an existing video under data/")
    if not math.isfinite(t) or t < 0:
        raise HTTPException(400, "Frame time must be a finite, nonnegative number")
    v = Video(p)
    try:
        if t >= v.duration:
            raise HTTPException(400, "Frame time is past the end of the video")
        im = v.frame_at(t)
    finally:
        v.cap.release()
    if im is None:
        raise HTTPException(404, "Could not decode this frame")
    if event:
        _event_key(event)
        cal = Calibration.load(event)
        if cal is None:
            raise HTTPException(400, "Calibrate this event first")
        cal = cal.scaled((im.shape[1], im.shape[0]))
        rect = tuple(int(n) for n in cal.view)
    else:
        rect = (0, 0, im.shape[1], im.shape[0])
    x, y, w, h = rect
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > im.shape[1] or y + h > im.shape[0]:
        raise HTTPException(400, "Calibration crop is outside this video")
    return p, im, rect


@app.post("/api/label/seeds")
def add_seeds(body: dict, background: BackgroundTasks):
    path = config.CONFIG_DIR / "seeds" / "dashboard.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    segs = (yaml.safe_load(path.read_text()) if path.exists() else None) or []
    seg = {"video": body["video"], "event": body["event"], "t": float(body["t"]),
           "duration": int(body.get("duration", 30)), "fps": int(body.get("fps", 4)),
           "robots": [[r[0], *[int(v) for v in r[1:]]] for r in body["robots"]]}
    segs.append(seg)
    path.write_text(yaml.safe_dump(segs, sort_keys=False))
    if body.get("propagate", True):
        one = config.DATA_DIR / "tmp" / "seed_once.yaml"
        one.parent.mkdir(parents=True, exist_ok=True)
        one.write_text(yaml.safe_dump([seg]))
        with open(config.DATA_DIR / "label_jobs.log", "a") as log:
            JOBS["label"] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "label-seeds", str(one)],
                                             stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"ok": True, "segments": len(segs)}


@app.get("/api/label/suggest")
def suggest(video: str, t: float, event: str = ""):
    """Detector boxes for a frame (full-frame pixels) to pre-fill the labeling canvas."""
    from ..vision.detect import alliance_from_pixels

    _, im, (x, y, w, h) = _label_frame(video, t, event)
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
    fingerprint = (w, w.stat().st_mtime_ns, w.stat().st_size) if w else None
    if "det" not in _DET or _DET.get("w") != fingerprint:
        _DET.update(w=fingerprint, det=default_detector())
    return _DET["det"]


@app.post("/api/label/save")
def save_labels(body: dict):
    """Save hand-corrected boxes for one frame directly as a training example."""
    from ..train.sam_label import write_label
    try:
        t = float(body["t"])
        p, im, (x, y, w, h) = _label_frame(body["video"], t, body.get("event", ""))
        for r in body["robots"]:
            if (len(r) != 5 or r[0] not in ("red", "blue")
                    or not all(math.isfinite(float(n)) for n in r[1:])
                    or not (x <= r[1] < r[3] <= x + w and y <= r[2] < r[4] <= y + h)
                    or r[3] - r[1] < 6 or r[4] - r[2] < 6):
                raise ValueError("Box must be at least 6 pixels wide/high inside the calibrated field view")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid frame labels: {e}") from e
    boxes = [(r[0], r[1] - x, r[2] - y, r[3] - x, r[4] - y) for r in body["robots"]]
    stem = f"{p.stem}_u{float(body['t']):.1f}".replace(".", "_")
    write_label(im[y: y + h, x: x + w], boxes, stem, dataset=config.DATA_DIR / "datasets" / "hand")
    return {"ok": True, "stem": stem}


@app.post("/api/train")
def train():
    if (_job_status("train") and _job_status("train")["running"]) or _training_status()["running"]:
        raise HTTPException(409, "Detector training is already running")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / "train.log", "w") as log:
        JOBS["train"] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "train-robots"],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.get("/api/dataset")
def dataset():
    from ..train.sam_label import DATASET

    n = len(list((DATASET / "images").glob("*.jpg"))) if (DATASET / "images").exists() else 0
    hand = len(list((config.DATA_DIR / "datasets" / "hand" / "images").glob("*.jpg")))
    return {"images": n + hand, "hand_images": hand, "pseudo_images": n,
            "label_job": _job_status("label"), "train_job": _training_status()}


def _training_status():
    """CLI training remains visible after a dashboard restart."""
    from filelock import FileLock, Timeout

    running = False
    if config.DATA_DIR.exists():
        try:
            with FileLock(config.DATA_DIR / "detector-training.lock", timeout=0):
                pass
        except Timeout:
            running = True
    job = _job_status("train")
    running = running or bool(job and job["running"])
    out = {"running": running}
    candidates = sorted((config.DATA_DIR / "runs").glob("robots_*/results.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        try:
            rows = list(csv.DictReader(io.StringIO(candidates[0].read_text())))
            if rows:
                out.update(epoch=int(rows[-1]["epoch"]), run=candidates[0].parent.name)
        except (OSError, ValueError, KeyError):
            pass
    reports = sorted((config.DATA_DIR / "models" / "experiments").glob("*/evaluation.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        try:
            out["evaluation"] = json.loads(reports[0].read_text())
        except (OSError, ValueError):
            pass
    return out


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


@app.get("/api/annotated/{name}.jpg")
def annotated_thumb(name: str):
    """A still from 40 s into an annotated video, cut once and cached next to it: a gallery of live
    <video> thumbnails makes the browser run one HD decoder per card, and Firefox-based browsers stall."""
    video = DEMOS / f"{name}.mp4"
    if not re.fullmatch(r"[\w-]+_annotated", name) or not video.exists():
        raise HTTPException(404, "no such video")
    thumb = DEMOS / "thumbs" / f"{name}.jpg"
    if not thumb.exists() or thumb.stat().st_mtime < video.stat().st_mtime:
        thumb.parent.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "40", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=640:-2", str(thumb)], check=False, timeout=60)
    if not thumb.exists():
        raise HTTPException(404, "no thumbnail")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/annotated/{name}")
def annotated_file(name: str, download: bool = False):
    if not re.fullmatch(r"[\w-]+_annotated\.mp4", name) or not (DEMOS / name).exists():
        raise HTTPException(404, "no such video")
    # ?download=1 saves the file instead of playing it in the page
    return FileResponse(DEMOS / name, media_type="video/mp4", filename=name if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.post("/api/annotate/{match_key}")
def annotate(match_key: str):
    """Render the annotated video of a processed match in the background."""
    if not re.fullmatch(r"20\d\d[a-z0-9]+_(?:qm\d+|(?:ef|qf|sf|f)\d+m\d+)", match_key):
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
    with open(config.DATA_DIR / f"render_{match_key}.log", "w") as log:
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

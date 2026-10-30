"""Dashboard + API (FastAPI). Run with `autoscout serve`."""
from __future__ import annotations

import json
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
    return (STATIC / "index.html").read_text()


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
    return r


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
    if _job_status(event_key) and _job_status(event_key)["running"]:
        return {"started": False, "reason": "already running"}
    log = open(config.DATA_DIR / f"run_{event_key}.log", "w")
    JOBS[event_key] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "run-event", event_key,
                                        "--limit", str(limit)], stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.get("/api/run/{event_key}/log")
def run_log(event_key: str):
    p = config.DATA_DIR / f"run_{event_key}.log"
    return Response(p.read_text()[-6000:] if p.exists() else "", media_type="text/plain")


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
    from ..vision.detect import RobotDetector, find_weights

    w = find_weights()
    if _DET.get("w") != w:
        _DET.update(w=w, det=RobotDetector(w))
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
    log = open(config.DATA_DIR / "train.log", "w")
    JOBS["train"] = subprocess.Popen([sys.executable, "-m", "autoscout.cli", "train-robots"],
                                     stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
    return {"started": True}


@app.get("/api/dataset")
def dataset():
    from ..train.sam_label import DATASET

    n = len(list((DATASET / "images").glob("*.jpg"))) if (DATASET / "images").exists() else 0
    return {"images": n, "label_job": _job_status("label"), "train_job": _job_status("train")}

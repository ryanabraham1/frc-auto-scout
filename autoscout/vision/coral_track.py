"""Coral placements from the CORAL detector and the fixed REEF close-up cameras.

White-pixel changes say that something changed near a REEF. This module says what: a detected CORAL
that appears and then stays, put on a branch (and so a REEF face and level) with the solved close-up
camera. A held coral moves and a robot's white parts come and go, so neither survives the persistence
check. Near and far faces can share pixels: a CORAL's size says how far away it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import field3d as F

CORAL_LEN = 0.30  # m
LINK_PX = 14.0  # px: detections of one resting coral
HOLD_S, HOLD_FRAC = 2.0, 0.6  # it must be seen this much of the time this long after it first appears
MIN_SEEN = 20  # detections in the whole match
END_S = 25.0  # scored coral stays: last seen this close to the end (the last ~20 s show a cropped close-up)
PANEL_Y = 645  # top of the close-up band in 1080p
PANEL_X = {"blue": 4, "red": 960}
MAX_PX, MIN_REACH_PX = 1.2, 45.0  # a branch is a candidate within this many coral lengths (floor: a tip is a coral end, not its centre)


def load_cameras(path: Path) -> dict:
    """{alliance: (pinhole camera, panel origin)} from configs/events/<event>_reef_cams.json ({}: not solved)."""
    if not path.exists():
        return {}
    out = {}
    for a, c in json.loads(path.read_text()).items():
        if a in PANEL_X:
            w, h = c["size"]
            K = np.array([[c["f"], 0, w / 2], [0, c["f"], h / 2], [0, 0, 1.0]])
            out[a] = ((K, np.array(c["rvec"], float), np.array(c["tvec"], float), 0.0),
                      tuple(c.get("origin", (PANEL_X[a], PANEL_Y))))
    return out


def rests(obs: list, alliance: str, top: float = PANEL_Y) -> list[dict]:
    """Coral at rest in a close-up: {t, x, y, size, seen} from [[t, [[x0, y0, x1, y1, conf], ...]], ...]."""
    lo, hi = (0, 958) if alliance == "blue" else (958, 1920)
    clusters: list[dict] = []
    for t, boxes in obs:
        for x0, y0, x1, y1, c in boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if y0 < top or not lo <= cx < hi:
                continue
            near = [(float(np.hypot(k["x"] - cx, k["y"] - cy)), i) for i, k in enumerate(clusters)]
            near = [n for n in near if n[0] < LINK_PX]
            if near:
                k = clusters[min(near)[1]]
                k["obs"].append((t, max(x1 - x0, y1 - y0)))
                n = min(len(k["obs"]), 10)
                k["x"] += (cx - k["x"]) / n
                k["y"] += (cy - k["y"]) / n
            else:
                clusters.append({"x": cx, "y": cy, "obs": [(t, max(x1 - x0, y1 - y0))]})
    times = np.array([t for t, _ in obs])
    out = []
    for k in clusters:
        seen = np.array([o[0] for o in k["obs"]])
        if len(seen) < MIN_SEEN or seen[-1] < times[-1] - END_S:
            continue
        for t in seen:
            frames = ((times >= t) & (times < t + HOLD_S)).sum()
            if ((seen >= t) & (seen < t + HOLD_S)).sum() >= HOLD_FRAC * max(frames, 1):
                out.append({"t": float(t), "x": k["x"], "y": k["y"], "seen": len(seen),
                            "size": float(np.median([o[1] for o in k["obs"]]))})
                break
    return sorted(out, key=lambda r: r["t"])


def _faces(keys: list, cost: np.ndarray) -> dict:
    """{level: {face: chance}}: near and far tips of a level can share pixels, so a coral at that level
    could be on any face whose tip fits about as well."""
    out: dict = {}
    for level in ("L1", "L2", "L3", "L4"):
        w: dict = {}
        best = min((c for k, c in zip(keys, cost) if k[1] == level), default=np.inf)
        if not np.isfinite(best):
            continue
        for k, c in zip(keys, cost):
            if k[1] == level and np.isfinite(c):
                face = "ABCDEFGHIJKL".index(k[0]) // 2
                w[face] = max(w.get(face, 0.0), float(np.exp(-0.5 * (c - best))))
        tot = sum(w.values())
        out[level] = {str(f): round(v / tot, 3) for f, v in w.items() if v / tot >= 0.02}
    return out


def placements(obs: list, cam, alliance: str, origin=None) -> list[dict]:
    """Each branch's first resting coral: {elem, t, x, y, branch, level, face}. A coral seen again after
    a robot uncovers it is the same coral, so a branch or trough spot is only placed once."""
    import cv2

    origin = origin or (PANEL_X[alliance], PANEL_Y)
    tips = F.branch_tips(alliance)
    keys = list(tips)
    P = F.project(cam, [tips[k] for k in keys]) + np.array(origin)
    K, rv, tv, _ = cam
    R, _ = cv2.Rodrigues(rv)
    pos = (-R.T @ tv).ravel()
    expected = np.array([K[0, 0] * CORAL_LEN / np.linalg.norm(tips[k] - pos) for k in keys])
    taken, out = set(), []
    for r in rests(obs, alliance, origin[1]):
        d = np.hypot(P[:, 0] - r["x"], P[:, 1] - r["y"])
        cost = (d / 15.0) ** 2 + (np.log(max(r["size"], 1.0) / expected) / 0.3) ** 2
        cost[d > max(MIN_REACH_PX, MAX_PX * r["size"])] = np.inf
        for i in np.argsort(cost):
            if not np.isfinite(cost[i]):
                break
            if keys[i] in taken:
                continue
            taken.add(keys[i])
            letter, level = keys[i]
            out.append({"elem": f"{alliance}_reef@coral", "t": round(r["t"], 2), "x": round(r["x"], 1),
                        "y": round(r["y"], 1), "branch": letter, "level": level,
                        "face": "ABCDEFGHIJKL".index(letter) // 2, "cost": round(float(cost[i]), 2),
                        "faces": _faces(keys, cost)})
            break
    return out


def weights() -> Path | None:
    from .. import config

    for p in (config.DATA_DIR / "models/coral/reef.pt", config.ROOT / "models/autoscout-coral-r2.pt"):
        if p.exists():
            return p
    return None


def scan(video, t0: float, t1: float, normalize, hz: float = 5.0, conf: float = 0.2) -> list:
    """CORAL boxes in the whole broadcast frame through the match: [[t, [[x0, y0, x1, y1, conf], ...]], ...]."""
    from ultralytics import YOLO

    from .. import config

    model = YOLO(str(weights()))
    device = config.device()
    out: list = []
    frames: list = []
    times: list = []

    def flush():
        for t, res in zip(times, model.predict(frames, device=device, imgsz=1920, conf=conf, verbose=False)):
            b = res.boxes
            out.append([round(t, 3), [[round(v, 1) for v in x] + [round(c, 3)]
                                      for x, c in zip(b.xyxy.tolist(), b.conf.tolist())]])
        frames.clear()
        times.clear()

    for t, frame in video.iter_frames(t0, t1, step=max(1, round(video.fps / hz))):
        frames.append(normalize(frame))
        times.append(t)
        if len(frames) == 4:
            flush()
    if frames:
        flush()
    return out

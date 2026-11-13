"""Fit every saved launch of processed matches as a 3D arc and cache the result (data/flight3d/<match>.json).

Also measures, from the tags, how far each match's picture sits from the one the event camera was solved
on. Needs configs/events/<event>_cam3d.json (scripts/calibrate_camera3d.py).

usage: python scripts/flight3d_cache.py MATCH_KEY... [--workers 6] [--force]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402

OUT = config.DATA_DIR / "flight3d"


def one(match_key: str, force: bool = False) -> str:
    import cv2

    from autoscout.field import FieldModel
    from autoscout.vision import camera3d as c3
    from autoscout.vision.shooters import fit_launch, flight_record
    from scripts.calibrate_camera3d import video_path

    out = OUT / f"{match_key}.json"
    if out.exists() and not force:
        return f"{match_key}: cached"
    t = time.time()
    res = json.loads((config.DATA_DIR / "results" / f"{match_key}.json").read_text())
    event = res["event_key"]
    cam = c3.load_event_camera(event)
    if cam is None:
        return f"{match_key}: no camera for {event}"
    fm = FieldModel.load(res["year"])
    shift = (0.0, 0.0, 0)
    try:
        cap = cv2.VideoCapture(video_path(res))
        c = res["clock"]
        frames = []
        for vt in np.linspace(c["auto_end_vt"] - 12, c["teleop_end_vt"] - 10, 6):
            cap.set(cv2.CAP_PROP_POS_MSEC, vt * 1000)
            ok, f = cap.read()
            if ok:
                frames.append(f)
        region = json.loads(c3.event_camera_path(event).read_text())["wide"]["region"]
        shift = c3.frame_shift(event, frames, region)
    except FileNotFoundError:
        pass
    if shift[2]:
        cam = cam.shifted(shift[0], shift[1])
    flights = {}
    for piece, launches in res["launches"].items():
        flights[piece] = [flight_record(fit_launch(cam, L, fm, piece)) for L in launches]
    n = sum(len(v) for v in flights.values())
    ok = sum(1 for v in flights.values() for f in v if f and "rel_xy" in f)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"match_key": match_key, "camera": cam.to_dict(), "shift": list(shift), "flights": flights}))
    return f"{match_key}: {ok}/{n} flights with a release, shift {shift[0]:+.1f},{shift[1]:+.1f} px ({shift[2]} tags), {time.time() - t:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matches", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(one, a.matches, [a.force] * len(a.matches)):
            print(msg, flush=True)

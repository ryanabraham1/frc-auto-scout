"""Solve an event's wide-shot camera in 3D from the field's AprilTags (autoscout/vision/camera3d.py).

Tags big enough to decode are found by themselves. In a wide shot of the whole field the wall tags are
often too small to decode but easy to see: give each one's centre with --click ID:X,Y (full-frame px),
which pins down the camera's distance and focal length (hub tags alone all lie in one plane).

usage: python scripts/calibrate_camera3d.py MATCH_KEY [--click 13:142.6,528 ...] [--bottom 645] [--draw out.jpg]
Writes configs/events/<event>_cam3d.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402
from autoscout.field import Calibration, FieldModel  # noqa: E402
from autoscout.vision import camera3d as c3  # noqa: E402


def video_path(result: dict) -> str:
    v = str(result["video"])
    for p in (Path(v), config.DATA_DIR / "videos" / f"{v}.mp4", config.DATA_DIR / "integration" / "clips" / f"{v}.mp4"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(v)


def match_frames(match_key: str, n: int = 30) -> tuple[dict, list]:
    d = json.loads((config.DATA_DIR / "results" / f"{match_key}.json").read_text())
    cap = cv2.VideoCapture(video_path(d))
    c = d["clock"]
    out = []
    for t in np.linspace(c["auto_end_vt"] - 15, c["teleop_end_vt"] - 5, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            out.append(f)
    return d, out


def draw(cam, img, fm, year, seen=()):
    L, W = fm.length, fm.width

    def line(a, b, col):
        P = cam.project([np.array(a) + (np.array(b) - np.array(a)) * s for s in np.linspace(0, 1, 30)]).astype(np.int32)
        cv2.polylines(img, [P], False, col, 1)

    edges = [((0, 0, 0), (L, 0, 0)), ((L, 0, 0), (L, W, 0)), ((L, W, 0), (0, W, 0)), ((0, W, 0), (0, 0, 0)),
             ((L / 2, 0, 0), (L / 2, W, 0))]
    edges += [((x, y, 0), (x, y, 2)) for x in (0, L) for y in (0, W)]
    for a, b in edges:
        line(a, b, (0, 255, 255))
    tags, _ = c3.tag_layout(year)
    for i in tags:
        cv2.polylines(img, [cam.project(c3.tag_corners(year, i)).astype(np.int32)], True,
                      (0, 255, 0) if i in seen else (0, 140, 255), 1)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match")
    ap.add_argument("--click", action="append", default=[], help="ID:X,Y centre of a tag too small to decode")
    ap.add_argument("--drop", type=int, action="append", default=[], help="decoded tag id to ignore (a misread)")
    ap.add_argument("--bottom", type=int, help="last frame row of the wide shot (default: bottom of the calibrated view)")
    ap.add_argument("--draw", type=Path)
    a = ap.parse_args()
    res, frames = match_frames(a.match)
    event, year = res["event_key"], res["year"]
    cal, fm = Calibration.load(event), FieldModel.load(year)
    h, w = frames[0].shape[:2]
    bottom = a.bottom or min(h, cal.view[1] + cal.view[3] + 3)
    region = (0, 0, w, bottom)
    obs = {i: c for i, c in c3.observe_tags(frames, region).items() if i not in a.drop}
    tags, _ = c3.tag_layout(year)
    clicks = {int(c.split(":")[0]): [float(v) for v in c.split(":")[1].split(",")] for c in a.click}
    extra = [(tags[i][:3], px) for i, px in clicks.items()]
    cam = c3.solve_best(obs, (w, h), year, extra=extra, centre=(w / 2, bottom / 2))
    if cam is None:
        sys.exit(f"not enough tags: decoded {sorted(obs)}, clicked {sorted(clicks)}")
    print(f"decoded {sorted(obs)} clicked {sorted(clicks)}: rms {cam.rms:.2f} px over {cam.n_points} points, "
          f"f {cam.f:.0f}, k1 {cam.k1:.3f}, camera at {np.round(cam.centre, 2).tolist()}")
    out = {"wide": {"camera": cam.to_dict(), "from_match": a.match, "region": list(region),
                    "ref_tags": {str(i): np.round(c, 2).tolist() for i, c in obs.items()},
                    "clicks": {str(i): v for i, v in clicks.items()}}}
    p = c3.event_camera_path(event)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    if a.draw:
        img = draw(cam, np.median(np.array(frames), axis=0).astype(np.uint8), fm, year, obs)
        cv2.imwrite(str(a.draw), img[:bottom], [cv2.IMWRITE_JPEG_QUALITY, 92])

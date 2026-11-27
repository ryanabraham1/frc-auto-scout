"""Tile raw frames around one moment of a 2025 match, to check a call on footage in one picture.

Each tile is a wide-shot crop of the alliance's REEF above its close-up panel. Nothing from the scouter
is drawn unless --labels is given (then the wide-shot robot boxes carry the team we call them), so the
sheet can be read without knowing the answer first. Read bumpers and elevators; one sheet per play.

usage: python scripts/play_sheet.py MATCH ALLIANCE VIDEO_SECONDS OUT.jpg [--dts=-3,-2,-1,0,1,2] [--labels]
"""
import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402
from autoscout.video import VIDEO_DIR, Video  # noqa: E402
from autoscout.vision import camera3d, field3d as F  # noqa: E402

PANEL_Y, PANEL_H, PANEL_W = 645, 370, 958  # the close-up band of the 1080p two-panel layout

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match")
    ap.add_argument("alliance", choices=("red", "blue"))
    ap.add_argument("t", type=float, help="video time (s), as in the result's events")
    ap.add_argument("out")
    ap.add_argument("--dts", default="-3,-2,-1,0,1,2", help="seconds around t, one tile each")
    ap.add_argument("--labels", action="store_true", help="draw the wide-shot robot boxes with the team we call them")
    a = ap.parse_args()
    r = json.loads((config.DATA_DIR / "results" / f"{a.match}.json").read_text())
    video = Video(str(VIDEO_DIR / f"{r['video']}.mp4"))
    cam = camera3d.load_event_camera(r["event_key"])
    shift = config.DATA_DIR / "cam3d_shift" / f"{a.match}.json"
    if cam is not None and shift.exists():
        dx, dy, n = json.loads(shift.read_text())
        cam = cam.shifted(dx, dy) if n else cam
    centre = cam.project([[*F.REEF_CENTRE[a.alliance], 1.0]])[0] if cam is not None else (960.0, 320.0)
    spans = r.get("identity_timeline") or {}
    tiles = []
    for dt in (float(x) for x in a.dts.split(",")):
        frame = video.frame_at(a.t + dt)
        if frame is None:
            continue
        if a.labels:
            for slot, track in r["raw_tracks"].items():
                row = min(track, key=lambda q: abs(q[0] - (a.t + dt)))
                if abs(row[0] - (a.t + dt)) < 0.15 and row[3] and row[4]:
                    team = next((tm for s, e, tm in spans.get(slot, []) if s <= row[0] < e), r["identity"].get(slot))
                    x0, y0, x1, y1 = (int(v) for v in row[4])
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255) if slot.startswith("red") else (255, 80, 0), 1)
                    cv2.putText(frame, str(team), (x0, y0 - 3), 0, 0.45, (0, 255, 255), 1)
        x0 = int(np.clip(centre[0] - 230, 0, frame.shape[1] - 460))
        y0 = int(np.clip(centre[1] - 150, 0, PANEL_Y - 255))
        wide = cv2.resize(frame[y0:y0 + 250, x0:x0 + 460], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        px = 0 if a.alliance == "blue" else frame.shape[1] - PANEL_W - 2
        close = cv2.resize(frame[PANEL_Y:PANEL_Y + PANEL_H, px:px + PANEL_W], (920, 355))
        tile = np.vstack([wide, close])
        cv2.putText(tile, f"{a.match} {a.alliance} t={a.t + dt:.1f} ({dt:+.0f})", (8, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        sys.exit("no frames at that time")
    cols = min(3, len(tiles))
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    k = 1900 / sheet.shape[1]
    cv2.imwrite(a.out, cv2.resize(sheet, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", a.out)

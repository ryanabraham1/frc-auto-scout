"""Contact sheet of attributed scoring events for checking attribution by eye.

Each tile is the video frame at the event, cropped around it, with every robot box labelled by the
team the pipeline thinks it is, the launch path (green) and the team it credited (title).

usage: python scripts/event_sheet.py MATCH_KEY VIDEO OUT.jpg [kinds=shot,feed] [n=24] [seed=0]
"""
import json
import random
import sys

import cv2
import numpy as np

from autoscout import games
from autoscout.field import Calibration
from autoscout.games.base import GamePlugin, MatchContext
from autoscout.pipeline import RESULTS_DIR
from autoscout.video import Video

key, vid, out = sys.argv[1:4]
kinds = set((sys.argv[4] if len(sys.argv) > 4 else "shot,feed").split(","))
n = int(sys.argv[5]) if len(sys.argv) > 5 else 24
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
d = json.loads((RESULTS_DIR / f"{key}.json").read_text())
c = d["calibration"]
cal = Calibration(event_key=d["event_key"], year=d["year"], view=tuple(c["view"]),
                  points={k: tuple(v) for k, v in c["points"].items()})
tracks = {k: [tuple(r) for r in v] for k, v in d["raw_tracks"].items()}
ctx = MatchContext(match=None, field=games.get(d["year"]).field, cal=cal, clock=None, timeline=None,
                   tracks=tracks, identity=d["identity"], identity_conf={}, launches={})
ctx.identity_tl = {k: [tuple(x) for x in v] for k, v in d.get("identity_timeline", {}).items()}
launch_pts = {}
for L in d["launches"].get("fuel", d["launches"].get("algae", [])):
    launch_pts[round(L["t0"], 2)] = L["pts"]
evs = [e for e in d["events"] if e.get("kind") in kinds and e.get("team")]
random.Random(seed).shuffle(evs)
evs = sorted(evs[:n], key=lambda e: e["t"])
v = Video(vid)
tiles = []
for e in evs:
    t = e["t"] - (1.2 if e["kind"] == "scored" else 0.0)  # score posts ~1 s after the release
    f = v.frame_at(t)
    if f is None:
        continue
    for slot in tracks:
        rb = GamePlugin.robot_box(ctx, slot, t)
        if rb is None:
            continue
        (x0, y0, x1, y1), obs = rb
        col = (0, 0, 255) if slot.startswith("red") else (255, 0, 0)
        cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), col, 2 if obs else 1)
        cv2.putText(f, str(ctx.team_at(slot, t)), (int(x0), int(y1) + 16), 0, 0.6, (255, 255, 255), 2)
    pts = launch_pts.get(round(t, 2))
    if pts:
        p = np.array([(q[1], q[2]) for q in pts], np.int32)
        cv2.polylines(f, [p], False, (0, 255, 0), 2)
        cx, cy = p[0]
    elif "start" in e:
        cx, cy = e["start"]
    else:  # centre on the credited robot
        slot = ctx.slot_of(e["team"], t)
        rb = GamePlugin.robot_box(ctx, slot, t) if slot else None
        cx, cy = ((rb[0][0] + rb[0][2]) / 2, rb[0][1]) if rb else (960, 540)
    x0 = int(np.clip(cx - 260, 0, f.shape[1] - 520))
    y0 = int(np.clip(cy - 180, 0, f.shape[0] - 360))
    tile = f[y0: y0 + 360, x0: x0 + 520].copy()
    cv2.rectangle(tile, (0, 0), (520, 22), (0, 0, 0), -1)
    cv2.putText(tile, f"t={e['t']:.1f} {e['kind']} -> {e['team']} n={e.get('n')}", (4, 16), 0, 0.55, (0, 255, 255), 1)
    tiles.append(tile)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros((360, 520, 3), np.uint8))
grid = np.vstack([np.hstack(tiles[i: i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(out, cv2.resize(grid, None, fx=0.75, fy=0.75))
print(len(evs), "events")

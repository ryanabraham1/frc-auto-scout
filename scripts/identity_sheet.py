"""Contact sheet of a processed match's robot tracklets grouped by the team they were labelled as,
for checking identity by eye (bumper numbers are visible in most thumbnails).

usage: python scripts/identity_sheet.py MATCH_KEY OUT.jpg [min_obs]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from autoscout.pipeline import RESULTS_DIR

key, out = sys.argv[1], sys.argv[2]
min_obs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
d = json.loads((RESULTS_DIR / f"{key}_tracklets.json").read_text())
rows = {}
for t in d["tracklets"]:
    p = RESULTS_DIR / f"{key}_robots" / f"{t['id']}.jpg"
    if len(t["obs"]) < min_obs or not p.exists():
        continue
    rows.setdefault(t["team"], []).append((len(t["obs"]), t["id"], p))
H, W = 110, 150
lines = []
for team in sorted(rows, key=lambda x: (x is None, x or 0)):
    items = sorted(rows[team], reverse=True)[:12]
    tiles = []
    for n, tid, p in items:
        im = cv2.imread(str(p))
        im = cv2.resize(im, (W, H))
        cv2.putText(im, f"#{tid} n{n}", (2, 14), 0, 0.45, (0, 255, 0), 1)
        tiles.append(im)
    while len(tiles) < 12:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    label = np.zeros((H, 90, 3), np.uint8)
    cv2.putText(label, str(team), (4, 60), 0, 0.8, (255, 255, 255), 2)
    lines.append(np.hstack([label] + tiles))
cv2.imwrite(out, np.vstack(lines))
print({k: len(v) for k, v in rows.items()})

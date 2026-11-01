"""How often is a scoring robot even visible? For each live score increase, is any robot of that
alliance actually detected (not coasted) in its own ALLIANCE ZONE during the 3 s before it?

usage: python scripts/visibility.py EVENT [EVENT ...]
"""
import json
import sys

import numpy as np

from autoscout import games
from autoscout.pipeline import RESULTS_DIR

for ev in sys.argv[1:]:
    tot = vis = 0
    pts_tot = pts_vis = 0
    by_side = {"red": [0, 0], "blue": [0, 0]}
    for p in sorted(RESULTS_DIR.glob(f"{ev}_qm*.json")):
        if p.stem.count("_") != 1:
            continue
        d = json.loads(p.read_text())
        if d.get("year") != 2026:
            continue
        fm = games.get(2026).field
        tr = {k: np.array([(r[0], r[1], r[2], r[3]) for r in v], float) for k, v in d["raw_tracks"].items()}
        t = np.array(d["timeline"]["t"])
        clk = d["clock"]
        for a in ("red", "blue"):
            x0, _, x1, _ = fm.zones[f"{a}_alliance"]
            s = np.array(d["timeline"][a], float)
            for i in range(1, len(s)):
                dp = s[i] - s[i - 1]
                if not (0 < dp < 10) or not (clk["auto_end_vt"] - clk["auto_len"] < t[i] < clk["teleop_end_vt"] + 1):
                    continue
                seen = False
                for k in (f"{a}0", f"{a}1", f"{a}2"):
                    R = tr.get(k)
                    if R is None or not len(R):
                        continue
                    m = (R[:, 0] > t[i] - 3) & (R[:, 0] < t[i] - 0.3) & (R[:, 3] > 0)
                    m &= (R[:, 1] >= x0 - 0.5) & (R[:, 1] <= x1 + 0.5)
                    seen |= bool(m.any())
                tot += 1
                vis += seen
                pts_tot += dp
                pts_vis += dp * seen
                by_side[a][0] += 1
                by_side[a][1] += seen
    print(f"{ev}: a robot of the scoring alliance is visible in its zone before {vis}/{tot} = {vis / max(tot, 1):.0%} of "
          f"score increases ({pts_vis / max(pts_tot, 1):.0%} of points); red {by_side['red'][1]}/{by_side['red'][0]}, "
          f"blue {by_side['blue'][1]}/{by_side['blue'][0]}")

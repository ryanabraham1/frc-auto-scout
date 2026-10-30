"""Independent check of per-robot fuel: compare video estimates with each team's fuel OPR.

Fuel OPR = least-squares per-team contribution to the official alliance hub counts over all of an
event's qualification matches (from TBA). It is a statistical average, not a per-match truth, but a
video system that attributes fuel correctly should correlate with it across teams.
Usage: python scripts/fuel_opr_check.py 2026dal data/results/2026dal_*.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from autoscout.tba import TBA, MatchInfo  # noqa: E402

event, files = sys.argv[1], [f for f in sys.argv[2:] if f.endswith(".json")]
ms = [MatchInfo.from_tba(m) for m in TBA().event_matches(event) if m["comp_level"] == "qm" and m.get("score_breakdown")]
teams = sorted({t for m in ms for t in m.red + m.blue})
idx = {t: i for i, t in enumerate(teams)}
A, b = [], []
for m in ms:
    for a in ("red", "blue"):
        row = np.zeros(len(teams))
        for t in m.teams(a):
            row[idx[t]] = 1
        A.append(row)
        b.append(m.breakdown[a]["hubScore"]["totalCount"])
opr = np.linalg.lstsq(np.array(A), np.array(b, float), rcond=None)[0]
ours, theirs, shots = [], [], []
print(f"{'match':16s} team   video-scored  video-shots  fuel-OPR")
for f in files:
    r = json.load(open(f))
    for t, m in r["teams"].items():
        t = int(t)
        ours.append(m["fuel_scored"]); shots.append(m["fuel_shot"]); theirs.append(opr[idx[t]])
        print(f"{r['match_key']:16s} {t:5d}  {m['fuel_scored']:8.0f}  {m['fuel_shot']:10.0f}  {opr[idx[t]]:9.1f}")
ours, theirs, shots = map(np.array, (ours, theirs, shots))
print(f"\nn={len(ours)} team-matches; correlation(video scored, fuel OPR) = {np.corrcoef(ours, theirs)[0, 1]:.2f}; "
      f"(video shots, OPR) = {np.corrcoef(shots, theirs)[0, 1]:.2f}")
# baseline: equal split of the alliance total (what you'd get with no video at all)

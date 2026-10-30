"""Compare detected fuel launches with the official per-period hub counts (2026).

For every processed match and alliance, per period: official fuel counted by the field,
detected shots attributed to the alliance's robots, and detected flights seen entering the hub
(regardless of robot attribution). Prints per-match lines and overall ratios/correlations.
Usage: python scripts/fuel_accuracy.py data/results/2026*.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from autoscout.games.rebuilt2026 import PERIODS  # noqa: E402

rows = []
for f in sys.argv[1:]:
    if f.endswith("_debug.mp4") or "_blobs" in f:
        continue
    r = json.load(open(f))
    if r.get("year") != 2026 or not r.get("launches"):
        continue
    clock = r["clock"]
    ts = clock["teleop_end_vt"]
    bounds = {"auto": (clock["auto_end_vt"] - clock["auto_len"], clock["auto_end_vt"] + 3),
              "transition": (ts - 140, ts - 130), "shift1": (ts - 130, ts - 105), "shift2": (ts - 105, ts - 80),
              "shift3": (ts - 80, ts - 55), "shift4": (ts - 55, ts - 30), "endgame": (ts - 30, ts + 3.5)}
    hub_in = {a: {p: 0.0 for p in PERIODS} for a in ("red", "blue")}
    for L in r["launches"]["fuel"]:
        tgt = L.get("target")
        if not tgt:
            continue
        a = tgt.split("_")[0]
        t = L["t0"] + 1.0
        for p, (lo, hi) in bounds.items():
            if lo <= t < hi:
                hub_in[a][p] += L["n"]
    for a in ("red", "blue"):
        off = {p: 0 for p in PERIODS}
        # official counts from TBA breakdown are already reflected in team splits; recompute sums
        for t, m in r["teams"].items():
            if m["alliance"] == a:
                for p in PERIODS:
                    off[p] += m["fuel_scored_by_period"][p]
        sh = {p: 0.0 for p in PERIODS}
        for e in r["events"]:
            if e.get("alliance") == a and e.get("kind") == "shot" and e.get("period") in sh:
                sh[e["period"]] += e["n"]
        for p in PERIODS:
            rows.append((r["match_key"], a, p, off[p], sh[p], hub_in[a][p]))

R = np.array([(o, s, h) for *_, o, s, h in rows], dtype=float)
by_match = {}
for mk, a, p, o, s, h in rows:
    k = (mk, a)
    by_match.setdefault(k, [0, 0, 0])
    by_match[k][0] += o; by_match[k][1] += s; by_match[k][2] += h
print(f"{'match':18s} all  official  shots(ratio)   seen-into-hub(ratio)")
for (mk, a), (o, s, h) in sorted(by_match.items()):
    print(f"{mk:18s} {a:4s} {o:6.0f}   {s:6.0f} ({s / max(o, 1):.2f})   {h:6.0f} ({h / max(o, 1):.2f})")
m = R[:, 0] > 0
if m.sum() > 2:
    print(f"\nper-period correlation with official: shots r={np.corrcoef(R[m, 0], R[m, 1])[0, 1]:.2f}, "
          f"into-hub r={np.corrcoef(R[m, 0], R[m, 2])[0, 1]:.2f}  (n={m.sum()} alliance-periods)")
    print(f"overall ratio: shots {R[:, 1].sum() / R[:, 0].sum():.2f}, into-hub {R[:, 2].sum() / R[:, 0].sum():.2f}")

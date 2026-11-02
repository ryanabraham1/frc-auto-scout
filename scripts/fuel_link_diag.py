"""usage: python scripts/fuel_link_diag.py MATCH OUT.json

Why tracked makes don't link to a robot: per make, how it links now and what the robots were doing."""
import json, pickle, sys
import numpy as np
from collections import Counter
sys.path.insert(0, ".")
from autoscout import pipeline
from autoscout.games import rebuilt2026 as R
from autoscout.games.base import GamePlugin
from autoscout.tba import TBA
key = sys.argv[1]
cap = {}
orig = R.Rebuilt2026.analyze
def an(self, ctx):
    cap["ctx"], cap["self"] = ctx, self
    return orig(self, ctx)
R.Rebuilt2026.analyze = an
pipeline.reanalyze(key, TBA(), save=False)
ctx, P = cap["ctx"], cap["self"]
hz = P.hub_top_zone(ctx)
sh = P.history_shooters(ctx, ctx.launches["fuel"], hz)
links = P.link_hub_flights(ctx, ctx.launches["fuel"], sh, hz)
rows, why = [], Counter()
for L in ctx.launches["fuel"]:
    if not (L.target and L.target.endswith("_hub")):
        continue
    a = L.target[:-4]
    at_hub = hz(np.array([L.start[0]]), np.array([L.start[1]]))[0]
    rising = getattr(L, "rising", True)
    s = links.get(id(L))
    how = "linked"
    ds = P.launch_distances(ctx, L)
    near = sorted(((v[0], k, v[1]) for k, v in ds.items()))[:3]
    if s is None:
        kind = "none"
    elif s[:-1] != a:
        kind = "wrong_alliance"
    else:
        kind = "ok"
    why[(kind, how, "at_hub" if at_hub else ("rising" if rising else "descent"))] += L.n_pieces
    rows.append(dict(t=L.t0, a=a, kind=kind, how=how, at_hub=bool(at_hub), rising=rising, start=L.start,
                     n=L.n_pieces, boxes={k: [list(map(float, b[0])), b[1]] for k in ctx.tracks if (b := P.robot_box(ctx, k, L.t0))},
                     near=[(round(d, 2), k, o) for d, k, o in near], slot=s,
                     pts=[tuple(p) for p in L.pts]))
for k, v in why.most_common():
    print(f"{v:5.0f} {k}")
json.dump(rows, open(sys.argv[2], "w"), default=float)

"""Conservative rejoining of a rising release and a later hub-entry fragment."""
from dataclasses import replace

import numpy as np

from .launches import fit_flight


def join_hub_fragments(launches, hub_zone, shooters, pixel_scale=1.0):
    """Return flights and (source, entry, merged) triples without mutating inputs.

    Require a known same-alliance shooter, a good rising ballistic fit, and a
    unique continuation in both directions. One-to-one matches prevent a single
    release being reused for several balls in a volley. Short/noisy entry arcs
    never supply the extrapolation. Merging consumes both originals, so one ball
    contributes once, at its original release time and identity.
    """
    sources, entries = [], []
    for flight in launches:
        at_hub = bool(hub_zone(np.array([flight.start[0]]), np.array([flight.start[1]]))[0])
        if at_hub and flight.target in ('red_hub', 'blue_hub') and flight.n_pieces == 1:
            entries.append(flight)
        elif not at_hub and flight.rising and flight.target is None and flight.n_pieces == 1:
            slot = shooters.get(id(flight))
            p = np.asarray(flight.pts, dtype=float)
            if slot is None or len(p) < 6 or p[-1,0] - p[0,0] < .2 or not np.isfinite(p).all():
                continue
            if np.any(np.diff(p[:,0]) <= 0):
                continue
            cx, cy, rms = fit_flight(p)
            if rms > 4 * pixel_scale or cy[0] <= 20 * pixel_scale or np.polyval(np.polyder(cy), 0) >= 0:
                continue
            sources.append((flight, slot, cx, cy))
    costs = {}
    for i, (source, slot, cx, cy) in enumerate(sources):
        for j, entry in enumerate(entries):
            gap = entry.t0 - source.t1
            if not 0 <= gap <= .8 or entry.t0 - source.t0 > 2 or slot[:-1] != entry.target[:-4]:
                continue
            p = np.asarray(entry.pts[:4], dtype=float)
            if len(p) < 2 or p.shape[1] != 3 or not np.isfinite(p).all():
                continue
            t = p[:,0] - source.t0
            err = np.hypot(np.polyval(cx,t)-p[:,1], np.polyval(cy,t)-p[:,2])
            if np.max(err) > 12 * pixel_scale:
                continue
            # The complete arc must also remain ballistic after joining.
            _, joined_y, rms = fit_flight(np.vstack([source.pts,entry.pts]))
            if rms > 4 * pixel_scale or joined_y[0] <= 20 * pixel_scale:
                continue
            costs[i,j] = float(np.mean(err))
    accepted = []
    for (i,j), cost in sorted(costs.items(), key=lambda kv: kv[1]):
        rivals = [v for (a,b),v in costs.items() if (a==i or b==j) and (a,b)!=(i,j)]
        if rivals and min(rivals)-cost < 5 * pixel_scale:
            continue
        source = sources[i][0]
        entry = entries[j]
        pts = sorted([*source.pts,*entry.pts], key=lambda p:p[0])
        merged = replace(source, t1=entry.t1, end=entry.end, target=entry.target, entry_t=entry.entry_t,
                         pts=pts, apex=tuple(min(pts,key=lambda p:p[2])[1:3]))
        accepted.append((source,entry,merged))
    replacements = {id(s): m for s,e,m in accepted}
    consumed = {id(e) for s,e,m in accepted}
    return [replacements.get(id(f),f) for f in launches if id(f) not in consumed], accepted

"""Robot tracks in the close-up camera views, so a robot keeps its name between bumper reads.

A close-up box is labelled only when its bumper number reads in that very frame, and a robot raising
its elevator at the REEF is exactly when its bumper turns away or is blocked. Each close-up's boxes are
linked frame to frame (nearest box of a similar size and the same bumper colour); every box on a track
then takes the team of the nearest confident read on that same track, earlier or later in time. A track
with no read at all takes the team of the wide-shot robot it was fused with (vision/fusion.py), so the
three views name each other's robots.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

LINK_GAP_S = 1.2  # s a close-up track survives without a box (the detector misses a frame or two)
LINK_DIST = 0.9  # box widths a robot's box centre may move between linked close-up boxes
SIZE_RATIO = 2.0  # max ratio of linked box widths (a robot doesn't change size abruptly)
READ_MIN = 0.5  # read confidence for a box to name its track
LABEL_REACH_S = 3.0  # s along a track a read names other boxes (a long track may still hold a bad link)


def link(boxes: list) -> list[int]:
    """Track id per close-up box ([t, view, x0, y0, x1, y1, alliance, team|None, conf])."""
    from scipy.optimize import linear_sum_assignment

    track = [-1] * len(boxes)
    by_view = defaultdict(lambda: defaultdict(list))
    for i, b in enumerate(boxes):
        by_view[b[1]][round(float(b[0]), 2)].append(i)
    next_id = 0
    for view, frames in by_view.items():
        active: list[dict] = []  # {"id", "t", "box", "alliance"}
        for t in sorted(frames):
            idx = frames[t]
            active = [a for a in active if t - a["t"] <= LINK_GAP_S]
            cost = np.full((len(active), len(idx)), 1e6)
            for r, a in enumerate(active):
                ax0, ay0, ax1, ay1 = a["box"]
                aw = max(ax1 - ax0, 1.0)
                for c, i in enumerate(idx):
                    x0, y0, x1, y1 = boxes[i][2:6]
                    w = max(x1 - x0, 1.0)
                    if not 1 / SIZE_RATIO <= w / aw <= SIZE_RATIO:
                        continue
                    al = boxes[i][6]
                    if al and a["alliance"] and al != a["alliance"]:
                        continue
                    d = np.hypot((x0 + x1 - ax0 - ax1) / 2, (y0 + y1 - ay0 - ay1) / 2) / ((w + aw) / 2)
                    if d <= LINK_DIST * (1 + (t - a["t"] - 0.25)):
                        cost[r, c] = d
            used = set()
            if active and idx:
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < 1e5:
                        i = idx[c]
                        a = active[r]
                        track[i] = a["id"]
                        a.update(t=t, box=boxes[i][2:6], alliance=a["alliance"] or boxes[i][6])
                        used.add(c)
            for c, i in enumerate(idx):
                if c not in used:
                    track[i] = next_id
                    active.append({"id": next_id, "t": t, "box": boxes[i][2:6], "alliance": boxes[i][6]})
                    next_id += 1
    return track


def label(boxes: list, slot_team=None) -> list[int | None]:
    """Team per close-up box: its own read, else the nearest confident read on its track within
    LABEL_REACH_S, else the wide-shot robot its track was fused with (slot_team(i) -> team | None)."""
    track = link(boxes)
    members = defaultdict(list)
    for i, k in enumerate(track):
        members[k].append(i)
    out: list[int | None] = [None] * len(boxes)
    for k, idx in members.items():
        reads = [(boxes[i][0], boxes[i][7]) for i in idx if boxes[i][7] is not None and boxes[i][8] >= READ_MIN]
        fused = defaultdict(int)
        if slot_team is not None:
            for i in idx:
                tm = slot_team(i)
                if tm is not None:
                    fused[tm] += 1
        fallback = max(fused, key=fused.get) if fused and not reads else None
        for i in idx:
            t = boxes[i][0]
            if boxes[i][7] is not None and boxes[i][8] >= READ_MIN:
                out[i] = boxes[i][7]
                continue
            before = [(t - rt, tm) for rt, tm in reads if 0 <= t - rt <= LABEL_REACH_S]
            after = [(rt - t, tm) for rt, tm in reads if 0 < rt - t <= LABEL_REACH_S]
            b, f = (min(before)[1] if before else None), (min(after)[1] if after else None)
            if b is not None and f is not None and b != f:
                out[i] = None  # reads either side disagree: the link between them may be wrong
            else:
                out[i] = b if b is not None else f if f is not None else fallback
    return out

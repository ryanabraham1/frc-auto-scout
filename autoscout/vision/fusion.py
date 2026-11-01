"""Fuse the broadcast's close-up cameras (e.g. 2025 reef cams) with the wide shot.

Each close-up camera is calibrated from the match itself: a robot whose bumper is read in the close-up
while the wide shot tracks the same team gives one (close-up pixel -> field metres) pair, and a few
hundred such pairs over an event fit the camera's floor homography (RANSAC drops misreads and
tracking errors). With it, every robot seen in a close-up has a field position, so a robot the wide
shot loses behind a REEF or in a crowd keeps being tracked while the close-up sees it.
"""
from __future__ import annotations

import bisect

import cv2
import numpy as np

MIN_PAIRS = 25  # pairs needed before a close-up camera is trusted
RANSAC_M = 0.75  # m: a pair further than this from the fitted homography is an outlier
PAIR_DT = 0.3  # s: wide-shot observation this close in time to a close-up sighting pairs with it
FILL_DT = 0.25  # s: wide-shot rows this close to a close-up sighting take its position if not observed
MATCH_M = 1.5  # m: an unread close-up robot joins the alliance slot expected within this distance


def foot(box) -> tuple[float, float]:
    return (box[2] + box[0]) / 2, box[3]


def _team_track(result: dict, team: int):
    """Observed wide-shot rows of the slot holding `team` at each time: [(t, x, y)]."""
    rows = []
    for slot, tr in result.get("raw_tracks", {}).items():
        segs = result.get("identity_timeline", {}).get(slot) or [[-1e9, 1e9, result["identity"].get(slot)]]
        for r in tr:
            if r[3] and any(a <= r[0] < b and tm == team for a, b, tm in segs):
                rows.append((r[0], r[1], r[2]))
    rows.sort()
    return rows


def calibration_pairs(results: list[dict], min_conf: float = 0.4) -> dict[str, list]:
    """{view: [(px, py, fx, fy)]} from robots read in a close-up and tracked in the wide shot."""
    pairs: dict[str, list] = {}
    for r in results:
        cache = {}
        for t, view, x0, y0, x1, y1, _al, team, conf in r.get("aux_boxes", []):
            if team is None or conf < min_conf:
                continue
            if team not in cache:
                cache[team] = _team_track(r, team)
            rows = cache[team]
            ts = [q[0] for q in rows]
            i = bisect.bisect_left(ts, t)
            near = [rows[j] for j in (i - 1, i) if 0 <= j < len(rows) and abs(rows[j][0] - t) <= PAIR_DT]
            if not near:
                continue
            q = min(near, key=lambda q: abs(q[0] - t))
            px, py = foot((x0, y0, x1, y1))
            pairs.setdefault(view, []).append((px, py, q[1], q[2]))
    return pairs


def calibrate_views(results: list[dict]) -> dict[str, dict]:
    """{view: {"H": 3x3 pixel->field, "pairs": n, "inliers": n, "err_m": median inlier error}}."""
    out = {}
    for view, P in calibration_pairs(results).items():
        if len(P) < MIN_PAIRS:
            continue
        P = np.array(P, np.float64)
        H, mask = cv2.findHomography(P[:, :2], P[:, 2:], cv2.RANSAC, RANSAC_M)
        if H is None:
            continue
        inl = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(P[inl, None, :2], H)[:, 0]
        err = np.linalg.norm(proj - P[inl, 2:], axis=1)
        out[view] = {"H": H.round(8).tolist(), "pairs": int(len(P)), "inliers": int(inl.sum()),
                     "err_m": round(float(np.median(err)), 2)}
    return out


def to_field(H, px: float, py: float) -> tuple[float, float]:
    p = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), np.asarray(H, np.float64))[0, 0]
    return float(p[0]), float(p[1])


def fuse_tracks(ctx, views: dict[str, dict]) -> int:
    """Fill the wide-shot slot tracks with robots seen in calibrated close-ups. A read bumper decides
    the slot; an unread robot joins the only slot of its alliance expected nearby. Rows the wide shot
    observed are kept; coasting rows near the sighting take its position, and gaps get a new row.
    Returns the number of sightings used."""
    if not views or not ctx.aux_boxes:
        return 0
    fm = ctx.field
    used = 0
    ctx.aux_slot = {}  # aux_boxes index -> slot it was matched to
    for k, (t, view, x0, y0, x1, y1, al, team, conf) in enumerate(ctx.aux_boxes):
        v = views.get(view)
        if not v:
            continue
        p = to_field(v["H"], *foot((x0, y0, x1, y1)))
        if not fm.in_field(p, 0.3):
            continue
        slot = ctx.slot_of(team, t) if team is not None else None
        if slot is None:
            cands = []
            for s in ctx.tracks:
                if al and not s.startswith(al):
                    continue
                r = ctx.robot_at(s, t)
                if r is not None and np.hypot(r[0] - p[0], r[1] - p[1]) < MATCH_M:
                    cands.append(s)
            if len(cands) != 1:
                continue
            slot = cands[0]
        tr = ctx.tracks.get(slot)
        if tr is None:
            continue
        ctx.aux_slot[k] = slot
        ts = [r[0] for r in tr]
        i0, i1 = bisect.bisect_left(ts, t - FILL_DT), bisect.bisect_right(ts, t + FILL_DT)
        if any(tr[j][3] for j in range(i0, i1)):
            continue  # the wide shot saw it: keep that
        if i1 > i0:
            for j in range(i0, i1):
                tr[j] = (tr[j][0], p[0], p[1], True, None)
        else:
            tr.insert(i0, (t, p[0], p[1], True, None))
        used += 1
    ctx._ts.clear()
    return used

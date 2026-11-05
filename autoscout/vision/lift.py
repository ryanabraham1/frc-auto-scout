"""How high each robot is reaching, from the pixels above its box (wide shot and reef close-ups).

Robot detections box the chassis; a raised elevator or arm sticks up out of the box. A background model
of each view (the field, the crowd, the REEF) marks what is moving or new; for every robot box, the strip
of image right above it is scanned for foreground rising out of the box: how far up it goes, in box
heights, is the robot's reach at that moment. Other robots' boxes are blanked out of the mask first, so
a robot driving past behind doesn't read as a raised elevator. In the close-ups robots are several
times bigger, so a raised elevator next to the REEF is plain to see there.

A robot scoring L2-L4 lifts its mechanism for a second or two right before the score posts and drops it
after; `lift_score` looks for that rise over the robot's own usual reach.
"""
from __future__ import annotations

import cv2
import numpy as np

HZ = 6.0  # samples per second
SCALE = 0.5  # background model resolution
ABOVE = 1.3  # how far above a box to look, in box heights
ROW_FRAC = 0.5  # a row counts as "something there" when this share of a column band is foreground
BAND_FRAC = 0.12  # width of that column band, as a share of the robot box's width (a mast is narrow)


def reach_above(fg: np.ndarray, box, origin, scale: float = SCALE, others=()) -> float | None:
    """Reach of foreground rising out of the top of `box` (full-frame px) in mask `fg` (pixels of the
    view at `origin`, times `scale`), in box heights: 0 = nothing above the chassis. `others`: boxes of
    other robots, blanked out first."""
    ox, oy = origin
    def to_mask(b):
        return [(v - o) * scale for v, o in zip(b, (ox, oy, ox, oy))]
    x0, y0, x1, y1 = to_mask(box)
    h = y1 - y0
    if h < 4 or x1 - x0 < 4:
        return None
    H, W = fg.shape
    xa, xb = int(max(0, x0 + 0.1 * (x1 - x0))), int(min(W, x1 - 0.1 * (x1 - x0)))
    top = int(max(0, y0 - ABOVE * h))
    yb = int(min(H, max(0, y0 + 0.15 * h)))
    if xb - xa < 3 or yb - top < 3:
        return None
    strip = (fg[top:yb, xa:xb] > 0).copy()
    for ob in others:
        a0, b0, a1, b1 = to_mask(ob)
        ca, cb = int(max(a0, xa)) - xa, int(min(a1, xb)) - xa
        ra, rb = int(max(b0, top)) - top, int(min(b1, yb)) - top
        if ca < cb and ra < rb:
            strip[ra:rb, ca:cb] = False
    # an elevator or arm is a narrow upright piece: a row counts when enough of its pixels are foreground
    # within any one narrow column band, not across the whole width of the robot
    band = max(2, int(BAND_FRAC * (xb - xa)))
    occ = cv2.blur(strip.astype(np.float32), (band, 1), borderType=cv2.BORDER_CONSTANT)
    rows = occ.max(axis=1) >= ROW_FRAC
    # walk up from the box top while rows stay occupied (small gaps allowed): a connected reach
    reach, gap = 0, 0
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]:
            reach, gap = len(rows) - i, 0
        else:
            gap += 1
            if gap > max(2, 0.08 * h):
                break
    return round(max(0.0, (reach - 0.15 * h) / h), 2)


def _subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=int(20 * HZ), varThreshold=25, detectShadows=False)


def _mask(bg, img):
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    fg = bg.apply(small)
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def collect(video, result: dict, view, aux_views: dict | None = None, normalize_aux: bool = False) -> tuple[dict[str, list], list]:
    """({slot: [[t, reach], ...]} for the wide shot's observed boxes, [reach or None per aux_boxes entry])."""
    vx, vy, vw, vh = view
    clk = result["clock"]
    t0, t1 = clk["auto_end_vt"] - clk["auto_len"] - 3, clk["teleop_end_vt"] + 1
    wide_bg = _subtractor()
    rows = {s: [r for r in tr if r[3] and r[4] is not None] for s, tr in result["raw_tracks"].items()}
    ts = {s: np.array([r[0] for r in rr]) for s, rr in rows.items()}
    out = {s: [] for s in rows}
    aux = result.get("aux_boxes", []) if aux_views else []
    aux_t = np.array([b[0] for b in aux]) if aux else np.zeros(0)
    aux_reach: list = [None] * len(aux)
    cam_bg = {k: _subtractor() for k in (aux_views or {})}
    step = max(1, int(round(video.fps / HZ)))
    from .aux_views import AuxViewNormalizer
    normalize = AuxViewNormalizer(video.frame_at(clk["teleop_end_vt"] - clk["teleop_len"] + 20),
                                  aux_views if normalize_aux else None)
    for t, frame in video.iter_frames(t0, t1, step=step):
        frame = normalize(frame)
        fg = _mask(wide_bg, frame[vy: vy + vh, vx: vx + vw])
        now = {}
        for s, rr in rows.items():
            if len(ts[s]):
                i = int(np.argmin(np.abs(ts[s] - t)))
                if abs(ts[s][i] - t) <= 0.12:
                    now[s] = rr[i][4]
        for s, box in now.items():
            v = reach_above(fg, box, (vx, vy), others=[b for k, b in now.items() if k != s])
            if v is not None:
                out[s].append([round(t, 2), v])
        if cam_bg:
            masks = {k: _mask(cam_bg[k], frame[y: y + h, x: x + w]) for k, (x, y, w, h) in aux_views.items()}
            if len(aux_t):
                idx = np.nonzero(np.abs(aux_t - t) <= 0.09)[0]
                for i in idx:
                    b = aux[i]
                    if b[1] not in masks:
                        continue
                    x, y = aux_views[b[1]][:2]
                    others = [aux[j][2:6] for j in idx if j != i and aux[j][1] == b[1]]
                    aux_reach[i] = reach_above(masks[b[1]], b[2:6], (x, y), others=others)
    return out, aux_reach


def lift_score(samples: list, t0: float, t1: float, base: float) -> float | None:
    """Lift in [t0, t1] from (t, reach) samples: the reach held for at least two samples in a row (the
    lower of each consecutive pair, so one noisy frame doesn't count), over the robot's usual reach."""
    v = sorted((t, r) for t, r in samples if t0 <= t <= t1)
    # Simultaneous boxes and sightings seconds apart cannot establish a held lift.
    pairs = [min(a, b) for (ta, a), (tb, b) in zip(v, v[1:]) if 0 < tb - ta <= 0.75]
    if not pairs:
        return None
    held = max(pairs)
    return max(0.0, held - base)

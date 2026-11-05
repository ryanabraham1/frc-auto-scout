"""Game pieces appearing on a scoring element, seen in the wide shot (2025: CORAL placed on a REEF).

A placed piece stays put, so it shows as a lasting change: pixels of the piece's colour that were not
there for a few seconds and then stay for several seconds. Robots driving in front of the element
change pixels only for a moment, which the persistence test ignores. Each change gives the time a
piece went on, where on the element it is (image x tells which face, so which robot placed it) and
how high (the level).
"""
from __future__ import annotations

import cv2
import numpy as np

HZ = 4.0  # samples per second kept per element
BEFORE = (-4.0, -0.5)  # s: window in which the pixel must have been clear
AFTER = (0.5, 6.0)  # s: window in which it must stay the piece's colour
CLEAR, PRESENT = 0.25, 0.75  # max / min share of samples of the piece colour in those windows


def element_region(cal, fm, name: str, height_k: float = 0.95, pad: float = 0.12) -> tuple[int, int, int, int] | None:
    """Image box (x0, y0, x1, y1) covering a round element's footprint and, above it, its height
    (taken as height_k footprint widths: the camera looks down at a similar angle at every event)."""
    e = fm.elements.get(name)
    if not e or "radius" not in e:
        return None
    c, r = e["center"], e["radius"]
    P = cal.to_image([(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 13)])
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    w = x1 - x0
    return int(x0 - pad * w), int(y0 - height_k * w), int(x1 + pad * w), int(y1 + 0.08 * w)


def piece_mask(crop: np.ndarray) -> np.ndarray:
    """CORAL: white PVC (bright, unsaturated)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 60) & (hsv[..., 2] > 180)


class ElementWatcher:
    """Feed every wide-shot frame (full-frame pixels) with step(); changes() returns the placements."""

    def __init__(self, regions: dict[str, tuple[int, int, int, int]], frame_size: tuple[int, int]):
        W, H = frame_size
        self.regions = {k: (max(0, x0), max(0, y0), min(W, x1), min(H, y1)) for k, (x0, y0, x1, y1) in regions.items()}
        self.t: list[float] = []
        self.masks: dict[str, list[np.ndarray]] = {k: [] for k in self.regions}
        self._next = -1e9

    def step(self, t: float, frame: np.ndarray) -> None:
        if t < self._next:
            return
        self._next = t + 1.0 / HZ - 1e-3
        self.t.append(t)
        for k, (x0, y0, x1, y1) in self.regions.items():
            self.masks[k].append(np.packbits(piece_mask(frame[y0:y1, x0:x1]), axis=-1))

    def changes(self, min_area: float | None = None) -> list[dict]:
        out = []
        T = np.array(self.t)
        if len(T) < 10:
            return out
        for k, (x0, y0, x1, y1) in self.regions.items():
            w = x1 - x0
            M = np.unpackbits(np.array(self.masks[k]), axis=-1, count=w).astype(np.float32)  # (n, h, w)
            C = np.concatenate([np.zeros((1,) + M.shape[1:], np.float32), np.cumsum(M, 0)])
            best = np.zeros(M.shape[1:], np.float32)
            best_t = np.zeros(M.shape[1:], np.float32)
            closed = np.zeros(M.shape[1:], bool)
            for i, t in enumerate(T):
                b0, b1 = np.searchsorted(T, [t + BEFORE[0], t + BEFORE[1]])
                a0, a1 = np.searchsorted(T, [t + AFTER[0], t + AFTER[1]])
                # the full windows must exist (start and end of the match can't show a placement)
                if T[b0] > t + BEFORE[0] + 1.0 or a1 >= len(T) or b1 - b0 < 4 or a1 - a0 < 8:
                    continue
                before = (C[b1] - C[b0]) / (b1 - b0)
                after = (C[a1] - C[a0]) / (a1 - a0)
                s = np.where((before <= CLEAR) & (after >= PRESENT), after - before, 0)
                # Keep the first persistent appearance. A robot can later hide
                # and uncover an already-scored coral with greater contrast;
                # choosing the strongest transition over the whole match would
                # move that original placement to the later robot's visit.
                closed |= (best > 0) & (s == 0)
                upd = (s > best) & ~closed
                best[upd], best_t[upd] = s[upd], t
            on = (best > 0).astype(np.uint8)
            on &= cv2.morphologyEx(on, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # (even kernels shift: keep within)
            area_min = min_area if min_area is not None else max(15.0, (0.05 * w) ** 2)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
            for j in range(1, n):
                if stats[j, cv2.CC_STAT_AREA] < area_min:
                    continue
                ys, xs = np.nonzero(lab == j)
                ts = best_t[ys, xs]
                # one blob can hold pieces placed at different times (stacked on one branch): split by time
                order = np.argsort(ts)
                cuts = np.nonzero(np.diff(ts[order]) > 1.5)[0] + 1
                for grp in np.split(order, cuts):
                    if len(grp) < area_min:
                        continue
                    gy, gx = ys[grp], xs[grp]
                    out.append({"elem": k, "t": round(float(np.median(ts[grp])), 2),
                                "x": round(float(gx.mean() + x0), 1), "y": round(float(gy.mean() + y0), 1),
                                "x0": int(gx.min() + x0), "x1": int(gx.max() + x0),
                                "y0": int(gy.min() + y0), "y1": int(gy.max() + y0), "area": int(len(grp)),
                                # 0 = bottom of the region (trough), 1 = top (L4)
                                "h": round(float(1 - (gy.mean()) / max(y1 - y0, 1)), 2)})
        return sorted(out, key=lambda c: c["t"])

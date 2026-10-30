"""Launch extraction v2: physically-validated flights from per-frame moving-piece blobs.

Input is every newly-appeared piece-colored blob per frame (t, x, y, area, w, h) in view
pixels, as collected by LaunchDetector.step. Working offline on the whole match lets us:

  1. link blobs into tracks with gates scaled to real distances (no cross-field jumps),
  2. keep only tracks that are ballistic: x(t) linear, y(t) a downward-curving parabola,
     with a good fit (robot-pushed / rolling / bouncing pieces and mis-links fail this),
  3. re-join one flight split by occlusion or by leaving the top of the frame,
  4. drop bounces (a "launch" that starts where a falling piece just landed),
  5. count pieces per track relative to typical single-piece tracks at that depth
     (motion blur inflates blob area, so a fixed area-per-piece over-counts),
  6. flag flights that come down into a HUB/target opening as made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .trajectories import Launch


@dataclass
class _Trk:
    pts: list  # rows (t, x, y, area, w, h)
    miss: int = 0

    def arr(self):
        return np.array(self.pts, dtype=np.float64)

    def predict(self, t):
        P = self.pts
        if len(P) == 1:
            return P[-1][1], P[-1][2]
        (t1, x1, y1), (t2, x2, y2) = P[-2][:3], P[-1][:3]
        dt = max(t2 - t1, 1e-3)
        vx, vy = (x2 - x1) / dt, (y2 - y1) / dt
        ay = 0.0
        if len(P) >= 3:
            t0, x0, y0 = P[-3][:3]
            ay = (vy - (y1 - y0) / max(t1 - t0, 1e-3)) / max((t2 - t0) / 2, 1e-3)
            ay = float(np.clip(ay, -300, 4000))
        h = t - t2
        return x2 + vx * h, y2 + vy * h + 0.5 * ay * h * h


def fit_flight(P: np.ndarray):
    """Fit x(t), y(t) as quadratics (image px). Ballistic motion is linear in x and quadratic in y
    in 3D; perspective bends x slightly when the piece moves toward/away from the camera, which a
    quadratic absorbs. Returns (coefs_x, coefs_y, rms)."""
    t = P[:, 0] - P[0, 0]
    cx = np.polyfit(t, P[:, 1], 2) if len(t) >= 6 else np.r_[0.0, np.polyfit(t, P[:, 1], 1)]
    cy = np.polyfit(t, P[:, 2], 2) if len(t) >= 4 else np.r_[0.0, np.polyfit(t, P[:, 2], 1)]
    res = np.r_[P[:, 1] - np.polyval(cx, t), P[:, 2] - np.polyval(cy, t)]
    return cx, cy, float(np.sqrt(np.mean(res ** 2)))


class LaunchExtractor:
    def __init__(self, view, cal=None, field_model=None, targets: dict | None = None):
        """targets: name -> (x, y, radius_px) full-frame image location of scoring openings."""
        self.view = view
        self.cal = cal
        self.fm = field_model
        self.targets = targets or {}
        self.stats = {}
        # tracking knobs (benchmarked against the live score: share of HUB entries tracked)
        self.gate_est = 0.35  # m: link gate for an established track (>= 3 points)
        self.gate_new = 0.9  # m: link gate for a fresh track with no velocity yet
        self.flow = True  # a fresh track borrows the velocity of nearby tracks (pieces in a volley fly together)
        self.max_miss = None  # frames a track may go unseen (None: ~0.09 s at the blob frame rate)
        self.min_len = 5  # points for a flight
        self.min_dur = None  # s a flight must span (None: 4 frame intervals, at least 0.08 s)
        self.split = True  # cut a track that fails the physics check at its worst kink (a mislink)

    # ---- geometry helpers ---------------------------------------------------------------
    def ppm(self, x, y):
        """Pixels per meter near a (full-frame) image point; ~60 if uncalibrated."""
        if self.cal is None:
            return 60.0
        y = min(y, self.view[1] + self.view[3] - 5)
        return max(self.cal.px_per_meter((x, y)), 8.0)

    def ball_diam_px(self, xs, ys, diameter_m: float = 0.15) -> np.ndarray:
        """Expected on-screen diameter of a piece at image points (full-frame), using the floor
        scale at the nearest on-field location (pieces in the air above the far wall map off-field)."""
        if self.cal is None:
            return np.full(len(xs), 14.0)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        f = self.cal.to_field(pts)
        L = self.fm.length if self.fm else 17.0
        W = self.fm.width if self.fm else 8.0
        bad = ~np.isfinite(f).all(1)
        f[bad] = (L / 2, W)
        f[:, 0] = np.clip(f[:, 0], 0, L)
        f[:, 1] = np.clip(f[:, 1], 0, W)
        a = self.cal.to_image(f)
        b = self.cal.to_image(f + [diameter_m, 0.0])
        return np.linalg.norm(b - a, axis=1)

    def filter_blobs(self, B: np.ndarray) -> np.ndarray:
        """Keep blobs that look like a whole piece in flight (not a sliver of a slow mover).

        A piece moving more than its own size per frame appears as a full piece-sized blob;
        pieces rolling, jiggling in a hopper, or pushed by a robot leave thin crescents.
        """
        d = self.ball_diam_px(B[:, 1] + self.view[0], B[:, 2] + self.view[1])
        exp_area = np.pi * (d / 2) ** 2
        w, h, area = B[:, 4], B[:, 5], B[:, 3]
        ok = (area >= 0.35 * exp_area) & (np.minimum(w, h) >= 0.45 * d) & (np.maximum(w, h) <= 5.0 * d)
        ok &= area <= 6.0 * exp_area  # not a person / big object
        self.stats["blobs_in"] = int(len(B))
        self.stats["blobs_kept"] = int(ok.sum())
        return B[ok]

    # ---- 1. linking ---------------------------------------------------------------------------
    def link(self, B: np.ndarray) -> list[_Trk]:
        vx0, vy0 = self.view[0], self.view[1]
        order = np.argsort(B[:, 0], kind="stable")
        B = B[order]
        times = np.unique(B[:, 0])
        idx = np.searchsorted(B[:, 0], times)
        idx = np.r_[idx, len(B)]
        live: list[_Trk] = []
        done: list[_Trk] = []
        for k, t in enumerate(times):
            D = B[idx[k]: idx[k + 1]]
            if live and len(D):
                cost = np.full((len(live), len(D)), 1e9)
                flow = self._flow(live) if self.flow else None
                for i, tr in enumerate(live):
                    px, py = tr.predict(t)
                    ppm = self.ppm(px + vx0, py + vy0)
                    gate_m = self.gate_est if len(tr.pts) >= 3 else self.gate_new
                    if flow is not None and len(tr.pts) == 1:
                        v = self._flow_at(flow, tr.pts[-1][1], tr.pts[-1][2], 1.5 * ppm)
                        if v is not None:  # moving with the volley around it: predict like an established track
                            h = t - tr.pts[-1][0]
                            px, py = px + v[0] * h, py + v[1] * h
                            gate_m = 0.5 * (self.gate_est + self.gate_new)
                    gate = gate_m * ppm + 6 * (tr.miss + 1)
                    d = np.hypot(D[:, 1] - px, D[:, 2] - py)
                    ok = d < gate
                    cost[i, ok] = d[ok]
                ri, ci = linear_sum_assignment(cost)
                used_t, used_d = set(), set()
                for i, j in zip(ri, ci):
                    if cost[i, j] < 1e9:
                        live[i].pts.append(tuple(D[j]))
                        live[i].miss = 0
                        used_t.add(i)
                        used_d.add(j)
            else:
                used_t, used_d = set(), set()
            nxt = []
            for i, tr in enumerate(live):
                if i not in used_t:
                    tr.miss += 1
                (done if tr.miss > self.max_miss else nxt).append(tr)
            live = nxt
            for j in range(len(D)):
                if j not in used_d:
                    live.append(_Trk([tuple(D[j])]))
        return done + live

    @staticmethod
    def _flow(live: list) -> np.ndarray | None:
        """(x, y, vx, vy) of established tracks, for the flow prior."""
        rows = []
        for tr in live:
            if len(tr.pts) >= 3:
                (t1, x1, y1), (t2, x2, y2) = tr.pts[-2][:3], tr.pts[-1][:3]
                dt = max(t2 - t1, 1e-3)
                rows.append((x2, y2, (x2 - x1) / dt, (y2 - y1) / dt))
        return np.array(rows) if len(rows) >= 2 else None

    @staticmethod
    def _flow_at(flow: np.ndarray, x, y, radius):
        near = np.hypot(flow[:, 0] - x, flow[:, 1] - y) < radius
        if near.sum() < 2:
            return None
        return np.median(flow[near, 2]), np.median(flow[near, 3])

    # ---- 2. physics ---------------------------------------------------------------------------
    def ballistic(self, P: np.ndarray, strict: bool = False):
        """Return fit info if the track is a plausible free flight, else None. `strict`: only clear
        gravity counts (for pieces cut out of a longer track, which have less to vouch for them)."""
        if len(P) < self.min_len or P[-1, 0] - P[0, 0] < self.min_dur:
            return None
        x0, y0 = P[0, 1] + self.view[0], P[0, 2] + self.view[1]
        ppm = self.ppm(x0, y0)
        cx, cy, rms = fit_flight(P)
        path = float(np.sum(np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2]))))
        if rms > max(3.5, 0.05 * ppm + 0.01 * path):
            return None  # jittery / mis-linked / bouncing in a pile
        speed0 = np.hypot(cx[1], cy[1])  # px/s at release
        if speed0 < 1.5 * ppm:
            return None  # too slow to be launched
        ay = 2 * cy[0]
        dur = P[-1, 0] - P[0, 0]
        rises = P[0, 2] - P[:, 2].min()
        # gravity: clear downward curvature, or a rising segment too fast to be on the floor
        g_like = ay > 1.5 * ppm
        floor_speed = self._floor_speed(P)
        if strict and ay < 4.0 * ppm:  # ~0.4 g: a real free fall curves at ~9.8 px-per-metre per s^2
            return None
        if not (g_like or floor_speed > 9.0 or (rises > 0.5 * ppm and dur < 0.4 and cy[1] < -2.5 * ppm)):
            return None
        if ay < -1.0 * ppm:
            return None  # accelerating upward: not a free flight
        return cx, cy, rms

    def _floor_speed(self, P):
        if self.cal is None:
            return 0.0
        pts = np.stack([P[:, 1] + self.view[0], P[:, 2] + self.view[1]], 1)
        f = self.cal.to_field(pts)
        dt = np.diff(P[:, 0])
        ok = dt > 1e-3
        if ok.sum() < 2:
            return 0.0
        return float(np.median(np.linalg.norm(np.diff(f, axis=0), axis=1)[ok] / dt[ok]))

    def split_ballistic(self, P: np.ndarray, depth: int = 0) -> list:
        """Ballistic pieces of a track. In a volley the linker sometimes hops from one piece to its
        neighbour, so a whole track fails the physics check although each part is a clean flight:
        cut it where the path kinks hardest and test each part."""
        fit = self.ballistic(P, strict=depth > 0)
        if fit is not None:
            return [(P, fit)]
        if len(P) < 2 * self.min_len or depth >= 3:
            return []
        acc = np.hypot(np.diff(P[:, 1], 2), np.diff(P[:, 2], 2))  # second difference ~ kink
        lo, hi = self.min_len - 1, len(P) - self.min_len - 1
        if hi < lo:
            return []
        k = lo + int(np.argmax(acc[lo: hi + 1])) + 1
        return self.split_ballistic(P[:k], depth + 1) + self.split_ballistic(P[k:], depth + 1)

    # ---- 3. re-join split flights ----------------------------------------------------------------
    def rejoin(self, flights: list[tuple]) -> list[tuple]:
        """flights: [(P, fit)] sorted by start. Merge B into A when B continues A's parabola."""
        flights = sorted(flights, key=lambda f: f[0][0, 0])
        out = []
        for P, fit in flights:
            merged = False
            for k in range(len(out) - 1, max(-1, len(out) - 40), -1):
                A, (cx, cy, _) = out[k]
                gap = P[0, 0] - A[-1, 0]
                if gap <= 0 or gap > 1.5:
                    continue
                t = P[0, 0] - A[0, 0]
                px, py = np.polyval(cx, t), np.polyval(cy, t)
                tol = 12 + 25 * gap
                if np.hypot(px - P[0, 1], py - P[0, 2]) < tol:
                    Q = np.vstack([A, P])
                    out[k] = (Q, fit_flight(Q))
                    merged = True
                    break
            if not merged:
                out.append((P, fit))
        return out

    # ---- 4. bounces -------------------------------------------------------------------------------
    @staticmethod
    def is_bounce(P: np.ndarray, ends: np.ndarray) -> bool:
        """A flight starting where a falling piece ended moments ago is a bounce."""
        if not len(ends):
            return False
        t0, x0, y0 = P[0, 0], P[0, 1], P[0, 2]
        m = (ends[:, 0] < t0) & (ends[:, 0] > t0 - 0.35) & (ends[:, 3] > 0)
        if not m.any():
            return False
        d = np.hypot(ends[m, 1] - x0, ends[m, 2] - y0)
        return bool((d < 30).any())

    # ---- 6. made ----------------------------------------------------------------------------------
    def made_into(self, P: np.ndarray, fit) -> str | None:
        """Target opening the piece *disappears into* while descending.

        A make ends at the opening: the track stops right after passing through it (the piece
        drops inside), or the visible track ends shortly before it and the fitted arc lands in
        it. A lob that passes over/near the opening and keeps flying is not a make.
        """
        cx, cy, _ = fit
        vx, vy = self.view[0], self.view[1]
        dur = P[-1, 0] - P[0, 0]
        t = np.linspace(0, dur + 0.6, 90)
        xs, ys = np.polyval(cx, t) + vx, np.polyval(cy, t) + vy
        vys = np.polyval(np.polyder(cy), t)
        for name, (tx, ty, r) in self.targets.items():
            d = np.hypot(xs - tx, (ys - ty) * 1.5)
            hit = np.nonzero((d < r) & (vys > 0))[0]
            if not len(hit):
                continue
            th = t[hit[0]]
            if th <= dur:
                # observed passing through: it must vanish there (not keep flying past)
                if dur - th <= 0.25:
                    return name
            elif th - dur <= 0.6:
                return name  # track lost just before the opening (occluded by the hub/net)
        return None

    # ---- main ---------------------------------------------------------------------------------------
    def extract(self, blobs: np.ndarray) -> list[Launch]:
        if blobs is None or len(blobs) == 0:
            return []
        B = np.asarray(blobs, dtype=np.float64)
        self.stats = {}
        ts = np.unique(B[:, 0])
        fps = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 2 else 30.0
        if self.max_miss is None:
            self.max_miss = max(2, int(round(0.09 * fps)))
        if self.min_dur is None:
            self.min_dur = max(0.08, 4.0 / fps)
        B = self.filter_blobs(B)
        tracks = self.link(B)
        flights, ends = [], []
        rej = {"short": 0, "not_ballistic": 0, "bounce": 0, "origin": 0}
        for tr in tracks:
            P = tr.arr()
            if len(P) >= 2:
                vy_end = (P[-1, 2] - P[-2, 2]) / max(P[-1, 0] - P[-2, 0], 1e-3)
                ends.append((P[-1, 0], P[-1, 1], P[-1, 2], vy_end))
            if len(P) < self.min_len:
                rej["short"] += 1
                continue
            parts = self.split_ballistic(P) if self.split else []
            if not parts:
                fit = self.ballistic(P)
                if fit is None:
                    rej["not_ballistic"] += 1
                    continue
                parts = [(P, fit)]
            flights += parts
        ends = np.array(ends).reshape(-1, 4)
        flights = self.rejoin(flights)
        kept = []
        for P, fit in flights:
            # A flight that drops into a goal opening is a shot even if it was first seen in mid
            # air: its first sighting maps to a meaningless floor point (origin check), and inside
            # a volley the piece ahead ends right where it starts (bounce check). Ejections leave a
            # HUB from its base and floor bounces can't reach the opening, so neither applies.
            if not (self.targets and self.made_into(P, fit)):
                if self.is_bounce(P, ends):
                    rej["bounce"] += 1
                    continue
                if not self._plausible_origin(P):
                    rej["origin"] += 1
                    continue
            kept.append((P, fit))
        # piece count relative to typical single-piece tracks at similar image height
        med_area = np.array([np.median(P[:, 3]) for P, _ in kept]) if kept else np.zeros(0)
        start_y = np.array([P[0, 2] for P, _ in kept]) if kept else np.zeros(0)
        out = []
        vx, vy = self.view[0], self.view[1]
        for i, (P, fit) in enumerate(kept):
            band = np.abs(start_y - start_y[i]) < 80
            ref = np.percentile(med_area[band], 40) if band.sum() >= 5 else np.percentile(med_area, 40)
            n = float(np.clip(np.round(med_area[i] / max(ref, 1.0) / 1.3), 1, 3))
            ia = int(np.argmin(P[:, 2]))
            L = Launch(
                t0=float(P[0, 0]), t1=float(P[-1, 0]), start=(P[0, 1] + vx, P[0, 2] + vy),
                end=(P[-1, 1] + vx, P[-1, 2] + vy), apex=(P[ia, 1] + vx, P[ia, 2] + vy),
                n_pieces=n, rises=float(P[0, 2] - P[ia, 2]),
                pts=[(float(p[0]), float(p[1] + vx), float(p[2] + vy)) for p in P],
                exited_top=bool(P[-1, 2] < 15),
            )
            L.target = self.made_into(P, fit) if self.targets else None
            # Was the release seen? A launch leaves the robot moving up. Tracks first seen already
            # falling (ascent hidden by graphics/occlusion) are kept for hub counting only.
            cx, cy, _ = fit
            L.rising = bool(cy[1] < -0.5 * self.ppm(L.start[0], L.start[1]) and L.rises > 3)
            out.append(L)
        self.stats.update(rej, tracks=len(tracks), flights=len(out))
        return out

    def _plausible_origin(self, P):
        if self.cal is None or self.fm is None:
            return True
        f = self.cal.to_field([(P[0, 1] + self.view[0], P[0, 2] + self.view[1])])[0]
        # HUB exits eject every scored piece into the NEUTRAL ZONE from the hub's base: not a launch
        for name, e in self.fm.elements.items():
            if e.get("ejects"):
                c = np.array(e["center"], float)
                half = np.array(e["size"], float) / 2 + 0.9
                if np.all(np.abs(f - c) <= half):
                    return False
        if self.fm.in_field(f, 1.5):
            return True
        for name, e in self.fm.elements.items():
            if "outpost" in name and np.linalg.norm(f - np.array(e["center"])) < 3.0:
                return True
        return False

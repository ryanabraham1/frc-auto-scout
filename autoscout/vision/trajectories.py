"""In-flight game-piece trajectories (launches) from the fixed wide shot.

Pieces resting on the carpet are static, so we look only at *newly appeared* piece-colored
pixels each frame (color mask minus the dilated previous mask). Those blobs are linked into
tracks with a constant-acceleration predictor (gravity bends arcs downward in the image).
A track that rises before falling is a launch; its first point is the release location.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    pts: list = field(default_factory=list)  # (t, x, y, area) in view coords
    misses: int = 0

    def predict(self, t: float):
        if len(self.pts) == 1:
            return self.pts[-1][1], self.pts[-1][2], 40.0
        (t1, x1, y1, _), (t2, x2, y2, _) = self.pts[-2], self.pts[-1]
        dt = max(t2 - t1, 1e-3)
        vx, vy = (x2 - x1) / dt, (y2 - y1) / dt
        if len(self.pts) >= 3:
            t0, x0, y0, _ = self.pts[-3]
            dt0 = max(t1 - t0, 1e-3)
            ay = (vy - (y1 - y0) / dt0) / ((dt + dt0) / 2)
            ay = float(np.clip(ay, -500, 3000))
        else:
            ay = 0.0
        h = t - t2
        speed = np.hypot(vx, vy) * h
        return x2 + vx * h, y2 + vy * h + 0.5 * ay * h * h, 12 + 0.6 * speed

    @property
    def duration(self):
        return self.pts[-1][0] - self.pts[0][0]


@dataclass
class Launch:
    t0: float  # video time of first sighting
    t1: float
    start: tuple[float, float]  # full-frame pixel of first sighting
    end: tuple[float, float]
    apex: tuple[float, float]
    n_pieces: float  # estimated pieces in this track (blobs can merge)
    rises: float  # pixels risen from start to apex
    pts: list  # [(t, x, y)] full-frame
    exited_top: bool = False
    # filled in by game logic:
    robot: int | None = None
    kind: str = ""  # "shot" | "feed" | "other"
    zone: str = ""
    target: str | None = None  # scoring opening the flight came down into (e.g. "blue_hub")
    rising: bool = True  # first seen moving up (a release); False = only the descent was visible

    def to_dict(self):
        return {"t0": round(self.t0, 3), "t1": round(self.t1, 3), "start": [round(v, 1) for v in self.start],
                "end": [round(v, 1) for v in self.end], "apex": [round(v, 1) for v in self.apex],
                "n": round(self.n_pieces, 2), "robot": self.robot, "kind": self.kind, "zone": self.zone,
                "target": self.target, "rising": self.rising}


class LaunchDetector:
    def __init__(self, hsv_lo, hsv_hi, view, piece_area_fn=None, min_area: int = 6,
                 cal=None, field_model=None, lag: int = 1):
        self.lo = np.array(hsv_lo, np.uint8)
        self.hi = np.array(hsv_hi, np.uint8)
        self.view = view
        self.prev = None
        self.tracks: list[Track] = []
        self.done: list[Track] = []
        self.next_id = 0
        self.min_area = min_area
        self.piece_area_fn = piece_area_fn  # (x, y) full-frame -> expected blob area of one piece
        self.kernel = np.ones((5, 5), np.uint8)
        self.ignore = None  # mask of regions to ignore (e.g. people/crowd); view coords
        self.cal = cal
        self.fm = field_model
        self.rejected_ground = 0
        self.blobs: list[tuple] = []  # (t, x, y, area, w, h) view coords, for offline re-tracking
        # Compare against the mask `lag` frames back: at 60 fps a piece moves too little per frame
        # to appear as a whole new blob, but sampling every frame makes linking unambiguous.
        self.lag = max(1, lag)
        self.history: list = []
        # bg_frames > 0: a piece is "moving" where the colour is now but not in the background, the
        # pixels piece-coloured in most of the last bg_frames frames. In a stream every ball fills the
        # spot the ball ahead just left, so a short-lag difference erases most of the stream; a pixel a
        # stream passes through is only piece-coloured part of the time, so it never joins the background.
        self.bg_frames = 0
        self.bg_count = None
        self.bg_hist: list = []
        # live ball tracks are only drawn in the debug video (launches come from the offline
        # LaunchExtractor over self.blobs), and linking every frame costs more than finding blobs
        self.online = True

    def step(self, t: float, frame_view: np.ndarray) -> None:
        m = cv2.inRange(cv2.cvtColor(frame_view, cv2.COLOR_BGR2HSV), self.lo, self.hi)
        if self.prev is None:  # reset (start or after a camera cut)
            self.history = []
            self.bg_hist, self.bg_count = [], None
        self.prev = m
        if self.bg_frames:
            bit = (m > 0).astype(np.uint8)
            if self.bg_count is None:
                self.bg_count = np.zeros(m.shape, np.uint8)
            self.bg_hist.append(bit)
            self.bg_count += bit
            if len(self.bg_hist) > self.bg_frames:
                self.bg_count -= self.bg_hist.pop(0)
            if len(self.bg_hist) < self.bg_frames:
                return
            ref = np.where(self.bg_count * 2 > self.bg_frames, 255, 0).astype(np.uint8)
        else:
            self.history.append(m)
            if len(self.history) <= self.lag:
                return
            ref = self.history.pop(0)
        mv = cv2.bitwise_and(m, cv2.bitwise_not(cv2.dilate(ref, self.kernel)))
        mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if self.ignore is not None:
            mv[self.ignore > 0] = 0
        n, _, st, c = cv2.connectedComponentsWithStats(mv)
        keep = [i for i in range(1, n) if st[i, 4] >= self.min_area]
        for i in keep:
            self.blobs.append((t, c[i][0], c[i][1], st[i, 4], st[i, 2], st[i, 3]))
        if self.online:
            self._associate(t, [(c[i][0], c[i][1], st[i, 4]) for i in keep])

    def _associate(self, t, dets):
        live = self.tracks
        if live and dets:
            preds = [tr.predict(t) for tr in live]
            cost = np.full((len(live), len(dets)), 1e6)
            for i, (px, py, gate) in enumerate(preds):
                for j, (x, y, a) in enumerate(dets):
                    d = np.hypot(x - px, y - py)
                    if d < gate:
                        cost[i, j] = d
            ri, ci = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    live[i].pts.append((t, *dets[j]))
                    live[i].misses = 0
                    matched_t.add(i)
                    matched_d.add(j)
        else:
            matched_t, matched_d = set(), set()
        for i, tr in enumerate(live):
            if i not in matched_t:
                tr.misses += 1
        keep = []
        for tr in live:
            if tr.misses > 6:
                self.done.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(self.next_id, [(t, *d)]))
                self.next_id += 1

    @staticmethod
    def _merge_fragments(tracks: list[Track], max_gap: float = 0.3) -> list[Track]:
        """Join tracks that are pieces of one flight (broken by occlusion or a missed frame).

        Track B continues track A if B starts shortly after A ends, close to where A's motion
        (with its fitted image-space gravity) predicts the piece to be.
        """
        tracks = sorted((t for t in tracks if t.pts), key=lambda t: t.pts[0][0])
        merged: list[Track] = []
        open_ends: list[Track] = []
        for tr in tracks:
            t0, x0, y0, _ = tr.pts[0]
            best, best_d = None, 1e9
            for a in open_ends:
                ta = a.pts[-1][0]
                if not (0 < t0 - ta <= max_gap) or len(a.pts) < 3:
                    continue
                px, py, gate = a.predict(t0)
                d = np.hypot(px - x0, py - y0)
                if d < max(25.0, gate) and d < best_d:
                    best, best_d = a, d
            if best is not None:
                best.pts.extend(tr.pts)
            else:
                merged.append(tr)
                open_ends.append(tr)
            open_ends = [a for a in open_ends if t0 - a.pts[-1][0] <= max_gap]
        return merged

    def finish(self) -> list[Launch]:
        self.done.extend(self.tracks)
        self.tracks = []
        self.done = self._merge_fragments(self.done)
        vx, vy = self.view[0], self.view[1]
        out = []
        for tr in self.done:
            if len(tr.pts) < 5:
                continue
            P = np.array([(p[0], p[1], p[2], p[3]) for p in tr.pts])
            ys = P[:, 2]
            disp = np.hypot(P[-1, 1] - P[0, 1], P[-1, 2] - P[0, 2])
            path = np.sum(np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2])))
            if disp < 35 or path < 50:
                continue
            ia = int(np.argmin(ys))
            rises = ys[0] - ys[ia]
            # launches rise first; balls rolling on the floor or carried by robots don't
            if rises < 20:
                continue
            # straightness check: real flights are smooth (path ~ displacement for short arcs)
            if path > 3.0 * max(disp, rises * 2):
                continue
            if not self.plausible_origin(P) or not self.compact(P):
                continue
            if not self.is_airborne(P):
                self.rejected_ground += 1
                continue
            x0, y0 = P[0, 1] + vx, P[0, 2] + vy
            area = np.median(P[:, 3])
            unit = self.piece_area_fn(x0, y0) if self.piece_area_fn else 30.0
            n = float(np.clip(area / max(unit, 1.0), 1.0, 6.0))
            out.append(Launch(
                t0=float(P[0, 0]), t1=float(P[-1, 0]), start=(x0, y0),
                end=(P[-1, 1] + vx, P[-1, 2] + vy), apex=(P[ia, 1] + vx, P[ia, 2] + vy),
                n_pieces=n, rises=float(rises),
                pts=[(float(p[0]), float(p[1] + vx), float(p[2] + vy)) for p in P],
                exited_top=bool(P[-1, 2] < 15),
            ))
        out.sort(key=lambda l: l.t0)
        return out

    def plausible_origin(self, P: np.ndarray) -> bool:
        """Launches start over the field (robots) or at an OUTPOST/human-player station."""
        if self.cal is None or self.fm is None:
            return True
        f = self.cal.to_field([(P[0, 1] + self.view[0], P[0, 2] + self.view[1])])[0]
        if self.fm.in_field(f, 1.5):
            return True
        for name, e in self.fm.elements.items():
            if "outpost" in name and np.linalg.norm(f - np.array(e["center"])) < 3.0:
                return True
        return False

    def compact(self, P: np.ndarray) -> bool:
        """Reject big moving blobs (people in yellow shirts) - pieces are small and similar-sized."""
        if self.piece_area_fn is None:
            return True
        unit = self.piece_area_fn(P[0, 1] + self.view[0], P[0, 2] + self.view[1])
        return float(np.median(P[:, 3])) < 6 * max(unit, 4.0)

    def is_airborne(self, P: np.ndarray) -> bool:
        """Reject pieces rolling/pushed along the carpet; keep only real launches.

        Perspective makes a ball rolling away from the camera move *up* the image, so rising
        alone proves nothing. A launched piece must show at least one airborne signature:
          * gravity: clear downward (positive-y) curvature of y(t),
          * impossible ground speed: interpreted as a floor point it would move > 9 m/s,
          * leaving the floor: it projects outside the field boundary (above the far wall).
        """
        t = P[:, 0] - P[0, 0]
        xs, ys = P[:, 1] + self.view[0], P[:, 2] + self.view[1]
        ppm = 60.0
        if self.cal is not None:
            ppm = max(self.cal.px_per_meter((xs[0], ys[0])) / 2, 5.0)
        # 1) gravity curvature in image space (px/s^2). 9.8 m/s^2 seen near-side-on ~ ppm*9.8.
        if len(t) >= 6 and t[-1] > 0.08:
            a = np.polyfit(t, ys, 2)[0] * 2
            resid = ys - np.polyval(np.polyfit(t, ys, 2), t)
            if a > 2.5 * ppm and np.std(resid) < 6:
                return True
        if self.cal is None or self.fm is None:
            return False
        f = self.cal.to_field(np.stack([xs, ys], 1))
        # 3) projects off the field (balls in the air above/behind the far guardrail)
        out_of_field = np.array([not self.fm.in_field(p, 0.3) for p in f])
        if out_of_field.mean() > 0.3 and not out_of_field[0]:
            return True
        # 2) implausible floor speed
        dt = np.diff(t)
        ok = dt > 1e-3
        if ok.sum() >= 3:
            sp = np.linalg.norm(np.diff(f, axis=0), axis=1)[ok] / dt[ok]
            if np.median(sp) > 9.0:
                return True
        return False

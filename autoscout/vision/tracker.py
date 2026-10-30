"""Fixed-slot robot tracker.

An FRC match always has exactly three robots per alliance on the field, so instead of a
generic multi-object tracker we keep 3 slots per alliance and associate detections to them
in *field coordinates* (meters). Slots coast through occlusions (refs, other robots, piles of
game pieces) instead of spawning new identities, which is what makes per-robot stats work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Det

MAX_SPEED = 5.5  # m/s, generous upper bound for FRC robots


@dataclass
class Slot:
    alliance: str
    index: int
    pos: np.ndarray | None = None  # field (x, y)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_t: float = -1.0
    hist: np.ndarray | None = None  # appearance (HSV histogram)
    box: Det | None = None  # last matched image box (full-frame)
    track: list = field(default_factory=list)  # (t, x, y, observed, box)

    @property
    def key(self) -> str:
        return f"{self.alliance}{self.index}"

    def predicted(self, t: float) -> np.ndarray:
        if self.pos is None:
            return None
        dt = min(max(t - self.last_t, 0.0), 1.0)
        return self.pos + self.vel * dt


def appearance(img: np.ndarray, d: Det) -> np.ndarray | None:
    crop = img[max(0, int(d.y0)): int(d.y1), max(0, int(d.x0)): int(d.x1)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, None).flatten()


class SlotTracker:
    def __init__(self, cal, field_model):
        self.cal = cal
        self.fm = field_model
        self.slots = {a: [Slot(a, i) for i in range(3)] for a in ("red", "blue")}

    def all_slots(self) -> list[Slot]:
        return self.slots["red"] + self.slots["blue"]

    def update(self, t: float, frame: np.ndarray, dets: list[Det]) -> None:
        foot = self.cal.to_field([d.foot for d in dets]) if dets else np.zeros((0, 2))
        for a in ("red", "blue"):
            idx = [i for i, d in enumerate(dets) if d.alliance == a and self.fm.in_field(foot[i], 0.6)]
            self._update_alliance(t, frame, a, [dets[i] for i in idx], foot[idx] if idx else np.zeros((0, 2)))
        for s in self.all_slots():
            if s.pos is not None and (not s.track or s.track[-1][0] != t):
                p = s.predicted(t)
                s.track.append((t, float(p[0]), float(p[1]), False, None))

    def _update_alliance(self, t, frame, a, dets, pts):
        slots = self.slots[a]
        apps = [appearance(frame, d) for d in dets]
        live = [s for s in slots if s.pos is not None]
        free = [s for s in slots if s.pos is None]
        used_d = set()
        if live and dets:
            cost = np.full((len(live), len(dets)), 1e6)
            for i, s in enumerate(live):
                p = s.predicted(t)
                gate = 1.2 + MAX_SPEED * min(t - s.last_t, 4.0)
                for j in range(len(dets)):
                    dist = float(np.linalg.norm(pts[j] - p))
                    if dist > gate:
                        continue
                    app = 0.0
                    if s.hist is not None and apps[j] is not None:
                        app = cv2.compareHist(s.hist, apps[j], cv2.HISTCMP_BHATTACHARYYA)
                    cost[i, j] = dist + 2.0 * app - 0.5 * dets[j].conf
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e5:
                    self._assign(live[i], t, dets[j], pts[j], apps[j])
                    used_d.add(j)
        rest = [j for j in range(len(dets)) if j not in used_d]
        rest.sort(key=lambda j: -dets[j].conf)
        for j in rest:
            # Never put two slots on the same robot.
            if any(s.pos is not None and np.linalg.norm(s.predicted(t) - pts[j]) < 0.8 for s in slots):
                continue
            if free:
                self._assign(free.pop(0), t, dets[j], pts[j], apps[j], init=True)
            else:
                # Re-acquire a lost slot only with a robot that could be the same one: reachable
                # from where it was last seen, and looking like it. Otherwise a slot whose robot
                # left the view would jump onto a different robot and inherit its stats.
                best, best_c = None, 1e9
                for s in slots:
                    gone = t - s.last_t
                    if gone <= 1.5 or s.pos is None:
                        continue
                    reach = 1.0 + MAX_SPEED * min(gone, 6.0)
                    dist = float(np.linalg.norm(pts[j] - s.pos))
                    app = 0.5
                    if s.hist is not None and apps[j] is not None:
                        app = cv2.compareHist(s.hist, apps[j], cv2.HISTCMP_BHATTACHARYYA)
                    ok = (dist <= reach and app < 0.55) or (gone > 8.0 and app < 0.4)
                    c = app + dist / max(reach, 1.0)
                    if ok and c < best_c:
                        best, best_c = s, c
                if best is not None and dets[j].conf > 0.4:
                    self._assign(best, t, dets[j], pts[j], apps[j], init=True)

    def _assign(self, s: Slot, t, d: Det, p, app, init=False):
        if s.pos is not None and not init:
            dt = max(t - s.last_t, 1e-3)
            v = (p - s.pos) / dt
            sp = np.linalg.norm(v)
            if sp > MAX_SPEED:
                v = v / sp * MAX_SPEED
            s.vel = 0.6 * s.vel + 0.4 * v if dt < 1.0 else np.zeros(2)
            s.pos = 0.5 * s.pos + 0.5 * p if dt < 0.15 else p
        else:
            s.pos = np.array(p, dtype=float)
            s.vel = np.zeros(2)
        s.last_t = t
        s.box = d
        if app is not None:
            s.hist = app if s.hist is None else 0.9 * s.hist + 0.1 * app
        s.track.append((t, float(s.pos[0]), float(s.pos[1]), True,
                        (round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1))))

    # ---- queries used by game logic ----------------------------------------------------
    def position_at(self, key: str, t: float):
        s = next(s for s in self.all_slots() if s.key == key)
        return slot_position_at(s.track, t)


def slot_position_at(track, t):
    """(x, y, observed, box) of a slot track nearest in time to t (None if no data)."""
    if not track:
        return None
    ts = np.array([r[0] for r in track])
    i = int(np.argmin(np.abs(ts - t)))
    if abs(ts[i] - t) > 1.0:
        return None
    r = track[i]
    # Use the most recent observed box within 1 s for image-space association.
    box = None
    for k in range(i, max(-1, i - 15), -1):
        if track[k][3] and t - track[k][0] < 1.0:
            box = track[k][4]
            break
    return r[1], r[2], r[3], box

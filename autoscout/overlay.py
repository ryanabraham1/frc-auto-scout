"""Broadcast scoreboard reader.

Every FRC broadcast overlay shows a match timer flanked by the blue and red alliance
scores, but the layout differs per event. We locate those three boxes automatically by
OCR-ing the top band of a handful of frames, then read just those boxes every ~0.5 s to
build a score timeline: the backbone for attributing points to robots in time.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import ocr
from .digits import DigitReader
from .video import Video

TIMER_RE = re.compile(r"^(\d)[:.;,](\d\d)$")
INT_RE = re.compile(r"^\d{1,3}$")

Box = tuple[int, int, int, int]


@dataclass
class OverlayLayout:
    timer: Box
    blue_score: Box
    red_score: Box
    frame_size: tuple[int, int]  # (w, h)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: tuple(d[k]) for k in ("timer", "blue_score", "red_score", "frame_size")})


def _center(b: Box):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def _alliance_of(img: np.ndarray, box: Box) -> str | None:
    """Classify the score box background as blue/red from the dominant saturated hue."""
    x, y, w, h = box
    pad = max(4, h // 3)
    crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1] > 90
    if sat.sum() < 20:
        return None
    hue = hsv[..., 0][sat]
    blue = np.sum((hue > 95) & (hue < 130))
    red = np.sum((hue < 10) | (hue > 165))
    if max(blue, red) < 0.25 * sat.sum():
        return None
    return "blue" if blue > red else "red"


def _detect_in_frame(img: np.ndarray, band_frac: float = 0.3):
    band = img[: int(img.shape[0] * band_frac)]
    toks = ocr.read_text(band)
    timers = [(b, t) for b, t, c in toks if TIMER_RE.match(t.replace(" ", ""))]
    if not timers:
        return None
    tbox, _ = max(timers, key=lambda bt: bt[0][3])
    tcx, tcy = _center(tbox)
    th = tbox[3]
    nums = [b for b, t, c in toks if INT_RE.match(t.replace(" ", "")) and b[3] >= 0.6 * th
            and abs(_center(b)[1] - tcy) < 0.6 * th]
    left = [b for b in nums if _center(b)[0] < tcx]
    right = [b for b in nums if _center(b)[0] > tcx]
    if not left or not right:
        return None
    lb = min(left, key=lambda b: tcx - _center(b)[0])
    rb = min(right, key=lambda b: _center(b)[0] - tcx)
    return tbox, lb, rb


def _median_box(boxes: list[Box], width_mult: float) -> Box:
    a = np.array(boxes, dtype=float)
    cx = np.median(a[:, 0] + a[:, 2] / 2)
    cy = np.median(a[:, 1] + a[:, 3] / 2)
    h = np.median(a[:, 3]) * 1.15
    w = max(np.max(a[:, 2]) * 1.1, h * width_mult)
    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


def detect_layout(video: Video, n_samples: int = 10) -> OverlayLayout:
    ts = np.linspace(video.duration * 0.15, video.duration * 0.8, n_samples)
    hits = []
    img_ref = None
    for t in ts:
        img = video.frame_at(float(t))
        if img is None:
            continue
        r = _detect_in_frame(img)
        if r:
            hits.append(r)
            img_ref = img
    if len(hits) < 2:
        raise RuntimeError("could not locate scoreboard (timer + scores) in overlay")
    timer = _median_box([h[0] for h in hits], 2.6)
    left = _median_box([h[1] for h in hits], 2.1)
    right = _median_box([h[2] for h in hits], 2.1)
    # Guard against score boxes overlapping the timer box.
    lx, ly, lw, lh = left
    if lx + lw > timer[0]:
        lw = timer[0] - lx - 2
    rx, ry, rw, rh = right
    if rx < timer[0] + timer[2]:
        shift = timer[0] + timer[2] + 2 - rx
        rx, rw = rx + shift, rw - shift
    left, right = (lx, ly, lw, lh), (rx, ry, rw, rh)
    blue, red = left, right
    la, ra = _alliance_of(img_ref, left), _alliance_of(img_ref, right)
    if la == "red" or ra == "blue":
        blue, red = right, left
    return OverlayLayout(timer=timer, blue_score=blue, red_score=red,
                         frame_size=(video.width, video.height))


def _crop(img, box: Box):
    x, y, w, h = box
    return img[max(0, y): y + h, max(0, x): x + w]


def _parse_timer(txt: str) -> int | None:
    m = TIMER_RE.match(txt)
    if not m:
        digits = re.sub(r"\D", "", txt)
        if len(digits) == 3:
            m = (digits[0], digits[1:])
        else:
            return None
        return int(m[0]) * 60 + int(m[1]) if int(m[1]) < 60 else None
    mins, secs = int(m.group(1)), int(m.group(2))
    return mins * 60 + secs if secs < 60 else None


@dataclass
class ScoreSample:
    t: float
    timer: int | None
    blue: int | None
    red: int | None


@dataclass
class MatchClock:
    """Maps video time <-> match time. Match time 0 = start of AUTO."""
    auto_len: int
    teleop_len: int
    auto_end_vt: float  # video time when the AUTO clock hits 0
    teleop_end_vt: float  # video time when the TELEOP clock hits 0

    @property
    def start_vt(self) -> float:
        return self.auto_end_vt - self.auto_len

    @property
    def teleop_start_vt(self) -> float:
        return self.teleop_end_vt - self.teleop_len

    def phase(self, vt: float) -> tuple[str, float]:
        """(phase, clock seconds remaining) at a video time."""
        if vt < self.start_vt:
            return "pre", float(self.auto_len)
        if vt <= self.auto_end_vt:
            return "auto", self.auto_end_vt - vt
        if vt < self.teleop_start_vt:
            return "pause", float(self.teleop_len)
        if vt <= self.teleop_end_vt:
            return "teleop", self.teleop_end_vt - vt
        return "post", 0.0

    def video_time(self, phase: str, remaining: float) -> float:
        return (self.auto_end_vt if phase == "auto" else self.teleop_end_vt) - remaining

    def to_dict(self):
        return asdict(self)


@dataclass
class ScoreTimeline:
    samples: list[ScoreSample]
    clock: MatchClock | None
    blue: np.ndarray = field(default=None)  # cleaned, aligned with samples
    red: np.ndarray = field(default=None)
    # extra per-alliance counters some broadcasts show (2025: algae scored), cleaned, aligned with samples
    counters: dict = field(default_factory=dict)

    @property
    def times(self) -> np.ndarray:
        return np.array([s.t for s in self.samples])

    def deltas(self, alliance: str) -> list[tuple[float, int]]:
        """(video_time, +points) for every increase in the cleaned score."""
        arr = self.blue if alliance == "blue" else self.red
        t = self.times
        out = []
        for i in range(1, len(arr)):
            d = int(arr[i] - arr[i - 1])
            if d > 0:
                out.append((float(t[i]), d))
        return out

    def counter_deltas(self, name: str) -> list[tuple[float, int]]:
        """(video_time, +count) for every increase of an overlay counter (e.g. "red_algae")."""
        arr = self.counters.get(name)
        if arr is None:
            return []
        t = self.times
        return [(float(t[i]), int(arr[i] - arr[i - 1])) for i in range(1, len(arr)) if arr[i] > arr[i - 1]]

    def to_dict(self):
        d = {
            "t": self.times.round(3).tolist(),
            "blue": self.blue.astype(int).tolist(),
            "red": self.red.astype(int).tolist(),
            "timer": [s.timer for s in self.samples],
            "clock": self.clock.to_dict() if self.clock else None,
        }
        if self.counters:
            d["counters"] = {k: np.asarray(v).astype(int).tolist() for k, v in self.counters.items()}
        return d


def clean_scores(raw: list[int | None], confirm: int = 2, window: int = 4, max_rise: int = 15) -> np.ndarray:
    """Turn noisy OCR reads into a non-decreasing score series.

    A new value is accepted only once `confirm` of the next `window` reads are consistent with
    it: equal, or higher by no more than `max_rise` points per read (scores that climb ball by
    ball, as in 2026, never repeat a value). A misread like 4 -> 423 -> 4 is rejected.
    """
    out = np.zeros(len(raw), dtype=float)
    cur = 0
    for i, v in enumerate(raw):
        if v is not None and v != cur:
            nxt = [(k, x) for k, x in enumerate(raw[i + 1: i + 1 + window], 1) if x is not None]
            support = sum(1 for k, x in nxt if v <= x <= v + max_rise * k)
            nxt = [x for _, x in nxt]
            if v > cur and support >= min(confirm, len(nxt)) and len(nxt) > 0:
                cur = v
            elif v < cur and support >= window - 1 and len(nxt) >= window - 1:
                cur = v  # genuine correction (e.g. overlay reset); rare
        out[i] = cur
    # second pass: running max keeps it monotone except for confirmed corrections
    return out


def fit_clock(samples: list[ScoreSample], auto_len: int, teleop_len: int, pause: float = 3.0) -> MatchClock | None:
    obs = [(s.t, s.timer) for s in samples if s.timer is not None]
    if len(obs) < 10:
        return None
    tele = np.array([t + r for t, r in obs if r > auto_len + 1 and r <= teleop_len])
    if len(tele) < 5:
        return None
    # mode of the anchor estimates (robust to misreads)
    def mode_anchor(a):
        best, best_n = None, -1
        for c in a:
            n = np.sum(np.abs(a - c) <= 1.2)
            if n > best_n:
                best, best_n = c, n
        sel = a[np.abs(a - best) <= 1.2]
        return float(np.median(sel)) + 0.5  # clock shows ceil; +0.5 centers it

    teleop_end = mode_anchor(tele)
    teleop_start = teleop_end - teleop_len
    auto = np.array([t + r for t, r in obs if r <= auto_len and t < teleop_start - 0.5])
    auto = auto[np.abs(auto - (teleop_start - pause)) < 8] if len(auto) else auto
    auto_end = mode_anchor(auto) if len(auto) >= 3 else teleop_start - pause
    return MatchClock(auto_len, teleop_len, auto_end, teleop_end)


def read_timeline(video: Video, layout: OverlayLayout, auto_len: int, teleop_len: int,
                  interval: float = 0.5, t0: float = 0.0, t1: float | None = None,
                  counters: dict | None = None) -> ScoreTimeline:
    """Match clock and live scores; `counters` ({name: [x, y, w, h]}, e.g. "red_algae") are extra
    overlay numbers read from the same frames."""
    samples = []
    raw_counters = {k: [] for k in counters or {}}
    step = max(1, int(round(video.fps * interval)))
    dr = DigitReader()
    for t, img in video.iter_frames(t0, t1, step=step):
        for k, box in (counters or {}).items():
            c = dr.read(_crop(img, tuple(box)), "counter")
            raw_counters[k].append(int(c) if c and len(c) <= 2 else None)
        tt = dr.read(_crop(img, layout.timer), "timer")
        b = dr.read(_crop(img, layout.blue_score), "score")
        r = dr.read(_crop(img, layout.red_score), "score")
        timer = None
        if tt and len(tt) in (2, 3):
            m, ss = (int(tt[0]), int(tt[1:])) if len(tt) == 3 else (0, int(tt))
            timer = m * 60 + ss if ss < 60 else None
        samples.append(ScoreSample(
            t=t, timer=timer,
            blue=int(b) if b and len(b) <= 3 else None,
            red=int(r) if r and len(r) <= 3 else None,
        ))
    clock = fit_clock(samples, auto_len, teleop_len)
    tl = ScoreTimeline(samples=samples, clock=clock)
    # Only trust scores within the match window (+ a few seconds of post-match counting).
    if clock:
        lo, hi = clock.start_vt - 2, clock.teleop_end_vt + 6
        for i, s in enumerate(samples):
            if s.t < lo:
                s.blue = s.red = 0
            elif s.t > hi:
                s.blue = s.red = None  # post-match graphics/replays: ignore
            for v in raw_counters.values():
                v[i] = 0 if s.t < lo else None if s.t > hi else v[i]
    tl.blue = clean_scores([s.blue for s in samples])
    tl.red = clean_scores([s.red for s in samples])
    tl.counters = {k: clean_scores(v, max_rise=3) for k, v in raw_counters.items()}
    return tl

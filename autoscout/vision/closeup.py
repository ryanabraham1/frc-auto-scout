"""Bumper numbers read in broadcast close-up views (picture-in-picture reef / hub cams).

A close-up view is a rectangle of the broadcast frame named after the field element it shows
(e.g. 'red_hub'), set per event in the calibration (`aux_views`). Its robots are large, so their
bumper numbers read far more often than in the wide shot; the read says *which team* was at
that element at that moment, which the season plugin / identity solver turn into evidence.
"""
from __future__ import annotations

import re

import numpy as np

from .. import ocr

MIN_PARTIAL = 3  # shortest digit fragment accepted as a partial read ("678" of 1678)


def match_team(digits: str, teams: list[int]) -> tuple[int, float] | None:
    """(team, fraction of its number read) when `digits` identifies exactly one team.

    Exact reads always count; a fragment counts when it is a substring of exactly one of the
    match's team numbers (bumpers are often half hidden behind the element or a banner).
    """
    digits = re.sub(r"\D", "", digits)
    if not digits:
        return None
    for t in teams:
        if digits == str(t):
            return t, 1.0
    if len(digits) < MIN_PARTIAL:
        return None
    hits = [t for t in teams if digits in str(t)]
    if len(hits) == 1:
        return hits[0], len(digits) / len(str(hits[0]))
    return None


def read_view(crop: np.ndarray, teams: list[int]) -> dict[int, float]:
    """Teams read in one close-up crop -> best confidence (scaled down for partial reads)."""
    import cv2

    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return {}
    if crop.shape[0] < 360:  # small PiP: upscale so bumper digits reach OCR size
        s = 360 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    out: dict[int, float] = {}
    for img in (crop, clahe):
        for _, txt, conf in ocr.reader().readtext(img, allowlist="0123456789"):
            m = match_team(txt, teams)
            if m is None or conf < 0.3:
                continue
            team, frac = m
            c = float(conf) * (frac if frac < 1 else 1.0)
            if c > out.get(team, 0.0):
                out[team] = c
    return out


def sample_times(views: dict, timeline, clock, offsets=(4.0, 2.5, 1.0), min_gap: float = 1.5,
                 every: float = 3.0, budget: int = 300) -> list[tuple[float, str]]:
    """(video time, view) pairs to read.

    Shortly before every score jump of the view's alliance (the scorer is at the element then),
    plus every `every` seconds through the match: identity needs reads at any time, and a view
    that is mostly covered (sponsor banner, element in the way) only shows a bumper now and then.
    At most one read per view per `min_gap` s; `budget` caps the total.
    """
    jump, steady = [], []
    for name in views:
        alliance = next((a for a in ("red", "blue") if name.startswith(a)), None)
        if timeline is not None and alliance:
            for t, _ in timeline.deltas(alliance):
                jump += [(round(t - off, 1), name) for off in offsets]
        if clock is not None and every:
            steady += [(round(float(t), 1), name)
                       for t in np.arange(clock.start_vt, clock.teleop_end_vt + 2, every)]
    done = set()

    def dedupe(pairs):
        out = []
        for tt, name in pairs:
            bucket = (round(tt / min_gap), name)
            if tt >= 0 and bucket not in done:
                done.add(bucket)
                out.append((tt, name))
        return out

    jump, steady = dedupe(sorted(jump)), dedupe(sorted(steady))  # jump samples win shared buckets
    if len(jump) + len(steady) > budget:
        n_jump = min(len(jump), max(budget - len(steady), 2 * budget // 3))
        jump, steady = _evenly(jump, n_jump), _evenly(steady, budget - n_jump)
    return sorted(jump + steady)


def _evenly(items: list, n: int) -> list:
    if n <= 0:
        return []
    if len(items) <= n:
        return items
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]

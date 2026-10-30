"""Assign tracked robot slots to team numbers.

Evidence sources (combined as log-likelihoods over the 3! slot->team permutations per alliance):
  1. Bumper-number OCR on robot crops, matched against the alliance's 3 known team numbers.
  2. Official per-station results (e.g. which station climbed) vs. each slot's behaviour.
  3. Manual overrides (dashboard / event config) - always win.
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict

import numpy as np

from .. import ocr


def digit_similarity(read: str, team: int) -> float:
    """0..1 similarity between an OCR read and a team number (handles partial reads)."""
    a, b = re.sub(r"\D", "", read), str(team)
    if not a:
        return 0.0
    if a == b:
        return 1.0
    # longest common substring, normalized
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best / max(len(a), len(b)) * (0.9 if len(a) >= 2 else 0.5)


def read_bumper(frame, box) -> list[tuple[str, float]]:
    """OCR candidates for the number on a robot's bumper (lower part of its box)."""
    import cv2

    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = y1 - y0, x1 - x0
    crop = frame[max(0, y0 + int(0.35 * h)): y1 + 3, max(0, x0 - 4): x1 + 4]
    if crop.shape[0] < 10 or crop.shape[1] < 20:
        return []
    scale = max(2.0, 96 / crop.shape[0])
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    out = []
    for img in (big, clahe):
        for _, txt, conf in ocr.reader().readtext(img, allowlist="0123456789"):
            if txt:
                out.append((txt, float(conf)))
    return out


def bumper_color(frame, box) -> str | None:
    """Alliance colour of a robot's bumper ("red" / "blue"), or None when unclear."""
    import cv2

    x0, y0, x1, y1 = [int(v) for v in box]
    strip = frame[max(0, y0 + int(0.6 * (y1 - y0))): y1, max(0, x0): x1]
    if strip.size == 0:
        return None
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 90) & (hsv[..., 2] > 50)
    hue = hsv[..., 0]
    red = (sat & ((hue < 10) | (hue > 165))).mean()
    blue = (sat & (hue > 98) & (hue < 130)).mean()
    if max(red, blue) < 0.08:
        return None
    return "red" if red > 2 * blue else "blue" if blue > 2 * red else None


def read_team_numbers(frame, box, all_teams: list[int]) -> dict[int, float]:
    """Team numbers read on a robot's bumper -> evidence weight.

    Apple Vision (macOS) rarely invents digits, so only exact team numbers count. The EasyOCR
    fallback is noisier: a partial read counts only if it names exactly one team.
    """
    import cv2

    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = y1 - y0, x1 - x0
    out: dict[int, float] = {}
    if not ocr.has_vision():
        for txt, conf in read_bumper(frame, box):
            d = re.sub(r"\D", "", txt)
            hits = [t for t in all_teams if len(d) >= 3 and d in str(t)]
            if len(hits) == 1:
                out[hits[0]] = out.get(hits[0], 0.0) + (0.4 + 0.8 * len(d) / len(str(hits[0]))) * max(conf, 0.2)
        return out
    crop = frame[max(0, y0 + int(0.35 * h)): y1 + int(0.12 * h), max(0, x0 - int(0.12 * w)): x1 + int(0.12 * w)]
    if crop.shape[0] < 8 or crop.shape[1] < 16:
        return out
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for txt, conf in ocr.read_lines(crop, [str(t) for t in all_teams]):
        for d in re.findall(r"\d{3,5}", txt):
            if int(d) in all_teams:
                out[int(d)] = out.get(int(d), 0.0) + 1.5 * max(conf, 0.3)
    return out


class IdentityEvidence:
    def __init__(self):
        # score[(slot_key, team)] -> accumulated log-likelihood-ish score
        self.score: dict[tuple[str, int], float] = defaultdict(float)
        self.notes: list[str] = []
        self.timed: list[tuple[float, str, int, float]] = []  # (t, slot, team, value) from OCR
        self.t: float | None = None  # time of the observation being added

    def add(self, slot_key: str, team: int, value: float, why: str = ""):
        self.score[(slot_key, team)] += value
        if self.t is not None:
            self.timed.append((round(self.t, 2), slot_key, team, round(value, 3)))
        if why:
            self.notes.append(f"{slot_key}->{team}: {value:+.2f} {why}")

    def ocr_bumper(self, frame, slot_key: str, box, teams: list[int], all_teams: list[int] | None = None) -> None:
        """Read the bumper number and score it against the alliance's teams.

        Reads are often partial ("12", "696"); a fragment is evidence for a team when it is a
        substring of exactly one candidate number in the match (all six teams if given).
        """
        for txt, conf in read_bumper(frame, box):
            digits = re.sub(r"\D", "", txt)
            if len(digits) < 2:
                continue
            pool = all_teams or teams
            hits = [t for t in pool if digits in str(t)]
            if len(hits) == 1 and hits[0] in teams:
                frac = len(digits) / len(str(hits[0]))
                self.add(slot_key, hits[0], (0.6 + 1.4 * frac) * max(conf, 0.2), f"ocr '{digits}'")
                continue
            for team in teams:
                s = digit_similarity(digits, team)
                if s > 0.6:
                    self.add(slot_key, team, s * conf)

    def vision_bumper(self, frame, slot_key: str, box, teams: list[int], all_teams: list[int]) -> None:
        """Bumper read of a slot's box; a box whose bumper is the other alliance's colour means the
        slot is on the wrong robot, so it gives no evidence."""
        col = bumper_color(frame, box)
        if col and col != slot_key[:-1]:
            return
        for team, v in read_team_numbers(frame, box, all_teams).items():
            if team in teams:
                self.add(slot_key, team, v, f"bumper '{team}'")

    def solve(self, slot_keys: list[str], teams: list[int], overrides: dict[str, int] | None = None):
        """Best permutation; returns ({slot_key: team}, {slot_key: confidence 0..1}).

        A slot's confidence is how much worse the best assignment gets if that slot were given
        any other team (so one well-identified robot is confident even if the other two tie).
        """
        overrides = overrides or {}
        scored = []
        for perm in itertools.permutations(teams):
            m = dict(zip(slot_keys, perm))
            if any(m.get(k) != v for k, v in overrides.items() if k in m):
                continue
            scored.append((sum(self.score.get((k, t), 0.0) for k, t in m.items()), m))
        if not scored:
            return dict(zip(slot_keys, teams)), {k: 0.0 for k in slot_keys}
        best_s, best = max(scored, key=lambda x: x[0])
        conf = {}
        for k in slot_keys:
            if k in overrides:
                conf[k] = 1.0
                continue
            alt = [s for s, m in scored if m[k] != best[k]]
            margin = best_s - max(alt) if alt else 10.0
            conf[k] = round(float(1 - np.exp(-max(margin, 0.0))), 2)
        return best, conf


def identity_timeline(timed, slot_keys, teams, global_map, t0, t1, overrides=None,
                      step: float = 5.0, tau: float = 8.0, prior: float = 1.0):
    """Time-varying slot->team mapping.

    The tracker can swap two robots' slots during a collision, so one mapping per match smears
    one robot's stats into another's. Every `step` seconds the permutation is re-solved from
    bumper reads near that time (weight exp(-|dt|/tau)) plus a small pull toward the global
    mapping. Returns {slot: [(t_start, t_end, team), ...]}.
    """
    overrides = overrides or {}
    perms = [dict(zip(slot_keys, p)) for p in itertools.permutations(teams)]
    perms = [m for m in perms if all(m.get(k) == v for k, v in overrides.items() if k in m)] or perms
    obs = [(t, s, tm, v) for t, s, tm, v in timed if s in slot_keys]
    out = {k: [] for k in slot_keys}
    prev = None
    for ts in np.arange(t0, t1 + step, step):
        score = defaultdict(float)
        for t, s, tm, v in obs:
            w = np.exp(-abs(t - ts) / tau)
            if w > 0.02:
                score[(s, tm)] += v * w
        best = max(perms, key=lambda m: sum(score[(k, m[k])] for k in slot_keys)
                   + prior * sum(m[k] == global_map.get(k) for k in slot_keys)
                   + 0.5 * prior * (prev is not None and m == prev))
        for k in slot_keys:
            seg = out[k]
            if seg and seg[-1][2] == best[k]:
                seg[-1] = (seg[-1][0], ts + step / 2, best[k])
            else:
                seg.append((ts - step / 2, ts + step / 2, best[k]))
        prev = best
    return out

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


class IdentityEvidence:
    def __init__(self):
        # score[(slot_key, team)] -> accumulated log-likelihood-ish score
        self.score: dict[tuple[str, int], float] = defaultdict(float)
        self.notes: list[str] = []

    def add(self, slot_key: str, team: int, value: float, why: str = ""):
        self.score[(slot_key, team)] += value
        if why:
            self.notes.append(f"{slot_key}->{team}: {value:+.2f} {why}")

    def ocr_bumper(self, frame, slot_key: str, box, teams: list[int]) -> None:
        x0, y0, x1, y1 = [int(v) for v in box]
        h = y1 - y0
        crop = frame[max(0, y0 + h // 3): y1 + 2, max(0, x0 - 4): x1 + 4]
        if crop.shape[0] < 12 or crop.shape[1] < 24:
            return
        for (_, txt, conf) in ocr.read_text(crop):
            if conf < 0.3:
                continue
            for team in teams:
                s = digit_similarity(txt, team)
                if s > 0.45:
                    self.add(slot_key, team, s * conf * 2.0)

    def solve(self, slot_keys: list[str], teams: list[int], overrides: dict[str, int] | None = None):
        """Best permutation; returns ({slot_key: team}, confidence 0..1)."""
        overrides = overrides or {}
        best, best_s, second = None, -1e9, -1e9
        for perm in itertools.permutations(teams):
            m = dict(zip(slot_keys, perm))
            if any(m.get(k) != v for k, v in overrides.items() if k in m):
                continue
            s = sum(self.score.get((k, t), 0.0) for k, t in m.items())
            if s > best_s:
                best, best_s, second = m, s, best_s
            elif s > second:
                second = s
        if best is None:
            best = dict(zip(slot_keys, teams))
        margin = best_s - second if second > -1e9 else best_s
        conf = float(1 - np.exp(-max(margin, 0.0)))
        if overrides and all(k in overrides for k in slot_keys):
            conf = 1.0
        return best, conf

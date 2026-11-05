"""Bumper reads that add evidence across a tracklet's best views.

The live reader judges one frame at a time, and each frame must pass its thresholds alone, so a legible
bumper in a blurry stream often yields nothing. Here the alliance's teams are scored on a tracklet's
largest views and the evidence is added up, so several weak frames can agree. Only tracklets with no
clear read are looked at, after labelling, and the reads feed the targeted repair rather than a re-solve.
"""
from __future__ import annotations

import cv2
import numpy as np

from .bumper_ocr import bumper_windows

MIN_OBS = 15  # shorter tracklets have too few distinct views
VIEWS = 10  # largest views scored per tracklet
VIEW_GAP = 0.2  # s between views, so they are not near-copies of one frame
MIN_BOX_W = 24  # px, as for live reads
SCALE = 3  # views are enlarged before the recognizer's windows are cut
FRAME_CLIP = 4.0  # one frame can move a team's evidence this far at most
MARGIN = 15.0  # lead over the next team needed to accept a read
READ_WEIGHT = 2.5  # bumper evidence of an accepted read


def best_views(tr, n: int = VIEWS) -> list:
    """Up to n observations: the largest boxes, at least VIEW_GAP apart."""
    obs = [o for o in tr.obs if o[3] and o[3][2] - o[3][0] >= MIN_BOX_W]
    obs.sort(key=lambda o: -(o[3][3] - o[3][1]) * (o[3][2] - o[3][0]))
    out: list = []
    for o in obs:
        if all(abs(o[0] - p[0]) >= VIEW_GAP for p in out):
            out.append(o)
            if len(out) == n:
                break
    return out


def _windows(frame: np.ndarray, box) -> list[np.ndarray]:
    """The live reader's windows, cut from an enlarged crop around the box."""
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = 12
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    crop = frame[ya:y1 + pad + 8, xa:x1 + pad]
    if not crop.size:
        return []
    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)
    return bumper_windows(big, ((x0 - xa) * SCALE, (y0 - ya) * SCALE, (x1 - xa) * SCALE, (y1 - ya) * SCALE))


def fused_scores(frames: list, teams: list[int], reader) -> dict[int, float]:
    """{team: evidence} over [(frame, box)]: each frame adds each team's best window relative to the
    teams' mean, clipped."""
    wins, owner = [], []
    for k, (frame, box) in enumerate(frames):
        ws = _windows(frame, box)
        wins += ws
        owner += [k] * len(ws)
    total = np.zeros(len(teams))
    if wins:
        scores, _ = reader.score(wins, [str(t) for t in teams])
        owner = np.array(owner)
        for k in sorted(set(owner.tolist())):
            best = scores[owner == k].max(0)
            total += np.clip(best - best.mean(), -FRAME_CLIP, FRAME_CLIP)
    return {t: float(v) for t, v in zip(teams, total)}


def _candidates(tracker, tr) -> list[int]:
    """The tracklet's alliance when its bumper colour is clear, otherwise all six teams."""
    votes = [o[4] for o in tr.obs if o[4]]
    red, blue = votes.count("red"), votes.count("blue")
    if red > 2 * blue:
        return tracker.teams["red"]
    if blue > 2 * red:
        return tracker.teams["blue"]
    return tracker.teams["red"] + tracker.teams["blue"]


def add_fused_reads(tracker, frame_at, reader=None) -> int:
    """Add a read to every long tracklet with no clear read whose best team leads by MARGIN. frame_at(t)
    returns the video frame at time t. Returns the reads added."""
    if reader is None:
        from .bumper_ocr import reader as shared_reader

        reader = shared_reader()
    added = 0
    for tr in tracker.done + tracker.active:
        if len(tr.obs) < MIN_OBS or (tr.reads and max(tr.reads.values()) >= 1.0):
            continue
        frames = []
        for o in best_views(tr):
            frame = frame_at(o[0])
            if frame is not None:
                frames.append((frame, o[3]))
        if not frames:
            continue
        ranked = sorted(fused_scores(frames, _candidates(tracker, tr), reader).items(), key=lambda x: -x[1])
        if ranked[0][1] - ranked[1][1] >= MARGIN:
            team = ranked[0][0]
            tr.reads[team] = tr.reads.get(team, 0.0) + READ_WEIGHT
            added += 1
    return added

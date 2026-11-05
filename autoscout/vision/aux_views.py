"""Keep close-up camera coordinates fixed when a broadcast inserts another panel.

The endgame layout crops/scales the same reef feeds around a barge feed. Match
static image features back to the normal two-panel frame before detecting or
measuring anything. Unseen parts stay black, rather than becoming another camera.
"""
from __future__ import annotations

import cv2
import numpy as np


def panels(frame, band):
    y, h = band
    margin = max(1, int(h * .04))
    hsv = cv2.cvtColor(frame[y + margin:y + h - margin], cv2.COLOR_BGR2HSV)
    saturated = ((hsv[..., 1] > 120) & (hsv[..., 2] > 120)).mean(0)
    cols = np.flatnonzero(saturated > .85)
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 3) + 1) if len(cols) else []
    separators = [int(g.mean()) for g in groups if 0 < len(g) <= 12]
    edges = [0, *separators, frame.shape[1]]
    return [(a, y, b - a, h) for a, b in zip(edges, edges[1:]) if b - a > frame.shape[1] * .1]


def match_view(src, dst):
    """Similarity mapping source camera pixels to reference camera pixels."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.)
    return matrix if matrix is not None and inliers.sum() >= 15 else None


class AuxViewNormalizer:
    def __init__(self, reference, views):
        self.views = sorted((views or {}).items(), key=lambda item: item[1][0])
        self.reference = reference
        self.transforms = {}
        self.band = None
        if reference is not None and len(self.views) == 2:
            y = min(r[1] for _, r in self.views)
            bottom = max(r[1] + r[3] for _, r in self.views)
            self.band = y, bottom - y

    def __call__(self, frame):
        if self.band is None:
            return frame
        regions = panels(frame, self.band)
        if len(regions) <= 2:
            return frame
        out = frame.copy()
        for (name, (x, y, w, h)), region in zip(self.views, (regions[0], regions[-1])):
            rx, ry, rw, rh = region
            pane = frame[ry:ry + rh, rx:rx + rw]
            key = name, region
            matrix = self.transforms.get(key)
            if matrix is None:
                matrix = match_view(pane, self.reference[y:y + h, x:x + w])
                if matrix is not None:
                    self.transforms[key] = matrix
            out[y:y + h, x:x + w] = (cv2.warpAffine(pane, matrix, (w, h))
                                     if matrix is not None else 0)
        return out

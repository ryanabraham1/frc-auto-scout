"""Static-scene model of the wide shot (field + structures without robots)."""
from __future__ import annotations

import cv2
import numpy as np

from ..video import Video


def median_background(video: Video, view, t0: float, t1: float, n: int = 45,
                      ref: np.ndarray | None = None, min_sim: float = 0.6) -> np.ndarray:
    """Per-pixel median over n frames sampled in [t0, t1] (robots move, so they vanish)."""
    from ..field import view_similarity

    x, y, w, h = view
    frames = []
    for t in np.linspace(t0, t1, n):
        f = video.frame_at(float(t))
        if f is None:
            continue
        if ref is not None and view_similarity(f, ref, view) < min_sim:
            continue  # camera cut
        frames.append(f[y: y + h, x: x + w])
    if not frames:
        raise RuntimeError("no frames for background")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def foreground_mask(frame_view: np.ndarray, bg: np.ndarray, thresh: int = 35) -> np.ndarray:
    d = cv2.absdiff(cv2.GaussianBlur(frame_view, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m

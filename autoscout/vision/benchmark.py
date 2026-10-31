"""One-to-one, alliance-aware scoring of hand-marked robot centres."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def score_points(detections, points, margin: float = 4.0) -> dict:
    """Each prediction and each robot may count at most once.

    Wrong-alliance boxes are both a missed robot and a false prediction. Maximise
    match cardinality rather than greedily consuming a box covering two robots.
    """
    valid = np.zeros((len(detections), len(points)), dtype=bool)
    for i, d in enumerate(detections):
        for j, (x, y, alliance) in enumerate(points):
            valid[i, j] = (d.alliance[:1] == alliance[:1]
                           and d.x0 - margin <= x <= d.x1 + margin
                           and d.y0 - margin <= y <= d.y1 + margin)
    rows, cols = linear_sum_assignment(-valid.astype(int))
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]
    tp = len(matches)
    fp, fn = len(detections) - tp, len(points) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * tp / max(2 * tp + fp + fn, 1), matches=matches)


def evaluate_model(detector, directory, veto):
    """Evaluate on the same frozen set, without downloading videos or touching results."""
    import hashlib
    import json
    import cv2
    from .detect import alliance_from_pixels

    raw = (directory / "points.json").read_bytes()
    points = json.loads(raw)
    if not points or not any(points.values()):
        raise ValueError("Detector benchmark must contain labelled robots")
    tp = fp = fn = 0
    per_frame = []
    for stem, labels in points.items():
        image = cv2.imread(str(directory / f"{stem}.jpg"))
        if image is None:
            raise FileNotFoundError(directory / f"{stem}.jpg")
        predictions = veto(image, detector(image))
        for d in predictions:
            if not d.alliance:
                d.alliance, _ = alliance_from_pixels(image, d)
        score = score_points(predictions, labels)
        tp += score["tp"]
        fp += score["fp"]
        fn += score["fn"]
        per_frame.append(dict(frame=stem, **score))
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                frames=len(points), robots=tp + fn, per_frame=per_frame,
                scoring="one-to-one-alliance-v2", dataset_sha256=hashlib.sha256(raw).hexdigest())


def improves(candidate, baseline):
    """Require a real gain without sacrificing either precision or recall."""
    return (candidate["f1"] > baseline["f1"] + 1e-9
            and candidate["precision"] >= baseline["precision"]
            and candidate["recall"] >= baseline["recall"])

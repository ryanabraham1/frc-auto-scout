"""Data engine: CORAL detector training labels (2025) from broadcast video, with no hand labelling.

CORAL is a white PVC tube. White pixels alone also catch field rails, signs, tape and white robot parts,
but those are all there when the match starts, while every CORAL that matters (carried, dropped, placed
on a REEF) appears after it. So per match:

  * background = pixels white in most frames of the first seconds (REEFs still empty),
  * candidates = white blobs in later frames that are mostly not background, CORAL-sized for their
    spot in the wide shot (about 0.3 x 0.11 m) and tube-shaped, inside the field or on a REEF close-up.

Frames go to a YOLO dataset under data/datasets/coral (one class). The labels are noisy on purpose;
the trained detector is the cleaner (relabel with it, then train again).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..field import Calibration
from ..video import Video

DATASET = config.DATA_DIR / "datasets" / "coral"
CORAL_LEN, CORAL_DIA = 0.30, 0.11  # m


def white_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 55) & (hsv[..., 2] > 175)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _cal(result: dict) -> Calibration:
    c = result["calibration"]
    return Calibration(event_key=result["event_key"], year=result["year"], view=tuple(c["view"]),
                       points={k: tuple(v) for k, v in c["points"].items()})


def regions(result: dict, base: Calibration, shape) -> list[tuple[str, np.ndarray]]:
    """(name, mask) of where CORAL may be labelled: the field in the wide shot, and each REEF close-up."""
    from ..games import get

    H, W = shape[:2]
    cal = _cal(result)
    wide = np.zeros((H, W), np.uint8)
    cv2.fillPoly(wide, [cal.field_polygon_image(get(result["year"]).field, margin=0.3).astype(np.int32)], 1)
    x, y, w, h = cal.view
    wide[:y] = 0
    wide[y + h:] = 0
    out = [("wide", wide)]
    for name, (x0, y0, x1, y1) in (base.aux_watch or {}).items():
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = 1
        out.append((name, m))
    return out


def candidates(frame: np.ndarray, bg: np.ndarray, region: np.ndarray, size_px) -> list[tuple[int, int, int, int]]:
    """CORAL-like white blobs: new (mostly not background), sized like a CORAL there, tube-shaped.
    size_px(x, y) -> (min_area, max_area) at that image spot."""
    m = white_mask(frame) & region
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        lo, hi = size_px(x + w / 2, y + h)
        if not lo <= a <= hi:
            continue
        blob = lab[y:y + h, x:x + w] == j
        if bg[y:y + h, x:x + w][blob].mean() > 0.3:
            continue
        ys, xs = np.nonzero(blob)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(float))))[::-1]
        elong = float(np.sqrt(ev[0] / max(ev[1], 1e-3)))
        if 1.4 <= elong <= 6.0 or (a >= 0.6 * hi and elong <= 6.0):  # a tube seen end-on is round
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def label_match(key: str, every: float = 2.0, start_s: float = 3.0) -> int:
    """Write labelled frames of one processed match; returns how many frames were written."""
    from ..overlay import MatchClock
    from ..pipeline import RESULTS_DIR, VIDEO_DIR

    r = json.loads((RESULTS_DIR / f"{key}.json").read_text())
    base = Calibration.load(r["event_key"])
    cal = _cal(r)
    clk = MatchClock(**r["clock"])
    v = Video(str(next(Path(VIDEO_DIR).glob(f"{r['video']}.*"))))
    first = [v.frame_at(clk.start_vt + 0.3 + k * 0.5) for k in range(int(start_s / 0.5))]
    first = [f for f in first if f is not None]
    if not first:
        return 0
    bg = (np.mean([white_mask(f) for f in first], 0) > 0.4).astype(np.float32)
    bg = cv2.dilate(bg, np.ones((5, 5), np.uint8))
    regs = regions(r, base, first[0].shape)

    def size_px(x, y, close=False):
        if close:
            return 60, 4000
        ppm = cal.px_per_meter_x(tuple(cal.to_field([(x, y)])[0]))
        full = ppm * ppm * CORAL_LEN * CORAL_DIA
        return max(12.0, 0.25 * full), 3.0 * full + 40

    (DATASET / "images").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for t in np.arange(clk.start_vt + start_s + 2, clk.teleop_end_vt, every):
        f = v.frame_at(float(t))
        if f is None:
            continue
        boxes = []
        for name, reg in regs:
            close = name != "wide"
            boxes += candidates(f, bg, reg, lambda x, y, c=close: size_px(x, y, c))
        if not boxes:
            continue
        H, W = f.shape[:2]
        stem = f"{key}_{t:.1f}"
        cv2.imwrite(str(DATASET / "images" / f"{stem}.jpg"), f)
        pad = 2
        (DATASET / "labels" / f"{stem}.txt").write_text("".join(
            f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0 + 2 * pad) / W:.6f} {(y1 - y0 + 2 * pad) / H:.6f}\n"
            for x0, y0, x1, y1 in boxes))
        n += 1
    return n


def relabel(weights: str, src: Path = DATASET, dst: Path | None = None, keep: float = 0.4, support: float = 0.15,
            iou: float = 0.3) -> tuple[int, int]:
    """Round two: label the same frames with a trained CORAL detector. Its confident boxes (>= keep) are
    labels; a round-one box stays only if the detector gives something there at least `support` (round-one
    boxes it rejects outright are mostly shirts, signs and glare). Returns (frames, boxes)."""
    from ultralytics import YOLO

    dst = dst or src.parent / f"{src.name}_r2"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)

    def iou_xyxy(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)

    n_img = n_box = 0
    for img in sorted((src / "images").glob("*.jpg")):
        f = cv2.imread(str(img))
        H, W = f.shape[:2]
        res = model.predict(f, imgsz=1920, conf=support, verbose=False)[0].boxes
        preds = list(zip(res.xyxy.tolist(), res.conf.tolist()))
        boxes = [b for b, c in preds if c >= keep]
        lab = src / "labels" / f"{img.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            _, cx, cy, w, h = map(float, line.split())
            b = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            if any(iou_xyxy(b, p) >= iou for p, _ in preds) and not any(iou_xyxy(b, k) >= 0.5 for k in boxes):
                boxes.append(b)
        dst_img = dst / "images" / img.name
        if not dst_img.exists():
            try:
                dst_img.hardlink_to(img)
            except OSError:
                cv2.imwrite(str(dst_img), f)
        (dst / "labels" / f"{img.stem}.txt").write_text("".join(
            f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n"
            for b in boxes))
        n_img += 1
        n_box += len(boxes)
    return n_img, n_box

"""Fast, self-teaching digit reader for broadcast overlays.

Overlay digits are crisp, single-font glyphs, so connected-component segmentation plus
template matching is both faster and more reliable than a general OCR model. Templates
are learned on the fly: any glyph that doesn't match an existing template confidently is
sent once to EasyOCR, and the answer becomes a new template.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ocr

GW, GH = 20, 30


def binarize(crop: np.ndarray) -> np.ndarray:
    """Foreground (text) mask: pixels that differ in polarity from the box background."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if crop.ndim == 3:
        # Colored score boxes: white text has low saturation and high value.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_bg = np.median(hsv[..., 1])
        if sat_bg > 80:
            return ((hsv[..., 1] < 70) & (hsv[..., 2] > 170)).astype(np.uint8)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = th > 0
    # Background = the majority class in the central band (edges may contain video).
    h = fg.shape[0]
    band = fg[h // 4: 3 * h // 4]
    if band.mean() > 0.5:
        fg = ~fg
    return fg.astype(np.uint8)


def segment(mask: np.ndarray, min_h_frac: float = 0.4) -> list[tuple[int, int, int, int]]:
    H, W = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h_frac * H or h > 0.97 * H or w > 0.9 * h * 1.6 or area < 0.15 * w * h:
            continue
        if x == 0 or x + w >= W:  # cut off at the crop edge
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return []
    # Keep glyphs whose height is consistent with the tallest (drops stray UI bits).
    hmax = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= 0.75 * hmax]
    boxes.sort(key=lambda b: b[0])
    # Merge components that overlap heavily in x (broken glyphs).
    merged = []
    for b in boxes:
        if merged and b[0] < merged[-1][0] + merged[-1][2] * 0.6:
            x0 = min(merged[-1][0], b[0]); y0 = min(merged[-1][1], b[1])
            x1 = max(merged[-1][0] + merged[-1][2], b[0] + b[2]); y1 = max(merged[-1][1] + merged[-1][3], b[1] + b[3])
            merged[-1] = (x0, y0, x1 - x0, y1 - y0)
        else:
            merged.append(b)
    return merged


def _norm(mask: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    g = mask[y: y + h, x: x + w].astype(np.float32)
    # Pad to a fixed aspect so narrow glyphs ("1") keep their shape.
    target_w = int(round(h * GW / GH))
    if w < target_w:
        pad = target_w - w
        g = np.pad(g, ((0, 0), (pad // 2, pad - pad // 2)))
    return cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA)


class DigitReader:
    def __init__(self, min_score: float = 0.8):
        self.templates: dict[str, list[tuple[str, np.ndarray, float]]] = {}
        self.min_score = min_score
        self.ocr_calls = 0

    def _match(self, key: str, g: np.ndarray, aspect: float) -> tuple[str | None, float]:
        best, best_s = None, -1.0
        for label, tpl, asp in self.templates.get(key, []):
            if abs(asp - aspect) > 0.25:
                continue
            a, b = g - g.mean(), tpl - tpl.mean()
            s = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
            if s > best_s:
                best, best_s = label, s
        return best, best_s

    def _learn(self, key: str, crop: np.ndarray, mask: np.ndarray, box) -> str | None:
        x, y, w, h = box
        pad = max(3, h // 4)
        # Render the glyph as black-on-white for the OCR model.
        sub = mask[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        img = np.where(sub > 0, 0, 255).astype(np.uint8)
        img = np.pad(img, pad, constant_values=255)
        self.ocr_calls += 1
        txt, conf = ocr.recognize(img, "0123456789")
        if len(txt) == 1 and conf > 0.4:
            g = _norm(mask, box)
            self.templates.setdefault(key, []).append((txt, g, w / h))
            return txt
        return None

    def _learn_line(self, key: str, mask: np.ndarray, boxes) -> str | None:
        """OCR the whole (cleaned) line; if the digit count matches the glyphs, learn them all."""
        x0 = min(b[0] for b in boxes); x1 = max(b[0] + b[2] for b in boxes)
        y0 = min(b[1] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        keep = np.zeros_like(mask)
        for x, y, w, h in boxes:
            keep[y: y + h, x: x + w] = mask[y: y + h, x: x + w]
        sub = keep[y0: y1, x0: x1]
        pad = max(4, (y1 - y0) // 3)
        img = np.pad(np.where(sub > 0, 0, 255).astype(np.uint8), pad, constant_values=255)
        self.ocr_calls += 1
        txt, conf = ocr.recognize(img, "0123456789")
        if len(txt) != len(boxes) or conf < 0.5:
            return None
        for ch, b in zip(txt, boxes):
            g = _norm(mask, b)
            if self._match(key, g, b[2] / b[3])[1] < 0.95:
                self.templates.setdefault(key, []).append((ch, g, b[2] / b[3]))
        return txt

    def read(self, crop: np.ndarray, key: str = "default") -> str | None:
        mask = binarize(crop)
        boxes = segment(mask)
        if not boxes or len(boxes) > 4:
            return None
        out = []
        for b in boxes:
            g = _norm(mask, b)
            label, s = self._match(key, g, b[2] / b[3])
            if s < self.min_score:
                if self.ocr_calls >= 400:
                    return None
                return self._learn_line(key, mask, boxes) or None
            out.append(label)
        return "".join(out)

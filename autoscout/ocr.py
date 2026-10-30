"""Shared OCR engine (EasyOCR; uses the GPU when available)."""
from __future__ import annotations

import functools
import warnings

import cv2
import numpy as np


@functools.lru_cache(maxsize=1)
def reader():
    import easyocr

    from . import config

    warnings.filterwarnings("ignore")
    # EasyOCR's GPU path is CUDA-only; on Apple silicon it runs on the (fast) CPU cores.
    return easyocr.Reader(["en"], gpu=config.device() == "cuda", verbose=False)


def read_text(img: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Detect + recognize. Returns [((x, y, w, h), text, conf)]."""
    out = []
    for box, txt, conf in reader().readtext(img):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0 = int(min(xs)), int(min(ys))
        out.append(((x0, y0, int(max(xs)) - x0, int(max(ys)) - y0), txt, float(conf)))
    return out


def recognize(crop: np.ndarray, allow: str = "0123456789") -> tuple[str, float]:
    """Recognize a single text line in a pre-cropped region."""
    if crop.size == 0:
        return "", 0.0
    h = crop.shape[0]
    if h < 48:
        s = 48 / h
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    res = reader().recognize(gray, allowlist=allow, detail=1)
    if not res:
        return "", 0.0
    best = max(res, key=lambda r: r[2])
    return best[1].replace(" ", ""), float(best[2])


@functools.lru_cache(maxsize=1)
def _vision():
    """Apple's Vision text recognizer (macOS), or None. It reads small, blurred, stylized bumper
    numbers far better than EasyOCR and almost never invents digits."""
    import sys

    if sys.platform != "darwin":
        return None
    try:
        import Vision
        from Foundation import NSData
    except ImportError:
        return None
    return Vision, NSData


def has_vision() -> bool:
    return _vision() is not None


def read_lines(img: np.ndarray, words: list[str] | None = None) -> list[tuple[str, float]]:
    """Text lines in an image as [(text, conf)]: Apple Vision when available, else EasyOCR (digits).

    `words` biases Vision toward expected strings (e.g. the match's team numbers).
    """
    vn = _vision()
    if vn is None:
        return [(txt, float(conf)) for _, txt, conf in reader().readtext(img, allowlist="0123456789")]
    Vision, NSData = vn
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return []
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(bool(words))
    if words:
        req.setCustomWords_(words)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(buf.tobytes(), len(buf)), None)
    handler.performRequests_error_([req], None)
    return [(str(c.string()), float(c.confidence())) for o in (req.results() or []) for c in o.topCandidates_(1)]

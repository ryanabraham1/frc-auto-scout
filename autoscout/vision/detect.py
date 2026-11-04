"""Robot and game-piece detectors.

RobotDetector wraps an Ultralytics YOLO model whose classes include red/blue robots
(e.g. our fine-tuned `autoscout-robots.pt`, or FRC 578's Spy Glass bumper model).
Game pieces of a single distinctive color (2026 FUEL, 2025 ALGAE) are found by color
segmentation, which is fast enough to run on every frame for trajectory tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

from .. import config


@dataclass
class Det:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    alliance: str  # "red" | "blue" | "" (unknown)
    src: int = 0  # which model of an ensemble found it (0 = our own model)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-center (where the robot touches the carpet)."""
        return (self.x0 + self.x1) / 2, self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


DEFAULT_WEIGHTS = [
    config.DATA_DIR / "models" / "autoscout-robots.engine",
    config.DATA_DIR / "models" / "autoscout-robots.pt",
    config.ROOT / "models" / "autoscout-robots.pt",
    config.DATA_DIR / "models" / "spyglass" / "model11n.pt",
]


def find_weights() -> Path | None:
    for p in available_weights():
        return p
    return None


def available_weights():
    """TensorRT is CUDA-only and must not shadow newer portable weights."""
    for p in map(Path, DEFAULT_WEIGHTS):
        if p.suffix == ".engine":
            if not p.exists():
                continue  # retired when newer weights were deployed (train_robots.promote)
            portable = p.with_suffix(".pt")
            if config.device() != "cuda" or (portable.exists() and portable.stat().st_mtime > p.stat().st_mtime):
                continue
        if Path(p).exists():
            yield p


class RobotDetector:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.25, imgsz: int | None = None):
        from ultralytics import YOLO

        weights = Path(weights) if weights else find_weights()
        if weights is None:
            raise FileNotFoundError("no robot detector weights found; run `autoscout train-robots`")
        self.weights = weights
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz or (1280 if weights.suffix == ".engine" else config.detector_imgsz())
        self.device = config.device()
        names = {int(k): v.lower() for k, v in self.model.names.items()}
        self.cls_alliance = {}
        for k, v in names.items():
            if "red" in v:
                self.cls_alliance[k] = "red"
            elif "blue" in v:
                self.cls_alliance[k] = "blue"
            elif "robot" in v:
                self.cls_alliance[k] = ""
        self.half = self.device == "cuda"

    def __call__(self, images: list[np.ndarray] | np.ndarray, conf: float | None = None,
                 imgsz: int | None = None) -> list[list[Det]]:
        single = isinstance(images, np.ndarray)
        batch = [images] if single else images
        if not batch:
            return []
        res = self.model.predict(batch, conf=self.conf if conf is None else conf, imgsz=imgsz or self.imgsz, verbose=False,
                                 classes=list(self.cls_alliance), device=self.device, **precision_args(self.half))
        out = []
        for r in res:
            dets = []
            for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append(Det(x0, y0, x1, y1, c, self.cls_alliance.get(int(k), "")))
            out.append(nms_merge(dets))
        return out[0] if single else out


def merge_scales(predictions: list[list[Det]], threshold: float = 0.3) -> list[Det]:
    """Accept confident boxes, or moderate boxes supported at another image scale.

    Unsupported moderate detections remain weak when a tracker requested weak
    evidence: they may extend an existing robot but cannot start a new track.
    """
    out = []
    for i, own in enumerate(predictions):
        others = [d for j, group in enumerate(predictions) if j != i for d in group]
        for d in own:
            if d.conf < threshold:
                continue
            supported = d.conf >= .5 or any(
                other.conf >= .2 and other.alliance == d.alliance and iou(d, other) >= .3
                for other in others)
            if d.conf >= .3 and not supported:
                if threshold >= .3:
                    continue
                d = replace(d, conf=.29)
            out.append(d)
    return nms_merge(out, thr=.4)


class MultiScaleDetector:
    """Reuse one model at two scales to recover small/occluded robots on a GPU."""
    def __init__(self, detector: RobotDetector, sizes=(1280, 1600)):
        self.detector = detector
        self.weights = detector.weights
        self.sizes = tuple(dict.fromkeys(sizes))
        self.imgsz = max(self.sizes)
        self.conf = detector.conf
        self.device = detector.device

    def __call__(self, images, conf: float | None = None):
        if not isinstance(images, np.ndarray):
            return [self(image, conf=conf) for image in images]
        threshold = self.conf if conf is None else conf
        predictions = [self.detector(images, conf=min(threshold, .2), imgsz=size) for size in self.sizes]
        return merge_scales(predictions, threshold)


def _robot_detector(weights, conf, imgsz=None, multiscale=None):
    detector = RobotDetector(weights, conf=conf, imgsz=imgsz)
    if multiscale is None:
        import os
        multiscale = os.environ.get("AUTOSCOUT_MULTISCALE", "1") != "0"
    if multiscale and detector.device in ("cuda", "mps") and detector.weights.suffix == ".pt" and detector.imgsz > 1280:
        return MultiScaleDetector(detector, sizes=(1280, detector.imgsz))
    return detector


class EnsembleDetector:
    """Union of several robot detectors (e.g. our full-robot model + Spy Glass bumper model).

    Boxes from different models describe the same robot differently (whole robot vs. bumper),
    but their ground-contact point (bottom-center) agrees, so detections of the same alliance
    whose foot points are close are merged, keeping the larger (whole-robot) box.
    """

    def __init__(self, detectors: list["RobotDetector"]):
        self.detectors = detectors
        self.weights = Path("+".join(d.weights.name for d in detectors))

    def __call__(self, image: np.ndarray, conf: float | None = None) -> list[Det]:
        dets = []
        for i, d in enumerate(self.detectors):
            for b in d(image, conf=conf):
                if not b.alliance:
                    b.alliance, _ = alliance_from_pixels(image, b)
                b.src = i
                dets.append(b)
        dets.sort(key=lambda b: -(b.w * b.h))
        out: list[Det] = []
        for b in dets:
            fx, fy = b.foot
            dup = False
            for k in out:
                kx, ky = k.foot
                if (b.alliance == k.alliance or not b.alliance) and abs(fx - kx) < 0.45 * k.w and abs(fy - ky) < 0.5 * k.h:
                    k.conf = max(k.conf, b.conf)
                    dup = True
                    break
            if not dup:
                out.append(b)
        return out


class StaticSuppressor:
    """Drop detections that sit at the same image spot for much of the match (field elements
    such as alliance-colored ramps or a REEF that look like robots). Robots never park on one spot
    for a quarter of a match (an endgame climb is ~15%)."""

    def __init__(self, cell: int = 24, min_frames: int = 40, frac: float = 0.25, expected_frames: int = 0):
        self.cell = cell
        self.counts: dict[tuple, int] = {}
        self.frames = 0
        self.min_frames = min_frames
        self.frac = frac
        # "A quarter of the match" means of the whole match. Measured against the frames seen so far,
        # a robot parked for a few seconds near the start (a shooter emptying its preload behind its HUB)
        # passed for a field element and was dropped for the rest of the match.
        self.expected_frames = expected_frames

    def _key(self, d: Det):
        return (d.alliance, int(d.foot[0] // self.cell), int(d.foot[1] // self.cell), int(d.w // (2 * self.cell)))

    def __call__(self, dets: list[Det]) -> list[Det]:
        self.frames += 1
        for d in dets:
            k = self._key(d)
            self.counts[k] = self.counts.get(k, 0) + 1
        if self.frames < self.min_frames:
            return dets
        lim = self.frac * max(self.frames, self.expected_frames)
        return [d for d in dets if self.counts.get(self._key(d), 0) < lim]


class PersonVeto:
    """Drop 'robots' that are people (referees, field staff, people at the rail).

    A COCO-pretrained YOLO finds people; a robot detection that *is* a person overlaps one almost
    entirely, while a robot partly hidden behind a referee only shares a slice of its box.
    """

    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.35, imgsz: int = 1280):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        self.device = config.device()

    def people(self, image: np.ndarray) -> list[Det]:
        r = self.model.predict(image, conf=self.conf, classes=[0], imgsz=self.imgsz, verbose=False,
                               device=self.device, **precision_args(self.device == "cuda"))[0]
        return [Det(*b, 0.0, "") for b in r.boxes.xyxy.tolist()]

    def __call__(self, image: np.ndarray, dets: list[Det]) -> list[Det]:
        if not dets:
            return dets
        ppl = self.people(image)
        return [d for d in dets if not any(is_person_box(d, p) for p in ppl)]


@lru_cache(maxsize=2)
def precision_args(half: bool) -> dict:
    """Support installed and older Ultralytics without a warning on every video frame."""
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return {"quantize": 16 if half else 32} if "quantize" in DEFAULT_CFG_DICT else {"half": half}


def is_person_box(d: Det, p: Det) -> bool:
    ix = max(0.0, min(d.x1, p.x1) - max(d.x0, p.x0))
    iy = max(0.0, min(d.y1, p.y1) - max(d.y0, p.y0))
    inter = ix * iy
    return iou(d, p) > 0.5 or (inter > 0.7 * d.w * d.h and d.w < 1.3 * p.w)


def ensure_spyglass() -> Path | None:
    """Download FRC Team 578's Spy Glass bumper model (Hugging Face, GPL) on first use."""
    p = Path(DEFAULT_WEIGHTS[3])
    if p.exists():
        return p
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download("fairportrobotics/spy-glass", "model11n.pt", local_dir=str(p.parent)))
    except Exception as e:  # offline: fall back to our own model only
        print("Spy Glass download skipped:", e)
        return None


def default_detector(weights: str | Path | None = None, conf: float = 0.3, ensemble: bool = False,
                     imgsz: int | None = None, multiscale: bool | None = None):
    """Our fine-tuned model; Spy Glass only when there is no model of our own.

    Ensembling Spy Glass was measured on hand-marked broadcast frames (scripts/det_eval.py): +3
    points of recall for -11 points of precision, and every false box becomes a junk tracklet that
    the identity step has to explain away, so it is off unless asked for."""
    if weights:
        return _robot_detector(weights, conf, imgsz, multiscale)
    own = [p for p in available_weights() if p in DEFAULT_WEIGHTS[:3]]
    if own and not ensemble:
        for p in own:
            try:
                return _robot_detector(p, conf, imgsz, multiscale)
            except Exception:
                if p.suffix != ".engine":
                    raise
                import warnings
                warnings.warn(f"TensorRT model {p} could not load; trying portable weights.", RuntimeWarning)
    ensure_spyglass()
    spy = DEFAULT_WEIGHTS[3]
    if own and Path(spy).exists():
        return EnsembleDetector([RobotDetector(own[0], conf=conf), RobotDetector(spy, conf=conf)])
    return RobotDetector(conf=conf)


def iou(a: Det, b: Det) -> float:
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6)


def nms_merge(dets: list[Det], thr: float = 0.5) -> list[Det]:
    """Class-agnostic NMS (a robot is one alliance; keep the confident label)."""
    dets = sorted(dets, key=lambda d: -d.conf)
    keep: list[Det] = []
    for d in dets:
        if all(iou(d, k) < thr for k in keep):
            keep.append(d)
    return keep


def alliance_from_pixels(img: np.ndarray, d: Det) -> tuple[str, float]:
    """Vote red vs blue from saturated bumper-colored pixels in the lower part of a box."""
    x0, y0, x1, y1 = int(d.x0), int(d.y0 + 0.4 * d.h), int(d.x1), int(d.y1)
    crop = img[max(0, y0): y1, max(0, x0): x1]
    if crop.size == 0:
        return "", 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v, h = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    red = np.sum(((h < 8) | (h > 168)) & (s > 110) & (v > 70))
    blue = np.sum((h > 98) & (h < 128) & (s > 110) & (v > 60))
    tot = red + blue
    if tot < 15:
        return "", 0.0
    return ("red", red / tot) if red > blue else ("blue", blue / tot)


class ColorBlobDetector:
    """Detect single-color game pieces (2026 FUEL, 2025 ALGAE) by HSV thresholding."""

    def __init__(self, hsv_lo, hsv_hi, min_area: int = 6, max_area: int = 4000):
        self.lo = np.array(hsv_lo, dtype=np.uint8)
        self.hi = np.array(hsv_hi, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area

    def mask(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lo, self.hi)

    def __call__(self, img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Returns Nx4 array of (cx, cy, area, radius)."""
        m = self.mask(img) if mask is None else mask
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= a <= self.max_area:
                out.append((cents[i][0], cents[i][1], a, np.sqrt(a / np.pi)))
        return np.array(out, dtype=np.float32).reshape(-1, 4)

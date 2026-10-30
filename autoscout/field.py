"""Field geometry: season field models and per-event camera calibration.

A calibration maps the broadcast's wide-angle shot to field coordinates (meters) with a
homography fitted from >= 4 named landmarks. The wide shot is fixed for a whole event, so
one calibration covers every match; small per-video shifts (re-framing, different stream
encodes) are absorbed by aligning each video to the calibration's reference frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import config


@dataclass
class FieldModel:
    name: str
    year: int
    length: float
    width: float
    auto_len: int
    teleop_len: int
    landmarks: dict[str, tuple[float, float]]
    zones: dict[str, tuple[float, float, float, float]]
    elements: dict[str, dict]
    game_pieces: dict[str, dict]

    @classmethod
    def load(cls, year: int) -> "FieldModel":
        d = yaml.safe_load((config.FIELDS_DIR / f"{year}.yaml").read_text())
        return cls(
            name=d["name"], year=d["year"], length=d["length"], width=d["width"],
            auto_len=d["auto_len"], teleop_len=d["teleop_len"],
            landmarks={k: tuple(v) for k, v in d["landmarks"].items()},
            zones={k: tuple(v) for k, v in d["zones"].items()},
            elements=d.get("elements", {}), game_pieces=d.get("game_pieces", {}),
        )

    def zone_of(self, p) -> str | None:
        x, y = p
        for name, (x0, y0, x1, y1) in self.zones.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def element_center(self, name: str) -> np.ndarray:
        return np.array(self.elements[name]["center"], dtype=float)

    def dist_to_element(self, p, name: str) -> float:
        """Distance from a field point to the edge of an element (0 if inside)."""
        e = self.elements[name]
        c = np.array(e["center"], dtype=float)
        if "radius" in e:
            return max(0.0, float(np.linalg.norm(np.asarray(p) - c)) - e["radius"])
        half = np.array(e["size"], dtype=float) / 2
        d = np.maximum(np.abs(np.asarray(p) - c) - half, 0)
        return float(np.linalg.norm(d))

    def in_field(self, p, margin: float = 0.3) -> bool:
        return -margin <= p[0] <= self.length + margin and -margin <= p[1] <= self.width + margin


@dataclass
class Calibration:
    """Per-event camera calibration for the wide shot."""
    event_key: str
    year: int
    view: tuple[int, int, int, int]  # x, y, w, h of the wide shot in the full frame
    points: dict[str, tuple[float, float]]  # landmark name -> pixel (full-frame coords)
    frame_size: tuple[int, int] = (1920, 1080)
    reference: str | None = None  # path to reference frame (for per-video alignment)
    overlay: dict | None = None  # cached OverlayLayout for this event
    # Extra broadcast views (picture-in-picture close-ups), keyed by the field element they show,
    # e.g. {"blue_reef": [x, y, w, h]}. Used to read which robot is at that element.
    aux_views: dict | None = None
    H: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.H is None and len(self.points) >= 4:
            self.fit()

    def fit(self, fm: FieldModel | None = None) -> float:
        fm = fm or FieldModel.load(self.year)
        names = [n for n in self.points if n in fm.landmarks]
        src = np.array([self.points[n] for n in names], dtype=np.float64)
        dst = np.array([fm.landmarks[n] for n in names], dtype=np.float64)
        if len(names) < 4:
            raise ValueError("need >= 4 landmarks")
        self.H, _ = cv2.findHomography(src, dst, 0)
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return float(np.sqrt(((proj - dst) ** 2).sum(1)).mean())  # reprojection error (m)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def scaled(self, frame_size: tuple[int, int]) -> "Calibration":
        """Same calibration for a different video resolution."""
        sx = frame_size[0] / self.frame_size[0]
        sy = frame_size[1] / self.frame_size[1]
        if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6:
            return self
        v = self.view
        return Calibration(
            event_key=self.event_key, year=self.year,
            view=(int(v[0] * sx), int(v[1] * sy), int(v[2] * sx), int(v[3] * sy)),
            points={k: (p[0] * sx, p[1] * sy) for k, p in self.points.items()},
            frame_size=frame_size, reference=self.reference, overlay=None,
        )

    def to_field(self, pts) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(a, self.H).reshape(-1, 2)

    def to_image(self, pts) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(a, self.H_inv).reshape(-1, 2)

    def px_per_meter(self, img_pt) -> float:
        """Approximate image scale (pixels per meter) around an image point."""
        f = self.to_field([img_pt])[0]
        a = self.to_image([f, f + [0.5, 0], f + [0, 0.5]])
        return float((np.linalg.norm(a[1] - a[0]) + np.linalg.norm(a[2] - a[0])))

    def px_per_meter_x(self, field_pt) -> float:
        """Image pixels per meter along the field's length axis at a field point."""
        a = self.to_image([field_pt, (field_pt[0] + 0.5, field_pt[1])])
        return float(np.linalg.norm(a[1] - a[0]) * 2)

    def field_polygon_image(self, fm: FieldModel, margin: float = 0.0) -> np.ndarray:
        m = margin
        corners = [(-m, -m), (fm.length + m, -m), (fm.length + m, fm.width + m), (-m, fm.width + m)]
        # densify edges so the polygon follows the (projective) field boundary
        pts = []
        for i in range(4):
            a, b = np.array(corners[i]), np.array(corners[(i + 1) % 4])
            for s in np.linspace(0, 1, 12, endpoint=False):
                pts.append(a + s * (b - a))
        return self.to_image(pts).astype(np.int32)

    # ---- persistence -----------------------------------------------------------------
    @staticmethod
    def path_for(event_key: str) -> Path:
        return config.EVENTS_DIR / f"{event_key}.yaml"

    def save(self) -> Path:
        p = self.path_for(self.event_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "event_key": self.event_key, "year": self.year, "view": list(self.view),
            "frame_size": list(self.frame_size),
            "points": {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in self.points.items()},
            "reference": self.reference, "overlay": self.overlay,
        }
        if self.aux_views:
            d["aux_views"] = {k: [int(v) for v in r] for k, r in self.aux_views.items()}
        p.write_text(yaml.safe_dump(d, sort_keys=False))
        return p

    @classmethod
    def load(cls, event_key: str) -> "Calibration | None":
        p = cls.path_for(event_key)
        if not p.exists():
            return None
        d = yaml.safe_load(p.read_text())
        return cls(
            event_key=d["event_key"], year=d["year"], view=tuple(d["view"]),
            points={k: tuple(v) for k, v in d["points"].items()},
            frame_size=tuple(d.get("frame_size", (1920, 1080))),
            reference=d.get("reference"), overlay=d.get("overlay"), aux_views=d.get("aux_views"),
        )


def align_to_reference(cal: Calibration, frame: np.ndarray, ref: np.ndarray) -> Calibration:
    """Re-fit the calibration for a video whose framing moved slightly vs. the reference.

    ORB features in the wide-shot view are matched between this video's frame and the
    reference frame; the resulting image->image homography moves the landmark pixels.
    """
    x, y, w, h = cal.view
    a = cv2.cvtColor(ref[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return cal
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 30:
        return cal
    pa = np.float32([ka[m.queryIdx].pt for m in matches]) + [x, y]
    pb = np.float32([kb[m.trainIdx].pt for m in matches]) + [x, y]
    M, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if M is None or inl.sum() < 25:
        return cal
    names = list(cal.points)
    moved = cv2.perspectiveTransform(np.float64([cal.points[n] for n in names]).reshape(-1, 1, 2), M).reshape(-1, 2)
    if np.abs(moved - np.float64([cal.points[n] for n in names])).max() < 0.5:
        return cal
    return Calibration(event_key=cal.event_key, year=cal.year, view=cal.view,
                       points={n: tuple(p) for n, p in zip(names, moved)},
                       frame_size=cal.frame_size, reference=cal.reference, overlay=cal.overlay)


def view_similarity(frame: np.ndarray, ref: np.ndarray, view) -> float:
    """0..1 similarity of the wide-shot region to the reference (detects camera cuts)."""
    x, y, w, h = view
    a = cv2.resize(cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY), (160, 90))
    b = cv2.resize(cv2.cvtColor(ref[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY), (160, 90))
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))

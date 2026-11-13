"""A full 3D model of a broadcast camera, solved from the field's AprilTags.

The floor homography (field.Calibration) only says where a pixel lands on the carpet. Anything above the
carpet (a ball in flight, the top of a robot, a HUB opening) needs the whole camera: where it stands,
where it looks, its focal length and how much its lens bends straight lines. Every AprilTag on the field
has a published 3D pose, so a few tags seen in a view give all of that, with no clicking.

    cam = solve_from_tags(observe_tags(frames, region), frame_size, year)
    cam.project(points_3d)        -> pixels
    cam.floor(pixels, z=0.0)      -> field (x, y) where each pixel's ray meets height z
    cam.rays(pixels)              -> camera centre, unit ray directions

Coordinates are the repo's field frame (configs/fields/<year>.yaml): metres, x from the blue wall,
z up from the carpet. Pixels are full-frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .. import config

# cv2.aruco reports a tag's corners starting from the corner that is bottom-right for a viewer facing the
# tag as it hangs on the field (see tag_corners). Found by fitting every rotation; the others miss by ~20 px.
CORNER_ORDER = ("br", "bl", "tl", "tr")


@lru_cache(maxsize=None)
def tag_layout(year: int) -> tuple[dict[int, tuple], float]:
    """({tag id: (x, y, z, yaw deg, pitch deg)}, tag edge length in m) for a season's field."""
    d = json.loads((config.FIELDS_DIR / f"{year}_tags.json").read_text())
    return {int(k): tuple(v) for k, v in d["tags"].items()}, float(d["tag_size"])


def tag_corners(year: int, tid: int, order=CORNER_ORDER) -> np.ndarray:
    """4x3 world corners of a tag, in the order cv2.aruco reports them."""
    tags, size = tag_layout(year)
    x, y, z, yaw, pitch = tags[tid]
    a, p = np.radians(yaw), np.radians(pitch)
    n = np.array([np.cos(a) * np.cos(p), np.sin(a) * np.cos(p), np.sin(p)])  # the way the tag faces
    right = np.cross(-n, [0.0, 0.0, 1.0])  # the viewer's right, facing the tag
    right /= np.linalg.norm(right)
    up = np.cross(right, -n)
    c, h = np.array([x, y, z]), size / 2
    pt = {"tl": c - right * h + up * h, "tr": c + right * h + up * h,
          "br": c + right * h - up * h, "bl": c - right * h - up * h}
    return np.array([pt[k] for k in order])


@dataclass
class Camera:
    f: float
    cx: float
    cy: float
    k1: float
    rvec: np.ndarray
    tvec: np.ndarray
    size: tuple[int, int]
    rms: float = 0.0  # px, over the points it was solved from
    n_points: int = 0
    tags: list = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, 0.0, 0.0, 0.0, 0.0])

    @property
    def R(self) -> np.ndarray:
        import cv2

        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ np.asarray(self.tvec, float).reshape(3)

    def project(self, pts) -> np.ndarray:
        import cv2

        p = np.asarray(pts, np.float64).reshape(-1, 1, 3)
        if not len(p):
            return np.zeros((0, 2))
        out, _ = cv2.projectPoints(p, np.asarray(self.rvec, float), np.asarray(self.tvec, float), self.K, self.dist)
        return out.reshape(-1, 2)

    def depth(self, pts) -> np.ndarray:
        """Distance along the optical axis (m); <= 0 is behind the camera."""
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return (self.R @ p.T).T[:, 2] + float(np.asarray(self.tvec).reshape(3)[2])

    def rays(self, px) -> tuple[np.ndarray, np.ndarray]:
        """(camera centre, Nx3 unit directions in the field frame) of the rays through pixels."""
        import cv2

        p = np.asarray(px, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(p, self.K, self.dist).reshape(-1, 2)
        d = np.c_[n, np.ones(len(n))] @ self.R  # R.T @ d for every row
        return self.centre, d / np.linalg.norm(d, axis=1, keepdims=True)

    def floor(self, px, z: float = 0.0) -> np.ndarray:
        """Field (x, y) where each pixel's ray meets the horizontal plane at height z (nan above the horizon)."""
        c, d = self.rays(px)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (z - c[2]) / d[:, 2]
        out = c[:2] + lam[:, None] * d[:, :2]
        out[~(lam > 0)] = np.nan
        return out

    def to_dict(self) -> dict:
        return {"f": round(float(self.f), 3), "cx": round(float(self.cx), 2), "cy": round(float(self.cy), 2),
                "k1": round(float(self.k1), 5), "rvec": [round(float(v), 6) for v in np.ravel(self.rvec)],
                "tvec": [round(float(v), 5) for v in np.ravel(self.tvec)], "size": list(self.size),
                "rms": round(float(self.rms), 3), "n_points": int(self.n_points), "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(f=d["f"], cx=d["cx"], cy=d["cy"], k1=d.get("k1", 0.0), rvec=np.array(d["rvec"], float),
                   tvec=np.array(d["tvec"], float), size=tuple(d["size"]), rms=d.get("rms", 0.0),
                   n_points=d.get("n_points", 0), tags=d.get("tags", []))

    def shifted(self, dx: float, dy: float) -> "Camera":
        """The same camera for a picture moved (dx, dy) px inside the frame (a re-laid-out broadcast)."""
        return Camera(self.f, self.cx + dx, self.cy + dy, self.k1, self.rvec, self.tvec, self.size, self.rms,
                      self.n_points, self.tags)


def detect_tags(frame, region=None, scales=(1, 2, 3, 4)) -> dict[int, np.ndarray]:
    """{tag id: 4x2 corners, full-frame px} in one frame. Small tags only decode enlarged, so several
    scales are tried and the largest scale that found a tag gives its corners."""
    import cv2

    x0, y0, x1, y1 = region if region is not None else (0, 0, frame.shape[1], frame.shape[0])
    g = frame[y0:y1, x0:x1]
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    out = {}
    for det in _detectors():
        for s in scales:
            gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s != 1 else g
            corners, ids, _ = det.detectMarkers(gs)
            for c, i in zip(corners, ids.flatten() if ids is not None else []):
                # OpenCV puts pixel centres at integer coordinates: undo the resize about the half-pixel
                out[int(i)] = (c.reshape(4, 2) + 0.5) / s - 0.5 + [x0, y0]
    return out


@lru_cache(maxsize=1)
def _detectors() -> tuple:
    """The stock detector, then one tuned for tags only ~15 px across (a wide shot of the whole field):
    finer threshold windows, smaller allowed outlines, sub-pixel corners. The tuned one runs last so its
    corners are the ones kept."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 53, 4
    p.minMarkerPerimeterRate = 0.005
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()), cv2.aruco.ArucoDetector(d, p)


def observe_tags(frames, region=None, min_seen: int = 2) -> dict[int, np.ndarray]:
    """Tag corners over several frames of a fixed camera: {id: median 4x2 corners}. Robots and people
    cover tags part of the time, and a single frame's corners jitter by a pixel or so. A tag seen in
    two places (the same tag in the wide shot and in a close-up) must be separated by `region`."""
    seen: dict[int, list] = {}
    for f in frames:
        for i, c in detect_tags(f, region).items():
            seen.setdefault(i, []).append(c)
    out = {}
    for i, cs in seen.items():
        cs = np.array(cs)
        if len(cs) < min_seen:
            continue
        med = np.median(cs, axis=0)
        keep = np.abs(cs - med).reshape(len(cs), -1).max(1) < 4.0  # same place: drop misreads of another tag
        if keep.sum() >= min_seen:
            out[i] = np.median(cs[keep], axis=0)
    return out


def _residuals(params, obj, img, size, fit_k1, fit_cy):
    import cv2

    f, k1, cy_off = params[0], params[1] if fit_k1 else 0.0, params[2] if fit_cy else 0.0
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2 + cy_off], [0, 0, 1.0]])
    pr, _ = cv2.projectPoints(obj, params[3:6], params[6:9], K, np.array([k1, 0, 0, 0, 0.0]))
    return (pr.reshape(-1, 2) - img).ravel()


def solve(obj, img, size, fit_k1: bool = True, fit_cy: bool = False, f_range=(400, 8000), centre=None) -> Camera | None:
    """Camera from 3D points and their pixels. The principal point is the middle of `size` (the picture
    the camera made, which for a broadcast panel is that panel, not the whole frame): pass `centre` for
    a panel that sits elsewhere in the frame."""
    import cv2
    from scipy.optimize import least_squares

    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    if len(obj) < 6:
        return None
    cx, cy = centre if centre is not None else (size[0] / 2, size[1] / 2)
    shift = np.array([cx - size[0] / 2, cy - size[1] / 2])
    img0 = img - shift  # solve as if the panel were centred in `size`
    best = None
    for f in np.geomspace(*f_range, 40):
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(obj, img0, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        err = float(np.sqrt(((pr.reshape(-1, 2) - img0) ** 2).sum(1).mean()))
        if (R := cv2.Rodrigues(rv)[0]) is not None and ((R @ obj.T).T[:, 2] + tv[2]).min() <= 0:
            continue  # a mirror solution with the field behind the camera
        if best is None or err < best[0]:
            best = (err, f, rv.ravel(), tv.ravel())
    if best is None:
        return None
    _, f, rv, tv = best
    x0 = np.r_[f, 0.0, 0.0, rv, tv]
    # far fewer points than a calibration target: only let the lens bend when the points can show it
    fit_k1 = fit_k1 and len(obj) >= 16
    lo = np.r_[f_range[0], -0.6 if fit_k1 else -1e-9, -0.35 * size[1] if fit_cy else -1e-9, [-np.inf] * 6]
    hi = np.r_[f_range[1], 0.6 if fit_k1 else 1e-9, 0.35 * size[1] if fit_cy else 1e-9, [np.inf] * 6]
    sol = least_squares(_residuals, x0, args=(obj, img0, size, fit_k1, fit_cy), bounds=(lo, hi), loss="soft_l1",
                        f_scale=2.0, x_scale=np.r_[f, 0.1, 50.0, [0.1] * 3, [1.0] * 3])
    p = sol.x
    res = _residuals(p, obj, img0, size, fit_k1, fit_cy).reshape(-1, 2)
    return Camera(f=float(p[0]), cx=float(cx), cy=float(cy + (p[2] if fit_cy else 0.0)), k1=float(p[1] if fit_k1 else 0.0),
                  rvec=p[3:6].copy(), tvec=p[6:9].copy(), size=tuple(size),
                  rms=float(np.sqrt((res ** 2).sum(1).mean())), n_points=len(obj))


def solve_from_tags(obs: dict[int, np.ndarray], size, year: int, extra=None, **kw) -> Camera | None:
    """Camera from observe_tags output. `extra`: [(xyz, pixel)] other known points (clicked landmarks)."""
    tags, _ = tag_layout(year)
    ids = sorted(i for i in obs if i in tags)
    if len(ids) < 2:
        return None
    obj = np.concatenate([tag_corners(year, i) for i in ids])
    img = np.concatenate([np.asarray(obs[i], float) for i in ids])
    if extra:
        obj = np.vstack([obj, [e[0] for e in extra]])
        img = np.vstack([img, [e[1] for e in extra]])
    cam = solve(obj, img, size, **kw)
    if cam is not None:
        cam.tags = ids
    return cam


def solve_best(obs: dict[int, np.ndarray], size, year: int, extra=None, centre=None) -> Camera | None:
    """The simplest camera that fits: lens distortion and a shifted principal point are only kept when
    they cut the error by a quarter. With a dozen points a freer model fits noise, and a camera that
    fits noise is wrong everywhere the points are not."""
    best = None
    for fit_k1, fit_cy in ((False, False), (True, False), (True, True)):
        cam = solve_from_tags(obs, size, year, extra=extra, fit_k1=fit_k1, fit_cy=fit_cy, centre=centre)
        if cam is not None and (best is None or cam.rms < 0.75 * best.rms):
            best = cam
    return best


def event_camera_path(event_key: str):
    return config.EVENTS_DIR / f"{event_key}_cam3d.json"


@lru_cache(maxsize=None)
def _event_file(event_key: str):
    p = event_camera_path(event_key)
    return json.loads(p.read_text()) if p.exists() else None


def load_event_camera(event_key: str, view: str = "wide") -> Camera | None:
    """The event's solved camera for a view (scripts/calibrate_camera3d.py), or None."""
    d = _event_file(event_key)
    return Camera.from_dict(d[view]["camera"]) if d and view in d else None


def reference_tags(event_key: str, view: str = "wide") -> dict[int, np.ndarray]:
    d = _event_file(event_key)
    return {int(k): np.array(v, float) for k, v in d[view].get("ref_tags", {}).items()} if d and view in d else {}


def frame_shift(event_key: str, frames, region=None, view: str = "wide") -> tuple[float, float, int]:
    """(dx, dy, tags used): how far this video's picture sits from the one the event camera was solved
    on, from the tags both show. Broadcast framing moves between matches (a bumped tripod, a re-laid-out
    stream); the tags don't."""
    ref = reference_tags(event_key, view)
    obs = observe_tags(frames, region, min_seen=1)
    d = [np.mean(obs[i] - ref[i], axis=0) for i in obs if i in ref]
    if not d:
        return 0.0, 0.0, 0
    dx, dy = np.median(np.array(d), axis=0)
    return float(dx), float(dy), len(d)

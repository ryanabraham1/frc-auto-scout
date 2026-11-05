"""3D 2025 field geometry and camera solving from AprilTags (welded field layout, game manual).

A broadcast camera that sees a few AprilTags is solved as a full pinhole camera (focal length and pose,
principal point at the view centre, no distortion) from the tags' corners, whose 3D positions are
known. Then any 3D point of the field projects into that view: here, the tip of every REEF branch, so a
CORAL seen in a REEF close-up can be put on its branch (A-L) and level.
"""
from __future__ import annotations

import numpy as np

IN = 0.0254
TAG_SIZE = 6.5 * IN
# id: (x, y, z) in inches, yaw in degrees (the direction the tag faces). Barge tags (4, 5, 14, 15) are pitched: left out.
TAGS = {
    1: (657.37, 25.80, 58.50, 126), 2: (657.37, 291.20, 58.50, 234), 3: (455.15, 317.15, 51.25, 270),
    6: (530.49, 130.17, 12.13, 300), 7: (546.87, 158.50, 12.13, 0), 8: (530.49, 186.83, 12.13, 60),
    9: (497.77, 186.83, 12.13, 120), 10: (481.39, 158.50, 12.13, 180), 11: (497.77, 130.17, 12.13, 240),
    12: (33.51, 25.80, 58.50, 54), 13: (33.51, 291.20, 58.50, 306), 16: (235.73, -0.15, 51.25, 90),
    17: (160.39, 130.17, 12.13, 240), 18: (144.00, 158.50, 12.13, 180), 19: (160.39, 186.83, 12.13, 120),
    20: (193.10, 186.83, 12.13, 60), 21: (209.49, 158.50, 12.13, 0), 22: (193.10, 130.17, 12.13, 300),
}
REEF_CENTRE = {"blue": (4.489, 4.026), "red": (13.059, 4.026)}
# Branch geometry (approx., game manual drawings): tips of the L2-L4 branches and the L1 trough front.
BRANCH_HALF = 6.47 * IN  # each face's two branches sit this far either side of the face centre
LEVEL_TIP = {"L4": (0.76, 1.83), "L3": (0.83, 1.21), "L2": (0.83, 0.81), "L1": (0.86, 0.46)}  # (m from centre, height)


def tag_corners(tid: int) -> np.ndarray:
    """3D corners of a tag in the order cv2.aruco reports them."""
    x, y, z, yaw = TAGS[tid]
    c = np.array([x, y, z]) * IN
    n = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
    right = np.cross(-n, [0, 0, 1.0])  # the viewer's right, facing the tag
    up = np.array([0, 0, 1.0])
    h = TAG_SIZE / 2
    tl, tr, br, bl = c - right * h + up * h, c + right * h + up * h, c + right * h - up * h, c - right * h - up * h
    return np.array([br, bl, tl, tr])  # OpenCV's AprilTag corner order for this field (fits to ~2 px; others ~20)


def face_normal_deg(alliance: str, face: int) -> float:
    """Direction a REEF face looks. Face 0 (branches A, B) faces that alliance's driver station; faces go
    counter-clockwise seen from above."""
    start = 180.0 if alliance == "blue" else 0.0
    return start + 60.0 * face


def branch_tips(alliance: str) -> dict[tuple[str, str], np.ndarray]:
    """{(branch letter, level): 3D tip}. L1 (the trough) is one spot per branch side of each face."""
    out = {}
    cx, cy = REEF_CENTRE[alliance]
    for face in range(6):
        a = np.radians(face_normal_deg(alliance, face))
        n = np.array([np.cos(a), np.sin(a)])
        t = np.array([-n[1], n[0]])  # counter-clockwise along the face
        for k, side in enumerate((-1, 1)):  # A is the clockwise branch of face 0, as seen from the DS
            letter = "ABCDEFGHIJKL"[2 * face + k]
            for level, (r, z) in LEVEL_TIP.items():
                p = np.array([cx, cy]) + r * n + side * BRANCH_HALF * t
                out[(letter, level)] = np.array([p[0], p[1], z])
    return out


def solve_camera(obs: dict[int, np.ndarray], size: tuple[int, int], f_range=(300, 6000), f: float | None = None):
    """Pinhole camera (K, rvec, tvec, rms px) from {tag id: 4x2 image corners in view pixels}. With two
    tags the focal length is poorly determined: pass f (from frames that showed more tags)."""
    import cv2

    obj = np.concatenate([tag_corners(i) for i in obs]).astype(np.float64)
    img = np.concatenate([np.asarray(obs[i], float) for i in obs]).astype(np.float64)
    if len(obs) < 2:
        return None
    w, h = size

    def fit(focal):
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        return K, rv, tv, float(np.sqrt(((pr.reshape(-1, 2) - img) ** 2).sum(1).mean()))

    if f:
        return fit(f)
    fits = [c for c in (fit(x) for x in np.geomspace(*f_range, 60)) if c]
    if not fits:
        return None
    best = min(fits, key=lambda c: c[3])
    # refine the focal length around the best grid value
    fits = [c for c in (fit(x) for x in np.linspace(best[0][0, 0] * 0.9, best[0][0, 0] * 1.1, 41)) if c]
    return min(fits + [best], key=lambda c: c[3])


def project(cam, pts) -> np.ndarray:
    import cv2

    K, rv, tv, _ = cam
    pr, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 3), rv, tv, K, None)
    return pr.reshape(-1, 2)

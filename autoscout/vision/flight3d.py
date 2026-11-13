"""A ball's flight in 3D from one camera.

A pixel only gives a ray, so one sighting says nothing about how far away a ball is. But a ball in the
air is in free fall: gravity bends its path by a known amount in metres, and how much that bend measures
in pixels says how far it is from the camera. Fitting a real 3D arc (gravity plus air drag) through all
of a flight's sightings therefore puts the ball in the room, and running that arc back down to shooter
height gives the spot on the carpet it left from, without any robot having to be seen there.

In flat image space a ball rising past two robots that line up along the camera's view could have come
from either. On the carpet they are metres apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81
# Quadratic air drag per unit mass (1/m): 0.5 * air density * Cd * area / mass for a 0.15 m, ~0.215 kg FUEL.
DRAG = {"fuel": 0.0232, "algae": 0.035, "coral": 0.02}
RELEASE_Z = 0.55  # m above the carpet where a shooter lets go of a ball (robots are ~0.5-0.8 m tall there)
PIXEL_SIGMA = 1.5  # px: how well a blob centre marks the ball's centre
STEP = 1 / 60  # s, integration step (RK4: far finer than the arc needs)


def _accel(v: np.ndarray, k: float) -> np.ndarray:
    a = -k * np.linalg.norm(v, axis=-1, keepdims=True) * v
    a[..., 2] -= G
    return a


def integrate(p0, v0, ts, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities at times `ts` (s, relative to the state's own time, either sign) of a
    ball at p0 moving at v0. p0, v0: (..., 3); returns (..., len(ts), 3) each. Classical RK4."""
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    ts = np.asarray(ts, float)
    P = np.empty(p0.shape[:-1] + (len(ts), 3))
    V = np.empty_like(P)
    for sign in (1.0, -1.0):
        idx = np.nonzero(ts * sign >= 0)[0] if sign > 0 else np.nonzero(ts < 0)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(np.abs(ts[idx]))]
        p, v, t = p0.copy(), v0.copy(), 0.0
        for i in idx:
            span = abs(ts[i]) - t
            n = max(1, int(np.ceil(span / STEP))) if span > 1e-9 else 0
            h = sign * span / n if n else 0.0
            for _ in range(n):
                a1 = _accel(v, k)
                v2 = v + 0.5 * h * a1
                a2 = _accel(v2, k)
                v3 = v + 0.5 * h * a2
                a3 = _accel(v3, k)
                v4 = v + h * a3
                a4 = _accel(v4, k)
                p = p + h / 6 * (v + 2 * v2 + 2 * v3 + v4)
                v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            t = abs(ts[i])
            P[..., i, :], V[..., i, :] = p, v
    return P, V


@dataclass
class Flight3D:
    t0: float  # video time of the state below (the flight's first sighting)
    p0: np.ndarray  # position then (field m)
    v0: np.ndarray  # velocity then (m/s)
    rms: float  # px, reprojection error of the fit
    cov: np.ndarray  # 6x6 covariance of (p0, v0)
    k: float  # drag used
    n: int  # sightings
    release_t: float | None = None  # video time it was at shooter height, going up
    release_xy: np.ndarray | None = None  # where on the carpet that was
    release_cov: np.ndarray | None = None  # 2x2 (m^2)
    release_v: np.ndarray | None = None  # velocity at release
    ok: bool = True

    def at(self, t) -> np.ndarray:
        """Position at video time(s) t."""
        ts = np.atleast_1d(np.asarray(t, float)) - self.t0
        return integrate(self.p0, self.v0, ts, self.k)[0]

    def release_sigma(self) -> float:
        """One number for how well the release spot is known (m): the long axis of its error ellipse."""
        if self.release_cov is None:
            return float("inf")
        return float(np.sqrt(max(np.linalg.eigvalsh(self.release_cov)[-1], 0.0)))


def _linear_start(cam, T, UV):
    """Drag-free arc through the sightings by linear least squares: every sighting's ray must pass
    through p0 + v0 t + g t^2 / 2. Exact in the algebra, not in pixels, so it only seeds the real fit."""
    c, d = cam.rays(UV)
    A, b = [], []
    g = np.array([0.0, 0.0, -G])
    for t, di in zip(T, d):
        # two directions perpendicular to the ray: the point's offset from the ray must vanish along both
        e1 = np.cross(di, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(di, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(di, e1)
        for e in (e1, e2):
            A.append(np.r_[e, e * t])
            b.append(e @ (c - 0.5 * g * t * t))
    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return x[:3], x[3:]


def fit(cam, pts, k: float = DRAG["fuel"], anchor=None, anchor_sigma: float = 0.45, anchor_t: float | None = None,
        sigma_px: float = PIXEL_SIGMA, prior_speed: float = 14.0) -> Flight3D | None:
    """Fit a flight to its sightings `pts` [(t, u, v)] (video s, full-frame px) seen by `cam`.

    anchor: a field point (x, y) the ball is known to be over at `anchor_t` (default: its last
    sighting), within anchor_sigma m. A ball tracked into a HUB ends over that HUB: that one fact
    removes most of the depth uncertainty of a short track."""
    P = np.asarray(pts, float)
    if len(P) < 4:
        return None
    t0 = P[0, 0]
    T, UV = P[:, 0] - t0, P[:, 1:3]
    if T[-1] < 0.05:
        return None
    try:
        p0, v0 = _linear_start(cam, T, UV)
    except np.linalg.LinAlgError:
        return None
    ta = (anchor_t - t0) if anchor_t is not None else T[-1]
    x = np.r_[p0, v0]
    if cam.depth([p0])[0] <= 0.5 or not np.isfinite(x).all() or np.linalg.norm(v0) > 60:
        # the algebra put it behind the camera or on a wild arc: start from the middle of the field
        c, d = cam.rays(UV[:1])
        lam = (1.0 - c[2]) / d[0, 2] if abs(d[0, 2]) > 1e-6 else 15.0
        x = np.r_[c + abs(lam) * d[0], 0.0, 0.0, 3.0]
    ts_all = np.r_[T, ta]
    lam = 1e-3
    cost = None
    for _ in range(25):
        r, J = _residuals(cam, x, ts_all, len(T), UV, k, anchor, anchor_sigma, sigma_px, prior_speed)
        c0 = float(r @ r)
        H = J.T @ J
        g = J.T @ r
        for _try in range(8):
            try:
                dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r1, _ = _residuals(cam, x + dx, ts_all, len(T), UV, k, anchor, anchor_sigma, sigma_px, prior_speed, jac=False)
            c1 = float(r1 @ r1)
            if np.isfinite(c1) and c1 < c0:
                x, lam, cost = x + dx, max(lam / 5, 1e-7), c1
                break
            lam *= 5
        else:
            break
        if np.abs(dx[:3]).max() < 1e-3 and np.abs(dx[3:]).max() < 1e-3:
            break
    r, J = _residuals(cam, x, ts_all, len(T), UV, k, anchor, anchor_sigma, sigma_px, prior_speed)
    px = r[: 2 * len(T)] * sigma_px
    rms = float(np.sqrt((px ** 2).reshape(-1, 2).sum(1).mean()))
    try:
        # scale the pixel noise up when the fit is worse than a blob centre should be: a rough track
        # (merged balls, a mislink) knows its arc less well than the nominal noise says
        s2 = max(1.0, (rms / (sigma_px * np.sqrt(2))) ** 2)
        cov = np.linalg.inv(J.T @ J + 1e-9 * np.eye(6)) * s2
    except np.linalg.LinAlgError:
        return None
    f = Flight3D(t0=float(t0), p0=x[:3].copy(), v0=x[3:].copy(), rms=rms, cov=cov, k=k, n=len(T))
    f.ok = bool(np.isfinite(x).all() and cam.depth([f.p0])[0] > 0.5 and rms < 12.0)
    if f.ok:
        _release(f)
    return f


def _residuals(cam, x, ts, n, UV, k, anchor, anchor_sigma, sigma_px, prior_speed, jac=True):
    """Stacked residuals in units of their own noise: pixels, then the anchor, then a weak speed prior.
    The Jacobian treats drag as fixed along the current arc (dP/dp0 = I, dP/dv0 = t I), which is close
    enough to converge; the residuals themselves use the full drag model."""
    Pw, _ = integrate(x[:3], x[3:], ts, k)
    pr = cam.project(Pw[:n])
    r = [((pr - UV) / sigma_px).ravel()]
    if anchor is not None:
        r.append((Pw[n, :2] - np.asarray(anchor, float)) / anchor_sigma)
    speed = float(np.linalg.norm(x[3:]))
    r.append(np.array([max(0.0, speed - prior_speed) / 2.0]))
    r = np.concatenate(r)
    if not jac:
        return r, None
    eps = 1e-3
    J = np.zeros((len(r), 6))
    # d(pixel)/d(point) by central differences on the projection (cheap, and includes the lens)
    dP = np.zeros((n, 2, 3))
    for a in range(3):
        e = np.zeros(3)
        e[a] = eps
        dP[:, :, a] = (cam.project(Pw[:n] + e) - cam.project(Pw[:n] - e)) / (2 * eps)
    for a in range(3):
        J[: 2 * n, a] = (dP[:, :, a] / sigma_px).ravel()
        J[: 2 * n, 3 + a] = (dP[:, :, a] * ts[:n, None] / sigma_px).ravel()
    row = 2 * n
    if anchor is not None:
        for a in range(2):
            J[row + a, a] = 1.0 / anchor_sigma
            J[row + a, 3 + a] = ts[n] / anchor_sigma
        row += 2
    if speed > prior_speed:
        J[row, 3:] = x[3:] / speed / 2.0
    return r, J


def _release(f: Flight3D, z: float = RELEASE_Z, max_back: float = 2.5) -> None:
    """Run the arc back to where it was at shooter height on the way up; fill the release fields."""
    ts = -np.arange(0.0, max_back + 1e-9, 1 / 60)
    P, V = integrate(f.p0, f.v0, ts, f.k)
    if P[0, 2] <= z:
        # first seen below shooter height (a blob low on the robot): the release is the first sighting
        i, frac = 0, 0.0
    else:
        below = np.nonzero((P[:, 2] <= z) & (V[:, 2] > 0))[0]
        if not len(below):
            return
        i = int(below[0])
        if i == 0:
            frac = 0.0
        else:
            i -= 1
            frac = float((P[i, 2] - z) / max(P[i, 2] - P[i + 1, 2], 1e-9))
    if i + 1 < len(ts):
        p = P[i] + frac * (P[i + 1] - P[i])
        v = V[i] + frac * (V[i + 1] - V[i])
        tr = ts[i] + frac * (ts[i + 1] - ts[i])
    else:
        p, v, tr = P[i], V[i], ts[i]
    f.release_t = float(f.t0 + tr)
    f.release_xy = p[:2].copy()
    f.release_v = v.copy()
    # error ellipse: move (p0, v0) a little and see where the release lands. Along the arc the release
    # time shifts too (a higher start reaches shooter height earlier), which is what stretches the
    # ellipse along the direction of flight.
    Jr = np.zeros((2, 6))
    vz = v[2] if abs(v[2]) > 0.3 else 0.3 * np.sign(v[2] or 1.0)
    for a in range(6):
        dz = (1.0 if a == 2 else 0.0) + (tr if a == 5 else 0.0)  # change in height at the release time
        dt = -dz / vz
        dxy = np.zeros(2)
        if a < 2:
            dxy[a] = 1.0
        elif 3 <= a < 5:
            dxy[a - 3] = tr
        Jr[:, a] = dxy + v[:2] * dt
    f.release_cov = Jr @ f.cov @ Jr.T


def release_distance(f: Flight3D, xy, pos_sigma: float = 0.3) -> float:
    """How many standard deviations a point on the carpet (a robot's centre) is from the release spot,
    with the robot's own position known to pos_sigma m."""
    if f.release_xy is None or f.release_cov is None:
        return float("inf")
    d = np.asarray(xy, float) - f.release_xy
    S = f.release_cov + np.eye(2) * pos_sigma ** 2
    try:
        return float(np.sqrt(d @ np.linalg.solve(S, d)))
    except np.linalg.LinAlgError:
        return float("inf")

"""Where each REEF branch is in the wide shot (2025), so a coral seen appearing on a REEF says which face
it went on, and so which robot (the one in front of that face) placed it.

The broadcast camera is low and nearly level: verticals stay vertical in the image, and a point h m up
sits straight above its floor point. Around each REEF the image is modelled from the floor calibration's
local mapping (scaled by `a` and shifted by `d` px, since the 4-point floor fit drifts there) plus `sz`
px per meter of height. Those four numbers per REEF are fitted to coral already seen on the REEFs with
the level the score decoding gave them: placed coral sits at its branch's tip (fit_reef_model).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Branch tips (game manual, approx.): m from the REEF centre to the branch line, half-spacing of a face's two
# branches, and each level's height. L1 is the trough along the whole face.
APOTHEM, BRANCH_HALF = 0.78, 0.165
LEVEL_Z = {"L1": 0.46, "L2": 0.81, "L3": 1.21, "L4": 1.83}
FACE_PX = 22.0  # px: spread of a placement around its branch tip (the fit's rms is ~20-26 px)


def face_normals() -> list[np.ndarray]:
    """Unit outward normal of each of the 6 faces (face 0 faces the blue alliance wall, -x)."""
    return [np.array([np.cos(np.pi + k * np.pi / 3), np.sin(np.pi + k * np.pi / 3)]) for k in range(6)]


def tips(level: str) -> list[tuple[int, np.ndarray]]:
    """(face, floor offset from the REEF centre in m) of each place a coral of `level` can sit."""
    out = []
    for k, n in enumerate(face_normals()):
        t = np.array([-n[1], n[0]])
        offs = (-BRANCH_HALF, 0.0, BRANCH_HALF) if level == "L1" else (-BRANCH_HALF, BRANCH_HALF)
        out += [(k, APOTHEM * n + s * t) for s in offs]
    return out


def _local(cal, centre):
    p0 = cal.to_image([centre])[0]
    J = np.stack([cal.to_image([(centre[0] + 0.1, centre[1])])[0] - p0,
                  cal.to_image([(centre[0], centre[1] + 0.1)])[0] - p0], 1) / 0.1
    return p0, J


def project(cal, centre, params, offset, z: float) -> np.ndarray:
    """Image point of floor offset `offset` (m, from the REEF centre) at height z m."""
    p0, J = _local(cal, centre)
    dx, dy, a, sz = params
    return p0 + np.array([dx, dy]) + a * (J @ np.asarray(offset)) - np.array([0.0, sz * z])


def face_probs(cal, centre, params, x: float, y: float, level: str) -> tuple[dict[int, float], float]:
    """Chance a coral seen at image (x, y) went on each face, if it is a `level` coral, and the distance in
    px to the nearest tip of that level (large: the placement doesn't look like that level)."""
    d = {}
    for k, off in tips(level):
        p = project(cal, centre, params, off, LEVEL_Z[level])
        d[k] = min(d.get(k, 1e9), float(np.hypot(p[0] - x, p[1] - y)))
    best = min(d.values())
    w = {k: np.exp(-0.5 * ((v - best) / FACE_PX) ** 2) for k, v in d.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items() if v / tot >= 0.01}, best


def fit_reef_model(centres: dict[str, tuple[float, float]], placements: dict[str, list[tuple]]) -> dict:
    """Per REEF (dx, dy, a, sz) placing each level's tips where coral of that level were seen: robust least
    squares of each placement's distance to its nearest tip of its level. placements: {reef: [(x, y, level,
    cal)]}, each with its match's (aligned) calibration."""
    from scipy.optimize import minimize

    out = {}
    for name, pts in placements.items():
        if len(pts) < 20:
            continue
        cands = {L: [off for _, off in tips(L)] for L in LEVEL_Z}

        def cost(th):
            tot = 0.0
            for x, y, L, cal in pts:
                ds = min(np.hypot(*(project(cal, centres[name], th, off, LEVEL_Z[L]) - (x, y))) for off in cands[L])
                tot += min(ds, 60.0) ** 2
            # the floor fit's local scale is roughly right: keep `a` near 1 (else red's fit trades width for height)
            return tot / len(pts) + 2000.0 * (th[2] - 1.0) ** 2

        best = None
        for dx in (-90.0, -45.0, 0.0):
            for sz in (40.0, 70.0, 100.0):
                res = minimize(cost, [dx, -20.0, 1.0, sz], method="Nelder-Mead", options={"maxiter": 800, "xatol": 0.5})
                if best is None or res.fun < best.fun:
                    best = res
        rms = float(np.sqrt(max(best.fun - 2000.0 * (best.x[2] - 1.0) ** 2, 0.0)))
        out[name] = {"params": [round(float(v), 3) for v in best.x], "rms_px": round(rms, 1),
                     "n": len(pts)}
    return out


def model_path(event_key: str) -> Path:
    from ..config import ROOT

    return ROOT / "configs" / "events" / f"{event_key}_reef.json"


def load_reef_model(event_key: str) -> dict | None:
    p = model_path(event_key)
    return json.loads(p.read_text()) if p.exists() else None

"""Where FUEL was shot from, in field metres, and which robot stood there.

Every tracked flight is fitted as a 3D arc (flight3d) through the event's solved camera (camera3d), and
run back to shooter height. That release spot is compared with robots on the carpet, not with boxes in
the picture: a ball that rises past a robot's box on screen but left the floor two metres behind it
did not come from that robot.
"""
from __future__ import annotations

import numpy as np

from . import flight3d
from .camera3d import Camera

ROBOT_HALF_DEPTH = 0.42  # m from the bumper edge nearest the camera to the robot's centre
HUB_ANCHOR_SIGMA = 0.45  # m: a ball tracked into a HUB ends this close to the HUB's centre


def robot_centre(cam: Camera, box) -> np.ndarray:
    """Field (x, y) of a robot's centre from its box. The bottom middle of the box is the bumper edge
    nearest the camera, on the carpet; the centre is half a robot further along the line of sight."""
    foot = ((box[0] + box[2]) / 2.0, box[3])
    p = cam.floor([foot])[0]
    _, d = cam.rays([foot])
    h = d[0, :2] / max(np.linalg.norm(d[0, :2]), 1e-9)
    return p + ROBOT_HALF_DEPTH * h


def fit_launch(cam: Camera, launch: dict, fm, piece: str = "fuel") -> flight3d.Flight3D | None:
    """3D arc of one saved launch ({'pts', 'target', 'entry_t'}). A flight tracked into a HUB is tied
    to that HUB at its entry."""
    target = launch.get("target")
    anchor = fm.elements[target]["center"] if target and target in fm.elements else None
    return flight3d.fit(cam, launch["pts"], k=flight3d.DRAG.get(piece, flight3d.DRAG["fuel"]), anchor=anchor,
                        anchor_sigma=HUB_ANCHOR_SIGMA, anchor_t=launch.get("entry_t"))


def flight_record(f: flight3d.Flight3D | None) -> dict | None:
    """What is kept of a fitted flight (JSON)."""
    if f is None or not f.ok:
        return None
    out = {"t0": round(f.t0, 3), "p0": [round(float(v), 3) for v in f.p0], "v0": [round(float(v), 3) for v in f.v0],
           "rms": round(f.rms, 2), "n": f.n}
    if f.release_xy is not None:
        out.update(rel_t=round(f.release_t, 3), rel_xy=[round(float(v), 3) for v in f.release_xy],
                   rel_cov=[round(float(v), 4) for v in f.release_cov.ravel()],
                   rel_v=[round(float(v), 2) for v in f.release_v])
    return out

"""Refine an event's REEF close-up cameras from the coral FMS says was scored.

The close-up cameras are solved once from a few AprilTags (configs/events/<event>_reef_cams.json), which
pins them down near the tags and less well up at the branches. Every processed match then supplies
better points for free: the official breakdown names the branches that hold coral, the CORAL detector
boxes each one, and the two are matched with the current camera. Where a coral can only be on one
official branch, its box centre against that branch's expected coral centre is a correspondence;
hundreds of them over an event refit focal length and pose. Matching and refitting repeat a few times.

The fixed cameras do not move during an event, so this is run once per event, after a few matches.

usage: python scripts/refine_reef_cams.py EVENT [--matches FILE.json] [--rounds 3] [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402
from autoscout.vision import reef3d  # noqa: E402
from autoscout.vision.camera3d import Camera  # noqa: E402

RESULTS = config.DATA_DIR / "results"
MAX_COST = 4.0  # only coral that sit clearly on one branch teach the camera


def correspondences(event: str, keys: list[str], cams: dict[str, Camera]) -> dict[str, list]:
    """{alliance: [(3D coral centre, pixel centre)]} from coral matched to the one official branch it fits."""
    from autoscout.tba import TBA, MatchInfo

    tba = TBA()
    out: dict[str, list] = {a: [] for a in cams}
    for key in keys:
        m = MatchInfo.from_tba(tba.match(key))
        side = RESULTS / f"{key}_coral.json"
        if not m.breakdown or not side.exists():
            continue
        obs = json.loads(side.read_text())
        auto_end = json.loads((RESULTS / f"{key}.json").read_text())["clock"]["auto_end_vt"]
        for a, cam in cams.items():
            nodes = reef3d.official_nodes(m.breakdown[a])
            if not nodes:
                continue
            for i, (r, cost, alts) in reef3d._view_matches(obs, cam, "cam", a, nodes, auto_end).items():
                if alts or cost > MAX_COST:
                    continue
                top, bottom = reef3d.coral_ends(a, nodes[i]["branch"], nodes[i]["level"])
                out[a].append(((top + bottom) / 2, (r["x"], r["y"])))
    return out


def refit(cam: Camera, pairs: list) -> tuple[Camera, float, float]:
    """(camera, rms before, rms after) with focal length and pose refitted to the pairs (robust loss)."""
    from scipy.optimize import least_squares

    P = np.array([p for p, _ in pairs], float)
    UV = np.array([uv for _, uv in pairs], float)

    def make(x):
        return Camera(f=float(x[0]), cx=cam.cx, cy=cam.cy, k1=cam.k1, rvec=x[1:4], tvec=x[4:7], size=cam.size)

    def res(x):
        return (make(x).project(P) - UV).ravel()

    x0 = np.r_[cam.f, np.ravel(cam.rvec), np.ravel(cam.tvec)]
    sol = least_squares(res, x0, loss="soft_l1", f_scale=6.0, x_scale=np.r_[cam.f, [0.05] * 3, [0.5] * 3])
    rms = lambda x: float(np.sqrt((res(x).reshape(-1, 2) ** 2).sum(1).mean()))  # noqa: E731
    return make(sol.x), rms(x0), rms(sol.x)


def main(event: str, keys: list[str], rounds: int = 3, write: bool = False) -> dict:
    path = config.EVENTS_DIR / f"{event}_reef_cams.json"
    saved = json.loads(path.read_text())
    cams = {a: reef3d.views(event, a)["cam"] for a in ("blue", "red") if a in saved}
    for k in range(rounds):
        pairs = correspondences(event, keys, cams)
        for a in cams:
            if len(pairs[a]) < 30:
                print(f"  {a}: only {len(pairs[a])} coral to learn from, left as it is")
                continue
            cams[a], before, after = refit(cams[a], pairs[a])
            print(f"  round {k + 1} {a}: {len(pairs[a])} coral, {before:.1f} px -> {after:.1f} px, f {cams[a].f:.0f}")
    if write:
        for a, cam in cams.items():
            saved[a].update(f=float(cam.f), rvec=[float(v) for v in np.ravel(cam.rvec)], tvec=[float(v) for v in np.ravel(cam.tvec)])
        saved["refined"] = f"focal length and pose refitted to scored coral on official branches over {len(keys)} matches (scripts/refine_reef_cams.py)"
        path.write_text(json.dumps(saved, indent=1))
        print("wrote", path)
    return cams


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event")
    ap.add_argument("--matches", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--write", action="store_true", help="save the refined cameras to the event's reef_cams.json")
    a = ap.parse_args()
    keys = json.loads(a.matches.read_text()) if a.matches else sorted(
        p.stem for p in RESULTS.glob(f"{a.event}_qm*.json") if p.stem.count("_") == 1)
    main(a.event, keys, a.rounds, a.write)

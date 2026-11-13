"""When each scored CORAL went onto its REEF branch, from 3D cameras.

The official breakdown lists exactly which branches (A-L, L2-L4) an alliance filled in a match. The CORAL
detector's boxes say where coral came to rest in the picture and when. With every view solved as a 3D
camera (camera3d), each branch tip projects to a known pixel in the wide shot and in the REEF close-up, so
a coral that comes to rest there, and stays, is that branch being filled: a branch, a REEF face and a
moment. The robot standing at that face at that moment placed it, however long its partners sat at
other faces.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import config
from . import field3d as F
from .camera3d import Camera, load_event_camera

CORAL_LEN = 0.30  # m
LEVELS = ("L4", "L3", "L2")
ROWS = {"L4": "topRow", "L3": "midRow", "L2": "botRow"}
LETTERS = "ABCDEFGHIJKL"
LINK_PX = 12.0  # px between boxes of one resting coral
HOLD_S, HOLD_FRAC = 2.0, 0.6  # a coral rests when it is seen this much of the next HOLD_S
MIN_SEEN = 15
PANEL_Y, PANEL_SPLIT = 645, 958  # close-up band and where the two REEF panels meet (1080p two-panel layout)
MATCH_LEN = 1.3  # coral lengths from a tip within which a resting coral can be on that branch
ALT_COST = 4.0  # how much worse another branch may fit one view's coral and still be a candidate
ROBOT_REACH = 0.83 + 0.55  # m from the REEF centre to the centre of a robot placing on a face
ROBOT_SIGMA = 0.55  # m: how well a robot's centre is known from its box


def views(event_key: str, alliance: str) -> dict[str, Camera]:
    """{"wide": camera, "cam": that alliance's REEF close-up camera} (whichever are solved)."""
    out = {}
    wide = load_event_camera(event_key)
    if wide is not None:
        out["wide"] = wide
    p = config.EVENTS_DIR / f"{event_key}_reef_cams.json"
    if p.exists():
        c = json.loads(p.read_text()).get(alliance)
        if c:
            w, h = c["size"]
            ox, oy = c.get("origin", (4 if alliance == "blue" else 960, PANEL_Y))
            out["cam"] = Camera(f=c["f"], cx=ox + w / 2, cy=oy + h / 2, k1=0.0, rvec=np.array(c["rvec"], float),
                                tvec=np.array(c["tvec"], float), size=(1920, 1080))
    return out


def in_view(view: str, alliance: str, x: float, y: float, wide_bottom: float = 640.0) -> bool:
    if view == "wide":
        return 0 <= x < 1920 and 0 <= y < wide_bottom
    mine = (x < PANEL_SPLIT) if alliance == "blue" else (x >= PANEL_SPLIT)
    return PANEL_Y <= y < PANEL_Y + 370 and mine and 0 <= x < 1920


def official_nodes(breakdown_alliance: dict) -> list[dict]:
    """[{branch, level, phase}] the alliance filled, from the official breakdown."""
    out = []
    for level, row in ROWS.items():
        for b in LETTERS:
            if breakdown_alliance["teleopReef"][row].get(f"node{b}"):
                auto = bool(breakdown_alliance["autoReef"][row].get(f"node{b}"))
                out.append({"branch": b, "level": level, "phase": "auto" if auto else "teleop"})
    return out


def rests(obs: list, keep) -> list[dict]:
    """Coral that came to rest and stayed to the end: [{t, x, y, size, seen, frac}] from the detector's
    boxes [[t, [[x0, y0, x1, y1, conf], ...]], ...]; keep(x, y) limits it to one view."""
    clusters: list[dict] = []
    for t, boxes in obs:
        for x0, y0, x1, y1, _c in boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not keep(cx, cy):
                continue
            best, bd = None, LINK_PX
            for k in clusters:
                d = float(np.hypot(k["x"] - cx, k["y"] - cy))
                if d < bd:
                    best, bd = k, d
            if best is None:
                clusters.append({"x": cx, "y": cy, "obs": [(t, max(x1 - x0, y1 - y0))]})
            else:
                best["obs"].append((t, max(x1 - x0, y1 - y0)))
                n = min(len(best["obs"]), 10)
                best["x"] += (cx - best["x"]) / n
                best["y"] += (cy - best["y"]) / n
    times = np.array([o[0] for o in obs])
    if not len(times):
        return []
    out = []
    for k in clusters:
        seen = np.array(sorted({o[0] for o in k["obs"]}))
        if len(seen) < MIN_SEEN or seen[-1] < times[-1] - 30.0:
            continue
        for t in seen:
            frames = max(((times >= t) & (times < t + HOLD_S)).sum(), 1)
            if ((seen >= t) & (seen < t + HOLD_S)).sum() >= HOLD_FRAC * frames:
                after = times[times >= t]
                out.append({"t": float(t), "x": float(k["x"]), "y": float(k["y"]), "seen": int(len(seen)),
                            "size": float(np.median([o[1] for o in k["obs"]])),
                            "frac": float(len(seen[seen >= t]) / max(len(after), 1))})
                break
    return sorted(out, key=lambda r: r["t"])


def node_fills(obs: list, cams: dict[str, Camera], alliance: str, nodes: list[dict]) -> list[dict]:
    """Fill time of each official node where a resting coral shows it: nodes gain "t" (first rest, video s),
    "views" and "fit" (how well the resting coral sits on the branch, 0..1). Each view's resting coral
    are matched one-to-one to the official nodes it can show; a node seen filling in both views takes
    the earlier time when they agree within a few seconds, else the better-fitting view's."""
    tips = F.branch_tips(alliance)
    found: dict[int, list] = {}
    for view, cam in cams.items():
        if not nodes:
            continue
        P = cam.project([tips[(n["branch"], n["level"])] for n in nodes])
        dist = np.array([np.linalg.norm(tips[(n["branch"], n["level"])] - cam.centre) for n in nodes])
        length = np.maximum(cam.f * CORAL_LEN / dist, 14.0)  # px a coral spans at each node
        # Only coral near these branches can be on them. The detector boxes every coral in the frame
        # (loading stations, robots, the carpet): clustering all of them took most of a minute a match.
        pad = 2.0 * MATCH_LEN * float(length.max())
        x0, y0, x1, y1 = P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad
        R = rests(obs, lambda x, y, v=view: x0 <= x <= x1 and y0 <= y <= y1 and in_view(v, alliance, x, y))
        if not R:
            continue
        shown = np.array([in_view(view, alliance, *p) for p in P])
        big = 1e6
        cost = np.full((len(nodes), len(R)), big)
        for j, r in enumerate(R):
            d = np.hypot(P[:, 0] - r["x"], P[:, 1] - r["y"])
            ok = (d < MATCH_LEN * length) & shown
            c = (d / (0.5 * length)) ** 2 + (np.log(max(r["size"], 1.0) / length) / 0.35) ** 2
            cost[ok, j] = c[ok]
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < big:
                # other official branches of that level this coral fits nearly as well in this view: the
                # REEF is an open frame, so a front branch and the one behind it can share pixels
                alts = [k for k in range(len(nodes)) if k != i and nodes[k]["level"] == nodes[i]["level"]
                        and nodes[k]["phase"] == nodes[i]["phase"] and cost[k, j] < cost[i, j] + ALT_COST]
                found.setdefault(i, []).append((view, R[j], float(np.exp(-0.5 * cost[i, j])), alts))
    out = []
    for i, n in enumerate(nodes):
        n = dict(n)
        got = found.get(i, [])
        if got:
            ts = sorted(g[1]["t"] for g in got)
            agree = len(ts) > 1 and ts[-1] - ts[0] <= 6.0
            n["t"] = float(ts[0]) if len(ts) == 1 or agree else float(max(got, key=lambda g: g[2])[1]["t"])
            n["views"] = [g[0] for g in got]
            n["fit"] = float(max(g[2] for g in got))
            n["agree"] = bool(agree)
            # two views that agree on the moment have pinned the branch; one view alone has not
            n["alts"] = [] if agree else sorted({nodes[k]["branch"] for g in got for k in g[3]})
        out.append(n)
    return out


def pair_jumps(nodes: list[dict], jumps: list[dict], lag: float = 1.0, window=(-1.5, 6.0)) -> dict[int, int]:
    """{jump index: node index}: each decoded coral score ({t, level, phase}) with the official node it
    filled. A node seen filling pairs with the score of its level posted about `lag` s later; nodes
    nobody saw fill take the scores left over at their level."""
    out: dict[int, int] = {}
    for level in LEVELS:
        for phase in ("auto", "teleop"):
            J = [i for i, j in enumerate(jumps) if j["level"] == level and j["phase"] == phase]
            N = [i for i, n in enumerate(nodes) if n["level"] == level and n["phase"] == phase]
            if not J or not N:
                continue
            big = 1e6
            cost = np.full((len(J), len(N)), big)
            for r, ji in enumerate(J):
                for c, ni in enumerate(N):
                    t = nodes[ni].get("t")
                    if t is None:
                        cost[r, c] = 8.0  # unseen: any score of that level, once the seen ones are placed
                    else:
                        gap = jumps[ji]["t"] - t
                        if window[0] <= gap <= window[1]:
                            cost[r, c] = abs(gap - lag)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < big:
                    out[J[r]] = N[c]
    return out


def node_pose(alliance: str, branch: str) -> np.ndarray:
    """Field (x, y) of the centre of a robot placing on a branch: in front of its face, lined up with it."""
    k = LETTERS.index(branch)
    a = np.radians(F.face_normal_deg(alliance, k // 2))
    n = np.array([np.cos(a), np.sin(a)])
    t = np.array([-n[1], n[0]])
    return np.array(F.REEF_CENTRE[alliance]) + ROBOT_REACH * n + (-1 if k % 2 == 0 else 1) * F.BRANCH_HALF * t

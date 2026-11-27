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
CORAL_RADIUS = 0.055  # m
BRANCH_TILT = 35.0  # deg above level of the L2 and L3 arms
LINK_PX = 12.0  # px between boxes of one resting coral
MIN_SEEN = 15
MIN_CONF = 0.1  # detector confidence a box needs to count as a coral
STEP_LEVEL = 0.5  # sightings per frame above which a coral is there (the step that dates its arrival)
STAY_FRAC, STAY_S = 0.5, 10.0  # once there it is seen this much; one hidden later (a coral placed in front) stayed this long
FIT_SIGMA, SIZE_SIGMA = 0.35, 0.35  # box sizes a resting coral's centre may sit off its node's; log size ratio
MIN_BOX_PX = 14.0  # the detector's boxes do not get smaller than this, however far the coral
MAX_FIT_COST = 9.0  # beyond this a resting coral is not on that node
TROUGH_R, TROUGH_Z = 0.72, 0.52  # m from the REEF centre and above the carpet of a coral lying in the trough
TROUGH_SIGMA = 0.6  # box sizes a trough coral may lie from its trough spot
TROUGH_SAME_S = 2.5  # s between two views' sightings of one coral landing
PANEL_Y, PANEL_SPLIT = 645, 958  # close-up band and where the two REEF panels meet (1080p two-panel layout)
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
    """Coral that came to rest and stayed: [{t, x, y, w, h, size, seen, frac, t_end}] from the detector's
    boxes [[t, [[x0, y0, x1, y1, conf], ...]], ...]; keep(x, y) limits it to one view.

    Boxes are linked into tracks one box per track per frame, so two coral whose boxes nearly coincide
    (a near branch in front of a far one) stay two tracks and the second keeps its own arrival time. A
    track's coral arrived at the step that best splits its sightings into "not there" before and
    "there" after: a robot holding a coral at the branch for a moment earlier does not date it early,
    and a few missed frames after it lands do not date it late."""
    tracks: list[dict] = []
    for t, boxes in obs:
        cand = [((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, conf) for x0, y0, x1, y1, conf in boxes if conf >= MIN_CONF]
        cand = [c for c in cand if keep(c[0], c[1])]
        pairs = sorted((float(np.hypot(k["x"] - c[0], k["y"] - c[1])), i, j) for i, k in enumerate(tracks)
                       for j, c in enumerate(cand) if abs(k["x"] - c[0]) < LINK_PX and abs(k["y"] - c[1]) < LINK_PX)
        used_t, used_c = set(), set()
        for d, i, j in pairs:
            if d >= LINK_PX or i in used_t or j in used_c:
                continue
            used_t.add(i)
            used_c.add(j)
            k, c = tracks[i], cand[j]
            k["obs"].append((t, c[2], c[3], c[4]))
            n = min(len(k["obs"]), 10)
            k["x"] += (c[0] - k["x"]) / n
            k["y"] += (c[1] - k["y"]) / n
        for j, c in enumerate(cand):
            if j not in used_c:
                tracks.append({"x": c[0], "y": c[1], "obs": [(t, c[2], c[3], c[4])]})
    times = np.array([o[0] for o in obs])
    if not len(times):
        return []
    out = []
    for k in tracks:
        seen = np.array([o[0] for o in k["obs"]])
        if len(seen) < MIN_SEEN:
            continue
        hit = np.isin(times, seen).astype(float)
        last = int(np.searchsorted(times, seen[-1], side="right"))  # nothing is known after its last sighting
        s = hit[:last] - STEP_LEVEL
        after = np.cumsum(s[::-1])[::-1]  # sum of (hit - level) from each frame on
        before = np.cumsum(s) - s  # ... and before it
        i = int(np.argmax(after - before))
        stay = float(hit[i:last].mean())
        t_end = float(seen[-1])
        if stay < STAY_FRAC or (t_end < times[-1] - 30.0 and t_end - times[i] < STAY_S):
            continue
        obs_after = [o for o in k["obs"] if o[0] >= times[i]]
        if len(obs_after) < MIN_SEEN:
            continue  # the best step left almost nothing after it: never really there
        w, h = float(np.median([o[1] for o in obs_after])), float(np.median([o[2] for o in obs_after]))
        out.append({"t": float(times[i]), "x": float(k["x"]), "y": float(k["y"]), "w": w, "h": h, "size": max(w, h),
                    "seen": int(len(obs_after)), "frac": stay, "t_end": t_end,
                    "conf": float(np.median([o[3] for o in obs_after]))})
    return sorted(out, key=lambda r: r["t"])


def coral_ends(alliance: str, branch: str, level: str) -> tuple[np.ndarray, np.ndarray]:
    """3D ends of a coral sitting on a branch: its outer end at the branch tip, the rest back along the
    branch (straight down the vertical L4 post, down the slanted L2 and L3 arms). L1 is the trough: a
    coral lying along the face, centred on the trough spot."""
    tip = F.branch_tips(alliance)[(branch, level)]
    k = LETTERS.index(branch)
    a = np.radians(F.face_normal_deg(alliance, k // 2))
    n = np.array([np.cos(a), np.sin(a), 0.0])
    if level == "L4":
        axis = np.array([0.0, 0.0, 1.0])
    elif level == "L1":
        along = np.array([-n[1], n[0], 0.0])
        centre = np.array([*F.REEF_CENTRE[alliance], 0.0]) + TROUGH_R * n + (-1 if k % 2 == 0 else 1) * F.BRANCH_HALF * along
        centre[2] = TROUGH_Z
        return centre + 0.5 * CORAL_LEN * along, centre - 0.5 * CORAL_LEN * along
    else:
        tilt = np.radians(BRANCH_TILT)
        axis = np.cos(tilt) * n + np.array([0.0, 0.0, np.sin(tilt)])
    return tip, tip - CORAL_LEN * axis


def coral_boxes(cam: Camera, alliance: str, nodes: list[dict]) -> np.ndarray:
    """(N, 4) centre x, centre y, width, height in pixels of the box a coral on each node shows in a view."""
    out = np.zeros((len(nodes), 4))
    for i, n in enumerate(nodes):
        a, b = coral_ends(alliance, n["branch"], n["level"])
        P = cam.project([a, b])
        r = cam.f * CORAL_RADIUS / max(float(np.linalg.norm((a + b) / 2 - cam.centre)), 1e-6)
        x0, y0 = P.min(0) - r
        x1, y1 = P.max(0) + r
        out[i] = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return out


def _fit_cost(boxes: np.ndarray, r: dict) -> np.ndarray:
    """How badly a resting coral fits each node's expected box: its centre in units of the expected box
    size, and its size against the expected size (a near branch's coral is bigger than a far one's)."""
    size = np.maximum(np.maximum(boxes[:, 2], boxes[:, 3]), MIN_BOX_PX)
    d = np.hypot(boxes[:, 0] - r["x"], boxes[:, 1] - r["y"]) / size
    return (d / FIT_SIGMA) ** 2 + (np.log(max(r["size"], 1.0) / size) / SIZE_SIGMA) ** 2


def _view_matches(obs: list, cam: Camera, view: str, alliance: str, nodes: list[dict]) -> dict[int, tuple]:
    """{node index: (resting coral, cost, [other node indices it fits nearly as well])} in one view."""
    B = coral_boxes(cam, alliance, nodes)
    # Only coral near these branches can be on them. The detector boxes every coral in the frame
    # (loading stations, robots, the carpet): tracking all of them took most of a minute a match.
    pad = 2.0 * float(np.maximum(B[:, 2], B[:, 3]).max())
    x0, y0, x1, y1 = B[:, 0].min() - pad, B[:, 1].min() - pad, B[:, 0].max() + pad, B[:, 1].max() + pad
    R = rests(obs, lambda x, y: x0 <= x <= x1 and y0 <= y <= y1 and in_view(view, alliance, x, y))
    if not R:
        return {}
    shown = np.array([in_view(view, alliance, b[0], b[1]) for b in B])
    big = 1e6
    cost = np.full((len(nodes), len(R)), big)
    for j, r in enumerate(R):
        c = _fit_cost(B, r)
        ok = (c < MAX_FIT_COST) & shown
        cost[ok, j] = c[ok]
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(rows, cols):
        if cost[i, j] < big:
            # other branches of that level this coral fits nearly as well in this view: the REEF is an
            # open frame, so a front branch and the one behind it can share pixels
            alts = [k for k in range(len(nodes)) if k != i and nodes[k]["level"] == nodes[i]["level"]
                    and nodes[k].get("phase") == nodes[i].get("phase") and cost[k, j] < cost[i, j] + ALT_COST]
            out[i] = (R[j], float(cost[i, j]), alts)
    return out


def node_fills(obs: list, cams: dict[str, Camera], alliance: str, nodes: list[dict]) -> list[dict]:
    """Fill time of each node where a resting coral shows it: nodes gain "t" (when it came to rest, video
    s), "views" and "fit" (how well the resting coral sits on the branch, 0..1). Each view's resting coral
    are matched one-to-one to the nodes it can show; a node seen filling in both views takes the
    earlier time when they agree within a few seconds, else the better-fitting view's."""
    found: dict[int, list] = {}
    for view, cam in cams.items():
        if not nodes:
            continue
        for i, (r, c, alts) in _view_matches(obs, cam, view, alliance, nodes).items():
            found.setdefault(i, []).append((view, r, float(np.exp(-0.5 * c)), alts))
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


def read_reef(obs: list, cams: dict[str, Camera], alliance: str) -> list[dict]:
    """The REEF as vision alone reads it, with no official node map: [{branch, level, t, ...}] for every
    branch a coral was seen coming to rest on and staying. scripts/bench2025.py grades this against FMS."""
    every = [{"branch": b, "level": level, "phase": "teleop"} for level in LEVELS + ("L1",) for b in LETTERS]
    return [n for n in node_fills(obs, cams, alliance, every) if "t" in n]


def trough_fills(obs: list, cams: dict[str, Camera], alliance: str) -> list[dict]:
    """Coral seen coming to rest in the trough (L1): [{t, face, branch, views}], in time order. A resting
    coral is a trough coral when it sits on a trough spot and fits no branch better. The trough holds many
    coral, so several can land on one face; the same coral seen in both views is one arrival."""
    every = [{"branch": b, "level": level} for level in ("L1",) + LEVELS for b in LETTERS]
    found = []
    for view, cam in cams.items():
        B = coral_boxes(cam, alliance, every)
        size = np.maximum(np.maximum(B[:, 2], B[:, 3]), MIN_BOX_PX)
        pad = 2.0 * float(size[:12].max())
        x0, y0, x1, y1 = B[:12, 0].min() - pad, B[:12, 1].min() - pad, B[:12, 0].max() + pad, B[:12, 1].max() + pad
        shown = np.array([in_view(view, alliance, b[0], b[1]) for b in B])
        for r in rests(obs, lambda x, y: x0 <= x <= x1 and y0 <= y <= y1 and in_view(view, alliance, x, y)):
            c = _fit_cost(B, r)
            # a coral lies in the trough any way round: only where it is says it is there, not its size
            c[:12] = (np.hypot(B[:12, 0] - r["x"], B[:12, 1] - r["y"]) / size[:12] / TROUGH_SIGMA) ** 2
            c[~shown] = np.inf
            i = int(np.argmin(c))
            if i < 12 and c[i] < MAX_FIT_COST:
                found.append({"t": r["t"], "face": i // 2, "branch": LETTERS[i], "views": [view], "fit": float(np.exp(-0.5 * c[i]))})
    out: list[dict] = []
    for f in sorted(found, key=lambda f: f["t"]):
        same = next((o for o in out if o["face"] == f["face"] and f["t"] - o["t"] <= TROUGH_SAME_S
                     and f["views"][0] not in o["views"]), None)
        if same is not None:
            same["views"].append(f["views"][0])
        else:
            out.append(f)
    return out


def face_pose(alliance: str, face: int) -> np.ndarray:
    """Field (x, y) of the centre of a robot scoring on a REEF face (for the trough: anywhere along it)."""
    a = np.radians(F.face_normal_deg(alliance, face))
    return np.array(F.REEF_CENTRE[alliance]) + ROBOT_REACH * np.array([np.cos(a), np.sin(a)])


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

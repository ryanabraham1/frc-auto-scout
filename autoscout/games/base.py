"""Season plugin interface.

Everything season-independent (video, overlay clock/score, field calibration, robot tracking,
team identity, launch trajectories, reconciliation, storage, dashboard) lives outside this
package. A season plugin only declares:
  * `year`, and its field model in configs/fields/<year>.yaml
  * `launch_pieces`: game pieces whose airborne launches should be tracked (color ranges)
  * `analyze(ctx)`: turn the shared observations into per-team stats + events
  * `metric_defs`: how to display the per-team stats in the dashboard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..field import Calibration, FieldModel
from ..overlay import MatchClock, ScoreTimeline
from ..tba import MatchInfo
from ..vision.tracker import slot_position_at

REGISTRY: dict[int, type["GamePlugin"]] = {}


@dataclass
class MatchContext:
    match: MatchInfo
    field: FieldModel
    cal: Calibration
    clock: MatchClock | None
    timeline: ScoreTimeline | None
    tracks: dict[str, list]  # slot key ("red0") -> [(t, x, y, observed, box)]
    identity: dict[str, int]  # slot key -> team number
    identity_conf: dict[str, float]  # slot key -> confidence 0..1
    launches: dict[str, list]  # piece name -> [Launch]
    notes: list[str] = field(default_factory=list)
    identity_tl: dict[str, list] = field(default_factory=dict)  # slot -> [(t0, t1, team)]
    aux_reads: list = field(default_factory=list)  # [(t, view, team, conf)] from close-up views
    aux_boxes: list = field(default_factory=list)  # [t, view, x0, y0, x1, y1, alliance, team|None, conf] robots in close-ups
    blobs: dict = field(default_factory=dict, repr=False)  # piece -> moving-piece blobs (t, x, y, area, w, h)
    blob_view: tuple | None = None  # image offset of the blob coordinates
    lift_obs: dict = field(default_factory=dict)  # slot -> [[t, reach above the chassis]] (vision/lift.py)
    element_changes: list = field(default_factory=list)  # pieces seen appearing on elements (element_changes.py)
    _ts: dict = field(default_factory=dict, repr=False)  # slot -> track times (lookup cache)

    def seen_in_view(self, view: str, t0: float, t1: float) -> dict[int, float]:
        """Teams read in a close-up view (e.g. 'blue_reef') during [t0, t1] -> summed confidence."""
        out: dict[int, float] = {}
        for t, v, team, conf in self.aux_reads:
            if v == view and t0 <= t <= t1:
                out[team] = out.get(team, 0.0) + conf
        return out

    def team_at(self, slot: str, t: float) -> int | None:
        """Team in a slot at video time t (slots can swap robots during collisions)."""
        for a, b, team in self.identity_tl.get(slot, []):
            if a <= t < b:
                return team
        return self.identity.get(slot)

    def slot_of(self, team: int, t: float) -> str | None:
        for slot in self.tracks:
            if self.team_at(slot, t) == team:
                return slot
        return self.team_slot(team)

    def team_slot(self, team: int) -> str | None:
        for k, v in self.identity.items():
            if v == team:
                return k
        return None

    def station(self, team: int) -> int:
        """1-based driver station (TBA team order) for a team."""
        for a in ("red", "blue"):
            if team in self.match.teams(a):
                return self.match.teams(a).index(team) + 1
        raise KeyError(team)

    def alliance_of(self, team: int) -> str:
        return "red" if team in self.match.red else "blue"

    def robot_at(self, slot: str, t: float):
        tr = self.tracks.get(slot, [])
        if slot not in self._ts or len(self._ts[slot]) != len(tr):
            self._ts[slot] = np.array([r[0] for r in tr])
        return slot_position_at(tr, t, self._ts[slot])

    def match_time(self, vt: float) -> tuple[str, float]:
        return self.clock.phase(vt) if self.clock else ("unknown", 0.0)


class GamePlugin:
    year: int = 0
    # robot detection settings, measured per season on hand-marked frames and event OPR checks
    detector_ensemble: bool = False  # also run Spy Glass (more recall, less precision)
    static_frac: float = 0.25  # share of frames on one spot after which a detection is a field element
    launch_pieces: dict[str, tuple] = {}  # name -> (hsv_lo, hsv_hi)
    aux_every: float = 1.0  # s between close-up camera reads
    measure_reach: bool = False  # measure robots' elevator/arm reach above the chassis (vision/lift.py)
    watch_elements: tuple[str, ...] = ()  # elements whose placed pieces are watched in the wide shot
    metric_defs: list[dict] = []  # [{"key", "label", "group", "fmt"}]

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.year:
            REGISTRY[cls.year] = cls

    def __init__(self):
        self.field = FieldModel.load(self.year)

    def analyze(self, ctx: MatchContext) -> dict[str, Any]:
        raise NotImplementedError

    # ---- shared helpers ------------------------------------------------------------------
    @staticmethod
    def robot_box(ctx: MatchContext, slot: str, t: float):
        """Current detection or a motion-compensated box at the estimated field position.

        A recent detection supplies the robot's shape, never its old image position.
        Estimated boxes remain marked unobserved for attribution confidence.
        """
        r = ctx.robot_at(slot, t)
        if r is None:
            return None
        if r[3] is not None:
            return tuple(r[3]), bool(r[2])
        foot = ctx.cal.to_image([(r[0], r[1])])[0]
        w = 0.95 * ctx.cal.px_per_meter_x((r[0], r[1]))
        h = 0.8 * w
        # Preserve the measured mechanism height during a short detector dropout.
        # Search by elapsed time rather than a fixed number of rows (tracks can have
        # different frame rates, and fused close-up rows have no wide-shot box).
        tr = ctx.tracks.get(slot, [])
        ts = ctx._ts[slot]
        end = int(np.searchsorted(ts, t, side="right"))
        start = int(np.searchsorted(ts, t - 1.0, side="right"))
        for j in range(end - 1, start - 1, -1):
            row = tr[j]
            if row[3] and row[4] is not None:
                x0, y0, x1, y1 = row[4]
                old_scale = ctx.cal.px_per_meter_x(row[1:3])
                scale = ctx.cal.px_per_meter_x(r[:2]) / max(old_scale, 1e-6)
                w, h = (x1 - x0) * scale, (y1 - y0) * scale
                break
        return (foot[0] - w / 2, foot[1] - h, foot[0] + w / 2, foot[1]), False

    @staticmethod
    def _release_distance(box, start) -> float:
        """Distance to a possible release, with a soft preference for the robot's center.

        Extended shooter regions overlap in a crowd. A flat zero throughout that
        region gave ties to the first slot, independent of the release location.
        """
        x0, y0, x1, y1 = box
        sx, sy = start
        w, h = x1 - x0, y1 - y0
        outside = np.hypot(max(x0 - sx, 0, sx - x1),
                           max(y0 - max(1.5 * h, w) - sy, 0, sy - y1)) / max(w, 20)
        return float(outside + 0.25 * abs(sx - (x0 + x1) / 2) / max(w, 20)
                     + 0.1 * abs(sy - y0) / max(h, 20))

    @staticmethod
    def _box_penalty(ctx: MatchContext, slot: str, t: float, observed: bool) -> float:
        if observed:
            return 0.0
        tr = ctx.tracks.get(slot, [])
        ts = ctx._ts.get(slot)
        if ts is not None:
            lo, hi = np.searchsorted(ts, [t - 1.0, t], side="right")
            if any(r[3] and r[4] is not None for r in tr[lo:hi]):
                return 0.05  # short dropout anchored by a measured robot shape
        return 0.25  # no recent wide-shot detection

    @staticmethod
    def launch_distances(ctx: MatchContext, launch, alliances=("red", "blue")) -> dict[str, tuple[float, bool]]:
        """{slot: (distance in box widths from the release point to its box, box observed?)}"""
        out = {}
        sx, sy = launch.start
        for slot in ctx.tracks:
            if slot[:-1] not in alliances:
                continue
            rb = GamePlugin.robot_box(ctx, slot, launch.t0)
            if rb is None:
                continue
            box, observed = rb
            out[slot] = (GamePlugin._release_distance(box, (sx, sy)), bool(observed))
        return out

    @staticmethod
    def attribute_launch(ctx: MatchContext, launch, alliances=("red", "blue"), max_d: float = 1.0):
        """Find the robot slot that released a launch (nearest robot box to the release point).

        Returns (slot_key | None, distance). The release point is the first airborne sighting,
        which sits above/inside the shooter's image box.
        """
        best, best_d = None, 1e9
        for slot, (distance, observed) in GamePlugin.launch_distances(ctx, launch, alliances).items():
            d = distance + GamePlugin._box_penalty(ctx, slot, launch.t0, observed)
            if d < best_d:
                best, best_d = slot, d
        return (best, best_d) if best_d < max_d else (None, best_d)

    @staticmethod
    def trace_back(ctx: MatchContext, launch, alliances=("red", "blue"), max_back: float = 1.6,
                   with_time: bool = False):
        """Shooter of a flight whose release wasn't seen (ascent hidden in a dense volley, behind the
        goal, or the track picked up mid-air): run its fitted arc backward in time and take the robot
        whose shooter it comes out of. Only the rising part of the arc can be a release: a ball
        flying over a robot on its way down never came out of it. Returns (slot | None, distance in
        box widths), plus the release time with `with_time`."""
        ds = GamePlugin.trace_distances(ctx, launch, alliances, max_back)
        best = min(ds, key=lambda s: ds[s][0]) if ds else None
        best_d, best_t = ds[best] if best else (9.0, None)
        if best_d >= 0.6:
            best = best_t = None
        return (best, best_d, best_t) if with_time else (best, best_d)

    @staticmethod
    def trace_distances(ctx: MatchContext, launch, alliances=("red", "blue"), max_back: float = 1.6):
        """{slot: (closest approach in box widths, time)} of a flight's arc, run backward, to each robot's
        shooter (box penalty included), over the rising part of the arc only (see trace_back)."""
        from ..vision.launches import fit_flight

        P = np.asarray(launch.pts, dtype=float)
        if len(P) < 4:
            return {}
        cx, cy, rms = fit_flight(P)
        # Short noisy fragments produce explosive backwards extrapolations.
        # Their fit must explain the observed flight before it can name a robot.
        if not np.isfinite(rms) or rms > 6.0 or cy[0] <= 0:
            return {}
        dvy = np.polyder(cy)
        out = {}
        for dt in np.arange(0.0, -max_back, -0.05):
            if np.polyval(dvy, dt) >= 0:
                continue  # falling here: not where it left a shooter
            sx, sy = np.polyval(cx, dt), np.polyval(cy, dt)
            for slot in ctx.tracks:
                if slot[:-1] not in alliances:
                    continue
                rb = GamePlugin.robot_box(ctx, slot, P[0, 0] + dt)
                if rb is None:
                    continue
                box, observed = rb
                d = (GamePlugin._release_distance(box, (sx, sy))
                     + GamePlugin._box_penalty(ctx, slot, P[0, 0] + dt, observed))
                if d < out.get(slot, (9.0,))[0]:
                    out[slot] = (d, P[0, 0] + dt)
        return out

    @staticmethod
    def release_time(ctx: MatchContext, launch, slot: str) -> float:
        """Time at the selected shooter, rather than first sighting in mid-air."""
        direct = GamePlugin.launch_distances(ctx, launch, (slot[:-1],)).get(slot)
        if getattr(launch, "rising", True) and direct is not None:
            distance, observed = direct
            if observed and distance < 0.6:
                return launch.t0
        traced = GamePlugin.trace_distances(ctx, launch, (slot[:-1],)).get(slot)
        if traced is not None and traced[0] < 0.6:
            return float(traced[1])
        return launch.t0

    @staticmethod
    def proportional_split(total: float, weights: dict[str, float], fallback: dict[str, float] | None = None):
        """Split an exact (official) total among teams proportionally to weights (integers, largest remainder)."""
        w = {k: max(v, 0.0) for k, v in weights.items()}
        if sum(w.values()) <= 0:
            w = {k: max(v, 0.0) for k, v in (fallback or {}).items()}
        if sum(w.values()) <= 0 or total <= 0:
            return {k: 0 for k in weights}
        s = sum(w.values())
        raw = {k: total * v / s for k, v in w.items()}
        out = {k: int(np.floor(v)) for k, v in raw.items()}
        rem = int(round(total)) - sum(out.values())
        for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:max(rem, 0)]:
            out[k] += 1
        for k in weights:
            out.setdefault(k, 0)
        return out


def slot_positions_between(ctx: MatchContext, slot: str, t0: float, t1: float) -> np.ndarray:
    tr = ctx.tracks.get(slot, [])
    return np.array([(r[1], r[2]) for r in tr if t0 <= r[0] <= t1]).reshape(-1, 2)


def add_station_evidence(ctx: MatchContext, evidence, per_station: dict[str, list[bool | None]],
                         slot_flags: dict[str, bool | None], weight: float = 1.5, why: str = "") -> None:
    """Compare a per-station official boolean (e.g. 'climbed') with each slot's observed boolean.

    per_station: alliance -> [station1, station2, station3] booleans (None = unknown)
    slot_flags:  slot key -> observed boolean (None = unknown)
    """
    for a, flags in per_station.items():
        teams = ctx.match.teams(a)
        if all(f == flags[0] for f in flags):
            continue  # all equal: no information
        for i in range(3):
            slot = f"{a}{i}"
            obs = slot_flags.get(slot)
            if obs is None:
                continue
            for team, flag in zip(teams, flags):
                if flag is None:
                    continue
                evidence.add(slot, team, weight if flag == obs else -weight, why if flag == obs else "")


def add_calibrated_closeup_evidence(ctx: MatchContext, evidence, cams: dict, weight: float = 1.0,
                                    max_dist: float = 1.6, margin: float = 0.8) -> int:
    """Bumper reads in calibrated close-up cameras (vision/fusion.py): the read robot stands at a known
    field point, so the read names the wide-shot slot standing there at that moment (not just "the
    one nearest the element"). Returns the number of reads used."""
    from ..vision.fusion import foot, to_field

    used = 0
    for t, view, x0, y0, x1, y1, al, team, conf in ctx.aux_boxes:
        if team is None or view not in cams:
            continue
        p = to_field(cams[view]["H"], *foot((x0, y0, x1, y1)))
        a = "red" if team in ctx.match.teams("red") else "blue"
        d = {}
        for i in range(3):
            P = slot_positions_between(ctx, f"{a}{i}", t - 0.3, t + 0.3)
            if len(P):
                d[f"{a}{i}"] = float(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])))
        ranked = sorted(d.items(), key=lambda x: x[1])
        if not ranked or ranked[0][1] > max_dist:
            continue
        if len(ranked) > 1 and ranked[1][1] - ranked[0][1] < margin:
            continue  # two robots there: the read doesn't say which
        evidence.t = t
        evidence.add(ranked[0][0], team, weight * conf, f"close-up {view} at the robot")
        used += 1
    evidence.t = None
    return used


def add_closeup_evidence(ctx: MatchContext, evidence, weight: float = 0.6, max_dist: float = 3.0) -> None:
    """Close-up camera reads (a hub/reef cam named after its field element): the team read there is the
    robot of its alliance nearest that element at that moment. Reads of the opposing alliance identify
    defenders the same way."""
    red = set(ctx.match.teams("red"))
    for t, view, team, conf in ctx.aux_reads:
        if view not in ctx.field.elements:
            continue
        a = "red" if team in red else "blue"
        d = {}
        for i in range(3):
            P = slot_positions_between(ctx, f"{a}{i}", t - 0.75, t + 0.75)
            if len(P):
                d[f"{a}{i}"] = min(ctx.field.dist_to_element(p, view) for p in P)
        ranked = sorted(d.items(), key=lambda x: x[1])
        if not ranked or ranked[0][1] > max_dist:
            continue
        margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else max_dist
        if margin < 0.5:
            continue  # two robots equally close: the read doesn't say which
        evidence.t = t
        evidence.add(ranked[0][0], team, weight * conf * min(1.0, margin / 2), f"close-up {view}")
    evidence.t = None

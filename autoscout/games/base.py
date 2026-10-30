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
    identity_conf: dict[str, float]  # alliance -> confidence
    launches: dict[str, list]  # piece name -> [Launch]
    notes: list[str] = field(default_factory=list)

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
        return slot_position_at(self.tracks.get(slot, []), t)

    def match_time(self, vt: float) -> tuple[str, float]:
        return self.clock.phase(vt) if self.clock else ("unknown", 0.0)


class GamePlugin:
    year: int = 0
    launch_pieces: dict[str, tuple] = {}  # name -> (hsv_lo, hsv_hi)
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
    def attribute_launch(ctx: MatchContext, launch, alliances=("red", "blue")):
        """Find the robot slot that released a launch (nearest robot box to the release point).

        Returns (slot_key | None, distance_px). The release point is the first airborne sighting,
        which sits above/inside the shooter's image box.
        """
        best, best_d = None, 1e9
        sx, sy = launch.start
        for slot in ctx.tracks:
            if slot[:-1] not in alliances:
                continue
            r = ctx.robot_at(slot, launch.t0)
            if r is None or r[3] is None:
                continue
            x0, y0, x1, y1 = r[3]
            w, h = x1 - x0, y1 - y0
            # distance from the release point to the box, allowing release above the box (shooters
            # sit on top of the robot; a bumper-only box is short, so also allow ~1 box width)
            dx = max(x0 - sx, 0, sx - x1)
            dy = max((y0 - max(1.5 * h, 1.0 * w)) - sy, 0, sy - y1)
            d = float(np.hypot(dx, dy)) / max(w, 20)
            if d < best_d:
                best, best_d = slot, d
        return (best, best_d) if best_d < 1.0 else (None, best_d)

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

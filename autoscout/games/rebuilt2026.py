"""2026 REBUILT.

FUEL is scored into the alliance HUB (1 pt) only while that HUB is active. Hubs alternate
during the four 25 s ALLIANCE SHIFTS: the alliance that scored more FUEL in AUTO is inactive
in SHIFT 1 and 3. Robots may only score from their own ALLIANCE ZONE, so:

  * launch from own ALLIANCE ZONE           -> SHOT at the hub
  * launch from NEUTRAL ZONE / opponent zone -> FEED (passing FUEL back toward the own zone)
  * launch from outside the field at an OUTPOST -> HUMAN PLAYER throw

Exact per-period alliance FUEL counts come from the official score breakdown (hubScore);
per-robot scored FUEL = that count split by each robot's detected shots in the period.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import GamePlugin, MatchContext, add_station_evidence, slot_positions_between

PERIODS = ["auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame"]
TELEOP_BOUNDS = [  # (period, clock_hi, clock_lo) in TELEOP remaining seconds
    ("transition", 140, 130), ("shift1", 130, 105), ("shift2", 105, 80),
    ("shift3", 80, 55), ("shift4", 55, 30), ("endgame", 30, -4),
]
TOWER_PTS_AUTO = {"Level1": 15}
TOWER_PTS_END = {"Level1": 10, "Level2": 20, "Level3": 30}
COUNT_LAG = 1.0  # s between release and the hub sensor counting the FUEL


def period_of(ctx: MatchContext, vt: float) -> str | None:
    phase, rem = ctx.match_time(vt)
    if phase == "auto":
        return "auto"
    if phase == "pause":
        return "auto" if vt - ctx.clock.auto_end_vt < 3 else None  # auto FUEL still counting
    if phase in ("teleop", "post"):
        if phase == "post":
            return "endgame" if vt - ctx.clock.teleop_end_vt < 3.5 else None
        for name, hi, lo in TELEOP_BOUNDS:
            if lo < rem <= hi:
                return name
    return None


def hub_active(period: str, alliance: str, auto_winner: str | None) -> bool | None:
    if period in ("auto", "transition", "endgame"):
        return True
    if auto_winner is None:
        return None
    first_inactive = auto_winner  # auto winner's hub is inactive in SHIFT 1 and 3
    inactive = first_inactive if period in ("shift1", "shift3") else ("red" if first_inactive == "blue" else "blue")
    return alliance != inactive


def landing_point(ctx: MatchContext, L):
    """Estimated floor landing point (field meters) of a launch, or None if it can't be told.

    Fits the visible flight (x(t), y(t) quadratics in image px) and follows it past the apex
    until it is back down to about the release height (pieces leave a robot ~0.5 m up), then
    maps that image point to the floor.
    """
    P = np.array([(p[0], p[1], p[2]) for p in L.pts], float)
    if len(P) < 4:
        return None
    t = P[:, 0] - P[0, 0]
    cx = np.polyfit(t, P[:, 1], 2 if len(P) >= 6 else 1)
    cy = np.polyfit(t, P[:, 2], 2)
    if cy[0] <= 0:  # no downward curvature: can't extrapolate a landing
        return None
    ppm = ctx.cal.px_per_meter(tuple(P[0, 1:3]))
    y_land = P[0, 2] + 0.45 * ppm
    t_apex = max(0.0, -cy[1] / (2 * cy[0]))
    ts = np.linspace(t_apex, t[-1] + 2.5, 200)
    ys = np.polyval(cy, ts)
    k = np.nonzero(ys >= y_land)[0]
    if not len(k):
        return None
    tl = ts[k[0]]
    return ctx.cal.to_field([(np.polyval(cx, tl), np.polyval(cy, tl))])[0]


class Rebuilt2026(GamePlugin):
    year = 2026
    launch_pieces = {"fuel": ((18, 90, 110), (38, 255, 255))}
    metric_defs = [
        {"key": "fuel_scored", "label": "Fuel scored", "group": "Scoring"},
        {"key": "fuel_scored_auto", "label": "Auto fuel", "group": "Scoring"},
        {"key": "fuel_scored_teleop", "label": "Teleop fuel", "group": "Scoring"},
        {"key": "fuel_shot", "label": "Fuel shot (detected)", "group": "Scoring"},
        {"key": "est_accuracy", "label": "Est. accuracy %", "group": "Scoring"},
        {"key": "inactive_shots", "label": "Shots at inactive hub", "group": "Scoring"},
        {"key": "fuel_fed", "label": "Fuel fed", "group": "Feeding"},
        {"key": "fuel_fed_raw", "label": "Fuel fed (detected)", "group": "Feeding"},
        {"key": "feed_from_opp_zone", "label": "Fed from opp. zone", "group": "Feeding"},
        {"key": "auto_tower", "label": "Auto tower", "group": "Tower"},
        {"key": "endgame_tower", "label": "Endgame tower", "group": "Tower"},
        {"key": "tower_points", "label": "Tower pts", "group": "Tower"},
        {"key": "pct_alliance_zone", "label": "% time own zone", "group": "Positioning"},
        {"key": "pct_neutral_zone", "label": "% time neutral", "group": "Positioning"},
        {"key": "pct_opp_zone", "label": "% time opp. zone", "group": "Positioning"},
        {"key": "points_contributed", "label": "Points contributed", "group": "Summary"},
    ]

    def is_feed(self, ctx: MatchContext, L, alliance: str, robot) -> bool:
        land = landing_point(ctx, L)
        own = self.field.zones[f"{alliance}_alliance"]
        if land is not None and -1.5 <= land[1] <= self.field.width + 1.5:
            # landing on the own side of the zone line (past the wall counts: calibration error)
            return land[0] <= own[2] + 0.3 if alliance == "blue" else land[0] >= own[0] - 0.3
        # Landing not visible (arc left the frame): accept only a long flight toward the own zone.
        if robot is None or len(L.pts) < 3:
            return False
        a = ctx.cal.to_field([L.pts[0][1:3], L.pts[-1][1:3]])
        toward = (a[1][0] - a[0][0]) * (1 if alliance == "red" else -1)
        return toward > 2.0

    def zone_for(self, alliance: str, p) -> str:
        z = self.field.zone_of(p)
        if z == f"{alliance}_alliance":
            return "own"
        if z == "neutral":
            return "neutral"
        if z is None:
            return "off"
        return "opp"

    def outpost_alliance(self, ctx: MatchContext, launch) -> str | None:
        """If a launch starts off-field next to an OUTPOST, return that outpost's alliance."""
        f = ctx.cal.to_field([launch.start])[0]
        if self.field.in_field(f, 0.2):
            return None
        for a in ("red", "blue"):
            if np.linalg.norm(f - self.field.element_center(f"{a}_outpost")) < 3.0:
                return a
        return None

    def analyze(self, ctx: MatchContext) -> dict:
        m = ctx.match
        bd = m.breakdown or {}
        teams = {a: m.teams(a) for a in ("red", "blue")}
        slot_team = ctx.identity
        auto_winner = None
        if bd:
            ra = bd["red"]["hubScore"]["autoCount"]
            ba = bd["blue"]["hubScore"]["autoCount"]
            if ra != ba:
                auto_winner = "red" if ra > ba else "blue"
            else:  # tie: FMS picked randomly -> infer from which hub counted nothing in SHIFT 1
                if bd["red"]["hubScore"]["shift1Count"] == 0 and bd["blue"]["hubScore"]["shift1Count"] > 0:
                    auto_winner = "red"
                elif bd["blue"]["hubScore"]["shift1Count"] == 0 and bd["red"]["hubScore"]["shift1Count"] > 0:
                    auto_winner = "blue"

        events = []
        shots = defaultdict(lambda: defaultdict(float))  # team -> period -> pieces
        made = defaultdict(lambda: defaultdict(float))  # team -> period -> pieces seen entering own hub
        have_targets = any(getattr(L, "target", None) for L in ctx.launches.get("fuel", []))
        inactive = defaultdict(float)
        feeds = defaultdict(lambda: defaultdict(float))
        feed_opp = defaultdict(float)
        knocked = defaultdict(float)
        hp_shots = defaultdict(lambda: defaultdict(float))  # alliance -> period -> pieces
        unattributed = 0.0
        for L in ctx.launches.get("fuel", []):
            period = period_of(ctx, L.t0 + COUNT_LAG)
            if period is None or not getattr(L, "rising", True):
                continue  # descent-only tracks: release not seen, can't attribute
            slot, _ = self.attribute_launch(ctx, L)
            ev = {"t": round(L.t0, 2), "period": period, "n": round(L.n_pieces, 2),
                  "start": [round(v) for v in L.start], "end": [round(v) for v in L.end]}
            if slot is None:
                hp = self.outpost_alliance(ctx, L)
                if hp:
                    hp_shots[hp][period] += L.n_pieces
                    ev.update(kind="human_player", alliance=hp)
                    events.append(ev)
                else:
                    unattributed += L.n_pieces
                continue
            alliance = slot[:-1]
            team = ctx.team_at(slot, L.t0)
            r = ctx.robot_at(slot, L.t0)
            zone = self.zone_for(alliance, r[:2]) if r else "own"
            if zone == "neutral" and getattr(L, "target", None) == f"{alliance}_hub" and r is not None:
                # a robot at the zone line whose piece drops into its own hub was shooting
                # (foot-point / calibration error is ~0.5 m)
                x0 = self.field.zones[f"{alliance}_alliance"]
                dx = max(x0[0] - r[0], 0, r[0] - x0[2])
                if dx < 1.0:
                    zone = "own"
            ev.update(team=team, alliance=alliance, zone=zone, robot_xy=[round(r[0], 2), round(r[1], 2)] if r else None)
            if zone == "own":
                active = hub_active(period, alliance, auto_winner)
                if active is False:
                    inactive[team] += L.n_pieces
                    ev["kind"] = "shot_inactive"
                else:
                    shots[team][period] += L.n_pieces
                    ev["kind"] = "shot"
                    if getattr(L, "target", None) == f"{alliance}_hub":
                        made[team][period] += L.n_pieces
                        ev["made"] = True
            else:
                # A FEED is a piece *shot* from the neutral/opponent zone that lands in the robot's
                # own ALLIANCE ZONE. Pieces knocked up while a robot drives through a pile land
                # nearby (or go elsewhere) and are not feeds.
                if self.is_feed(ctx, L, alliance, r):
                    feeds[team][period] += L.n_pieces
                    if zone == "opp":
                        feed_opp[team] += L.n_pieces
                    ev["kind"] = "feed"
                else:
                    knocked[team] += L.n_pieces
                    ev["kind"] = "knocked"
            events.append(ev)

        # Zone occupancy (for positioning stats and as fallback weights).
        zone_time = defaultdict(lambda: defaultdict(float))  # team -> zone -> seconds
        zone_time_period = defaultdict(lambda: defaultdict(float))  # team -> period -> secs in own zone
        for slot, tr in ctx.tracks.items():
            alliance = slot[:-1]
            for i in range(1, len(tr)):
                t, x, y = tr[i][0], tr[i][1], tr[i][2]
                team = ctx.team_at(slot, t)
                dt = min(tr[i][0] - tr[i - 1][0], 0.5)
                per = period_of(ctx, t)
                if per is None:
                    continue
                z = self.zone_for(alliance, (x, y))
                zone_time[team][z] += dt
                if z == "own":
                    zone_time_period[team][per] += dt

        # Split official per-period hub counts among robots (+ human player).
        scored = defaultdict(lambda: defaultdict(int))
        source = "breakdown" if bd else "overlay"
        alliance_summary = {}
        for a in ("red", "blue"):
            hs = bd.get(a, {}).get("hubScore", {}) if bd else {}
            tot_raw = tot_off = 0.0
            for per in PERIODS:
                official = hs.get(f"{per}Count")
                if official is None:
                    official = self._overlay_count(ctx, a, per)
                # Evidence: pieces seen dropping into the hub count fully; other shots partially
                # (make detection misses some, e.g. when the opening is occluded).
                if have_targets:
                    weights = {t: made[t][per] + 0.3 * (shots[t][per] - made[t][per]) for t in teams[a]}
                else:
                    weights = {t: shots[t][per] for t in teams[a]}
                weights["HP"] = hp_shots[a][per]
                fallback = {t: zone_time_period[t][per] for t in teams[a]}
                split = self.proportional_split(official or 0, weights, fallback)
                for k, v in split.items():
                    scored[k][per] = v
                tot_raw += sum(shots[t][per] for t in teams[a]) + hp_shots[a][per]
                if hub_active(per, a, auto_winner) is not False:
                    tot_off += official or 0
            det_made = sum(made[t][p] for t in teams[a] for p in PERIODS)
            alliance_summary[a] = {
                "official_fuel": int(tot_off), "detected_shots": round(tot_raw, 1),
                "detected_made": round(det_made, 1),
                "made_ratio": round(det_made / tot_off, 2) if tot_off else None,
                "detection_ratio": round(tot_raw / tot_off, 2) if tot_off else None,
                "hp_fuel_scored": int(sum(scored["HP"][p] for p in PERIODS)) if a else 0,
                "auto_winner": auto_winner == a,
                "unattributed_launches": round(unattributed, 1),
            }
            scored.pop("HP", None)

        teams_out = {}
        for a in ("red", "blue"):
            hs_tot = alliance_summary[a]
            # detection recall correction for feeds (shots/official fuel); bounded
            ratio = hs_tot["detection_ratio"]
            k = float(np.clip(1.0 / ratio, 0.25, 2.5)) if ratio else 1.0
            for i, team in enumerate(teams[a]):
                st = i + 1
                at = bd.get(a, {}).get(f"autoTowerRobot{st}", "None") if bd else "Unknown"
                et = bd.get(a, {}).get(f"endGameTowerRobot{st}", "None") if bd else "Unknown"
                tower_pts = TOWER_PTS_AUTO.get(at, 0) + TOWER_PTS_END.get(et, 0)
                fuel_auto = scored[team]["auto"]
                fuel_tele = sum(scored[team][p] for p in PERIODS[1:])
                zt = zone_time[team]
                ztot = sum(zt.values()) or 1.0
                fed_raw = sum(feeds[team].values())
                teams_out[team] = {
                    "alliance": a, "station": st,
                    "fuel_scored": fuel_auto + fuel_tele,
                    "fuel_scored_auto": fuel_auto, "fuel_scored_teleop": fuel_tele,
                    "fuel_scored_by_period": {p: scored[team][p] for p in PERIODS},
                    "fuel_shot": round(sum(shots[team].values()), 1),
                    "fuel_made_seen": round(sum(made[team].values()), 1),
                    "est_accuracy": round(100 * (fuel_auto + fuel_tele) / sum(shots[team].values()))
                    if sum(shots[team].values()) >= 3 else None,
                    "fuel_shot_by_period": {p: round(shots[team][p], 1) for p in PERIODS},
                    "inactive_shots": round(inactive[team], 1),
                    "fuel_fed_raw": round(fed_raw, 1),
                    "fuel_fed": int(round(fed_raw * k)),
                    "fuel_fed_by_period": {p: round(feeds[team][p] * k, 1) for p in PERIODS},
                    "feed_from_opp_zone": int(round(feed_opp[team] * k)),
                    "knocked_pieces": round(knocked[team], 1),
                    "auto_tower": at, "endgame_tower": et, "tower_points": tower_pts,
                    "pct_alliance_zone": round(100 * zt["own"] / ztot),
                    "pct_neutral_zone": round(100 * zt["neutral"] / ztot),
                    "pct_opp_zone": round(100 * zt["opp"] / ztot),
                    "points_contributed": fuel_auto + fuel_tele + tower_pts,
                }
        return {"teams": teams_out, "alliances": alliance_summary, "events": events,
                "auto_winner": auto_winner, "count_source": source}

    def identity_evidence(self, ctx: MatchContext, evidence) -> None:
        """Climb results per station vs. which slot ends at its TOWER."""
        bd = ctx.match.breakdown
        if not bd or ctx.clock is None:
            return
        end = ctx.clock.teleop_end_vt
        flags = {}
        for slot in ctx.tracks:
            P = slot_positions_between(ctx, slot, end - 4, end + 1)
            if len(P):
                d = min(self.field.dist_to_element(p, f"{slot[:-1]}_tower") for p in P)
                flags[slot] = d < 0.7
        per_station = {a: [bd[a].get(f"endGameTowerRobot{i}", "None") != "None" for i in (1, 2, 3)]
                       for a in ("red", "blue")}
        add_station_evidence(ctx, evidence, per_station, flags, 1.5, "endgame tower")
        aend = ctx.clock.auto_end_vt
        flags = {}
        for slot in ctx.tracks:
            P = slot_positions_between(ctx, slot, aend - 3, aend + 1)
            if len(P):
                d = min(self.field.dist_to_element(p, f"{slot[:-1]}_tower") for p in P)
                flags[slot] = d < 0.7
        per_station = {a: [bd[a].get(f"autoTowerRobot{i}", "None") != "None" for i in (1, 2, 3)]
                       for a in ("red", "blue")}
        add_station_evidence(ctx, evidence, per_station, flags, 1.0, "auto tower")

    def _overlay_count(self, ctx: MatchContext, alliance: str, period: str) -> int:
        """Fallback when no official breakdown: score increase in the period from the overlay."""
        tl = ctx.timeline
        if tl is None or ctx.clock is None:
            return 0
        total = 0
        for t, d in tl.deltas(alliance):
            if period_of(ctx, t) == period:
                total += d
        return total

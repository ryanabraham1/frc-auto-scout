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
COLUMN_WEIGHT = 0.3  # release blobs per detected shot (a ball is seen leaving its shooter for ~3 blob frames)
FEED_MIN_DEPTH = 0.5  # m past the zone line: closer launches are shots taken from the line
FEED_SPEED = 1.0  # m/s of image-horizontal speed toward the own zone at release
FEED_ACCURACY = 0.8  # share of own-zone shots that score, for converting release blobs to balls
RELEASE_SPEED = 3.0  # m/s upward: launched FUEL leaves a shooter at ~8-12 m/s; crowd and hopper motion is slower
SHOT_WINDOW = 3.0  # s before a score increase in which the scoring shot was released
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
        shot_log = []  # (t0, team or "HP", alliance, pieces, made) for time-local attribution
        for L in ctx.launches.get("fuel", []):
            period = period_of(ctx, L.t0 + COUNT_LAG)
            if period is None:
                continue
            if getattr(L, "rising", True):
                # a release must be right at a robot's top: looser matches pick up crowd motion
                # and the overlay above far-side robots
                slot, _ = self.attribute_launch(ctx, L, max_d=0.6)
            elif L.target and L.target.endswith("_hub"):
                # descent-only (ascent hidden, typical inside a volley) but dropping into a HUB:
                # trace the arc back to the shooter of that HUB's alliance
                slot, _ = self.trace_back(ctx, L, alliances=(L.target[:-4],))
                if slot is None:
                    continue
            else:
                continue  # descent-only and not into a goal: release not seen, can't attribute
            ev = {"t": round(L.t0, 2), "period": period, "n": round(L.n_pieces, 2),
                  "start": [round(v) for v in L.start], "end": [round(v) for v in L.end]}
            if slot is None:
                hp = self.outpost_alliance(ctx, L)
                if hp:
                    hp_shots[hp][period] += L.n_pieces
                    shot_log.append((L.t0, "HP", hp, L.n_pieces, False))
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
                    is_made = getattr(L, "target", None) == f"{alliance}_hub"
                    if is_made:
                        made[team][period] += L.n_pieces
                        ev["made"] = True
                    shot_log.append((L.t0, team, alliance, L.n_pieces, is_made))
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
        live = {}
        columns = self.column_activity(ctx)
        for a in ("red", "blue"):
            live[a], credit = self.live_weights(ctx, a, teams[a], shot_log, columns)
            events += credit
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
                if live[a] is not None:
                    weights = {t: live[a][t][per] for t in teams[a]}
                    weights["HP"] = live[a]["HP"][per]
                elif have_targets:
                    weights = {t: made[t][per] + 0.3 * (shots[t][per] - made[t][per]) for t in teams[a]}
                    weights["HP"] = hp_shots[a][per]
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

        # Feeds from release evidence: release blobs seen while a robot was in the NEUTRAL or the
        # opponent's zone, converted to balls with this match's own-zone ratio of officially counted
        # FUEL to release blobs (scored FUEL is a share of shots, hence the accuracy allowance).
        ctimes, csum = columns if columns else (None, {})
        own_blobs = sum(float(c[-1]) for c in csum.values() if len(c))
        off_fuel = sum(alliance_summary[a]["official_fuel"] for a in ("red", "blue"))
        balls_per_blob = float(np.clip(off_fuel / FEED_ACCURACY / own_blobs, 0.05, 1.0)) if own_blobs > 50 else 0.0
        feed_blobs = getattr(self, "feed_blobs", {})
        if balls_per_blob:  # feed events for the timeline / annotated video, one per robot-second
            for e in events:  # whole tracked flights stay listed, but the release count is the feed tally
                if e.get("kind") == "feed":
                    e["kind"] = "feed_flight"
            buckets = defaultdict(float)
            for t, team, a, c in getattr(self, "feed_log", []):
                buckets[(int(t), team, a)] += c
            for (t, team, a), c in sorted(buckets.items()):
                n = c * balls_per_blob
                if n >= 0.5:
                    events.append({"t": float(t), "kind": "feed", "team": team, "alliance": a, "n": round(n, 2),
                                   "period": period_of(ctx, t), "by": "release"})

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
                fb = feed_blobs.get(team, {})
                fed_released = sum(fb.values()) * balls_per_blob
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
                    # whole flights tracked into the own zone miss most long lobs; releases from the
                    # neutral/opponent zone catch them, so the larger of the two estimates is used
                    "fuel_fed": int(round(max(fed_raw * k, fed_released))),
                    "fuel_fed_released": round(fed_released, 1),
                    "fuel_fed_by_period": {p: round(max(feeds[team][p] * k, fb.get(p, 0.0) * balls_per_blob), 1)
                                           for p in PERIODS},
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

    def column_activity(self, ctx: MatchContext):
        """FUEL leaving each robot's shooter, per blob frame: (times, {slot: cumulative counts}).

        Inside a dense volley single balls can't be followed (identical pieces a few cm apart get
        linked to each other), but every ball of a volley leaves its shooter fast and upward.
        Moving-FUEL blobs are linked to their nearest blob in the previous blob frame to get a
        velocity; a blob counts as a release when it is in the zone just above a robot (from its
        top to ~1.6 robot heights up), moving up at launch speed (> RELEASE_SPEED m/s). Each such
        blob goes to the one robot whose release zone it is nearest, so a ball shot over a
        teammate is not credited twice, and the crowd above far-side robots (slow, not launched)
        and pieces falling into hoppers or the HUB (moving down) never count."""
        from scipy.spatial import cKDTree

        self.feed_blobs = defaultdict(lambda: defaultdict(float))  # team -> period -> release blobs
        self.feed_log = []  # (t, team, alliance, blobs) for the event list
        B = ctx.blobs.get("fuel")
        if B is None or not len(B) or ctx.blob_view is None:
            return None, {}
        order = np.argsort(B[:, 0], kind="stable")
        B = B[order]
        T = B[:, 0]
        X, Y = B[:, 1] + ctx.blob_view[0], B[:, 2] + ctx.blob_view[1]
        times, starts = np.unique(T, return_index=True)
        ends = np.r_[starts[1:], len(T)]
        slots = list(ctx.tracks)
        counts = {slot: np.zeros(len(times)) for slot in slots}
        prev_tree, prev_t, prev_xy = None, None, None
        for k in range(len(times)):
            xy = np.stack([X[starts[k]: ends[k]], Y[starts[k]: ends[k]]], 1)
            tree = cKDTree(xy) if len(xy) else None
            if prev_tree is not None and tree is not None and 0 < times[k] - prev_t < 0.1:
                dt = times[k] - prev_t
                zones = []
                for slot in slots:
                    r = ctx.robot_at(slot, times[k])
                    z = self.zone_for(slot[:-1], r[:2]) if r is not None else "off"
                    if z == "off":
                        continue
                    rb = self.robot_box(ctx, slot, times[k])
                    if rb is None:
                        continue
                    (x0, y0, x1, y1), _ = rb
                    ppm = ctx.cal.px_per_meter_x(r[:2])
                    zones.append((slot, x0, y0, x1, y1, ppm, z == "own", self._home_dir(ctx, slot[:-1], r[:2], rb[0]),
                                  self._dist_from_own_zone(slot[:-1], r[:2])))
                if zones:
                    # velocity from the nearest blob one frame back (within a fast ball's travel)
                    vmax = max(z[5] for z in zones) * 25.0 * dt
                    d, j = prev_tree.query(xy, distance_upper_bound=vmax)
                    ok = np.isfinite(d)
                    vx = np.full(len(xy), 0.0)
                    vy = np.full(len(xy), 0.0)
                    vx[ok] = (xy[ok, 0] - prev_xy[j[ok], 0]) / dt
                    vy[ok] = (xy[ok, 1] - prev_xy[j[ok], 1]) / dt
                    best = np.full(len(xy), -1)
                    best_d = np.full(len(xy), np.inf)
                    for zi, (slot, x0, y0, x1, y1, ppm, *_z) in enumerate(zones):
                        w, h = x1 - x0, max(y1 - y0, 0.5 * (x1 - x0))
                        cx = 0.5 * (x0 + x1)
                        inside = ((np.abs(xy[:, 0] - cx) < 0.7 * w) & (xy[:, 1] < y0 + 0.15 * h)
                                  & (xy[:, 1] > y0 - 1.6 * h) & (vy < -RELEASE_SPEED * ppm))
                        dd = np.hypot((xy[:, 0] - cx) / w, (xy[:, 1] - y0) / h)
                        take = inside & (dd < best_d)
                        best[take], best_d[take] = zi, dd[take]
                    for zi, (slot, *_r, ppm, own, home, away) in enumerate(zones):
                        mine = best == zi
                        if own:  # FUEL only scores when shot from the ALLIANCE ZONE
                            counts[slot][k] = np.count_nonzero(mine)
                        elif home and away >= FEED_MIN_DEPTH:
                            # a FEED: launched from well inside the NEUTRAL / opponent zone and heading
                            # back toward the robot's own ALLIANCE ZONE (where it has to land)
                            c = np.count_nonzero(mine & (vx * home > FEED_SPEED * ppm))
                            if c:
                                team = ctx.team_at(slot, times[k])
                                self.feed_blobs[team][period_of(ctx, times[k]) or "teleop"] += c
                                self.feed_log.append((float(times[k]), team, slot[:-1], c))
            prev_tree, prev_t, prev_xy = tree, times[k], xy
        return times, {slot: np.cumsum(c) for slot, c in counts.items()}

    def _home_dir(self, ctx: MatchContext, alliance: str, p, box) -> float:
        """+1 / -1: the image-x direction (at this robot) that leads toward its ALLIANCE ZONE."""
        foot = ((box[0] + box[2]) / 2, box[3])
        f = ctx.cal.to_field([foot, (foot[0] + 20.0, foot[1])])
        dfx = f[1][0] - f[0][0]
        if not np.isfinite(dfx) or abs(dfx) < 1e-6:
            return 0.0
        return float(np.sign(dfx)) * (1.0 if alliance == "red" else -1.0)

    def _dist_from_own_zone(self, alliance: str, p) -> float:
        """Metres from the robot to its ALLIANCE ZONE along the field (0 inside it)."""
        x0, _, x1, _ = self.field.zones[f"{alliance}_alliance"]
        return float(max(x0 - p[0], 0.0, p[0] - x1))

    def live_weights(self, ctx: MatchContext, alliance: str, teams: list[int], shot_log: list, columns=None):
        """Credit every live score increase to whoever was shooting just before it.

        The broadcast score rises ball by ball as the HUB counts FUEL, so it says *when* each
        alliance scored even when a volley is too dense to track ball by ball. Each increase is
        shared among the alliance's robots by their detected shots in the few seconds before it
        (a seen make counts double) and by how much moving FUEL filled the launch column above
        each of them (what reveals a volley); with neither, among robots in shooting position: in
        their ALLIANCE ZONE, weighted toward the HUB.
        Returns ({team | "HP": {period: points}}, credit events), or (None, []) without a live
        score timeline.
        """
        tl = ctx.timeline
        if tl is None or ctx.clock is None:
            return None, []
        deltas = tl.deltas(alliance)
        if len(deltas) < 15:
            return None, []  # score posted in chunks, not live: no timing to work with
        ends = (ctx.clock.auto_end_vt, ctx.clock.teleop_end_vt)
        out = defaultdict(lambda: defaultdict(float))
        credit = []  # per score increase: who was credited, for the timeline / annotated video
        hub = f"{alliance}_hub"
        for t, d in deltas:
            if d >= 10 and any(-1.0 < t - e < 4.0 for e in ends):
                continue  # TOWER points posted at the end of AUTO / the match, not FUEL
            per = period_of(ctx, t - COUNT_LAG)
            if per is None:
                continue
            lo, hi = t - SHOT_WINDOW, t - 0.3
            w = defaultdict(float)
            for t0, who, a, n, is_made in shot_log:
                if a == alliance and lo <= t0 <= hi:
                    w[who] += n * (2.0 if is_made else 1.0)
            ctimes, csum = columns or (None, {})
            if ctimes is not None:
                i0, i1 = np.searchsorted(ctimes, lo), np.searchsorted(ctimes, hi) - 1
                for team in teams:
                    slot = ctx.slot_of(team, t)
                    if slot in csum and i1 >= max(i0, 0):
                        c = csum[slot][i1] - (csum[slot][i0 - 1] if i0 > 0 else 0.0)
                        if c > 0:
                            w[team] += COLUMN_WEIGHT * c
            src = "seen" if w else "position"
            if not w:
                for team in teams:
                    slot = ctx.slot_of(team, t)
                    pos = [ctx.robot_at(slot, x) for x in np.linspace(lo, hi, 5)] if slot else []
                    pos = [q for q in pos if q is not None and self.zone_for(alliance, q[:2]) == "own"]
                    for q in pos:
                        w[team] += 1.0 / (1.0 + self.field.dist_to_element(q[:2], hub) / 2.0)
            tot = sum(w.values())
            for who, v in w.items():
                out[who][per] += d * v / tot
                if who != "HP":
                    credit.append({"t": round(t, 2), "kind": "scored", "team": who, "alliance": alliance,
                                   "period": per, "n": round(d * v / tot, 2), "by": src})
        return out, credit

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

"""2025 REEFSCAPE.

The live overlay score says what was scored and when; the video says who. An alliance's score jumps
are decoded together so the actions add up to the official breakdown's counts per category
(decode_jumps): equal point values (teleop L3 = NET = 4, teleop L2 = LEAVE = 3) are told apart by
what the official counts still need and where the robots were. Each robot action is then credited
from coral seen appearing on the reef (which robot was at that spot), elevator lift, time at the
element and algae launches, one piece per robot at a time (assign_actions). The exact official
counts are split among robots by these credits. LEAVE and BARGE results are reported per station
in the breakdown, so they are exact per team.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import GamePlugin, MatchContext, add_station_evidence, slot_positions_between

AUTO_VALUES = {"L1": 3, "L2": 4, "L3": 6, "L4": 7, "net": 4, "processor": 6}
TELE_VALUES = {"L1": 2, "L2": 3, "L3": 4, "L4": 5, "net": 4, "processor": 6}
ENDGAME_PTS = {"DeepCage": 12, "ShallowCage": 6, "Parked": 2, "None": 0}
CATS = ["L1", "L2", "L3", "L4", "net", "processor"]
NON_ROBOT = {"foul_minor": 2, "foul_major": 6, "hp_net": 4}  # jumps no robot action explains
NEAR = {"reef": 1.1, "net": 1.0, "processor": 1.3}  # meters from element edge
LOOKBACK = 5.0  # s: scoring is often posted a few seconds after it happens
CREDIT_PRE, CREDIT_POST = 3.0, 1.5  # s around a post in which time at the element counts as credit
CAM_BOOST_AUTO = 3.0  # the same during AUTO
CAM_BOOST = 0.0  # per unit of reef close-up read confidence (off: hurt coral vs OPR; the close-up still feeds identity and decoding)
JOINT_CREDIT = True  # credit all of an alliance's actions together (one piece per robot at a time)
REFRACTORY = 3.0  # s: one robot doesn't score two pieces of the same kind closer together than this
REFRACTORY_COST = 1.0  # objective cost of crediting one robot inside that window anyway (log-likelihood)
AUTO_SHRINK = 0.0  # pseudo-share each robot gets in splitting AUTO coral before the video's shares
ELEM_TIME_W = 0.0  # weight of each robot's share of match time at the NET / PROCESSOR in algae credit
UNSEEN_W = 0.5  # NET credit for a robot not detected around the post (the NET side is a detection blind spot)
ALGAE_COUNTER = True  # use the overlay's per-alliance algae counter when the event has one
ALGAE_MISMATCH = 5.0  # decoding cost per algae action more/fewer than the counter moved at that jump
UNMATCHED_W = 0.0  # weight of a coral seen placed that no score jump explains (off: hurt vs OPR, 15 matches)
LEVEL_HEIGHTS = [("L4", 0.75), ("L3", 0.55), ("L2", 0.38), ("L1", -1.0)]  # placement height (0-1 of reef) -> level
REACH_PRE, REACH_POST = 3.0, 0.7  # s around a post in which a robot's reach counts
REACH_MIN = 0.2  # box heights of lift (over the robot's usual reach) that start to count
REACH_BOOST = 1.0  # conservative weight: stronger reach weighting hurt on the pinned 35-match comparison
REACH_ADD = 0.3  # credit (seconds-at-the-reef equivalent) for a robot at the reef that lifted
LIFT_BOOST = 3.0  # coral credit multiplier per unit of elevator lift (box taller than usual) near the post
CONF_WEIGHT = True  # weigh each action's credit by how much the video says about it
CONF_REF = {"reef": 1.5, "net": 1.0, "processor": 1.0}  # evidence (s at the element / closeness) = full confidence
ALGAE_CREDIT = "near"  # "near": closeness + seen launches | "dwell": time at the element like coral
LAUNCH_W = 1.0  # added credit for the robot an algae launch is seen leaving just before a NET post
COAST_MIN_SEEN = 3  # sightings in the seconds before a coast needed for it to count
STALE_S = 3.0  # s a coasting (unseen) position stays usable after a sighting (a robot lost behind a REEF is usually still there)
PLACE_MIN = 0.3  # a placement only credits robots near its spot (best robot's weight at least this)
CAM_PLACE_W = 1.0  # weight of a reef close-up placement next to a wide-shot one
CAM_PLACE_PX = 40.0  # px: falloff of close-up placement credit with distance from a robot's box
PLACE_MIX = 0.7  # weight of "whose robot was over the spot where the coral appeared" in coral credit
PLACE_WINDOW = (-1.0, 4.5)  # s: a post comes this long after the coral was seen placed
PLACE_LAG = 1.0  # s: typical delay from placement to post
PLACE_PX = 25.0  # px: falloff of placement credit with horizontal distance from the robot's box
DECODE = "joint"  # "joint": all jumps decoded together against official counts | "each": one at a time
SHARE_MODE = "hybrid"  # "nearest" | "soft" | "dwell" | "hybrid" (dwell for coral, soft for algae): see _shares
SOFT_SCALE = 1.0  # m
DWELL_SLACK = 0.5  # m added to an element's reach when timing how long a robot stayed there


CORAL = ("L1", "L2", "L3", "L4")
BARGE_TOKENS = {"park": 2, "shallow": 6, "deep": 12}
COUNT_SLACK = 2.0  # objective cost of each action the decoded jumps leave over/under an official count
FOUL_SLACK = 0.3  # per point of foul points not matched by decoded foul jumps (fouls often post late)
NOTHING = -4.0  # a jump explained by nothing (overlay misread)


def _decompose(points: int, allowed: dict[str, int]) -> list[list[str]]:
    """All multisets of actions (up to 3) whose values sum to `points`."""
    out = []
    names = list(allowed)
    for a in names:
        if allowed[a] == points:
            out.append([a])
        for b in names:
            if names.index(b) < names.index(a):
                continue
            if allowed[a] + allowed[b] == points:
                out.append([a, b])
            for c in names:
                if names.index(c) < names.index(b):
                    continue
                if allowed[a] + allowed[b] + allowed[c] == points:
                    out.append([a, b, c])
    return out


class Reefscape2025(GamePlugin):
    year = 2025
    prior: dict | None = None  # alliance -> {team: {cat: weight}}, set by an event-level pass
    # Spy Glass used to be ensembled for recall at the REEF; on 12 hand-marked 2025 frames (48 robots) our
    # own model with the field filters scores recall 75% / precision 90% vs the ensemble's 77% / 82%
    # (61% unfiltered), and every false box becomes a junk track
    detector_ensemble = False
    static_frac = 0.5
    launch_pieces = {"algae": ((78, 60, 80), (100, 255, 255))}
    measure_reach = True  # L2-L4 coral means lifting a mechanism: who reached up before a score
    aux_every = 0.5  # the reef close-ups see robots the wide shot loses behind the REEF: read them often
    watch_elements = ("blue_reef", "red_reef")  # coral placed on the reefs (vision/element_changes.py)
    metric_defs = [
        {"key": "coral_total", "label": "Coral", "group": "Coral"},
        {"key": "coral_L4", "label": "L4", "group": "Coral"},
        {"key": "coral_L3", "label": "L3", "group": "Coral"},
        {"key": "coral_L2", "label": "L2", "group": "Coral"},
        {"key": "coral_L1", "label": "L1", "group": "Coral"},
        {"key": "auto_coral", "label": "Auto coral", "group": "Auto"},
        {"key": "auto_leave", "label": "Leave", "group": "Auto"},
        {"key": "algae_net", "label": "Net algae", "group": "Algae"},
        {"key": "algae_processor", "label": "Processor algae", "group": "Algae"},
        {"key": "endgame", "label": "Barge", "group": "Endgame"},
        {"key": "pct_defense", "label": "% time on opp. side", "group": "Positioning"},
        {"key": "points_contributed", "label": "Points contributed", "group": "Summary"},
    ]

    @staticmethod
    def _fresh(ctx: MatchContext, slot: str) -> list:
        """A slot's track rows whose position is worth something: seen, or coasted from a sighting less than
        STALE_S ago. A robot the camera hasn't seen for a while keeps its last position in the tracker,
        and that guess (often right next to a REEF where it was lost) must not earn it credit."""
        cache = ctx.__dict__.setdefault("_fresh_rows", {})
        tr = ctx.tracks.get(slot, [])
        if slot not in cache or cache[slot][0] != len(tr):
            out, seen = [], []  # seen: times of recent sightings
            for r in tr:
                if r[3]:
                    seen.append(r[0])
                    out.append(r)
                    continue
                while seen and r[0] - seen[0] > STALE_S + 2.0:
                    seen.pop(0)
                # coast only from a real stretch of sightings, not a one-frame blip (often a false box)
                if seen and r[0] - seen[-1] <= STALE_S and len(seen) >= COAST_MIN_SEEN:
                    out.append(r)
            cache[slot] = (len(tr), out)
        return cache[slot][1]

    def _elem_dists(self, ctx: MatchContext, slot: str, t0: float, t1: float, alliance: str):
        """Min distance of a slot to each of its alliance's scoring elements during [t0, t1]."""
        best = {"reef": 9e9, "net": 9e9, "processor": 9e9}
        tr = self._fresh(ctx, slot)
        for r in tr:
            if t0 <= r[0] <= t1:
                p = (r[1], r[2])
                best["reef"] = min(best["reef"], self.field.dist_to_element(p, f"{alliance}_reef"))
                best["net"] = min(best["net"], self.field.dist_to_element(p, f"{alliance}_net_zone"))
                best["processor"] = min(best["processor"], self.field.dist_to_element(p, f"{alliance}_processor"))
        return best

    def _dwell(self, ctx: MatchContext, slot: str, t0: float, t1: float, alliance: str) -> dict[str, float]:
        """Seconds a slot spent within reach of each of its alliance's scoring elements in [t0, t1]."""
        out = {"reef": 0.0, "net": 0.0, "processor": 0.0}
        tr = self._fresh(ctx, slot)
        prev = None
        for r in tr:
            if not t0 <= r[0] <= t1:
                continue
            dt = min(r[0] - prev, 0.5) if prev is not None else 0.1
            prev = r[0]
            p = (r[1], r[2])
            if self.field.dist_to_element(p, f"{alliance}_reef") < NEAR["reef"] + DWELL_SLACK:
                out["reef"] += dt
            if self.field.dist_to_element(p, f"{alliance}_net_zone") < NEAR["net"] + DWELL_SLACK:
                out["net"] += dt
            if self.field.dist_to_element(p, f"{alliance}_processor") < NEAR["processor"] + DWELL_SLACK:
                out["processor"] += dt
        return out

    def official_counts(self, bd_a: dict) -> dict[str, dict[str, int]]:
        """Official per-category counts for one alliance, split auto/teleop."""
        ar, tr = bd_a["autoReef"], bd_a["teleopReef"]
        auto = {"L4": ar["tba_topRowCount"], "L3": ar["tba_midRowCount"], "L2": ar["tba_botRowCount"],
                "L1": ar["trough"], "net": 0, "processor": 0}
        tele = {"L4": max(tr["tba_topRowCount"] - ar["tba_topRowCount"], 0),
                "L3": max(tr["tba_midRowCount"] - ar["tba_midRowCount"], 0),
                "L2": max(tr["tba_botRowCount"] - ar["tba_botRowCount"], 0),
                "L1": tr["trough"],
                "net": bd_a.get("netAlgaeCount", 0), "processor": bd_a.get("wallAlgaeCount", 0)}
        return {"auto": auto, "teleop": tele}

    def analyze(self, ctx: MatchContext) -> dict:
        m = ctx.match
        bd = m.breakdown or {}
        teams = {a: m.teams(a) for a in ("red", "blue")}
        team_slot = {v: k for k, v in ctx.identity.items()}
        weights = {ph: defaultdict(lambda: defaultdict(float)) for ph in ("auto", "teleop")}
        events = []
        if ctx.timeline is not None and ctx.clock is not None:
            for a in ("red", "blue"):
                if DECODE == "joint":
                    self._decode_alliance(ctx, a, teams[a], weights, events)
                    continue
                for t, d in ctx.timeline.deltas(a):
                    phase, _ = ctx.match_time(t)
                    if phase == "pause":
                        phase = "auto"
                    if phase not in ("auto", "teleop"):
                        continue
                    vals = AUTO_VALUES if phase == "auto" else TELE_VALUES
                    extra = dict(NON_ROBOT)
                    if phase == "auto" or t - ctx.clock.auto_end_vt < 4:
                        extra["leave"] = 3
                    near, dwell = {}, {}
                    for team in teams[a]:
                        slot = ctx.slot_of(team, t)
                        if slot:
                            near[team] = self._elem_dists(ctx, slot, t - LOOKBACK, t, a)
                            dwell[team] = self._dwell(ctx, slot, t - LOOKBACK, t, a)
                        else:
                            near[team] = {"reef": 9e9, "net": 9e9, "processor": 9e9}
                            dwell[team] = {"reef": 0.0, "net": 0.0, "processor": 0.0}
                    # The reef close-up camera shows who is actually at the reef (the wide shot often
                    # has them hidden behind it): those teams are at the reef, others are not.
                    # Seen there = at the reef. Not seen proves nothing (bumper may face away from
                    # that camera), so unseen teams keep their wide-shot evidence.
                    seen = ctx.seen_in_view(f"{a}_reef", t - 3.5, t + 0.5)
                    for team, c in seen.items():
                        if team in near and c >= 0.5:
                            near[team]["reef"] = min(near[team]["reef"], 0.3)
                    ev = self._explain(d, vals, near, extra)
                    if ev is None:
                        events.append({"t": round(t, 2), "alliance": a, "phase": phase, "points": d, "kind": "unexplained"})
                        continue
                    for team, cat in ev:
                        for who, share in self._shares(near, team, cat, dwell).items():
                            weights[phase][who][cat] += share
                        events.append({"t": round(t, 2), "alliance": a, "phase": phase, "points": vals[cat],
                                       "team": team, "kind": cat})
        # Algae launches near the barge are direct evidence of NET shots.
        for L in ctx.launches.get("algae", []):
            if not getattr(L, "rising", True):
                continue
            slot, _ = self.attribute_launch(ctx, L)
            if not slot:
                continue
            a, team = slot[:-1], ctx.team_at(slot, L.t0)
            r = ctx.robot_at(slot, L.t0)
            if r and self.field.dist_to_element(r[:2], f"{a}_net_zone") < NEAR["net"]:
                phase, _ = ctx.match_time(L.t0)
                phase = "auto" if phase in ("auto", "pause") else "teleop"
                weights[phase][team]["net"] += 0.5
                events.append({"t": round(L.t0, 2), "alliance": a, "phase": phase, "team": team, "kind": "net_shot_seen"})

        # Time near each element (fallback weights) + defense time.
        near_time = defaultdict(lambda: defaultdict(float))
        opp_time = defaultdict(float)
        tot_time = defaultdict(float)
        for slot in ctx.tracks:
            tr = self._fresh(ctx, slot)
            a = slot[:-1]
            for i in range(1, len(tr)):
                team = ctx.team_at(slot, tr[i][0])
                dt = min(tr[i][0] - tr[i - 1][0], 0.5)
                phase, _ = ctx.match_time(tr[i][0])
                if phase not in ("auto", "teleop"):
                    continue
                p = (tr[i][1], tr[i][2])
                tot_time[team] += dt
                if self.field.dist_to_element(p, f"{a}_reef") < NEAR["reef"]:
                    near_time[team]["reef"] += dt
                if self.field.dist_to_element(p, f"{a}_net_zone") < NEAR["net"]:
                    near_time[team]["net"] += dt
                if self.field.dist_to_element(p, f"{a}_processor") < NEAR["processor"]:
                    near_time[team]["processor"] += dt
                own_side = p[0] < self.field.length / 2 if a == "blue" else p[0] > self.field.length / 2
                if not own_side:
                    opp_time[team] += dt

        out_teams, alliances = {}, {}
        for a in ("red", "blue"):
            counts = self.official_counts(bd[a]) if bd else None
            split = {ph: {} for ph in ("auto", "teleop")}
            for ph in ("auto", "teleop"):
                for cat in CATS:
                    official = counts[ph][cat] if counts else sum(weights[ph][t][cat] for t in teams[a])
                    w = {t: weights[ph][t][cat] for t in teams[a]}
                    if ph == "auto" and AUTO_SHRINK > 0 and sum(w.values()) > 0:
                        # in AUTO nearly every robot scores its preloaded coral: the video only tilts the split
                        tot = sum(w.values())
                        w = {t: v / tot + AUTO_SHRINK for t, v in w.items()}
                    elem = "reef" if cat.startswith("L") else cat
                    fb = {t: near_time[t][elem] for t in teams[a]}
                    if elem != "reef" and ELEM_TIME_W > 0 and sum(fb.values()) > 0 and sum(w.values()) > 0:
                        # who hangs around the NET / PROCESSOR all match is who scores there: steadier than
                        # positions at each post (the NET side is where robots go undetected)
                        tw, tf = sum(w.values()), sum(fb.values())
                        w = {t: w[t] / tw + ELEM_TIME_W * fb[t] / tf for t in w}
                    split[ph][cat] = self.proportional_split(official, w, fb)
            explained = sum(sum(weights[ph][t][c] for t in teams[a] for c in CATS) for ph in ("auto", "teleop"))
            off_total = sum(sum(counts[ph].values()) for ph in counts) if counts else None
            alliances[a] = {"official_actions": off_total, "explained_score_events": int(explained),
                            "coverage": round(explained / off_total, 2) if off_total else None}
            for i, team in enumerate(teams[a]):
                st = i + 1
                leave = bd[a].get(f"autoLineRobot{st}") == "Yes" if bd else None
                eg = bd[a].get(f"endGameRobot{st}", "None") if bd else "Unknown"
                c = {cat: split["auto"][cat][team] + split["teleop"][cat][team] for cat in CATS}
                auto_coral = sum(split["auto"][k][team] for k in ("L1", "L2", "L3", "L4"))
                pts = (sum(split["auto"][k][team] * AUTO_VALUES[k] for k in CATS)
                       + sum(split["teleop"][k][team] * TELE_VALUES[k] for k in CATS)
                       + (3 if leave else 0) + ENDGAME_PTS.get(eg, 0))
                out_teams[team] = {
                    "alliance": a, "station": st,
                    "coral_total": sum(c[k] for k in ("L1", "L2", "L3", "L4")),
                    "coral_L4": c["L4"], "coral_L3": c["L3"], "coral_L2": c["L2"], "coral_L1": c["L1"],
                    "auto_coral": auto_coral,
                    "auto_coral_by_level": {k: split["auto"][k][team] for k in ("L1", "L2", "L3", "L4")},
                    "auto_leave": leave, "algae_net": c["net"], "algae_processor": c["processor"],
                    "endgame": eg,
                    "pct_defense": round(100 * opp_time[team] / tot_time[team]) if tot_time[team] else 0,
                    "points_contributed": int(pts),
                }
        return {"teams": out_teams, "alliances": alliances, "events": events}

    def identity_evidence(self, ctx: MatchContext, evidence) -> None:
        """BARGE results and LEAVE per station vs. observed slot behaviour."""
        bd = ctx.match.breakdown
        if not bd or ctx.clock is None:
            return
        end = ctx.clock.teleop_end_vt
        cage, barge = {}, {}
        for slot in ctx.tracks:
            P = slot_positions_between(ctx, slot, end - 4, end + 1)
            if len(P):
                dx = np.abs(P[:, 0] - self.field.length / 2).min()
                cage[slot] = dx < 0.9
                barge[slot] = dx < 1.6
        eg = {a: [bd[a].get(f"endGameRobot{i}", "None") for i in (1, 2, 3)] for a in ("red", "blue")}
        add_station_evidence(ctx, evidence, {a: [v in ("DeepCage", "ShallowCage") for v in eg[a]] for a in eg},
                             cage, 1.5, "cage climb")
        add_station_evidence(ctx, evidence, {a: [v != "None" for v in eg[a]] for a in eg}, barge, 0.7, "barge")
        moved = {}
        for slot in ctx.tracks:
            P = slot_positions_between(ctx, slot, ctx.clock.start_vt, ctx.clock.auto_end_vt)
            if len(P) > 3:
                moved[slot] = float(np.linalg.norm(P[-1] - P[0])) > 0.6
        add_station_evidence(ctx, evidence, {a: [bd[a].get(f"autoLineRobot{i}") == "Yes" for i in (1, 2, 3)]
                                             for a in ("red", "blue")}, moved, 1.0, "leave")

    def _evidence(self, ctx: MatchContext, a: str, teams: list[int], t: float, pre: float = LOOKBACK, post: float = 0.0):
        """Per team: closest approach to, and time spent at, each scoring element in [t - pre, t + post]
        around a score posted at t (the reef close-up camera puts the teams it reads at the reef)."""
        near, dwell = {}, {}
        for team in teams:
            slot = ctx.slot_of(team, t)
            if slot:
                near[team] = self._elem_dists(ctx, slot, t - pre, t + post, a)
                dwell[team] = self._dwell(ctx, slot, t - pre, t + post, a)
            else:
                near[team] = {"reef": 9e9, "net": 9e9, "processor": 9e9}
                dwell[team] = {"reef": 0.0, "net": 0.0, "processor": 0.0}
        # Seen there = at the reef. Not seen proves nothing (bumper may face away from that camera),
        # so unseen teams keep their wide-shot evidence.
        for team, c in ctx.seen_in_view(f"{a}_reef", t - 3.5, t + 0.5).items():
            if team in near and c >= 0.5:
                near[team]["reef"] = min(near[team]["reef"], 0.3)
        # Repeated bumper reads are direct observations even when the floor mapping is
        # unreliable (a hidden bumper puts the detected box bottom above the carpet).
        # Count only intervals bracketed by reads, never extrapolate an isolated read,
        # and take the better view's dwell rather than counting the same time twice.
        sightings = defaultdict(set)
        for bt, view, *box in ctx.aux_boxes:
            team, conf = box[-2:]
            if view == f"{a}_reef" and team in near and conf >= 0.5:
                region = (getattr(ctx.cal, "aux_watch", None) or {}).get(view)
                if region and (box[2] < region[0] or box[0] > region[2]):
                    continue  # a readable robot at the loading wall is not reef dwell
                sightings[team].add(bt)
        for team, times in sightings.items():
            times = sorted(times)
            seconds = sum(max(0.0, min(right, t + post) - max(left, t - pre))
                          for left, right in zip(times, times[1:]) if 0 < right - left <= 0.75)
            dwell[team]["reef"] = max(dwell[team]["reef"], seconds)
        return near, dwell

    def _credit(self, ctx: MatchContext, a: str, teams: list[int], t: float, cat: str) -> tuple[dict[int, float], float]:
        """How likely each team made one decoded action posted at t (sums to 1; empty = no idea), and
        how much the video says at all (0..1: nobody near the element = little).

        The live score posts about a second after a piece is placed and the robot usually is still at
        the element then, so time at the element is counted in a short window around the post. Teams
        the reef close-up camera reads in that window are at the reef: their share is boosted."""
        elem = "reef" if cat in CORAL else cat
        if elem != "reef" and ALGAE_CREDIT == "near":
            # algae is shot / thrown from a little way off: closeness over the lookback, as before, plus
            # algae seen flying from a robot just before the post
            near, _ = self._evidence(ctx, a, teams, t)
            w = {tm: np.exp(-d[elem] / SOFT_SCALE) for tm, d in near.items() if d[elem] <= NEAR[elem] + 1.0}
            # The NET is on the far side, where robots often go undetected: a robot the tracker can't see
            # right now is a real candidate, one clearly seen elsewhere is not.
            for tm in teams:
                if UNSEEN_W > 0 and not self._seen(ctx, tm, t - 3.0, t + 0.5):
                    w[tm] = max(w.get(tm, 0.0), UNSEEN_W)
            for L in ctx.launches.get("algae", []):
                if getattr(L, "rising", True) and t - LOOKBACK <= L.t0 <= t + 0.5:
                    slot, _ = self.attribute_launch(ctx, L)
                    tm = ctx.team_at(slot, L.t0) if slot else None
                    if tm in teams:
                        w[tm] = w.get(tm, 0.0) + LAUNCH_W
            tot = sum(w.values())
            return ({tm: v / tot for tm, v in w.items() if v > 0} if tot > 0 else {}), min(1.0, tot / CONF_REF[elem])
        near, dwell = self._evidence(ctx, a, teams, t, CREDIT_PRE, CREDIT_POST)
        w = {tm: d[elem] for tm, d in dwell.items()}
        # The reef close-up names the robots at that reef. In AUTO the robots there are the ones scoring; in
        # TELEOP it also sees robots waiting or driving past, so it helps only in AUTO (measured vs OPR).
        boost = CAM_BOOST_AUTO if ctx.clock and t <= ctx.clock.auto_end_vt + 1.0 else CAM_BOOST
        if elem == "reef" and boost > 0:
            for tm, c in ctx.seen_in_view(f"{a}_reef", t - CREDIT_PRE, t + CREDIT_POST).items():
                if tm in w:
                    w[tm] = (w[tm] + 0.3) * (1 + boost * min(c, 2.0))
        if cat in ("L2", "L3", "L4") and getattr(ctx, "lift_obs", None) and REACH_BOOST > 0:
            # who actually reached up (elevator / arm raised out of the chassis, vision/lift.py) right before
            # the score: a robot at the reef that lifted beats one that only stood there
            reaches = {tm: self._reach(ctx, tm, t - REACH_PRE, t + REACH_POST) for tm in w}
            # No pixels for a plausible scorer means unknown reach, not a lowered
            # mechanism. Comparing only the visible robots rewards detection gaps.
            comparable = all(reaches[tm] is not None for tm in w if w[tm] > 0)
            for tm in w:
                reach = reaches[tm] if comparable else None
                if reach is None:
                    continue
                ev = min(max(reach - REACH_MIN, 0.0), 1.0)
                at_reef = near[tm]["reef"] <= NEAR["reef"] + DWELL_SLACK
                w[tm] = (w[tm] + (REACH_ADD * ev if at_reef else 0.0)) * (1 + REACH_BOOST * ev)
        elif cat in ("L2", "L3", "L4") and LIFT_BOOST > 0:
            for tm in w:
                lift = self._lift(ctx, tm, t - CREDIT_PRE, t + CREDIT_POST)
                w[tm] *= 1 + LIFT_BOOST * min(max(lift - 1.15, 0.0), 1.0)
        conf = min(1.0, sum(w.values()) / CONF_REF[elem])
        if sum(w.values()) <= 0:  # nobody stayed there: closeness instead
            w = {tm: np.exp(-d[elem] / SOFT_SCALE) for tm, d in near.items() if d[elem] <= NEAR[elem] + 1.0}
            conf = 0.3 * min(1.0, sum(w.values()))
        tot = sum(w.values())
        return ({tm: v / tot for tm, v in w.items() if v > 0} if tot > 0 else {}), conf

    def _seen(self, ctx: MatchContext, team: int, t0: float, t1: float) -> bool:
        """Was this team's robot actually detected (not just coasted) in [t0, t1]?"""
        slot = ctx.slot_of(team, (t0 + t1) / 2)
        return bool(slot) and any(t0 <= r[0] <= t1 and r[3] for r in ctx.tracks.get(slot, []))

    @staticmethod
    def _reach(ctx: MatchContext, team: int, t0: float, t1: float) -> float | None:
        """How much a robot lifted its mechanism in [t0, t1], over its own usual reach this match, from the
        wide shot and the reef close-ups (vision/lift.py; the larger of the two); None if never measured."""
        from ..vision.lift import lift_score

        cache = ctx.__dict__.setdefault("_reach_cache", {})
        if "cam" not in cache:  # close-up samples per team: read bumper, or the track fusion matched it to
            slots = getattr(ctx, "aux_slot", {})
            cam: dict = {}
            for i, rv in enumerate(getattr(ctx, "aux_reach", []) or []):
                if rv is None:
                    continue
                b = ctx.aux_boxes[i]
                tm = b[7] if b[7] is not None else (ctx.team_at(slots[i], b[0]) if i in slots else None)
                if tm is not None:
                    cam.setdefault(tm, []).append((b[0], rv))
            cache["cam"] = {k: sorted(v) for k, v in cam.items()}
        scores = []
        slot = ctx.slot_of(team, (t0 + t1) / 2)
        for key, samples in ((("wide", slot), ctx.lift_obs.get(slot, []) if slot else []),
                             (("cam", team), cache["cam"].get(team, []))):
            if key not in cache:
                cache[key] = float(np.median([r for _, r in samples])) if len(samples) >= 10 else 0.0
            # Motion behind a parked robot can persist for an entire reef visit.
            # A lift must exceed that recent background as well as its match norm.
            before = [r for t, r in samples if t0 - 3.0 <= t < t0]
            base = max(cache[key], float(np.median(before))) if len(before) >= 3 else cache[key]
            v = lift_score(samples, t0, t1, base)
            if v is not None:
                scores.append(v)
        return max(scores) if scores else None

    def _lift(self, ctx: MatchContext, team: int, t0: float, t1: float) -> float:
        """How far a robot raised its elevator in [t0, t1]: its tallest box (height / width) then over
        its usual shape this match. Scoring L2-L4 means lifting the mechanism; driving doesn't."""
        cache = ctx.__dict__.setdefault("_lift_base", {})
        slot = ctx.slot_of(team, (t0 + t1) / 2)
        if not slot:
            return 1.0
        rows = [r for r in ctx.tracks.get(slot, []) if r[3] and r[4] is not None]
        if slot not in cache:
            ar = [(b[3] - b[1]) / max(b[2] - b[0], 1) for *_, b in rows]
            cache[slot] = float(np.median(ar)) if len(ar) > 20 else None
        base = cache[slot]
        win = [(b[3] - b[1]) / max(b[2] - b[0], 1) for r in rows if t0 <= r[0] <= t1 for b in [r[4]]]
        if not base or len(win) < 3:
            return 1.0
        return float(np.percentile(win, 90)) / base

    def _placement_credit(self, ctx: MatchContext, a: str, teams: list[int], pl: dict) -> dict[int, float]:
        """Which robot put a coral seen appearing on the reef there: the alliance robot whose image box
        is over that spot (horizontally) when it appeared, and that is at the reef."""
        w = {}
        for team in teams:
            slot = ctx.slot_of(team, pl["t"])
            best = 0.0
            for dt in (-1.0, -0.5, 0.0, 0.5):
                rb = self.robot_box(ctx, slot, pl["t"] + dt) if slot else None
                if rb is None:
                    continue
                (x0, _y0, x1, _y1), obs = rb
                dx = max(x0 - pl["x"], 0, pl["x"] - x1)
                r = ctx.robot_at(slot, pl["t"] + dt)
                at = r is not None and self.field.dist_to_element(r[:2], f"{a}_reef") < NEAR["reef"] + DWELL_SLACK
                # a coasting (guessed) position must not claim a placement only a seen robot could have made
                best = max(best, np.exp(-dx / PLACE_PX) * (1.0 if at else 0.25) * (1.0 if obs else 0.15))
            w[team] = best
        if max(w.values(), default=0.0) < PLACE_MIN:
            return {}  # no robot we can see is at that spot: the placement doesn't say who
        tot = sum(w.values())
        return {tm: v / tot for tm, v in w.items() if v > 0} if tot > 0 else {}

    def _cam_placement_credit(self, ctx: MatchContext, a: str, teams: list[int], pl: dict) -> dict[int, float]:
        """Which robot put a coral seen appearing on the reef in the close-up camera there: the robot box
        in the same close-up under that spot around that moment, named by its bumper read or by the
        wide-shot track it was fused with (vision/fusion.py)."""
        view = pl["elem"].split("@")[0]
        slots = getattr(ctx, "aux_slot", {})
        w: dict[int, float] = {}
        for i, (bt, v, x0, y0, x1, y1, al, team, conf) in enumerate(ctx.aux_boxes):
            if v != view or abs(bt - pl["t"]) > 1.0 or (al and al != a):
                continue
            tm = team if team in teams else (ctx.team_at(slots[i], bt) if i in slots else None)
            if tm not in teams:
                continue
            dx = max(x0 - pl["x"], 0, pl["x"] - x1)
            w[tm] = w.get(tm, 0.0) + float(np.exp(-dx / CAM_PLACE_PX)) * (1.0 - abs(bt - pl["t"]) / 1.5)
        if max(w.values(), default=0.0) < PLACE_MIN:
            return {}  # the robot under that spot wasn't boxed in the close-up (often cut off at its edge)
        tot = sum(w.values())
        return {tm: v / tot for tm, v in w.items() if v > 0} if tot > 0 else {}

    def _pair(self, actions: list[dict], idx: list[int], pls: list[dict]) -> list[tuple[int, dict]]:
        """One-to-one pairs (action index, placement) with the placement up to PLACE_WINDOW before the post."""
        from scipy.optimize import linear_sum_assignment

        if not pls or not idx:
            return []
        big = 1e6
        cost = np.full((len(idx), len(pls)), big)
        for r, i in enumerate(idx):
            for c, p in enumerate(pls):
                lag = actions[i]["t"] - p["t"]
                if PLACE_WINDOW[0] <= lag <= PLACE_WINDOW[1]:
                    cost[r, c] = abs(lag - PLACE_LAG)
        rows, cols = linear_sum_assignment(cost)
        return [(idx[r], pls[c]) for r, c in zip(rows, cols) if cost[r, c] < big]

    def _match_placements(self, ctx: MatchContext, a: str, actions: list[dict]) -> None:
        """Pair decoded coral actions with coral seen placed on the alliance's reef, in the wide shot and in
        the reef close-up (each placement used once), and fold in who placed it: both views' answers are
        averaged when both saw it."""
        if PLACE_MIX <= 0:
            return
        idx = [i for i, A in enumerate(actions) if A["piece"] == "coral"]
        teams = list(ctx.match.teams(a))
        found: dict[int, list] = {}
        for i, pl in self._pair(actions, idx, [p for p in ctx.element_changes if p["elem"] == f"{a}_reef"]):
            pc = self._placement_credit(ctx, a, teams, pl)
            if pc:
                found.setdefault(i, []).append((pl, pc, 1.0))
        if CAM_PLACE_W > 0:
            for i, pl in self._pair(actions, idx, [p for p in ctx.element_changes if p["elem"] == f"{a}_reef@cam"]):
                pc = self._cam_placement_credit(ctx, a, teams, pl)
                if pc:
                    found.setdefault(i, []).append((pl, pc, CAM_PLACE_W))
        for i, got in found.items():
            A = actions[i]
            wsum = sum(k for _, _, k in got)
            pc = {tm: sum(k * c.get(tm, 0) for _, c, k in got) / wsum for tm in teams}
            A["placed"] = got[0][0]
            A["placed_views"] = [p["elem"] for p, _, _ in got]
            A["conf"] = 1.0
            A["t_place"] = min(p["t"] for p, _, _ in got)
            base = A["p"] or {tm: 1 / 3 for tm in teams}
            A["p"] = {tm: (1 - PLACE_MIX) * base.get(tm, 0) + PLACE_MIX * pc.get(tm, 0) for tm in teams}

    @staticmethod
    def assign_actions(actions: list[dict], teams: list[int], prior: dict | None = None) -> list[dict[int, float]]:
        """Credit robot actions of one alliance jointly.

        actions: [{"t", "piece" ("coral"/"algae"), "cat", "p": {team: likelihood}}, optional "t_place"].
        A robot carries one piece of a kind at a time, so it can't make two of them within REFRACTORY
        seconds: when two quick coral go on while two robots are at the reef, each robot most likely
        made one. The actions of each piece kind form a chain in time; crediting one robot with
        consecutive pieces closer than REFRACTORY costs a factor exp(-REFRACTORY_COST) (soft: the live
        score sometimes posts a batch), and forward-backward gives each action's credit. prior:
        {team: {cat: weight}} multiplies the likelihoods (event-wide tendencies). Returns per action
        {team: weight} summing to 1.
        """
        if not actions:
            return []
        teams = list(teams)
        nt = len(teams)
        lik = np.zeros((len(actions), nt))
        for i, A in enumerate(actions):
            p = np.array([A["p"].get(tm, 0.0) for tm in teams], float)
            p = p / p.sum() if p.sum() > 0 else np.full(nt, 1 / nt)
            if prior:
                p = p * np.array([prior.get(tm, {}).get(A["cat"], 1.0) for tm in teams])
                p = p / p.sum() if p.sum() > 0 else np.full(nt, 1 / nt)
            lik[i] = 0.98 * p + 0.02 / nt  # nothing is ever certain
        post = lik.copy()
        tm_of = lambda A: A.get("t_place", A["t"])  # noqa: E731
        same = np.exp(-REFRACTORY_COST)
        for piece in ("coral", "algae"):
            idx = sorted((i for i, A in enumerate(actions) if A["piece"] == piece), key=lambda i: tm_of(actions[i]))
            if len(idx) < 2:
                continue
            T = [np.where(np.eye(nt) > 0, same if tm_of(actions[j]) - tm_of(actions[i]) < REFRACTORY else 1.0, 1.0)
                 for i, j in zip(idx, idx[1:])]
            fw = [lik[idx[0]] / lik[idx[0]].sum()]
            for k in range(1, len(idx)):
                f = (fw[-1] @ T[k - 1]) * lik[idx[k]]
                fw.append(f / f.sum())
            bw = [np.ones(nt)]
            for k in range(len(idx) - 2, -1, -1):
                b = T[k] @ (lik[idx[k + 1]] * bw[0])
                bw.insert(0, b / b.sum())
            for k, i in enumerate(idx):
                m = fw[k] * bw[k]
                post[i] = m / m.sum()
        return [{tm: float(post[i, k]) for k, tm in enumerate(teams)} for i in range(len(actions))]

    def _decode_alliance(self, ctx: MatchContext, a: str, teams: list[int], weights, events) -> None:
        """Decode an alliance's score jumps jointly against the official counts, then credit each
        robot action to the robots at its element (see decode_jumps and assign_actions)."""
        clk = ctx.clock
        bd = (ctx.match.breakdown or {}).get(a)
        need, foul_pts = self.official_tokens(bd, self.official_counts(bd)) if bd else (None, 0)
        jumps, info = [], []
        for t, d in ctx.timeline.deltas(a):
            if not clk.start_vt - 1 < t < clk.teleop_end_vt + 3:
                continue
            allowed = {"net": 4, "processor": 6, "foul2": 2, "foul6": 6}
            if t < clk.auto_end_vt + 4:
                allowed.update({f"auto_{c}": AUTO_VALUES[c] for c in CORAL})
            if t > clk.auto_end_vt - 1:
                allowed.update({f"teleop_{c}": TELE_VALUES[c] for c in CORAL})
            if t < clk.auto_end_vt + 6:
                allowed["leave"] = 3
            if t > clk.teleop_end_vt - 3:
                allowed.update(BARGE_TOKENS)
            near, dwell = self._evidence(ctx, a, teams, t)
            score = {}
            for tok in allowed:
                cat = tok.split("_")[-1]
                if cat in CATS:
                    elem = "reef" if cat.startswith("L") else cat
                    best = min((nr[elem] for nr in near.values()), default=9e9)
                    score[tok] = 1.0 - best / (NEAR[elem] + 1.0) if best <= NEAR[elem] + 1.0 else -3.0
                else:
                    score[tok] = {"leave": 0.6, "foul2": -0.2, "foul6": -0.2}.get(tok, 0.3)
            jumps.append({"points": d, "allowed": allowed, "score": score, "t": t})
            info.append((t, d))
        # The broadcast's algae counter (NET + PROCESSOR) says how many of each jump's actions were algae.
        cdel = ctx.timeline.counter_deltas(f"{a}_algae") if ALGAE_COUNTER else []
        if f"{a}_algae" in getattr(ctx.timeline, "counters", {}) and ALGAE_COUNTER:
            for J in jumps:
                J["algae"] = 0
            for tc, n in cdel:
                near_j = [J for J in jumps if abs(J["t"] - tc) <= 1.0]
                if near_j:
                    min(near_j, key=lambda J: abs(J["t"] - tc))["algae"] += n
        actions = []
        for (t, d), combo in zip(info, self.decode_jumps(jumps, need, foul_pts)):
            phase = "auto" if t <= clk.auto_end_vt + 0.5 else "teleop"
            if combo is None:
                events.append({"t": round(t, 2), "alliance": a, "phase": phase, "points": d, "kind": "unexplained"})
                continue
            for tok in combo:
                cat = tok.split("_")[-1]
                if cat not in CATS:
                    events.append({"t": round(t, 2), "alliance": a, "phase": phase, "points": d, "kind": tok})
                    continue
                ph = tok.split("_")[0] if cat in CORAL else "teleop"
                p, conf = self._credit(ctx, a, teams, t, cat)
                actions.append({"t": t, "cat": cat, "phase": ph, "piece": "coral" if cat in CORAL else "algae",
                                "p": p, "conf": conf, "jump": d})
        self._match_placements(ctx, a, actions)
        # Vision first: coral seen going onto the reef with no score jump to pair it with (missed overlay
        # read, batched post) still says which robot scored, and its height roughly which level.
        used = {(A["placed"]["elem"], A["placed"]["t"]) for A in actions if "placed" in A}
        for pl in ctx.element_changes:
            if pl["elem"] != f"{a}_reef" or (pl["elem"], pl["t"]) in used or UNMATCHED_W <= 0:
                continue
            if not clk.start_vt <= pl["t"] <= clk.teleop_end_vt:
                continue
            pc = self._placement_credit(ctx, a, teams, pl)
            lvl = next(k for k, lo in LEVEL_HEIGHTS if pl["h"] >= lo)
            ph = "auto" if pl["t"] <= clk.auto_end_vt else "teleop"
            for who, share in pc.items():
                weights[ph][who][lvl] += UNMATCHED_W * share
            if pc:
                events.append({"t": round(pl["t"], 2), "alliance": a, "phase": ph, "kind": "coral_seen",
                               "level_guess": lvl, "team": max(pc, key=pc.get),
                               "shares": {str(k): round(v, 2) for k, v in pc.items() if v >= 0.01}, "placed": pl})
        if JOINT_CREDIT:
            credit = self.assign_actions(actions, teams, (self.prior or {}).get(a))
        else:
            credit = [A["p"] for A in actions]
        for A, cr in zip(actions, credit):
            # actions the video says little about weigh little in splitting the official counts
            k = A["conf"] if CONF_WEIGHT else 1.0
            for who, share in cr.items():
                weights[A["phase"]][who][A["cat"]] += share * max(k, 0.05)
            vals = AUTO_VALUES if A["phase"] == "auto" else TELE_VALUES
            events.append({"t": round(A["t"], 2), "alliance": a, "phase": A["phase"], "points": vals[A["cat"]],
                           "jump": A["jump"], "team": max(cr, key=cr.get) if cr else None, "kind": A["cat"],
                           "shares": {str(k): round(v, 2) for k, v in cr.items() if v >= 0.01},
                           "likelihood": {str(k): round(v, 2) for k, v in A["p"].items() if v >= 0.01},
                           **({"placed": A["placed"], "placed_views": A["placed_views"]} if "placed" in A else {})})

    @staticmethod
    def official_tokens(bd_a: dict, counts: dict) -> tuple[dict[str, int], int]:
        """Exact number of each decodable action an alliance's score jumps must add up to, and
        the foul points it was awarded (known as points only)."""
        need = {f"auto_{c}": counts["auto"][c] for c in CORAL}
        need.update({f"teleop_{c}": counts["teleop"][c] for c in CORAL})
        need["net"], need["processor"] = counts["teleop"]["net"], counts["teleop"]["processor"]
        need["leave"] = sum(bd_a.get(f"autoLineRobot{i}") == "Yes" for i in (1, 2, 3))
        eg = [bd_a.get(f"endGameRobot{i}", "None") for i in (1, 2, 3)]
        need["park"], need["shallow"], need["deep"] = (eg.count("Parked"), eg.count("ShallowCage"),
                                                       eg.count("DeepCage"))
        return need, int(bd_a.get("foulPoints", 0) or 0)

    @staticmethod
    def decode_jumps(jumps: list[dict], need: dict[str, int] | None, foul_pts: int) -> list[list[str] | None]:
        """Decode every score jump of one alliance into the actions behind it, jointly.

        jumps: [{"points", "allowed": {token: value}, "score": {token: evidence}}], where tokens are
        "auto_L4", "teleop_L2", "net", "processor", "leave", "foul2", "foul6", "park", "shallow",
        "deep" and evidence says how well the robots' positions fit that action. Decoding each
        jump alone confuses equal values (teleop L3 = NET = 4, teleop L2 = LEAVE = 3); here the
        decoded actions must also add up to the official per-category counts, so a +4 that a robot
        at the barge explains as well as one at the reef goes to whichever the official count
        still lacks. Returns per jump the chosen tokens (None = left unexplained).
        """
        from scipy.optimize import Bounds, LinearConstraint, milp

        opts, cols = [], []  # cols: (jump index, tokens, objective)
        for j, J in enumerate(jumps):
            for combo in _decompose(J["points"], J["allowed"]):
                s = sum(J["score"][k] for k in combo) - 1.2 * (len(combo) - 1)
                if J.get("algae") is not None:  # the overlay's algae counter moved by this much
                    s -= ALGAE_MISMATCH * abs(sum(k in ("net", "processor") for k in combo) - J["algae"])
                cols.append((j, combo, s))
            cols.append((j, None, NOTHING))
        if not cols:
            return []
        cats = sorted(need) if need else []
        n_x, n_c = len(cols), len(cats)
        # variables: x (one per option), then under/over slack per category, then foul under/over
        n = n_x + 2 * n_c + 2
        c = np.zeros(n)
        c[:n_x] = [-s for _, _, s in cols]
        c[n_x: n_x + 2 * n_c] = COUNT_SLACK
        c[n_x + 2 * n_c:] = FOUL_SLACK
        A, lo, hi = [], [], []
        for j in range(len(jumps)):
            row = np.zeros(n)
            row[[i for i, (jj, _, _) in enumerate(cols) if jj == j]] = 1
            A.append(row), lo.append(1), hi.append(1)
        for ci, cat in enumerate(cats):
            row = np.zeros(n)
            for i, (_, combo, _) in enumerate(cols):
                row[i] = combo.count(cat) if combo else 0
            row[n_x + 2 * ci], row[n_x + 2 * ci + 1] = 1, -1
            A.append(row), lo.append(need[cat]), hi.append(need[cat])
        if need is not None:
            row = np.zeros(n)
            for i, (_, combo, _) in enumerate(cols):
                row[i] = sum({"foul2": 2, "foul6": 6}.get(k, 0) for k in combo) if combo else 0
            row[n - 2], row[n - 1] = 1, -1
            A.append(row), lo.append(foul_pts), hi.append(foul_pts)
        integ = np.r_[np.ones(n_x), np.zeros(n - n_x)]
        res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ,
                   bounds=Bounds(np.zeros(n), np.r_[np.ones(n_x), np.full(n - n_x, np.inf)]),
                   options={"time_limit": 10})
        out: list[list[str] | None] = [None] * len(jumps)
        if res.x is None:
            return out
        for i, (j, combo, _) in enumerate(cols):
            if res.x[i] > 0.5:
                out[j] = combo
        return out

    @staticmethod
    def _shares(near: dict[int, dict[str, float]], team: int, cat: str, dwell=None) -> dict[int, float]:
        """How one explained action is credited. SHARE_MODE "nearest" gives it all to the robot
        nearest the element; "soft" shares it among the robots at the element by closeness,
        so a position error of a few tenths of a metre doesn't flip the whole action; "dwell"
        shares it by how long each robot stayed at the element before the score posted (scoring
        a piece takes a stop there; driving past doesn't)."""
        if SHARE_MODE == "nearest":
            return {team: 1.0}
        elem = "reef" if cat.startswith("L") else cat
        if (SHARE_MODE == "dwell" or (SHARE_MODE == "hybrid" and cat.startswith("L"))) and dwell:
            w = {t: d[elem] for t, d in dwell.items() if d[elem] > 0}
            tot = sum(w.values())
            if tot > 0:
                return {t: v / tot for t, v in w.items()}
        w = {t: np.exp(-d[elem] / SOFT_SCALE) for t, d in near.items() if d[elem] <= NEAR[elem] + 1.0}
        tot = sum(w.values())
        return {t: v / tot for t, v in w.items()} if tot > 0 else {team: 1.0}

    @staticmethod
    def _explain(points: int, vals: dict[str, int], near: dict[int, dict[str, float]], extra: dict[str, int] | None = None):
        """Pick the most plausible (team, category) actions behind a score increase.

        Score jumps also come from things no robot scoring action explains (LEAVE points posted
        at the end of AUTO, opponent fouls, human-player NET shots); those are allowed as
        explanations with a neutral score, so they absorb jumps no robot was positioned for.
        """
        allowed = dict(vals, **(extra or {}))
        best, best_s = None, -1e9
        for combo in _decompose(points, allowed):
            assigned, score = [], 0.0
            for cat in combo:
                if cat in (extra or {}):
                    score += 0.6 if cat == "leave" else -0.2
                    continue
                elem = "reef" if cat.startswith("L") else cat
                cands = sorted(((d[elem], t) for t, d in near.items()), key=lambda x: x[0])
                if not cands or cands[0][0] > NEAR[elem] + 1.0:
                    score -= 3.0  # nobody there: unlikely (or a foul / human player)
                    assigned.append((None, cat))
                    continue
                dist, team = cands[0]
                score += 1.0 - dist / (NEAR[elem] + 1.0)
                assigned.append((team, cat))
            score -= 1.2 * (len(combo) - 1)  # prefer simple explanations (> max per-action score)
            if score > best_s:
                best, best_s = assigned, score
        if best is None or best_s < -1.0:
            return None
        return [(t, c) for t, c in best if t is not None] or None

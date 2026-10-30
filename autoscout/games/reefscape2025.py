"""2025 REEFSCAPE.

Per-robot attribution: every increase of the live overlay score is matched to the scoring
element a robot of that alliance was working in the seconds before (REEF, NET side of the
BARGE, PROCESSOR); the point value of the increase identifies the action (e.g. teleop +5 at
the REEF = L4 CORAL). Exact alliance totals per category come from the official breakdown and
are split among robots in proportion to these attributed events. LEAVE and BARGE results are
reported per station in the breakdown, so they are exact per team.
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
    launch_pieces = {"algae": ((78, 60, 80), (100, 255, 255))}
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

    def _elem_dists(self, ctx: MatchContext, slot: str, t0: float, t1: float, alliance: str):
        """Min distance of a slot to each of its alliance's scoring elements during [t0, t1]."""
        best = {"reef": 9e9, "net": 9e9, "processor": 9e9}
        tr = ctx.tracks.get(slot, [])
        for r in tr:
            if t0 <= r[0] <= t1:
                p = (r[1], r[2])
                best["reef"] = min(best["reef"], self.field.dist_to_element(p, f"{alliance}_reef"))
                best["net"] = min(best["net"], self.field.dist_to_element(p, f"{alliance}_net_zone"))
                best["processor"] = min(best["processor"], self.field.dist_to_element(p, f"{alliance}_processor"))
        return best

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
                    near = {}
                    for team in teams[a]:
                        slot = ctx.slot_of(team, t)
                        if slot:
                            near[team] = self._elem_dists(ctx, slot, t - LOOKBACK, t, a)
                    ev = self._explain(d, vals, near, extra)
                    if ev is None:
                        events.append({"t": round(t, 2), "alliance": a, "phase": phase, "points": d, "kind": "unexplained"})
                        continue
                    for team, cat in ev:
                        weights[phase][team][cat] += 1
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
        for slot, tr in ctx.tracks.items():
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
                    elem = "reef" if cat.startswith("L") else cat
                    fb = {t: near_time[t][elem] for t in teams[a]}
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

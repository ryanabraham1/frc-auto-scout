"""Score FUEL credit against human scouting reports (the scouting app's match_scouting_report rows).

A scout watches one robot for a whole match and logs every volley, so a report says how much FUEL that
robot scored in AUTO and in TELEOP. Scouts' counts are rough (two scouts on one robot differ by about
45%), but which robot of an alliance did the shooting is what they are good at. So only alliances with
all three robots scouted are used, each report is turned into that robot's SHARE of its alliance, and
the share is applied to the official FMS count:

    truth(robot, phase) = official alliance count (phase) * scouted(robot) / scouted(alliance)

"credited right" is the share of official FUEL that went to the robot the scouts say scored it,
1 - sum|ours - truth| / (2 * official). An equal three-way split is the yardstick to beat.
"phantom" is FUEL given to robots whose scout logged no scoring at all in that phase.

The reports are not in the repo (data/labels/scouting_reports_2026.json, git-ignored).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPORTS = Path("data/labels/scouting_reports_2026.json")
PHASES = ("auto", "teleop")


def load_truth(path: Path = REPORTS) -> dict:
    """{(match_key, alliance): {team: {'auto': n, 'teleop': n}}} for alliances with all three robots scouted
    (duplicate reports of one robot are averaged)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in json.loads(path.read_text()):
        if r.get("no_show"):
            pass  # a no-show scored nothing: its zeros are the truth
        by[(r["match_key"], r["alliance_color"])][int(r["target_team_number"])].append(
            {"auto": r["auto_fuel"] or 0, "teleop": (r["teleop_fuel_active"] or 0) + (r["endgame_fuel"] or 0)})
    out = {}
    for key, teams in by.items():
        if len(teams) == 3:
            out[key] = {t: {p: float(np.mean([x[p] for x in reps])) for p in PHASES} for t, reps in teams.items()}
    return out


def official_counts(match_key: str) -> dict | None:
    """{alliance: {'auto': n, 'teleop': n, 'teams': [...]}} from the cached TBA match."""
    from autoscout.tba import TBA, MatchInfo

    m = MatchInfo.from_tba(TBA().match(match_key))
    if not m.breakdown:
        return None
    return {a: {"auto": m.breakdown[a]["hubScore"]["autoCount"], "teleop": m.breakdown[a]["hubScore"]["teleopCount"],
                "teams": list(m.teams(a))} for a in ("red", "blue")}


def score(credit: dict, truth: dict | None = None, min_fuel: int = 5, detail: bool = False) -> dict:
    """credit: {match_key: {team: {'auto': n, 'teleop': n}}} -> per-phase accuracy against the scouts.
    Alliances the scouts saw score fewer than min_fuel in a phase carry no share and are skipped."""
    truth = truth if truth is not None else load_truth()
    tot = {p: defaultdict(float) for p in PHASES}
    rows = []
    for (mk, a), scouted in sorted(truth.items()):
        if mk not in credit:
            continue
        off = official_counts(mk)
        if off is None or set(off[a]["teams"]) != set(scouted):
            continue
        for p in PHASES:
            ssum = sum(scouted[t][p] for t in scouted)
            n = off[a][p]
            if ssum < min_fuel or n < min_fuel:
                continue
            T = tot[p]
            for t in scouted:
                want = n * scouted[t][p] / ssum
                got = float(credit[mk].get(t, {}).get(p, 0.0))
                T["err"] += abs(got - want)
                T["equal_err"] += abs(n / 3 - want)
                if scouted[t][p] == 0:
                    T["phantom"] += got
                rows.append((mk, a, p, t, want, got))
            T["official"] += n
            T["alliances"] += 1
            T["credited"] += sum(float(credit[mk].get(t, {}).get(p, 0.0)) for t in scouted)
    out = {}
    for p in PHASES:
        T = tot[p]
        n = max(T["official"], 1.0)
        out[p] = {"alliances": int(T["alliances"]), "official": int(T["official"]),
                  "right": 1 - T["err"] / (2 * n), "equal_right": 1 - T["equal_err"] / (2 * n),
                  "mae": T["err"] / max(3 * T["alliances"], 1), "equal_mae": T["equal_err"] / max(3 * T["alliances"], 1),
                  "phantom": T["phantom"] / n, "credited": T["credited"] / n}
    n = sum(tot[p]["official"] for p in PHASES) or 1.0
    out["all"] = {"right": 1 - sum(tot[p]["err"] for p in PHASES) / (2 * n),
                  "equal_right": 1 - sum(tot[p]["equal_err"] for p in PHASES) / (2 * n),
                  "phantom": sum(tot[p]["phantom"] for p in PHASES) / n,
                  "official": int(n), "alliances": int(max(tot[p]["alliances"] for p in PHASES))}
    if detail:
        out["rows"] = rows
    return out


def line(name: str, s: dict) -> str:
    a, t, al = s["auto"], s["teleop"], s["all"]
    return (f"{name:28s} right: all {al['right']:.1%}  auto {a['right']:.1%}  teleop {t['right']:.1%} | "
            f"robot MAE auto {a['mae']:.1f} teleop {t['mae']:.1f} | phantom {al['phantom']:.1%}")


def saved_credit(match_key: str, results_dir: Path = Path("data/results")) -> dict:
    """{team: {'auto', 'teleop'}} as a saved result credits it."""
    r = json.loads((results_dir / f"{match_key}.json").read_text())
    return {int(t): {"auto": x["fuel_scored_auto"], "teleop": x["fuel_scored_teleop"]} for t, x in r["teams"].items()}


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, ".")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("data/results"))
    ap.add_argument("--event", action="append", help="limit to events (repeatable)")
    a = ap.parse_args()
    truth = load_truth()
    keys = sorted({mk for mk, _ in truth if (a.results / f"{mk}.json").exists()
                   and (not a.event or mk.split("_")[0] in a.event)})
    credit = {mk: saved_credit(mk, a.results) for mk in keys}
    s = score(credit, truth)
    print(f"{len(keys)} matches, {s['all']['alliances']} alliances with all three robots scouted, {s['all']['official']} FUEL")
    print(line("saved results", s))
    eq = {p: s[p]["equal_right"] for p in PHASES}
    print(f"{'equal split':28s} right: all {s['all']['equal_right']:.1%}  auto {eq['auto']:.1%}  teleop {eq['teleop']:.1%} | "
          f"robot MAE auto {s['auto']['equal_mae']:.1f} teleop {s['teleop']['equal_mae']:.1f}")

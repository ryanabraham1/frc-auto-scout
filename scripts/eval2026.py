"""Honest 2026 check: how much of the FUEL credit comes from balls actually tracked, and how well per-team
averages match fuel OPR from matches that were NOT evaluated.

Ball tracking, per alliance and scoring period (only periods whose HUB was active), against the official
FMS hub count, which says exactly how many balls went in:
  made   - balls seen dropping into the own HUB (tracked flights) / official count
  shots  - balls seen released by a robot in its own zone / official count
  shooter- share of tracked flights (shots, feeds, knocked, HP) traced to a shooter
Credit: share of scored points whose increase had one clear shooter (top share >= 0.7).

Per team: average credited FUEL vs fuel OPR of the event's other qualification matches. Half of the
matches not evaluated (alternating by number) are the OPR reference; only the rest, and the evaluated
matches, may inform the event prior (rebuilt2026.PRIOR_ALPHA, off by default).

usage: python scripts/eval2026.py EVENT [--saved] [--matches FILE.json] [--out FILE.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from autoscout import pipeline  # noqa: E402
from autoscout.evaluation import load_results  # noqa: E402
from autoscout.games.rebuilt2026 import COUNT_LAG, PERIODS, TELEOP_BOUNDS, hub_active  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, reanalyze  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402

METRICS = [("fuel_scored", "totalCount"), ("fuel_scored_auto", "autoCount"), ("fuel_scored_teleop", "teleopCount")]
CLEAR = 0.7  # top share of a score increase that counts as one clear shooter


def opr(ms, keys):
    teams = sorted({t for m in ms if m.key in keys for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    A, B = [], defaultdict(list)
    for m in ms:
        if m.key not in keys:
            continue
        for a in ("red", "blue"):
            row = np.zeros(len(teams))
            for t in m.teams(a):
                row[idx[t]] = 1
            A.append(row)
            for k, f in METRICS:
                B[k].append(m.breakdown[a]["hubScore"][f])
    A = np.array(A)
    return {k: dict(zip(teams, np.linalg.lstsq(A, np.array(v, float), rcond=None)[0])) for k, v in B.items()}


def tracking(r, m):
    """Per alliance: official, made seen, shots seen (active periods only); flights with/without shooter."""
    out = {}
    for a in ("red", "blue"):
        hs = m.breakdown[a]["hubScore"]
        aw = r.get("auto_winner")
        act = [p for p in PERIODS if hub_active(p, a, aw) is not False]
        off = sum(hs.get(f"{p}Count", 0) for p in act)
        mine = [t for t, x in r["teams"].items() if x["alliance"] == a]
        made = sum(r["teams"][t]["fuel_made_seen"] for t in mine)
        shots = sum(r["teams"][t]["fuel_shot"] for t in mine)
        per = {p: (hs.get(f"{p}Count", 0), sum(r["teams"][t]["fuel_shot_by_period"][p] for t in mine)) for p in act}
        out[a] = {"official": off, "made": made, "shots": shots, "per_period": per}
    flights = [e for e in r["events"] if e.get("kind") in ("shot", "shot_inactive", "feed_flight", "feed", "knocked",
                                                          "human_player") and e.get("by") is None]
    seen = sum(e["n"] for e in flights)
    lost = r["alliances"]["red"].get("unattributed_launches", 0.0)
    scored = defaultdict(list)
    for e in r["events"]:
        if e.get("kind") == "scored":
            scored[(e["t"], e["alliance"])].append(e["n"])
    pts = sum(sum(v) for v in scored.values())
    clear = sum(sum(v) for v in scored.values() if max(v) / sum(v) >= CLEAR)
    return out, seen, lost, pts, clear


def score_from_balls(r, m, bin_s=5.0):
    """The score with the scoreboard hidden: per alliance, every tracked flight that drops into its HUB
    while the HUB is active, against the live broadcast score. Returns (tracked, official, binned pairs)."""
    tl, clock = r["timeline"], r["clock"]
    ts = np.array(tl["t"])
    ends = (clock["auto_end_vt"], clock["teleop_end_vt"])
    t0, t1 = clock["auto_end_vt"] - clock["auto_len"], clock["teleop_end_vt"] + 3.0
    edges = np.arange(t0, t1 + bin_s, bin_s)
    out = []
    for a in ("red", "blue"):
        s = np.array(tl[a], float)
        d = np.diff(s, prepend=s[0])
        # TOWER points post in a lump at the end of AUTO and of the match (rebuilt2026.live_weights' rule)
        tower = (d >= 10) & np.array([any(-1.0 < t - e < 4.0 for e in ends) for t in ts])
        fuel_t = np.repeat(ts[(d > 0) & ~tower], d[(d > 0) & ~tower].astype(int))
        made = []
        for fl in r["launches"].get("fuel", []):
            if fl.get("target") != f"{a}_hub":
                continue
            per = period_at(clock, fl["t1"] + COUNT_LAG)
            if per and hub_active(per, a, r.get("auto_winner")) is not False:
                made += [fl["t1"] + COUNT_LAG] * int(round(fl["n"]))
        out.append((np.histogram(made, edges)[0], np.histogram(fuel_t, edges)[0]))
    return out


def period_at(clock, vt):
    """rebuilt2026.period_of from a saved clock (auto FUEL still counts for 3 s after AUTO ends)."""
    if vt < clock["auto_end_vt"] + 3:
        return "auto" if vt >= clock["auto_end_vt"] - clock["auto_len"] else None
    rem = clock["teleop_end_vt"] - vt
    if rem > clock["teleop_len"]:
        return None
    return next((p for p, hi, lo in TELEOP_BOUNDS if lo < rem <= hi), None)


def main(event, saved=False, matches=None):
    tba = TBA()
    ms = [MatchInfo.from_tba(m) for m in tba.event_matches(event) if m["comp_level"] == "qm"]
    ms = [m for m in ms if m.breakdown]
    byk = {m.key: m for m in ms}
    results = [r for r in load_results(event, RESULTS_DIR) if r["match_key"] in byk]
    if matches is not None:
        results = [r for r in results if r["match_key"] in set(matches)]
    if not results:
        raise ValueError("No qualification results to evaluate")
    done = {r["match_key"] for r in results}
    rest = sorted((m for m in ms if m.key not in done), key=lambda m: m.match_number)
    ref = {m.key for m in rest[::2]}
    if not saved:
        pipeline.PRIOR_MATCH_KEYS = {m.key for m in ms} - ref
        results = [reanalyze(r["match_key"], tba, save=False) for r in results]
    for r in results:
        r["teams"] = {str(k): v for k, v in r["teams"].items()}

    print(f"{event}: score from tracked balls alone (scoreboard hidden) vs the live score, per alliance:")
    G, S, per_match = [], [], []
    for r in sorted(results, key=lambda r: byk[r["match_key"]].match_number):
        row = []
        for a, (got, live) in zip(("red", "blue"), score_from_balls(r, byk[r["match_key"]])):
            G.append(got)
            S.append(live)
            row.append(f"{a} {got.sum():4d}/{live.sum():<4d}")
        per_match.append(f"    {r['match_key']:14s} " + "  ".join(row))
    print("\n".join(per_match))
    G, S = np.concatenate(G), np.concatenate(S)
    print(f"  total tracked {G.sum()} vs scored {S.sum()} ({G.sum() / max(S.sum(), 1):.0%}); 5 s windows: "
          f"r={np.corrcoef(G, S)[0, 1]:.3f}, |tracked - scored| {np.abs(G - S).sum() / max(S.sum(), 1):.0%} of scored")
    blind = {"ratio": float(G.sum() / max(S.sum(), 1)), "window_r": float(np.corrcoef(G, S)[0, 1]),
             "window_abs_error": float(np.abs(G - S).sum() / max(S.sum(), 1))}

    tot = defaultdict(float)
    for r in results:
        al, seen, lost, pts, clear = tracking(r, byk[r["match_key"]])
        for a in al.values():
            tot["official"] += a["official"]
            tot["made"] += a["made"]
            tot["shots"] += a["shots"]
            for p, (o, s) in a["per_period"].items():
                tot["period_abs"] += abs(s - o)
        tot["flights"] += seen
        tot["lost"] += lost
        tot["pts"] += pts
        tot["clear"] += clear
    print(f"{event}: {len(results)} matches; OPR reference {len(ref)} other matches")
    print(f"  ball tracking vs official hub counts ({tot['official']:.0f} FUEL scored in active periods):")
    print(f"    made seen   {tot['made'] / tot['official']:.0%}")
    print(f"    shots seen  {tot['shots'] / tot['official']:.0%}   per-period |shots - official| "
          f"{tot['period_abs'] / tot['official']:.0%}")
    print(f"    tracked flights with a shooter {tot['flights'] / max(tot['flights'] + tot['lost'], 1):.0%} "
          f"({tot['lost']:.0f} of {tot['flights'] + tot['lost']:.0f} pieces untraced)")
    print(f"    scored points with one clear shooter {tot['clear'] / max(tot['pts'], 1):.0%}")
    report = {"event": event, "matches": sorted(done), "reference_matches": sorted(ref), "score_from_balls": blind,
              "tracking": {"made_seen": tot["made"] / tot["official"], "shots_seen": tot["shots"] / tot["official"],
                           "period_abs_error": tot["period_abs"] / tot["official"],
                           "flights_traced": tot["flights"] / max(tot["flights"] + tot["lost"], 1),
                           "clear_points": tot["clear"] / max(tot["pts"], 1)}, "metrics": {}}

    o = opr(ms, ref)
    ours, eq = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in results:
        m = byk[r["match_key"]]
        for a in ("red", "blue"):
            for k, f in METRICS:
                for t in m.teams(a):
                    ours[k][t].append(r["teams"][str(t)][k])
                    eq[k][t].append(m.breakdown[a]["hubScore"][f] / 3)
    for k, _ in METRICS:
        teams = sorted(t for t in ours[k] if t in o[k])
        x = np.array([np.mean(ours[k][t]) for t in teams])
        e = np.array([np.mean(eq[k][t]) for t in teams])
        y = np.array([o[k][t] for t in teams])
        rr = lambda v: float(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 0.0  # noqa: E731
        report["metrics"][k] = {"r": rr(x), "equal_r": rr(e), "mae": float(np.abs(x - y).mean()),
                                "equal_mae": float(np.abs(e - y).mean()), "teams": len(teams)}
        print(f"  {k:18s} r={rr(x):.3f} (equal split {rr(e):.3f})  mae={np.abs(x - y).mean():.1f} "
              f"(equal {np.abs(e - y).mean():.1f})  teams={len(teams)}")
    k = "fuel_scored"
    top = sorted((t for t in ours[k] if t in o[k]), key=lambda t: -o[k][t])[:8]
    print("  top teams by reference fuel OPR (ours per match vs OPR):")
    for t in top:
        print(f"    {t:5d} {np.mean(ours[k][t]):6.0f} vs {o[k][t]:5.0f}   matches {[round(v) for v in ours[k][t]]}")
    report["top_teams"] = [{"team": t, "ours": float(np.mean(ours[k][t])), "opr": float(o[k][t])} for t in top]
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event")
    ap.add_argument("--saved", action="store_true", help="score the saved results instead of reanalyzing")
    ap.add_argument("--matches", type=Path, help="JSON list of qualification match keys, fixed across variants")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rep = main(a.event, a.saved, json.loads(a.matches.read_text()) if a.matches else None)
    if a.out:
        a.out.write_text(json.dumps(rep, indent=2))

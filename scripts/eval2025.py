"""Honest 2025 attribution check: reanalyze every processed match of an event (nothing saved) and score
per-team averages against OPR computed only from the event's qualification matches that were NOT
evaluated, so the reference never shares a match with what it checks. By default this
is every processed match; --matches pins a subset even after the rest are processed.

Also reports how well score increases were decoded: per alliance and category, the decoded count of
actions vs the official FMS count (before the official totals are split among robots).

usage: python scripts/eval2025.py EVENT [--saved]   (--saved: score the saved results instead)
"""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from autoscout.evaluation import load_results
from autoscout.games.reefscape2025 import CATS, Reefscape2025
from autoscout.pipeline import RESULTS_DIR, reanalyze
from autoscout.tba import TBA, MatchInfo

METRICS = [("coral_total", lambda c: sum(c[p][k] for p in c for k in ("L1", "L2", "L3", "L4"))),
           ("coral_L4", lambda c: sum(c[p]["L4"] for p in c)),
           ("auto_coral", lambda c: sum(c["auto"][k] for k in ("L1", "L2", "L3", "L4"))),
           ("algae_net", lambda c: c["teleop"]["net"]),
           ("algae_processor", lambda c: c["teleop"]["processor"])]


def heldout_opr(ms, skip):
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    A, B = [], defaultdict(list)
    for m in ms:
        if m.key in skip or not m.breakdown:
            continue
        for a in ("red", "blue"):
            row = np.zeros(len(teams))
            for t in m.teams(a):
                row[idx[t]] = 1
            A.append(row)
            c = Reefscape2025.official_counts(None, m.breakdown[a])
            for k, f in METRICS:
                B[k].append(f(c))
    if not A:
        raise ValueError("No disjoint qualification matches remain for held-out OPR; use a fixed --matches subset.")
    A = np.array(A)
    return {k: dict(zip(teams, np.linalg.lstsq(A, np.array(v, float), rcond=None)[0])) for k, v in B.items()}


def main(event, saved=False, matches=None, results_dir=RESULTS_DIR):
    tba = TBA()
    ms = [MatchInfo.from_tba(m) for m in tba.event_matches(event) if m["comp_level"] == "qm"]
    results = load_results(event, results_dir)
    qm = {m.key for m in ms}
    results = [r for r in results if r["match_key"] in qm]
    if matches is not None:
        want = set(matches)
        missing = want - {r["match_key"] for r in results}
        if missing:
            raise ValueError(f"Missing results for pinned matches: {sorted(missing)}")
        results = [r for r in results if r["match_key"] in want]
    if not results:
        raise ValueError("No qualification results to evaluate")
    if not saved:
        if Path(results_dir).resolve() != RESULTS_DIR.resolve():
            raise ValueError("--results-dir requires --saved; reanalysis reads the live results directory")
        results = [reanalyze(r["match_key"], tba, save=False) for r in results]
    done = {r["match_key"] for r in results}
    o = heldout_opr(ms, done)
    byk = {m.key: m for m in ms}
    ours, eq = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    dec = Counter()
    for r in results:
        m = byk[r["match_key"]]
        for a in ("red", "blue"):
            rows = [r["teams"].get(str(t)) or r["teams"].get(t) for t in m.teams(a)]
            for k, _ in METRICS:
                tot = sum(x[k] for x in rows)
                for t, x in zip(m.teams(a), rows):
                    ours[k][t].append(x[k])
                    eq[k][t].append(tot / 3)
            # decoding: decoded actions per (phase, cat) vs official
            off = Reefscape2025.official_counts(None, m.breakdown[a])
            got = Counter((e["phase"], e["kind"]) for e in r["events"] if e.get("alliance") == a and e["kind"] in CATS)
            for ph in ("auto", "teleop"):
                for c in CATS:
                    dec["abs"] += abs(got[(ph, c)] - off[ph][c])
                    dec["off"] += off[ph][c]
    report = {"event": event, "matches": sorted(done), "reference_matches": sorted(set(byk) - done),
              "metrics": {}}
    print(f"{event}: {len(results)} matches scored against OPR of {len(ms) - len(done)} other matches")
    for k, _ in METRICS:
        teams = sorted(t for t in ours[k] if t in o[k])
        x = np.array([np.mean(ours[k][t]) for t in teams])
        e = np.array([np.mean(eq[k][t]) for t in teams])
        y = np.array([o[k][t] for t in teams])
        r = lambda v: np.corrcoef(v, y)[0, 1] if np.std(v) > 0 else 0  # noqa: E731
        report["metrics"][k] = {"r": float(r(x)), "equal_r": float(r(e)),
                                 "mae": float(np.abs(x - y).mean()),
                                 "equal_mae": float(np.abs(e - y).mean()), "teams": len(teams)}
        print(f"  {k:16s} r={r(x):.3f} (equal split {r(e):.3f})  mae={np.abs(x - y).mean():.2f} "
              f"(equal {np.abs(e - y).mean():.2f})  teams={len(teams)}")
    print(f"  decoded vs official action counts: abs diff {dec['abs']} over {dec['off']} actions "
          f"({dec['abs'] / max(dec['off'], 1):.0%})")
    report["decoded"] = {"absolute_difference": dec['abs'], "official_actions": dec['off']}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event")
    ap.add_argument("--saved", action="store_true")
    ap.add_argument("--matches", type=Path, help="JSON list of qualification match keys, fixed across variants")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, help="write metrics and both match lists as JSON")
    args = ap.parse_args()
    report = main(args.event, args.saved, json.loads(args.matches.read_text()) if args.matches else None, args.results_dir)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

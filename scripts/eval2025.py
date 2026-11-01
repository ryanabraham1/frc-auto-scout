"""Honest 2025 attribution check: reanalyze every processed match of an event (nothing saved) and score
per-team averages against OPR computed only from the event's qualification matches that were NOT
processed, so the reference never shares a match with what it checks.

Also reports how well score increases were decoded: per alliance and category, the decoded count of
actions vs the official FMS count (before the official totals are split among robots).

usage: python scripts/eval2025.py EVENT [--saved]   (--saved: score the saved results instead)
"""
import json
import sys
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
    A = np.array(A)
    return {k: dict(zip(teams, np.linalg.lstsq(A, np.array(v, float), rcond=None)[0])) for k, v in B.items()}


def main(event, saved=False):
    tba = TBA()
    ms = [MatchInfo.from_tba(m) for m in tba.event_matches(event) if m["comp_level"] == "qm"]
    results = load_results(event, RESULTS_DIR)
    if not saved:
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
    print(f"{event}: {len(results)} matches scored against OPR of {len(ms) - len(done)} other matches")
    for k, _ in METRICS:
        teams = sorted(t for t in ours[k] if t in o[k])
        x = np.array([np.mean(ours[k][t]) for t in teams])
        e = np.array([np.mean(eq[k][t]) for t in teams])
        y = np.array([o[k][t] for t in teams])
        r = lambda v: np.corrcoef(v, y)[0, 1] if np.std(v) > 0 else 0  # noqa: E731
        print(f"  {k:16s} r={r(x):.3f} (equal split {r(e):.3f})  mae={np.abs(x - y).mean():.2f} "
              f"(equal {np.abs(e - y).mean():.2f})  teams={len(teams)}")
    print(f"  decoded vs official action counts: abs diff {dec['abs']} over {dec['off']} actions "
          f"({dec['abs'] / max(dec['off'], 1):.0%})")


if __name__ == "__main__":
    main(sys.argv[1], "--saved" in sys.argv)

"""How well does per-robot attribution agree with OPR across an event?

OPR (least squares over all qualification matches of the *official* alliance counts) is an
independent estimate of each team's average contribution: it never looks at video. Per-robot
numbers from the scouter are averaged per team over the processed matches and correlated with it.

Two reference points make the correlation meaningful:
  * equal split: every robot gets a third of its alliance's official count (no video at all);
    the scouter has to beat this to be adding information,
  * OPR split: the official count split in proportion to the three teams' OPRs, i.e. what a
    scouter that knew every team's true average would produce (a soft ceiling).

usage: python scripts/attribution_check.py EVENT [--results DIR] [--same-as DIR]
  --results: results folder to score (default: data/results)
  --same-as: only score matches that also exist in this other results folder (fair A/B)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from opr import opr  # noqa: E402

from autoscout.pipeline import RESULTS_DIR  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402

METRICS = {
    2026: [("fuel_scored", "hubScore.totalCount", lambda bd: bd["hubScore"]["totalCount"])],
    2025: [("coral_total", "teleopCoralCount", None),
           ("coral_L4", "teleopReef.tba_topRowCount", None),
           ("algae_net", "netAlgaeCount", lambda bd: bd["netAlgaeCount"]),
           ("algae_processor", "wallAlgaeCount", lambda bd: bd["wallAlgaeCount"])],
}


def main(event: str, results=RESULTS_DIR, same_as=None):
    tba = TBA()
    year = int(event[:4])
    res = []
    for p in sorted(Path(results).glob(f"{event}_qm*.json")):
        if p.stem.count("_") != 1 or (same_as and not (Path(same_as) / p.name).exists()):
            continue
        r = json.loads(p.read_text())
        if r.get("teams"):
            res.append(r)
    ms = {m["key"]: MatchInfo.from_tba(m) for m in tba.event_matches(event)}
    print(f"{event}: {len(res)} processed matches")
    for key, field, official in METRICS[year]:
        o = opr(event, field, tba)
        ours, eq, osplit = {}, {}, {}
        for r in res:
            m = ms[r["match_key"]]
            for a in ("red", "blue"):
                tm = m.teams(a)
                tot = sum(r["teams"][str(t)][key] for t in tm)
                s = sum(max(o[t], 0.1) for t in tm)
                for t in tm:
                    ours.setdefault(t, []).append(r["teams"][str(t)][key])
                    eq.setdefault(t, []).append(tot / 3)
                    osplit.setdefault(t, []).append(tot * max(o[t], 0.1) / s)
        teams = sorted(ours)
        x = np.array([np.mean(ours[t]) for t in teams])
        e = np.array([np.mean(eq[t]) for t in teams])
        s = np.array([np.mean(osplit[t]) for t in teams])
        y = np.array([o[t] for t in teams])
        c = lambda a: np.corrcoef(a, y)[0, 1]  # noqa: E731
        print(f"  {key:16s} vs OPR({field}): scouter r={c(x):.2f} | equal split r={c(e):.2f} | "
              f"OPR split (ceiling) r={c(s):.2f} | teams={len(teams)} | "
              f"mean abs err vs OPR: scouter {np.mean(np.abs(x - y)):.1f}, equal {np.mean(np.abs(e - y)):.1f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--same-as")
    a = ap.parse_args()
    main(a.event, a.results, a.same_as)

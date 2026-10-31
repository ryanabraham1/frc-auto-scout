"""How well does per-robot attribution agree with OPR across an event? (see autoscout/evaluation.py)

usage: python scripts/attribution_check.py EVENT [--results DIR] [--same-as DIR]
  --results: results folder to score (default: data/results)
  --same-as: only score matches that also exist in this other results folder (fair A/B)
"""
import argparse

from autoscout.evaluation import METRICS, attribution_accuracy, load_results  # noqa: F401
from autoscout.pipeline import RESULTS_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--same-as")
    a = ap.parse_args()
    acc = attribution_accuracy(a.event, load_results(a.event, a.results, a.same_as))
    print(f"{a.event}: {acc['matches']} processed matches")
    for m in acc["metrics"]:
        print(f"  {m['key']:16s} scouter r={m['r']:.2f} | equal split r={m['r_equal']:.2f} | OPR split (ceiling) "
              f"r={m['r_ceiling']:.2f} | teams={m['teams']} | mean abs err vs OPR: scouter {m['mae']:.1f}, "
              f"equal {m['mae_equal']:.1f}")

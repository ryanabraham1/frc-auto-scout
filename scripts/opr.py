"""Per-team least-squares contributions (OPR) to official alliance counts at an event.

usage: python scripts/opr.py EVENT [field=hubScore.totalCount] [team ...]
"""
import sys

from autoscout.evaluation import opr

if __name__ == "__main__":
    o = opr(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "hubScore.totalCount")
    want = [int(t) for t in sys.argv[3:]] or sorted(o, key=o.get, reverse=True)
    for t in want:
        print(t, round(o[t], 1))

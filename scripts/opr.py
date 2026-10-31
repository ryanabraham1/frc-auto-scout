"""Per-team least-squares contributions (OPR) to official alliance counts at an event.

usage: python scripts/opr.py EVENT [field=hubScore.totalCount] [team ...]
"""
import sys

import numpy as np

from autoscout.tba import TBA, MatchInfo


def opr(event: str, field: str = "hubScore.totalCount", tba=None) -> dict[int, float]:
    tba = tba or TBA()
    ms = [MatchInfo.from_tba(m) for m in tba.event_matches(event) if m["comp_level"] == "qm" and m.get("score_breakdown")]
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    A, b = [], []
    for m in ms:
        for a in ("red", "blue"):
            row = np.zeros(len(teams))
            for t in m.teams(a):
                row[idx[t]] = 1
            v = m.breakdown[a]
            for k in field.split("."):
                v = v[k]
            A.append(row)
            b.append(float(v))
    x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return {t: float(x[idx[t]]) for t in teams}


if __name__ == "__main__":
    o = opr(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "hubScore.totalCount")
    want = [int(t) for t in sys.argv[3:]] or sorted(o, key=o.get, reverse=True)
    for t in want:
        print(t, round(o[t], 1))

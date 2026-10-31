"""Event-level accuracy of per-robot attribution, measured against OPR.

OPR (least squares over all qualification matches of the *official* alliance counts) is an
independent estimate of each team's average contribution: it never looks at video. Per-robot
numbers from the scouter are averaged per team over the processed matches and correlated with it,
next to two reference points:
  * equal split: every robot gets a third of its alliance's official count (no video at all),
  * OPR split: the official count split by the three teams' OPRs (a soft ceiling).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tba import TBA, MatchInfo

METRICS = {
    2026: [("fuel_scored", "Fuel scored", "hubScore.totalCount")],
    2025: [("coral_total", "Coral", "teleopCoralCount"),
           ("coral_L4", "Coral L4", "teleopReef.tba_topRowCount"),
           ("algae_net", "Net algae", "netAlgaeCount"),
           ("algae_processor", "Processor algae", "wallAlgaeCount")],
}


def opr(event: str, field: str, tba: TBA | None = None) -> dict[int, float]:
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
    if not A:
        return {}
    x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return {t: float(x[idx[t]]) for t in teams}


def load_results(event: str, results: Path, same_as: Path | None = None) -> list[dict]:
    out = []
    for p in sorted(Path(results).glob(f"{event}_qm*.json")):
        if p.stem.count("_") != 1 or (same_as and not (Path(same_as) / p.name).exists()):
            continue
        r = json.loads(p.read_text())
        if r.get("teams"):
            out.append(r)
    return out


def attribution_accuracy(event: str, results: list[dict], tba: TBA | None = None) -> dict:
    """{matches, metrics: [{key, label, r, r_equal, r_ceiling, mae, mae_equal, teams}]}."""
    tba = tba or TBA()
    year = int(event[:4])
    ms = {m["key"]: MatchInfo.from_tba(m) for m in tba.event_matches(event)}
    out = []
    for key, label, field in METRICS.get(year, []):
        o = opr(event, field, tba)
        ours, eq, osplit = {}, {}, {}
        for r in results:
            m = ms.get(r["match_key"])
            if m is None:
                continue
            for a in ("red", "blue"):
                tm = m.teams(a)
                rows = [r["teams"].get(t) or r["teams"].get(str(t)) for t in tm]
                if any(x is None for x in rows) or any(t not in o for t in tm):
                    continue
                tot = sum(x[key] for x in rows)
                s = sum(max(o[t], 0.1) for t in tm)
                for t, x in zip(tm, rows):
                    ours.setdefault(t, []).append(x[key])
                    eq.setdefault(t, []).append(tot / 3)
                    osplit.setdefault(t, []).append(tot * max(o[t], 0.1) / s)
        teams = sorted(ours)
        if len(teams) < 4:
            continue
        x = np.array([np.mean(ours[t]) for t in teams])
        e = np.array([np.mean(eq[t]) for t in teams])
        s = np.array([np.mean(osplit[t]) for t in teams])
        y = np.array([o[t] for t in teams])
        c = lambda a: float(np.corrcoef(a, y)[0, 1]) if np.std(a) > 0 else 0.0  # noqa: E731
        out.append({"key": key, "label": label, "teams": len(teams), "r": round(c(x), 3),
                    "r_equal": round(c(e), 3), "r_ceiling": round(c(s), 3),
                    "mae": round(float(np.mean(np.abs(x - y))), 2), "mae_equal": round(float(np.mean(np.abs(e - y))), 2)})
    return {"event": event, "matches": len(results), "metrics": out}

"""Event-level team aggregates (averages, totals, consistency) from per-match rows."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def numeric_items(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, bool):
            yield prefix + k, float(v)
        elif isinstance(v, (int, float)):
            yield prefix + k, float(v)
        elif isinstance(v, dict):
            yield from numeric_items(v, prefix + k + ".")


def team_table(con, event_key: str) -> list[dict]:
    rows = con.execute("SELECT team, match_key, metrics FROM team_matches WHERE event_key=?", (event_key,)).fetchall()
    per_team: dict[int, list[dict]] = defaultdict(list)
    cats: dict[int, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        m = json.loads(r["metrics"])
        per_team[r["team"]].append(m)
        for k, v in m.items():
            if isinstance(v, str):
                cats[r["team"]][k][v] += 1
    out = []
    for team, ms in per_team.items():
        vals: dict[str, list[float]] = defaultdict(list)
        for m in ms:
            for k, v in numeric_items(m):
                if k not in ("station",):
                    vals[k].append(v)
        row = {"team": team, "matches": len(ms)}
        for k, v in vals.items():
            a = np.array(v)
            row[f"{k}.avg"] = round(float(a.mean()), 2)
            row[f"{k}.max"] = round(float(a.max()), 2)
            row[f"{k}.sd"] = round(float(a.std()), 2)
            row[f"{k}.total"] = round(float(a.sum()), 2)
        for k, c in cats[team].items():
            row[f"{k}.counts"] = dict(c)
        out.append(row)
    out.sort(key=lambda r: -r.get("points_contributed.avg", 0))
    return out

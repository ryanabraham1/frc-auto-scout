"""Validation report: what the system measured vs. the official FMS breakdown."""
from __future__ import annotations

import json

import numpy as np


def event_report(con, event_key: str) -> str:
    rows = con.execute("SELECT match_key, result_path FROM matches WHERE event_key=? AND status='done'",
                       (event_key,)).fetchall()
    lines = [f"Validation report for {event_key} ({len(rows)} matches)", ""]
    score_err, ratios, id_conf = [], [], []
    for r in rows:
        res = json.loads(open(r["result_path"]).read())
        tl = res["timeline"]
        for a in ("red", "blue"):
            off = res.get(f"{a}_score")
            if off is not None and tl[a]:
                score_err.append(abs(max(tl[a]) - off))
            al = res.get("alliances", {}).get(a, {})
            if al.get("detection_ratio"):
                ratios.append(al["detection_ratio"])
            if al.get("coverage"):
                ratios.append(al["coverage"])
            id_conf.append(res.get("identity_conf", {}).get(a, 0))
    if score_err:
        lines.append(f"Overlay score read vs official final score: mean abs diff {np.mean(score_err):.1f} pts "
                     f"(official includes post-match penalty adjustments), exact in {np.mean(np.array(score_err) == 0):.0%}")
    if ratios:
        lines.append(f"Detected actions / official actions: median {np.median(ratios):.2f} "
                     f"(1.0 = every scored piece seen; per-robot splits only need the *relative* shares)")
    if id_conf:
        lines.append(f"Robot identity confidence: median {np.median(id_conf):.2f}; "
                     f"{np.mean(np.array(id_conf) < 0.5):.0%} of alliances need a manual check")
    return "\n".join(lines)

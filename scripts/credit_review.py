"""Review sheets for 2025 scoring credit: one row per credited robot action of a match, for checking by eye.

Each row shows, at the moment the piece went on (the seen placement, else 1 s before the score posted):
the scoring alliance's half of the wide shot with every robot boxed and labelled with the team the
scouter thinks it is (the credited team in yellow), the alliance's reef close-up with its robot boxes,
and a caption with the decoded action and the credit split. Write verdicts into data/credit_truth.json:

    {"2025casj_qm2": {"blue@45.05": 1323, "red@58.06": "?"}}

(team that really scored; "?" when it can't be told). `score` reports how often the scouter's top
credit matches.

usage:
  python scripts/credit_review.py sheet MATCH OUT_PREFIX [--kinds L1,L2,L3,L4,net,processor]
  python scripts/credit_review.py score [RESULTS_DIR]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autoscout import games
from autoscout.field import Calibration
from autoscout.games.base import GamePlugin, MatchContext
from autoscout.pipeline import RESULTS_DIR
from autoscout.video import Video

TRUTH = RESULTS_DIR.parent / "credit_truth.json"


def _ctx(r):
    c = r["calibration"]
    cal = Calibration(event_key=r["event_key"], year=r["year"], view=tuple(c["view"]),
                      points={k: tuple(v) for k, v in c["points"].items()})
    tracks = {k: [tuple(x) for x in v] for k, v in r["raw_tracks"].items()}
    ctx = MatchContext(match=None, field=games.get(r["year"]).field, cal=cal, clock=None, timeline=None,
                       tracks=tracks, identity={k: int(v) for k, v in r["identity"].items()},
                       identity_conf={}, launches={})
    ctx.identity_tl = {k: [tuple(x) for x in v] for k, v in r.get("identity_timeline", {}).items()}
    return cal, ctx


def sheet(key, prefix, kinds, alliance=None):
    r = json.loads((RESULTS_DIR / f"{key}.json").read_text())
    base = Calibration.load(r["event_key"])
    cal, ctx = _ctx(r)
    v = Video(str(next(Path("data/videos").glob(f"{r['video']}.*"))))
    vx, vy, vw, vh = cal.view
    evs = [e for e in r["events"] if e.get("shares") and e.get("kind") in kinds and (not alliance or e["alliance"] == alliance)]
    out = []
    for n in range(0, len(evs), 2):
        blocks = []
        for e in evs[n: n + 2]:
            a = e["alliance"]
            pl = e.get("placed")
            t = pl["t"] if pl else e["t"] - 1.0
            f = v.frame_at(t).copy()
            top = max(e["shares"], key=e["shares"].get)
            for slot in ctx.tracks:
                rb = GamePlugin.robot_box(ctx, slot, t)
                if rb is None:
                    continue
                (x0, y0, x1, y1), obs = rb
                tm = ctx.team_at(slot, t)
                col = (0, 255, 255) if str(tm) == top else (0, 0, 255) if slot.startswith("red") else (255, 120, 0)
                cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if obs else 1)
                cv2.putText(f, str(tm), (int(x0), int(y0) - 6), 0, 0.9, col, 2)
            for bt, view, x0, y0, x1, y1, al, team, conf in r.get("aux_boxes", []):
                if view == f"{a}_reef" and abs(bt - t) <= 0.3:
                    col = (0, 0, 255) if al == "red" else (255, 120, 0)
                    cv2.rectangle(f, (x0, y0), (x1, y1), col, 2)
                    cv2.putText(f, str(team or "?"), (x0, y0 - 5), 0, 0.8, col, 2)
            for p in [p for p in r.get("element_changes", []) if abs(p["t"] - t) < 0.3 and p["elem"].startswith(a)]:
                cv2.circle(f, (int(p["x"]), int(p["y"])), 14, (0, 255, 0), 3)
            rx = cal.points[f"{a}_reef_center"][0]
            hx0 = int(min(max(rx - vw / 3, vx), vx + vw / 3))
            wide = f[vy: vy + vh, hx0: hx0 + 2 * vw // 3]
            ax, ay, aw, ah = base.aux_views[f"{a}_reef"]
            close = cv2.resize(f[ay: ay + ah, ax: ax + aw], (int(aw * vh / ah), vh))
            row = np.hstack([wide, np.full((vh, 4, 3), 255, np.uint8), close])
            sh = " ".join(f"{k} {int(100 * x)}%" for k, x in sorted(e["shares"].items(), key=lambda kv: -kv[1]))
            src = "placed:" + "+".join(x.split("_")[-1] for x in e.get("placed_views", [])) if pl else "not seen placed"
            cap = f"{a}@{e['t']} {e['kind']} -> {top}   [{sh}]  {src}"
            cv2.rectangle(row, (0, 0), (min(row.shape[1], 1300), 40), (0, 0, 0), -1)
            cv2.putText(row, cap, (8, 29), 0, 0.9, (0, 255, 255), 2)
            blocks.append(row)
        w = max(b.shape[1] for b in blocks)
        img = np.vstack([np.pad(b, ((0, 10), (0, w - b.shape[1]), (0, 0))) for b in blocks])
        p = f"{prefix}_{n // 2:02d}.jpg"
        cv2.imwrite(p, cv2.resize(img, None, fx=0.75, fy=0.75), [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(p)
    print(len(evs), "actions ->", len(out), "sheets")


def score(results_dir):
    truth = json.loads(TRUTH.read_text())
    ok = n = 0
    for key, marks in truth.items():
        p = Path(results_dir) / f"{key}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        by = {f"{e['alliance']}@{e['t']}": e for e in r["events"] if e.get("shares")}
        for tag, team in marks.items():
            e = by.get(tag)
            if e is None or team == "?":
                continue
            top = max(e["shares"], key=e["shares"].get)
            ok += top == str(team)
            n += 1
    print(f"top credit is the real scorer: {ok}/{n} = {ok / max(n, 1):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--kinds", default="L1,L2,L3,L4,net,processor")
    ap.add_argument("--alliance")
    a = ap.parse_args()
    if a.cmd == "sheet":
        sheet(a.args[0], a.args[1], set(a.kinds.split(",")), a.alliance)
    else:
        score(a.args[0] if a.args else RESULTS_DIR)

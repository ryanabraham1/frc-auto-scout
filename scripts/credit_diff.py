"""Sheets of the 2025 scoring actions two analysis variants credit to different teams, for checking by eye.

Each row: the scoring alliance's half of the wide shot just before / at / after the score posted, robots
boxed and labelled with their teams (yellow = variant A's pick, magenta = variant B's), and the reef
close-up. Write the verdicts into data/credit_truth.json: {"2025casj_qm11": {"red@41.54": 6238}}.

usage: python scripts/credit_diff.py MATCH OUT_PREFIX 'A_CHANGES' 'B_CHANGES'
  changes are Python dicts of reefscape2025 module settings, e.g. "{'CAM_BOOST': 0}"
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from autoscout.field import Calibration
from autoscout.games import reefscape2025 as R
from autoscout.games.base import GamePlugin, MatchContext
from autoscout import games
from autoscout.pipeline import reanalyze
from autoscout.video import Video

SHOW = (-1.5, -0.5, 0.7)


def picks(key, changes):
    old = {k: getattr(R, k) for k in changes}
    for k, v in changes.items():
        setattr(R, k, v)
    try:
        r = reanalyze(key, save=False)
    finally:
        for k, v in old.items():
            setattr(R, k, v)
    return r, {(e["alliance"], e["t"], e["kind"]): e.get("team") for e in r["events"] if e.get("shares")}


def main(key, prefix, a_ch, b_ch):
    ra, pa = picks(key, eval(a_ch))
    _, pb = picks(key, eval(b_ch))
    diff = sorted(k for k in pa if k in pb and pa[k] != pb[k])
    print(len(diff), "of", len(pa), "actions differ")
    cal = Calibration.load(ra["event_key"])
    c = ra["calibration"]
    ccal = Calibration(event_key=ra["event_key"], year=2025, view=tuple(c["view"]),
                       points={k: tuple(v) for k, v in c["points"].items()})
    tracks = {k: [tuple(x) for x in v] for k, v in ra["raw_tracks"].items()}
    ctx = MatchContext(match=None, field=games.get(2025).field, cal=ccal, clock=None, timeline=None,
                       tracks=tracks, identity=ra["identity"], identity_conf={}, launches={})
    ctx.identity_tl = {k: [tuple(x) for x in v] for k, v in ra["identity_timeline"].items()}
    v = Video(str(next(Path("data/videos").glob(f"{ra['video']}.*"))))
    vx, vy, vw, vh = cal.view
    out = []
    for n in range(0, len(diff), 2):
        blocks = []
        for a, t, kind in diff[n: n + 2]:
            rx = cal.points[f"{a}_reef_center"][0]
            hx0 = int(min(max(rx - vw / 4, vx), vx + vw / 2))
            tiles = []
            for dt in SHOW:
                f = v.frame_at(t + dt).copy()
                for slot in tracks:
                    if not slot.startswith(a):
                        continue
                    rb = GamePlugin.robot_box(ctx, slot, t + dt)
                    if rb is None:
                        continue
                    (x0, y0, x1, y1), obs = rb
                    tm = ctx.team_at(slot, t + dt)
                    col = (0, 255, 255) if tm == pa[(a, t, kind)] else (255, 0, 255) if tm == pb[(a, t, kind)] else (200, 200, 200)
                    cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), col, 2 if obs else 1)
                    cv2.putText(f, str(tm), (int(x0), int(y0) - 5), 0, 0.9, col, 2)
                g = f[vy: vy + vh, hx0: hx0 + vw // 2]
                cv2.putText(g, f"{dt:+.1f}s", (8, vh - 10), 0, 0.8, (255, 255, 255), 2)
                tiles.append(g)
            ax, ay, aw, ah = cal.aux_views[f"{a}_reef"]
            f = v.frame_at(t - 0.8)
            reef = cv2.resize(f[ay: ay + ah, ax: ax + aw], (int(aw * vh / ah), vh))
            top = np.hstack(tiles[:2])
            bot = np.hstack([tiles[2], cv2.resize(reef, (tiles[2].shape[1], vh))])
            blk = np.vstack([top, bot])
            cv2.rectangle(blk, (0, 0), (760, 36), (0, 0, 0), -1)
            cv2.putText(blk, f"{a}@{t} {kind}  A(yellow)={pa[(a, t, kind)]} B(magenta)={pb[(a, t, kind)]}",
                        (6, 26), 0, 0.8, (255, 255, 255), 2)
            blocks.append(blk)
        img = np.vstack([np.pad(b, ((0, 10), (0, 0), (0, 0))) for b in blocks])
        p = f"{prefix}_{n // 2:02d}.jpg"
        cv2.imwrite(p, cv2.resize(img, None, fx=0.8, fy=0.8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(p)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:5])

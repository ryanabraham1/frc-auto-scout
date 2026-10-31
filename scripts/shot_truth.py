"""Ground truth for 2026 scoring attribution, checked by eye.

`sheet` renders, for a sample of live score increases of one alliance, the wide shot at a few
moments before the increase (when the scoring FUEL was released) with every tracked robot box
labelled with the team the pipeline thinks it is. A person writes down which team(s) were really
shooting (checking bumpers where the label looks wrong) into the truth file:

    {"2026dal_qm103": {"red": {"88.5": [176], "93.0": [176, 6696], ...}}}

([] = nobody visible shooting, e.g. hidden behind the HUB; such increases are skipped.)

`score` compares the pipeline's credit (from data/results/<match>.json) with that truth: the
share of checked increases whose credit went mostly to a robot that really was shooting.

usage:
  python scripts/shot_truth.py sheet MATCH VIDEO ALLIANCE OUT.jpg [n=16] [seed=0]
  python scripts/shot_truth.py score MATCH truth.json
"""
import json
import random
import sys

import cv2
import numpy as np

from autoscout import games
from autoscout.field import Calibration
from autoscout.games.base import GamePlugin, MatchContext
from autoscout.pipeline import RESULTS_DIR

SLOTS = ["red0", "red1", "red2", "blue0", "blue1", "blue2"]


def load(key):
    d = json.loads((RESULTS_DIR / f"{key}.json").read_text())
    c = d["calibration"]
    cal = Calibration(event_key=d["event_key"], year=d["year"], view=tuple(c["view"]),
                      points={k: tuple(v) for k, v in c["points"].items()})
    tracks = {k: [tuple(r) for r in v] for k, v in d["raw_tracks"].items()}
    ctx = MatchContext(match=None, field=games.get(d["year"]).field, cal=cal, clock=None, timeline=None,
                       tracks=tracks, identity=d["identity"], identity_conf={}, launches={})
    ctx.identity_tl = {k: [tuple(x) for x in v] for k, v in d.get("identity_timeline", {}).items()}
    return d, ctx


def increases(d, alliance):
    t = np.array(d["timeline"]["t"])
    s = np.array(d["timeline"][alliance], float)
    out = []
    for i in range(1, len(s)):
        if s[i] > s[i - 1]:
            out.append((round(float(t[i]), 2), int(s[i] - s[i - 1])))
    return out


def sheet(key, video, alliance, out, n=16, seed=0):
    from autoscout.video import Video

    d, ctx = load(key)
    clk = d["clock"]
    inc = [x for x in increases(d, alliance) if clk["auto_end_vt"] - clk["auto_len"] + 1 < x[0] < clk["teleop_end_vt"] + 1 and x[1] < 10]
    random.Random(seed).shuffle(inc)
    inc = sorted(inc[:n])
    v = Video(video)
    x, y, w, h = ctx.cal.view
    rows = []
    for t, dpts in inc:
        tiles = []
        for dt in (2.0, 1.4, 0.8):
            f = v.frame_at(t - dt)
            if f is None:
                f = np.zeros((1080, 1920, 3), np.uint8)
            for i, slot in enumerate(SLOTS):
                rb = GamePlugin.robot_box(ctx, slot, t - dt)
                if rb is None:
                    continue
                (x0, y0, x1, y1), obs = rb
                col = (0, 0, 255) if slot.startswith("red") else (255, 0, 0)
                cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if obs else 1)
                cv2.putText(f, str(ctx.team_at(slot, t - dt)), (int(x0), int(y0) - 6), 0, 1.3, (255, 255, 255), 3)
            tiles.append(cv2.resize(f[y: y + h, x: x + w], (640, int(640 * h / w))))
        row = np.hstack(tiles)
        cv2.rectangle(row, (0, 0), (360, 30), (0, 0, 0), -1)
        cv2.putText(row, f"{alliance} +{dpts} @ {t}", (6, 22), 0, 0.8, (0, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(out, np.vstack(rows))
    print([t for t, _ in inc])


def score(key, truth_path):
    d, ctx = load(key)
    truth = json.loads(open(truth_path).read())[key]
    credits = [e for e in d["events"] if e.get("kind") == "scored"]
    ok = tot = 0
    rows = []
    for alliance, marks in truth.items():
        for ts, true_teams in marks.items():
            t = float(ts)
            if not true_teams:
                continue
            got = {}
            for e in credits:
                if abs(e["t"] - t) < 0.05 and e["alliance"] == alliance:
                    got[e["team"]] = got.get(e["team"], 0) + e["n"]
            best = max(got, key=got.get) if got else None
            hit = best in true_teams
            ok += hit
            tot += 1
            rows.append((alliance, t, true_teams, {k: round(v, 2) for k, v in got.items()}, hit))
    for r in rows:
        print(*r)
    print(f"{key}: credit to a real shooter {ok}/{tot} = {ok / max(tot, 1):.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], *(int(a) for a in sys.argv[6:]))
    else:
        score(sys.argv[2], sys.argv[3])

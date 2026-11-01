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
  python scripts/shot_truth.py sheet2 MATCH VIDEO [n=8] [seed=0]     # unlabelled sheets (v2)
  python scripts/shot_truth.py score2 [RESULTS_DIR] [-v]             # score against data/shot_truth.json
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


# ---- v2: ground truth independent of the pipeline -------------------------------------------------
# Sheets show unlabelled frames with a pixel grid; a person marks where the real shooter is (full-frame
# pixels at the middle shown frame) and its team when the bumper is readable:
#   data/shot_truth.json: {"2026cosp_qm3": {"red@88.5": [[1520, 430, 8044], [1650, 400, null]], ...}}
# An empty list = nobody visibly shooting (skipped).
TRUTH = RESULTS_DIR.parent / "shot_truth.json"
OFFSETS = (2.2, 1.5, 0.8)  # seconds before the score increase that are shown


def zone_columns(ctx, alliance, margin_m=1.5):
    """Image x-range covering an alliance's zone (plus a margin into the neutral zone) in the wide shot."""
    fm = ctx.field
    x0, _, x1, _ = fm.zones[f"{alliance}_alliance"]
    pts = [(x, y) for x in (max(x0 - margin_m, 0), min(x1 + margin_m, fm.length)) for y in (0.0, fm.width)]
    xs = ctx.cal.to_image(pts)[:, 0]
    vx, _, vw, _ = ctx.cal.view
    return int(max(vx, xs.min())), int(min(vx + vw, xs.max()))


def sheet2(key, video, n=8, seed=0, out_prefix="sheet"):
    """Sheets of n random live score increases per alliance (4 per image); returns the image paths."""
    from autoscout.video import Video

    d, ctx = load(key)
    clk = d["clock"]
    v = Video(video)
    vx, vy, vw, vh = ctx.cal.view
    rng = random.Random(seed)
    picks = []
    for alliance in ("red", "blue"):
        inc = [x for x in increases(d, alliance)
               if clk["auto_end_vt"] - clk["auto_len"] + 1.5 < x[0] < clk["teleop_end_vt"] + 0.5 and x[1] < 10]
        rng.shuffle(inc)
        # spread out: at most one increase per 3 s window, so the picks are different volleys
        chosen = []
        for t, dp in inc:
            if all(abs(t - c) > 3 for c, _ in chosen):
                chosen.append((t, dp))
            if len(chosen) >= n:
                break
        picks += [(alliance, t, dp) for t, dp in sorted(chosen)]
    paths = []
    for k in range(0, len(picks), 4):
        rows = []
        for alliance, t, dp in picks[k: k + 4]:
            # native resolution, cropped to the scoring alliance's side (shooters must be in its zone)
            zx0, zx1 = zone_columns(ctx, alliance)
            tiles = []
            for dt in OFFSETS:
                f = v.frame_at(t - dt)
                f = f if f is not None else np.zeros((1080, 1920, 3), np.uint8)
                g = f[vy: vy + vh, zx0: zx1].copy()
                for gx in range((zx0 // 100 + 1) * 100 - zx0, zx1 - zx0, 100):
                    cv2.line(g, (gx, 0), (gx, vh), (0, 200, 200), 1)
                    cv2.putText(g, str(gx + zx0), (gx + 2, 14), 0, 0.5, (0, 255, 255), 1)
                for gy in range((vy // 50 + 1) * 50 - vy, vh, 50):
                    cv2.line(g, (0, gy), (zx1 - zx0, gy), (0, 130, 130), 1)
                    cv2.putText(g, str(gy + vy), (2, gy + 14), 0, 0.5, (0, 255, 255), 1)
                cv2.putText(g, f"-{dt}s", (zx1 - zx0 - 90, vh - 12), 0, 0.8, (255, 255, 255), 2)
                tiles.append(np.pad(g, ((0, 0), (0, 6), (0, 0)), constant_values=255))
            row = np.hstack(tiles)
            cv2.rectangle(row, (0, vh - 34), (330, vh), (0, 0, 0), -1)
            cv2.putText(row, f"{alliance}@{t}  +{dp}", (6, vh - 10), 0, 0.8,
                        (0, 0, 255) if alliance == "red" else (255, 160, 0), 2)
            rows.append(row)
        wmax = max(r.shape[1] for r in rows)
        img = np.vstack([np.pad(r, ((0, 6), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
        p = f"{out_prefix}_{key}_{k // 4}.jpg"
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    print(paths)
    return paths


def score2(results_dir=RESULTS_DIR, truth_path=TRUTH, verbose=False):
    """For every marked increase, the team the pipeline credited most for it: is its tracked box on a
    marked shooter (physical) and is it a marked shooter's team where the team was readable (team)?"""
    from pathlib import Path

    truth = json.loads(Path(truth_path).read_text())
    phys = phys_n = team = team_n = 0
    for key, marks in truth.items():
        p = Path(results_dir) / f"{key}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        c = d["calibration"]
        cal = Calibration(event_key=d["event_key"], year=d["year"], view=tuple(c["view"]),
                          points={k: tuple(v) for k, v in c["points"].items()})
        tracks = {k: [tuple(r) for r in v] for k, v in d["raw_tracks"].items()}
        ctx = MatchContext(match=None, field=games.get(d["year"]).field, cal=cal, clock=None, timeline=None,
                           tracks=tracks, identity=d["identity"], identity_conf={}, launches={})
        ctx.identity_tl = {k: [tuple(x) for x in v] for k, v in d.get("identity_timeline", {}).items()}
        credits = [e for e in d["events"] if e.get("kind") == "scored"]
        for tag, shooters in marks.items():
            if not shooters:
                continue
            alliance, ts = tag.split("@")
            t = float(ts)
            got = {}
            for e in credits:
                if abs(e["t"] - t) < 0.3 and e["alliance"] == alliance:
                    got[int(e["team"])] = got.get(int(e["team"]), 0) + e["n"]
            best = max(got, key=got.get) if got else None
            on = False
            if best is not None:
                ts0 = t - OFFSETS[1]  # marks are made on the middle frame
                for slot in tracks:
                    if ctx.team_at(slot, ts0) != best:
                        continue
                    rb = GamePlugin.robot_box(ctx, slot, ts0)
                    if rb is None:
                        continue
                    (x0, y0, x1, y1), _ = rb
                    m = 0.35 * (x1 - x0)
                    on = any(x0 - m <= sx <= x1 + m and y0 - 1.0 * (y1 - y0) <= sy <= y1 + m for sx, sy, _ in shooters)
            phys += on
            phys_n += 1
            known = [tm for _, _, tm in shooters if tm]
            if known:
                team += best in known
                team_n += 1
            if verbose:
                print(key, tag, "credited", best, {k: round(v, 1) for k, v in got.items()}, "truth", shooters,
                      "on-shooter" if on else "WRONG-ROBOT")
    print(f"credited robot is a real shooter (position): {phys}/{phys_n} = {phys / max(phys_n, 1):.0%}")
    print(f"credited team is a real shooter (readable bumpers): {team}/{team_n} = {team / max(team_n, 1):.0%}")
    return phys / max(phys_n, 1), team / max(team_n, 1)


if __name__ == "__main__":
    if sys.argv[1] == "sheet2":
        sheet2(sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:6]))
    elif sys.argv[1] == "score2":
        score2(*(sys.argv[2:3] or [RESULTS_DIR]), verbose="-v" in sys.argv)
    elif sys.argv[1] == "sheet":
        sheet(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], *(int(a) for a in sys.argv[6:]))
    else:
        score(sys.argv[2], sys.argv[3])

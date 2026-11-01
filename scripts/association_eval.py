"""Replay short detector dropouts against known boxes; no training or video inference.

This measures image association under simulated occlusion, NOT real scoring accuracy.
Known boxes are removed before interpolation; the hidden box supplies only the target
release point and localization reference. Whole matches are the reporting units.
Run: python -m scripts.association_eval --out data/association_eval.json
"""
import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from autoscout.field import Calibration
from autoscout.games.base import GamePlugin, MatchContext
from autoscout.vision.tracker import _fill


def legacy_box(ctx, slot, t):
    tr = ctx.tracks[slot]
    if not tr:
        return None
    ts = np.array([r[0] for r in tr])
    i = int(np.argmin(abs(ts - t)))
    if abs(ts[i] - t) > 1:
        return None
    for j in range(i, max(-1, i - 15), -1):
        if tr[j][3] and t - tr[j][0] < 1 and tr[j][4] is not None:
            return tr[j][4], True
    r = tr[i]
    foot = ctx.cal.to_image([r[1:3]])[0]
    w = .95 * ctx.cal.px_per_meter_x(r[1:3])
    return (foot[0] - w/2, foot[1] - .8*w, foot[0] + w/2, foot[1]), False


def summarize(rows):
    out = {"samples": len(rows)}
    for method in rows[0]["methods"]:
        out[method] = {
            "correct_associations": sum(r[method]["correct"] for r in rows),
            "accuracy": float(np.mean([r[method]["correct"] for r in rows])),
            "median_box_error_widths": float(np.median([r[method]["error"] for r in rows])),
            "mean_box_error_widths": float(np.mean([r[method]["error"] for r in rows])),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("data/results"))
    ap.add_argument("--out", type=Path, default=Path("data/association_eval.json"))
    ap.add_argument("--glob", default="*_qm*.json")
    ap.add_argument("--tune", action="store_true", help="Include exploratory candidate weights")
    a = ap.parse_args()
    rows = []
    inputs = {}
    for path in sorted(a.results.glob(a.glob)):
        if path.stem.count("_") != 1:
            continue
        d = json.loads(path.read_text())
        if not d.get("raw_tracks") or not d.get("calibration"):
            continue
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        c = d["calibration"]
        cal = Calibration(event_key=d["event_key"], year=d["year"],
                          view=tuple(c["view"]), points=c["points"])
        tracks = d["raw_tracks"]
        for slot, track in tracks.items():
            last = -1e9
            for row in track:
                t, x, y, seen, box = row
                if not seen or box is None or t - last < 8:
                    continue
                last = t
                for gap in (.2, .5, .9):
                    obs = [(r[0], r[1], r[2], r[4], slot[:-1]) for r in track
                           if r[3] and r[4] is not None and not t-gap < r[0] <= t]
                    before = [o for o in obs if o[0] <= t-gap]
                    after = [o for o in obs if o[0] > t]
                    if not before or not after or t-before[-1][0] > 1 or after[0][0]-t > .3:
                        continue
                    masked = dict(tracks)
                    lo, hi = before[-1][0], after[0][0]
                    filled = _fill([before[-1], after[0]], [r[0] for r in track if lo <= r[0] <= hi])
                    replacements = {r[0]: r for r in filled}
                    masked[slot] = [replacements.get(r[0], r) for r in track]
                    ctx = MatchContext(None, None, cal, None, None, masked, {}, {}, {})
                    width = max(box[2]-box[0], 1)
                    seed = int.from_bytes(hashlib.sha256(f"{path.stem}/{slot}/{t}".encode()).digest()[:8], "little")
                    rng = np.random.default_rng(seed)
                    launch = SimpleNamespace(t0=t, start=((box[0]+box[2])/2 + rng.uniform(-.35,.35)*width,
                                                          box[1]+rng.uniform(-.5,.15)*(box[3]-box[1])))
                    r = {"match": path.stem, "slot": slot, "t": t, "gap": gap}
                    methods = [("legacy", legacy_box, None), ("current", GamePlugin.robot_box, None)]
                    if a.tune:
                        methods += [(f"center_{weight}", GamePlugin.robot_box, weight) for weight in (.25,.5,1.)]
                    r["methods"] = [m[0] for m in methods]
                    boxes = {s: GamePlugin.robot_box(ctx, s, t) for s in masked if s[:-1] == slot[:-1]}
                    for name, fn, center in methods:
                        b, _ = fn(ctx, slot, t)
                        error = np.hypot((b[0]+b[2]-box[0]-box[2])/2, b[1]-box[1])/width
                        # Same release association policy as production, box provider is the only difference.
                        best, bd = None, 1e9
                        for s in masked:
                            if s[:-1] != slot[:-1]:
                                continue
                            rb = boxes[s] if fn == GamePlugin.robot_box else fn(ctx, s, t)
                            if rb is None:
                                continue
                            (x0,y0,x1,y1), observed = rb
                            sx,sy = launch.start
                            w,h = x1-x0,y1-y0
                            dist = np.hypot(max(x0-sx,0,sx-x1), max(y0-max(1.5*h,w)-sy,0,sy-y1))/max(w,20)
                            if center is None:
                                dist += 0 if observed else .25
                            else:
                                dist += center * abs(sx-(x0+x1)/2)/max(w,20) + .1*abs(sy-y0)/max(h,20)
                                dist += 0 if observed else .05
                            if dist < bd:
                                best,bd = s,dist
                        if name == "current":
                            best, bd = GamePlugin.attribute_launch(ctx, launch, alliances=(slot[:-1],))
                        r[name] = {"correct": bool(best == slot and bd < 1), "error": float(error)}
                    rows.append(r)
        print(path.stem, len(rows), flush=True)
    if not rows:
        raise SystemExit("No evaluable detections")
    by_event = defaultdict(list)
    by_gap = defaultdict(list)
    for r in rows:
        by_event[r["match"].split("_")[0]].append(r)
        by_gap[str(r["gap"])].append(r)
    report = {"scope": "simulated detector dropouts; release randomized across hidden robot's upper region",
              "inputs_sha256": inputs,
              "overall": summarize(rows), "events": {k:summarize(v) for k,v in by_event.items()},
              "gaps": {k:summarize(v) for k,v in by_gap.items()}, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

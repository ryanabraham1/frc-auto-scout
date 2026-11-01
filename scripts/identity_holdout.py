"""Held-out robot identity accuracy of the tracker's offline labelling, from saved tracklets (no video).

Tracklets with a clear bumper read are ground truth. In each of K folds the reads of 1/K of them are
hidden, the match is relabelled, and we score the share of hidden observations given the team read.
That is how well the tracker names a robot nobody could read: the stretches stats get smeared over.

usage: python scripts/identity_holdout.py [GLOB] [--k 4] [--set NAME=VALUE ...]
  GLOB: match keys to score, e.g. "2025casj_*" (default: all)
  --set: override a tracker constant for an A/B, e.g. --set MATCH_APP_WEIGHT=0
"""
import argparse
import glob
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout.pipeline import RESULTS_DIR  # noqa: E402
from autoscout.vision import tracker as T  # noqa: E402


def _anchor(t):
    if len(t.obs) < 2 or not t.reads:
        return None
    top = max(t.reads, key=t.reads.get)
    rest = sum(v for k, v in t.reads.items() if k != top)
    return top if t.reads[top] >= 1.5 and rest <= 0.3 * t.reads[top] else None


def score(args):
    path, k, sets = args
    for name, v in sets.items():
        setattr(T, name, v)
    d = json.loads(Path(path).read_text())
    base = T.RobotTracker.from_dict(d)
    anchors = [(t.id, g) for t in base.done if (g := _anchor(t)) in base.alliance_of]
    if len(anchors) < k:
        return path, {}
    random.Random(0).shuffle(anchors)
    out = {"ok": 0, "none": 0, "teammate": 0, "opponent": 0}
    for f in range(k):
        held = dict(anchors[f::k])
        tr = T.RobotTracker.from_dict(d)
        for t in tr.done:
            if t.id in held:
                t.reads = {}
        tr.relabel()
        for t in tr.done:
            if t.id in held:
                g = held[t.id]
                kind = ("ok" if t.team == g else "none" if t.team is None
                        else "teammate" if tr.alliance_of[t.team] == tr.alliance_of[g] else "opponent")
                out[kind] += len(t.obs)
    return path, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", nargs="?", default="*")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    sets = {k: (float(v) if "." in v else int(v)) for k, v in (s.split("=") for s in a.set)}
    files = sorted(glob.glob(str(RESULTS_DIR / f"{a.glob}_tracklets.json")))
    events: dict = {}
    with ProcessPoolExecutor() as ex:
        for path, out in ex.map(score, [(f, a.k, sets) for f in files]):
            if out:
                e = events.setdefault(Path(path).name.split("_")[0], dict.fromkeys(out, 0))
                for kk, v in out.items():
                    e[kk] += v
    tot = dict.fromkeys(["ok", "none", "teammate", "opponent"], 0)
    for ev, e in sorted(events.items()):
        n = sum(e.values())
        print(f"{ev:10s} acc {e['ok'] / n:.3f} | dropped {e['none'] / n:.3f} teammate {e['teammate'] / n:.3f} "
              f"opponent {e['opponent'] / n:.3f} | {n} obs")
        for kk in tot:
            tot[kk] += e[kk]
    n = max(sum(tot.values()), 1)
    print(f"{'ALL':10s} acc {tot['ok'] / n:.3f} | dropped {tot['none'] / n:.3f} teammate {tot['teammate'] / n:.3f} "
          f"opponent {tot['opponent'] / n:.3f} | {n} obs")

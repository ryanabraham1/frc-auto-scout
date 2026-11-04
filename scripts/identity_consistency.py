"""Label-free identity check: do the close-up bumper reads agree with who the tracker says is there?

Credit goes to whichever team a slot is labelled as at that moment, so a swap between two alliance
partners sends every action of one robot to the other even when the tracking itself is right. Each
confident read of team X in a close-up of an element ("red_reef", "blue_hub", ...) is compared with
the saved slot tracks and identity timeline at that time:

  consistent   - the slot labelled X is within --near m of that element
  contradicted - it isn't, but a partner's slot is (likely identity swap: X's actions go to that partner)
  neither      - no slot of X's alliance is near (detection gap, or a robot read in passing)

Runs from saved results only (no video), for any season. "contradicted" is the share to drive down;
a few percent are real (a robot waiting behind the scorer is read too), so compare variants, not zero.

usage: python scripts/identity_consistency.py 2025casj [--near 1.6] [--conf 0.8] [--detail]
"""
import argparse
import bisect
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from autoscout import games  # noqa: E402
from autoscout.pipeline import RESULTS_DIR  # noqa: E402


def team_at(timeline: dict, slot: str, t: float):
    for a, b, team in timeline.get(slot, []):
        if a <= t < b:
            return team
    return None


def score(result: dict, near: float, conf_min: float) -> dict:
    fm = games.get(result["year"]).field
    tl = result.get("identity_timeline") or {s: [[-1e9, 1e9, t]] for s, t in result.get("identity", {}).items()}
    tracks = result.get("tracks", {})
    times = {s: [r[0] for r in tr] for s, tr in tracks.items()}
    out = {"consistent": 0, "contradicted": 0, "neither": 0, "swaps": []}
    for t, view, team, conf in result.get("aux_reads", []):
        if conf < conf_min or view not in fm.elements:
            continue
        a = view.split("_")[0]
        dist = {}
        for i in range(3):
            s = f"{a}{i}"
            tr = tracks.get(s) or []
            if not tr:
                continue
            k = min(bisect.bisect_left(times[s], t), len(tr) - 1)
            if abs(tr[k][0] - t) > 1.0:
                continue
            tm = team_at(tl, s, t)
            if tm is not None:
                dist[tm] = (fm.dist_to_element((tr[k][1], tr[k][2]), view), s)
        if team not in dist:
            continue
        if dist[team][0] <= near:
            out["consistent"] += 1
        elif min(d for d, _ in dist.values()) <= near:
            out["contradicted"] += 1
            other = min(dist, key=lambda k: dist[k][0])
            out["swaps"].append((round(t, 2), view, team, other, dist[other][1]))
        else:
            out["neither"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event")
    ap.add_argument("--near", type=float, default=1.6, help="m from the element that counts as there")
    ap.add_argument("--conf", type=float, default=0.8, help="minimum read confidence")
    ap.add_argument("--detail", action="store_true", help="list contradicted reads")
    a = ap.parse_args()
    tot = {"consistent": 0, "contradicted": 0, "neither": 0}
    for fn in sorted(glob.glob(str(RESULTS_DIR / f"{a.event}_*.json")), key=lambda p: (len(p), p)):
        if not Path(fn).stem.replace(f"{a.event}_", "").isalnum():
            continue  # *_tracklets.json, *_labels.json ...
        r = json.loads(Path(fn).read_text())
        if "aux_reads" not in r or "tracks" not in r:
            continue
        s = score(r, a.near, a.conf)
        n = s["consistent"] + s["contradicted"] + s["neither"]
        if not n:
            continue
        for k in tot:
            tot[k] += s[k]
        print(f"{r['match_key']:16s} reads {n:4d}  consistent {s['consistent'] / n:5.0%}  "
              f"contradicted {s['contradicted'] / n:5.0%}  neither {s['neither'] / n:5.0%}")
        if a.detail:
            for t, view, team, other, slot in s["swaps"]:
                print(f"    t={t:7.2f} {view}: read {team}, but {slot} (labelled {other}) is the one there")
    n = sum(tot.values())
    if n:
        print(f"{'ALL':16s} reads {n:4d}  consistent {tot['consistent'] / n:5.0%}  "
              f"contradicted {tot['contradicted'] / n:5.0%}  neither {tot['neither'] / n:5.0%}")


if __name__ == "__main__":
    main()

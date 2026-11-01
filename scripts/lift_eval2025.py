"""Compare reach attribution on identical saved observations and a pinned match list.

First collect lift observations (autoscout lift-pass EVENT), then copy the results
to an immutable directory. This command never changes the live results or DB.
"""
import argparse
import json
from pathlib import Path

from autoscout import pipeline
from autoscout.games import reefscape2025 as game
from autoscout.tba import TBA
from eval2025 import main as evaluate


def compare(event, matches, source, output):
    keys = json.loads(Path(matches).read_text())
    if not keys:
        raise ValueError('The pinned match list is empty')
    for key in keys:
        r = json.loads((source / f'{key}.json').read_text())
        if not any(r.get('lift_obs', {}).values()):
            raise ValueError(f'{key} has no reach samples; collect them before comparing')
    old_dir, old_reach, old_box = pipeline.RESULTS_DIR, game.REACH_BOOST, game.LIFT_BOOST
    reports = {}
    tba = TBA()
    try:
        pipeline.RESULTS_DIR = source
        for name, reach, box in [('reach', old_reach, old_box), ('box_only', 0., old_box), ('no_lift', 0., 0.)]:
            game.REACH_BOOST, game.LIFT_BOOST = reach, box
            dest = output / name
            dest.mkdir(parents=True, exist_ok=True)
            for key in keys:
                result = pipeline.reanalyze(key, tba, save=False)
                (dest / f'{key}.json').write_text(json.dumps(result, default=pipeline._json_default))
            print(name, flush=True)
            reports[name] = evaluate(event, saved=True, matches=keys, results_dir=dest)
            (output / 'comparison.json').write_text(json.dumps(reports, indent=2))
    finally:
        pipeline.RESULTS_DIR, game.REACH_BOOST, game.LIFT_BOOST = old_dir, old_reach, old_box
    return reports


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('event')
    ap.add_argument('--matches', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    compare(args.event, args.matches, args.results_dir, args.out_dir)

"""Reanalyze cached 2026cc matches with the code in this checkout, writing only to a replay folder.

Usage: python scripts/replay_current_code.py NAME   (writes data/learned_scout_replay/NAME_live)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import config, pipeline  # noqa: E402
from autoscout.tba import TBA  # noqa: E402


def main():
    out = config.DATA_DIR / 'learned_scout_replay' / f'{sys.argv[1]}_live'
    out.mkdir(parents=True, exist_ok=True)
    tba = TBA(max_age_s=1e12)
    keys = sorted(p.stem.replace('_tracklets', '') for p in pipeline.RESULTS_DIR.glob('2026cc_qm*_tracklets.json'))
    summary = []
    for key in keys:
        result = pipeline.reanalyze(key, tba, save=False)
        for alliance in ('red', 'blue'):
            a = result['alliances'][alliance]
            total = sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance'] == alliance)
            assert total + a['hp_fuel_scored'] + a['unattributed_fuel'] == a['official_fuel']
        (out / f'{key}.json').write_text(json.dumps(result, default=pipeline._json_default))
        summary.append(dict(match=key, unattributed={a: v['unattributed_fuel'] for a, v in result['alliances'].items()}))
        print(key, flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(experimental=True, matches=summary), indent=2))


if __name__ == '__main__':
    main()

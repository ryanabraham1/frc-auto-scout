"""Apply the identity anchor repair to cached 2026cc results, backing up every file it changes.

Only matches whose labels change are rewritten. Official alliance fuel totals are asserted before
each result is saved, and the dashboard database is updated with it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import config, db  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, _json_default, reanalyze, repair_identity  # noqa: E402
from autoscout.tba import TBA  # noqa: E402


def main():
    backup = config.DATA_DIR / '2026cc_before_identity_repair'
    backup.mkdir(exist_ok=True)
    con = db.connect()
    tba = TBA(max_age_s=1e12)
    keys = sorted(p.stem.replace('_tracklets', '') for p in RESULTS_DIR.glob('2026cc_qm*_tracklets.json'))
    report = {}
    for key in keys:
        paths = [RESULTS_DIR / f'{key}.json', RESULTS_DIR / f'{key}_tracklets.json']
        staged = [(p, p.read_bytes()) for p in paths]
        changed = repair_identity(key)
        if not changed:
            continue
        for p, data in staged:
            if not (backup / p.name).exists():
                (backup / p.name).write_bytes(data)
        previous = json.loads(staged[0][1])
        result = reanalyze(key, tba, save=False)
        for alliance in ('red', 'blue'):
            a = result['alliances'][alliance]
            assigned = sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance'] == alliance)
            assert assigned + a['hp_fuel_scored'] + a['unattributed_fuel'] == a['official_fuel']
        temporary = paths[0].with_suffix('.repair.tmp')
        temporary.write_text(json.dumps(result, default=_json_default))
        temporary.replace(paths[0])
        db.save_result(con, result, str(paths[0]))
        moved = {k: (previous['teams'].get(str(k), {}).get('fuel_scored'), v['fuel_scored'])
                 for k, v in result['teams'].items()
                 if v['fuel_scored'] != previous['teams'].get(str(k), {}).get('fuel_scored')}
        report[key] = dict(tracklets=changed, fuel=moved)
        print(key, len(changed), 'tracklets;', moved, flush=True)
    con.close()
    (config.DATA_DIR / '2026cc_identity_repair.json').write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()

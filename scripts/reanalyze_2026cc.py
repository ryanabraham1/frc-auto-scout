"""Reanalyze cached 2026cc matches with the current credit rules, backing up results and the database.

Usage: python scripts/reanalyze_2026cc.py BACKUP_NAME   (backups go to data/BACKUP_NAME)
A match that fails the official alliance total check stops the run before it is written.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import config, db, pipeline  # noqa: E402
from autoscout.tba import TBA  # noqa: E402


def main():
    backup = config.DATA_DIR / sys.argv[1]
    backup.mkdir(exist_ok=False)  # a re-run must not replace the originals
    shutil.copy2(db.DB_PATH, backup / db.DB_PATH.name)
    keys = sorted(p.stem.replace('_tracklets', '') for p in pipeline.RESULTS_DIR.glob('2026cc_qm*_tracklets.json'))
    for key in keys:
        shutil.copy2(pipeline.RESULTS_DIR / f'{key}.json', backup / f'{key}.json')
    con = db.connect()
    tba = TBA(max_age_s=1e12)
    for key in keys:
        path = pipeline.RESULTS_DIR / f'{key}.json'
        result = pipeline.reanalyze(key, tba, save=False)
        for alliance in ('red', 'blue'):
            a = result['alliances'][alliance]
            total = sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance'] == alliance)
            assert total + a['hp_fuel_scored'] + a['unattributed_fuel'] == a['official_fuel'], (key, alliance)
        temporary = path.with_suffix('.rules.tmp')
        temporary.write_text(json.dumps(result, default=pipeline._json_default))
        temporary.replace(path)
        db.save_result(con, result, str(path))
        print(key, flush=True)
    con.close()


if __name__ == '__main__':
    main()

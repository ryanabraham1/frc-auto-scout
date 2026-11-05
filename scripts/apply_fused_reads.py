"""Apply fused bumper reads (margin 15, targeted repair) to cached 2026cc results.

Backs up every file it changes and the dashboard database to data/2026cc_before_fused_reads. A match
already marked as done is skipped, so reads are never added twice. A match that fails the official
alliance total check is restored and stops the run.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, db, games, pipeline  # noqa: E402
from autoscout.tba import TBA  # noqa: E402
from autoscout.vision.tracker import RobotTracker  # noqa: E402
from replay_fused_reads import READ_WEIGHT, fused_reads  # noqa: E402
from replay_resolved_identity import order  # noqa: E402

MARGIN = 15.0


def main():
    source = pipeline.RESULTS_DIR
    backup = config.DATA_DIR / '2026cc_before_fused_reads'
    backup.mkdir(exist_ok=True)
    if not (backup / db.DB_PATH.name).exists():
        shutil.copy2(db.DB_PATH, backup / db.DB_PATH.name)
    reads = fused_reads(MARGIN)
    con = db.connect()
    tba = TBA(max_age_s=1e12)
    keys = sorted((p.stem.replace('_tracklets', '') for p in source.glob('2026cc_qm*_tracklets.json')), key=order)
    report = {}
    for key in keys:
        paths = [source / f'{key}.json', source / f'{key}_tracklets.json']
        old = json.loads(paths[0].read_text())
        if 'fused_reads' in old.get('tracker', {}):
            print(key, 'already applied', flush=True)
            continue
        staged = [(p, p.read_bytes()) for p in paths]
        for p, data in staged:
            if not (backup / p.name).exists():
                (backup / p.name).write_bytes(data)
        saved = json.loads(staged[1][1])
        tracker = RobotTracker.from_dict(saved, field_model=games.get(2026).field)
        for t in tracker.done:
            team = reads.get(key, {}).get(t.id)
            if team is not None:
                t.reads[team] = t.reads.get(team, 0.0) + READ_WEIGHT
        tracker.repair_saved()
        before = {x['id']: x['team'] for x in saved['tracklets']}
        changed = [t.id for t in tracker.done if t.team != before[t.id]]
        old.update(pipeline._track_fields({s.key: s.track for s in tracker.all_slots()}))
        old['identity_scores'] = {f'{s.key}:{old[s.alliance][s.index]}': pipeline.TRACKER_EVIDENCE
                                  for s in tracker.all_slots()}
        old['identity_timed'] = []
        old.setdefault('tracker', {})['fused_reads'] = len(reads.get(key, {}))
        try:
            paths[0].write_text(json.dumps(old, default=pipeline._json_default))
            result = pipeline.reanalyze(key, tba, save=False)
            for alliance in ('red', 'blue'):
                a = result['alliances'][alliance]
                total = sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance'] == alliance)
                assert total + a['hp_fuel_scored'] + a['unattributed_fuel'] == a['official_fuel'], (key, alliance)
        except Exception:
            for p, data in staged:
                p.write_bytes(data)
            raise
        temporary = paths[0].with_suffix('.fused.tmp')
        temporary.write_text(json.dumps(result, default=pipeline._json_default))
        temporary.replace(paths[0])
        paths[1].write_text(json.dumps(tracker.to_dict()))
        db.save_result(con, result, str(paths[0]))
        report[key] = dict(reads=len(reads.get(key, {})), tracklets=changed)
        print(key, len(reads.get(key, {})), 'reads;', len(changed), 'labels changed', flush=True)
    con.close()
    if report:  # a re-run applies nothing and must not blank the report
        (config.DATA_DIR / '2026cc_fused_reads.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

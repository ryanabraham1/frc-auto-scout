"""Replay cached 2026cc matches with fused bumper reads added to their tracklets.

Fused reads come from data/identity_review/strong_all.json. Works in a scratch copy: production results,
the dashboard database and the appearance bank are never written. Output goes to
data/learned_scout_replay/fused<margin>_<strategy>_live, with labels beside it.

Usage: python scripts/replay_fused_reads.py MARGIN {repair,resolve} SCRATCH_DIR
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, games, pipeline  # noqa: E402
from autoscout.tba import TBA  # noqa: E402
from autoscout.vision import appearance  # noqa: E402
from autoscout.vision.tracker import RobotTracker  # noqa: E402
from replay_resolved_identity import bank_for, order  # noqa: E402

READ_WEIGHT = 2.5


def fused_reads(margin):
    """{match: {tracklet id: team}} for target tracklets whose best team leads by at least margin."""
    out = {}
    for a in json.loads((config.DATA_DIR / 'identity_review/strong_all.json').read_text()):
        s = sorted(a['scores'].items(), key=lambda x: -x[1])
        if a['kind'] == 'target' and a['frames'] and s[0][1] - s[1][1] >= margin:
            out.setdefault(a['match'], {})[a['id']] = int(s[0][0])
    return out


def main():
    margin, strategy, scratch = float(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    source = pipeline.RESULTS_DIR
    scratch.mkdir(parents=True, exist_ok=True)
    name = f'fused{int(margin)}_{strategy}'
    root = config.DATA_DIR / 'learned_scout_replay'
    out = root / f'{name}_live'
    labels = root / f'{name}_labels'
    out.mkdir(parents=True, exist_ok=True)
    labels.mkdir(exist_ok=True)
    reads = fused_reads(margin)
    raw = appearance._load('2026cc')
    tba = TBA(max_age_s=1e12)
    keys = sorted((p.stem.replace('_tracklets', '') for p in source.glob('2026cc_qm*_tracklets.json')), key=order)
    summary = []
    for key in keys:
        old = json.loads((source / f'{key}.json').read_text())
        saved = json.loads((source / f'{key}_tracklets.json').read_text())
        tracker = RobotTracker.from_dict(saved, field_model=games.get(2026).field)
        for t in tracker.done:
            team = reads.get(key, {}).get(t.id)
            if team is not None:
                t.reads[team] = t.reads.get(team, 0.0) + READ_WEIGHT
        if strategy == 'repair':
            tracker.repair_saved()
        else:
            tracker.relabel(bank_for(raw, key, 'earlier'), pipeline.load_labels(key))
        before = {x['id']: x['team'] for x in saved['tracklets']}
        changed = [t.id for t in tracker.done if t.team != before[t.id]]
        old.update(pipeline._track_fields({s.key: s.track for s in tracker.all_slots()}))
        old['identity_scores'] = {f'{s.key}:{old[s.alliance][s.index]}': pipeline.TRACKER_EVIDENCE
                                  for s in tracker.all_slots()}
        old['identity_timed'] = []
        (scratch / f'{key}.json').write_text(json.dumps(old, default=pipeline._json_default))
        if (source / f'{key}_blobs.npz').exists():
            shutil.copy2(source / f'{key}_blobs.npz', scratch / f'{key}_blobs.npz')
        pipeline.RESULTS_DIR = scratch
        try:
            result = pipeline.reanalyze(key, tba, save=False)
        finally:
            pipeline.RESULTS_DIR = source
        for alliance in ('red', 'blue'):
            a = result['alliances'][alliance]
            total = sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance'] == alliance)
            assert total + a['hp_fuel_scored'] + a['unattributed_fuel'] == a['official_fuel']
        (out / f'{key}.json').write_text(json.dumps(result, default=pipeline._json_default))
        (labels / f'{key}.json').write_text(json.dumps({str(t.id): t.team for t in tracker.done}))
        row = dict(match=key, changed=len(changed), teams={k: v['fuel_scored'] for k, v in result['teams'].items()})
        summary.append(row)
        print(name, row, flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(experimental=True, margin=margin, strategy=strategy,
                                                      matches=summary), indent=2))


if __name__ == '__main__':
    main()

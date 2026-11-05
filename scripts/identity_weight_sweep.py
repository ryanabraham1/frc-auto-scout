"""Compare identity strategies on saved tracklets (no video); writes nothing to the results.

saved    the labels on disk
repair   saved labels, changing only stretches that fix a contradicted clear bumper read
resolve  a full re-solve with the current appearance bank
nobank   a full re-solve with no appearance bank
Usage: python scripts/identity_weight_sweep.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import games  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, load_bank, load_labels  # noqa: E402
from autoscout.vision import tracker as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / 'benchmarks/attribution/2026cc_identity_review.json'


def main():
    keys = sorted(p.stem.replace('_tracklets', '') for p in RESULTS_DIR.glob('2026cc_qm*_tracklets.json'))
    truth = json.loads(TRUTH.read_text())['tracklets']
    bank = load_bank('2026cc')
    out = {}
    for mode in ('saved', 'repair', 'resolve', 'nobank'):
        s, changed = Counter(), []
        for key in keys:
            d = json.loads((RESULTS_DIR / f'{key}_tracklets.json').read_text())
            tr = T.RobotTracker.from_dict(d, field_model=games.get(2026).field)
            saved = {x['id']: x['team'] for x in d['tracklets']}
            if mode == 'repair':
                tr.repair_saved()
            elif mode != 'saved':
                tr.relabel(bank if mode == 'resolve' else None, load_labels(key))
            for t in tr.done + tr.active:
                sec = t.t1 - t.t0
                if t.team != saved.get(t.id):
                    s['changed'] += 1
                    s['changed_sec'] += sec
                    changed.append((key, t.id, saved.get(t.id), t.team, round(t.t0, 1), round(t.t1, 1), len(t.obs),
                                    {k: round(v, 2) for k, v in t.reads.items()}))
                a = tr._anchor_team(t)
                if a is not None and t.team is not None:
                    s['anchored'] += 1
                    s['anchor_broken'] += t.team != a
                    s['anchor_broken_sec'] += (t.team != a) * sec
                want = truth.get(key, {}).get(str(t.id))
                if want is not None:
                    s['truth'] += 1
                    s['truth_ok'] += t.team == want
        out[mode] = dict(s, changes=changed)
        print(mode, {k: round(v, 1) for k, v in s.items()}, flush=True)
    (ROOT / 'data/identity_review/sweep.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

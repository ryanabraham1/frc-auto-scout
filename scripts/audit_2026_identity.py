"""Flag saved bumper/slot disagreements without changing identities.

Close-up associations use proximity to the named hub, so they are review candidates,
not independent ground truth. Tracklet reads were also used by the tracker itself.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.shot_truth import load
from autoscout.games.base import slot_positions_between
from autoscout.pipeline import RESULTS_DIR
from autoscout import config


def audit(d, ctx):
    checks = []
    for t, view, team, conf in d.get('aux_reads', []):
        if conf < 0.7 or view not in ctx.field.elements or team not in d['red'] + d['blue']:
            continue
        alliance = 'red' if team in d['red'] else 'blue'
        candidates = []
        for slot in ctx.tracks:
            if not slot.startswith(alliance):
                continue
            points = slot_positions_between(ctx, slot, t - .3, t + .3)
            if len(points):
                candidates.append((min(ctx.field.dist_to_element(p, view) for p in points), slot))
        candidates.sort()
        if not candidates or candidates[0][0] > 3:
            continue
        margin = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else 3
        if margin < .75:
            continue
        distance, slot = candidates[0]
        assigned = ctx.team_at(slot, t)
        checks.append(dict(t=t, view=view, slot=slot, read_team=team, assigned_team=assigned,
                           agrees=team == assigned, confidence=conf, distance_m=round(distance, 2),
                           margin_m=round(margin, 2)))
    return checks


def main():
    matches = []
    for p in sorted(RESULTS_DIR.glob('2026cc_qm*.json')):
        if not p.stem.split('_')[-1][2:].isdigit():
            continue
        d, ctx = load(p.stem)
        checks = audit(d, ctx)
        tp = p.with_name(p.stem + '_tracklets.json')
        conflicts = []
        if tp.exists():
            for tr in json.loads(tp.read_text())['tracklets']:
                reads = {int(k): v for k, v in tr['reads'].items()}
                if not reads:
                    continue
                team = max(reads, key=reads.get)
                if reads[team] >= 1.5 and sum(reads.values()) - reads[team] <= .3 * reads[team] and team != tr['team']:
                    conflicts.append(dict(tracklet=tr['id'], read_team=team, assigned_team=tr['team'],
                                          t0=tr['obs'][0][0], t1=tr['obs'][-1][0]))
        counts = Counter((r['slot'], r['read_team'], r['assigned_team']) for r in checks if not r['agrees'])
        matches.append(dict(match=p.stem, checks=len(checks), disagreements=sum(counts.values()),
                            repeated_conflicts=[dict(slot=s, read_team=t, assigned_team=a, reads=n)
                                                for (s,t,a), n in counts.most_common() if n >= 3],
                            tracklet_conflicts=conflicts, observations=checks))
    report = dict(scope='Review flags; proximity associations and training OCR are not independent truth.', matches=matches)
    out = config.DATA_DIR / '2026cc_identity_audit.json'
    out.write_text(json.dumps(report, indent=2))
    for m in matches:
        print(m['match'], f"{m['disagreements']}/{m['checks']} close-up disagreements;",
              len(m['tracklet_conflicts']), 'tracklet conflicts')
    print('Saved', out)


if __name__ == '__main__':
    main()

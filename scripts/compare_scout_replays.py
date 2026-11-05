"""Compare scouting replays against fuel OPR without alliance-total inflation.

Every variant is forced to the official alliance total, and OPR is fitted from those same totals,
so an equal split would look excellent against an all-match OPR. OPR here is fitted only on matches
outside the replay set. `within` compares each team's deviation from its alliance mean with the same
deviation in OPR, which only rewards dividing the alliance total correctly.
OPR is a noisy event average, so this ranks variants; it is not per-match accuracy.

Usage: python scripts/compare_scout_replays.py [--baseline current_live]
Writes data/learned_scout_replay/comparison.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import config  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402

BOOTSTRAPS = 4000


def fuel_opr(event, exclude):
    ms = [MatchInfo.from_tba(m) for m in TBA(max_age_s=1e12).event_matches(event)
          if m['comp_level'] == 'qm' and m.get('score_breakdown') and m['key'] not in exclude]
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    rows, totals = [], []
    for m in ms:
        for a in ('red', 'blue'):
            row = np.zeros(len(teams))
            row[[idx[t] for t in m.teams(a)]] = 1
            rows.append(row)
            totals.append(m.breakdown[a]['hubScore']['totalCount'])
    opr = np.linalg.lstsq(np.array(rows), np.array(totals, float), rcond=None)[0]
    print(f'OPR fitted on {len(ms)} held-out matches, {len(teams)} teams')
    return {t: opr[i] for t, i in idx.items()}


def load(folder, opr):
    """One row per team-match: (match index, alliance id, scored, opr, team)."""
    rows = []
    files = sorted(p for p in folder.glob('2026cc_qm*.json'))
    for path in files:
        r = json.loads(path.read_text())
        for team, v in r['teams'].items():
            if int(team) not in opr:
                continue
            rows.append((path.stem, f"{path.stem}/{v['alliance']}", v['fuel_scored'], opr[int(team)], int(team)))
    return rows


def within(rows):
    groups = {}
    for _, g, y, o, _ in rows:
        groups.setdefault(g, []).append((y, o))
    dy, do = [], []
    for vals in groups.values():
        a = np.array(vals, float)
        dy.extend(a[:, 0] - a[:, 0].mean())
        do.extend(a[:, 1] - a[:, 1].mean())
    return float(np.corrcoef(dy, do)[0, 1]) if np.std(dy) > 0 else 0.0


def team_means(rows):
    per = {}
    for _, _, y, o, t in rows:
        per.setdefault(t, [[], o])[0].append(y)
    return np.array([(np.mean(v[0]), v[1]) for v in per.values()])


def summary(rows):
    y = np.array([r[2] for r in rows], float)
    o = np.array([r[3] for r in rows], float)
    tm = team_means(rows)
    return dict(n=len(rows), raw=float(np.corrcoef(y, o)[0, 1]), within=within(rows),
                team_mean=float(np.corrcoef(tm[:, 0], tm[:, 1])[0, 1]),
                team_mae=float(np.abs(tm[:, 0] - tm[:, 1]).mean()))


def paired(base, other, rng):
    """Bootstrap whole matches, so a variant's rows stay paired with the baseline's."""
    by_b, by_o = {}, {}
    for r in base:
        by_b.setdefault(r[0], []).append(r)
    for r in other:
        by_o.setdefault(r[0], []).append(r)
    keys = sorted(by_b)
    diffs = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(len(keys), size=len(keys))
        b, o = [], []
        for n, i in enumerate(pick):
            for src, dst in ((by_b, b), (by_o, o)):
                dst.extend((r[0], f'{n}/{r[1]}', *r[2:]) for r in src[keys[i]])
        diffs.append(within(o) - within(b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(delta=float(np.mean(diffs)), lo=float(lo), hi=float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='current_live')
    args = ap.parse_args()
    root = config.DATA_DIR / 'learned_scout_replay'
    folders = [p for p in sorted(root.iterdir())
               if (p / 'summary.json').exists() and len(list(p.glob('2026cc_qm*.json'))) > 3]
    opr = fuel_opr('2026cc', {f.stem for f in folders[0].glob('2026cc_qm*.json')})
    variants = {p.name: load(p, opr) for p in folders}
    base = variants[args.baseline]
    # An even split of the alliance total is the no-video baseline.
    totals = {}
    for r in base:
        totals.setdefault(r[1], []).append(r[2])
    variants['equal_split'] = [(r[0], r[1], float(np.mean(totals[r[1]])), r[3], r[4]) for r in base]
    rng = np.random.default_rng(0)
    out = {}
    print(f"{'variant':16s} {'n':>4s} {'raw':>6s} {'within':>7s} {'teamMean':>9s} {'teamMAE':>8s}  within vs {args.baseline} (95% CI)")
    for name, rows in variants.items():
        s = summary(rows)
        if name not in (args.baseline, 'equal_split'):
            s['vs_baseline'] = paired(base, rows, rng)
        out[name] = s
        d = s.get('vs_baseline')
        tail = f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d else ''
        print(f"{name:16s} {s['n']:4d} {s['raw']:6.2f} {s['within']:7.3f} {s['team_mean']:9.2f} {s['team_mae']:8.1f}  {tail}")
    (root / 'comparison.json').write_text(json.dumps(dict(
        scope='Fuel OPR fitted on matches outside the replay set. Proxy that ranks variants only; not per-match accuracy.',
        baseline=args.baseline, variants=out), indent=2))


if __name__ == '__main__':
    main()

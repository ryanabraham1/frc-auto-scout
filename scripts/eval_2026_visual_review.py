"""Compare attribution against AI visual review; qm28 stays a separate holdout."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoscout import config
from autoscout.games.rebuilt2026 import Rebuilt2026
from autoscout.vision.trajectories import Launch
from autoscout.vision.flight_fragments import join_hub_fragments
from scripts.shot_truth import load


def flights(d):
    return [Launch(t0=l['t0'], t1=l['t1'], start=l['start'], end=l['end'], apex=l['apex'],
                   n_pieces=l['n'], rises=l['start'][1]-l['apex'][1], pts=l.get('pts', []),
                   target=l.get('target'), rising=l.get('rising', True)) for l in d['launches']['fuel']]


def predict(key, joined=True):
    d,ctx=load(key)
    f=flights(d)
    plugin=Rebuilt2026()
    hz=plugin.hub_top_zone(ctx)
    shooters=plugin.history_shooters(ctx,f,hz)
    processed,pairs=join_hub_fragments(f,hz,shooters,ctx.cal.view[2]/1920) if joined else (f,[])
    if pairs:
        shooters=plugin.history_shooters(ctx,processed,hz)
    links=plugin.link_hub_flights(ctx,processed,shooters,hz)
    aliases={id(e):m for s,e,m in pairs}
    aliases.update({id(s):m for s,e,m in pairs})
    out={}
    for i,l in enumerate(f):
        effective=aliases.get(id(l),l)
        slot=links.get(id(effective))
        out[f'{key}:{i}']=dict(slot=slot,team=ctx.team_at(slot,effective.t0) if slot else None)
    return out,len(pairs)


def main():
    labels=json.loads((config.ROOT/'benchmarks/attribution/2026cc_video_review.json').read_text())['labels']
    manifest=json.loads((config.ROOT/'benchmarks/attribution/2026cc_frozen_predictions.json').read_text())
    summaries=defaultdict(lambda:defaultdict(int))
    details=[]
    for key in sorted({r['match'] for r in labels}):
        pred,joins=predict(key)
        base,_=predict(key,False)
        print(key,'joined fragments:',joins,flush=True)
        for l in labels:
            if l['match']!=key or l['status']!='labeled':
                continue
            old=manifest[l['id']]
            now=pred[l['id']]
            split=summaries[l['split']]
            split['team_labels']+=1
            split['frozen_team_correct']+=old['prediction_team']==l['team']
            split['without_join_team_correct']+=base[l['id']]['team']==l['team']
            split['current_team_correct']+=now['team']==l['team']
            if l['slot']:
                split['box_labels']+=1
                split['frozen_box_correct']+=old['prediction_slot']==l['slot']
                split['without_join_box_correct']+=base[l['id']]['slot']==l['slot']
                split['current_box_correct']+=now['slot']==l['slot']
            details.append(dict(id=l['id'],team=l['team'],box=l['slot'],
                                frozen=dict(team=old['prediction_team'],slot=old['prediction_slot']),
                                without_join=base[l['id']],current=now))
    report=dict(scope='AI visual review; uncertain shots excluded, not human-verified truth or full-match accuracy.',
                summary=dict(summaries),details=details)
    out=config.DATA_DIR/'2026cc_visual_review_eval.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()

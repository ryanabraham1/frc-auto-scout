"""Run an experimental learned linker through full cached-match scouting.

Writes isolated results; never updates production JSON, models, or the dashboard.
Use --split-by made to test official-count allocation from linked tracked makes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from autoscout import config
from autoscout.pipeline import reanalyze,_json_default
from autoscout.games import rebuilt2026 as game
from autoscout.tba import TBA
from autoscout.vision import shot_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--matches',default='2026cc_qm10,2026cc_qm28,2026cc_qm62')
    ap.add_argument('--split-by',choices=['live','made'],default='live')
    args=ap.parse_args()
    torch.set_num_threads(2)
    label=args.checkpoint.stem if args.checkpoint else 'current'
    out=config.DATA_DIR/'learned_scout_replay'/f'{label}_{args.split_by}'
    out.mkdir(parents=True,exist_ok=True)
    original=game.Rebuilt2026.link_hub_flights
    original_split=game.SPLIT_BY
    try:
        if args.checkpoint:
            if args.checkpoint.suffix=='.pt':
                model=shot_model.load(str(args.checkpoint.resolve()))
                def link(self,ctx,launches,shooters,hub_zone):
                    return {k:v['slot'] for k,v in shot_model.predict(model,ctx,launches).items()}
            else:
                import joblib
                from threadpoolctl import threadpool_limits
                state=joblib.load(args.checkpoint)
                def link(self,ctx,launches,shooters,hub_zone):
                    valid=[l for l in launches if l.target in ('red_hub','blue_hub')]
                    if not valid:return {}
                    x=np.stack([shot_model.features(ctx,l,l.target[:-4]) for l in valid])
                    with threadpool_limits(limits=2):
                        p=state['model'].predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(x),3)
                    return {id(l):None if q.max()<state['threshold'] else f'{l.target[:-4]}{int(q.argmax())}'
                            for l,q in zip(valid,p)}
            game.Rebuilt2026.link_hub_flights=link
        game.SPLIT_BY=args.split_by
        summary=[]
        tba=TBA(max_age_s=1e12)
        for key in args.matches.split(','):
            result=reanalyze(key,tba,save=False)
            for alliance in ('red','blue'):
                a=result['alliances'][alliance]
                total=sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance']==alliance)
                assert total+a['hp_fuel_scored']+a['unattributed_fuel']==a['official_fuel']
            (out/f'{key}.json').write_text(json.dumps(result,default=_json_default))
            row=dict(match=key,teams={k:v['fuel_scored'] for k,v in result['teams'].items()},
                     unattributed={a:v['unattributed_fuel'] for a,v in result['alliances'].items()})
            summary.append(row);print(label,args.split_by,row,flush=True)
        (out/'summary.json').write_text(json.dumps(dict(experimental=True,checkpoint=str(args.checkpoint),
            scope='Scouting replay and count conservation; per-team correctness still requires video truth.',matches=summary),indent=2))
    finally:
        game.Rebuilt2026.link_hub_flights=original
        game.SPLIT_BY=original_split


if __name__=='__main__':main()

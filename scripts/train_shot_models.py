"""Train several attribution networks on cached synthetic dropouts and visual labels.

CPU-only experiments; no re-detection, no production changes. Whole events separate
synthetic training and validation; qm28 visual labels remain outside training.
"""
import hashlib
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoscout import config
from autoscout.games.base import GamePlugin,MatchContext
from autoscout.field import Calibration
from autoscout.vision.shot_model import ShotRanker,features
from scripts.eval_2026_visual_review import flights
from scripts.shot_truth import load

OUT=config.DATA_DIR/'models/experiments/shot_linker_v1'


def context(d):
    c=d['calibration']
    return MatchContext(None,None,Calibration(d['event_key'],d['year'],tuple(c['view']),c['points']),
                        None,None,d['raw_tracks'],{},{},{})


def build_synthetic(include_estimates=False):
    rng=np.random.default_rng(20261104)
    sets={'train':[],'validation':[]}
    hashes={}
    for path in sorted((config.DATA_DIR/'results').glob('2026*_qm*.json')):
        event=path.stem.split('_')[0]
        if event not in ('2026cosp','2026dal') or not path.stem.split('_')[-1][2:].isdigit():continue
        d=json.loads(path.read_text())
        if not d.get('raw_tracks'):continue
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        ctx=context(d)
        for slot,track in ctx.tracks.items():
            candidates=[r for r in track if r[3] and r[4] is not None]
            if not candidates:continue
            indices=rng.choice(len(candidates),min(24,len(candidates)),replace=False)
            for index in indices:
                row=candidates[int(index)];t=row[0];x0,y0,x1,y1=row[4];w=x1-x0;h=y1-y0
                sx=(x0+x1)/2+rng.uniform(-.35,.35)*w;sy=y0+rng.uniform(-.4,.15)*h
                # Synthetic ballistic paths supply weak pretraining, not true shot labels.
                vx=rng.uniform(-2.,2.)*w;vy=-rng.uniform(2.,6.)*h;ay=rng.uniform(2.,5.)*h
                pts=[(t+dt,sx+vx*dt,sy+vy*dt+ay*dt*dt) for dt in np.linspace(0,.6,8)]
                launch=SimpleNamespace(t0=t,start=(sx,sy),pts=pts,target=f'{slot[:-1]}_hub')
                masked=dict(ctx.tracks)
                absent=rng.random()<.12
                for s,tr in ctx.tracks.items():
                    if s[:-1]!=slot[:-1]:continue
                    gap=float(rng.choice([0.,.2,.5,.9]))
                    if absent and s==slot and include_estimates:
                        masked[s]=[]
                        continue
                    if absent and s==slot:lo,hi=t-1.5,t+1.
                    else:lo,hi=t-gap,t+.1
                    before=[r for r in tr if r[0]<=lo and r[3]]
                    after=[r for r in tr if r[0]>hi and r[3]]
                    def hidden(r):
                        if include_estimates and before and after:
                            a,b=before[-1],after[0]
                            alpha=(r[0]-a[0])/max(b[0]-a[0],1e-6)
                            return (r[0],a[1]+alpha*(b[1]-a[1]),a[2]+alpha*(b[2]-a[2]),False,None)
                        return (r[0],r[1],r[2],False,None)
                    masked[s]=[hidden(r) if lo<r[0]<=hi else r for r in tr]
                trial=MatchContext(None,None,ctx.cal,None,None,masked,{},{},{})
                target=3 if absent else int(slot[-1])
                base,_=GamePlugin.attribute_launch(trial,launch,alliances=(slot[:-1],))
                sets['train' if event=='2026cosp' else 'validation'].append(
                    (features(trial,launch,slot[:-1],include_estimates),target,path.stem,base==slot if not absent else base is None))
        print(path.stem,'synthetic examples',sum(map(len,sets.values())),flush=True)
    return sets,hashes


def visual_data(include_estimates=False):
    labels=json.loads((config.ROOT/'benchmarks/attribution/2026cc_video_review.json').read_text())['labels']
    sets={'development':[],'holdout':[]}
    for key in sorted({l['match'] for l in labels}):
        d,ctx=load(key);fs=flights(d)
        for l in labels:
            if l['match']!=key or l['status']!='labeled' or not l['slot']:continue
            launch=fs[int(l['id'].split(':')[1])]
            sets[l['split']].append((features(ctx,launch,l['slot'][:-1],include_estimates),int(l['slot'][-1]),key,l['id']))
    return sets


def tensors(rows):
    return torch.from_numpy(np.stack([r[0] for r in rows])),torch.tensor([r[1] for r in rows],dtype=torch.long)


def evaluate(model,rows):
    x,y=tensors(rows)
    with torch.inference_mode():
        p=model(x).softmax(dim=1);pred=p.argmax(dim=1)
    known=y<3
    ranking=p[:,:3].argmax(dim=1)
    return dict(examples=len(rows),correct=int((pred==y).sum()),accuracy=float((pred==y).float().mean()),
                unknown_predictions=int((pred==3).sum()),
                known_examples=int(known.sum()),known_ranking_correct=int(((ranking==y)&known).sum()),
                accepted_correct=int(((pred==y)&(pred<3)).sum()),accepted=int((pred<3).sum()),
                predictions=[dict(id=r[3],target=r[1],prediction=int(i),confidence=float(q.max()))
                             for r,i,q in zip(rows,pred,p)])


def train(model,rows,epochs,lr,seed):
    torch.manual_seed(seed)
    x,y=tensors(rows)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    for epoch in range(epochs):
        for indices in torch.randperm(len(x)).split(128):
            # Reorder candidates so station/slot order cannot become a shortcut.
            order=torch.randperm(3)
            target=y[indices].clone()
            inverse=torch.argsort(order)
            known=target<3;target[known]=inverse[target[known]]
            loss=nn.functional.cross_entropy(model(x[indices][:,order]),target)
            optimizer.zero_grad();loss.backward();optimizer.step()


def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--estimates',action='store_true',help='Include tracked position estimates and visibility masks')
    args=ap.parse_args()
    if args.estimates:OUT=OUT.with_name('shot_linker_v2')
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'dataset.npz'
    if cache.exists():
        data=np.load(cache,allow_pickle=False)
        sets={k:[(x,int(y),str(m),bool(b)) for x,y,m,b in zip(data[k+'_x'],data[k+'_y'],data[k+'_match'],data[k+'_baseline'])]
              for k in ('train','validation')}
        hashes=json.loads((OUT/'sources.json').read_text())
    else:
        sets,hashes=build_synthetic(args.estimates)
        arrays={}
        for key,rows in sets.items():
            arrays.update({key+'_x':np.stack([r[0] for r in rows]),key+'_y':np.array([r[1] for r in rows]),
                           key+'_match':np.array([r[2] for r in rows]),key+'_baseline':np.array([r[3] for r in rows])})
        np.savez_compressed(cache,**arrays);(OUT/'sources.json').write_text(json.dumps(hashes,indent=2))
    visual=visual_data(args.estimates)
    experiments=[]
    # Fixed variants, no hyperparameter search on the held-out real shots.
    for hidden,epochs in ((16,15),(32,25),(64,25)):
        torch.manual_seed(7)
        model=ShotRanker(hidden)
        train(model,sets['train'],epochs,.001,7)
        before=evaluate(model,visual['development'])
        synthetic=evaluate(model,sets['validation'])
        train(model,visual['development'],100,.0001,11)
        name=f'mlp_{hidden}'
        torch.save(dict(hidden=hidden,state_dict=model.state_dict(),feature_version=2 if args.estimates else 1,
                        status='experimental',training_scope='2026cosp synthetic + qm10/qm62 visual'),OUT/f'{name}.pt')
        report=dict(name=name,synthetic_validation=synthetic,visual_before_finetune=before,
                    synthetic_after_finetune=evaluate(model,sets['validation']),
                    development=evaluate(model,visual['development']),holdout=evaluate(model,visual['holdout']))
        experiments.append(report)
        print(name,{k:{a:b for a,b in v.items() if a!='predictions'} for k,v in report.items() if isinstance(v,dict)},flush=True)
    report=dict(scope='Synthetic dropout pretraining and AI visual labels; not end-to-end scouting accuracy.',
                training_examples=len(sets['train']),validation_examples=len(sets['validation']),
                synthetic_baseline_accuracy=float(np.mean([r[3] for r in sets['validation']])),
                independent_visual_holdout='2026cc_qm28',experiments=experiments,promoted=False)
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()

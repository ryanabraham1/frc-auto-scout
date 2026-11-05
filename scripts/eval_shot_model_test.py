"""Evaluate frozen trained linkers on the fresh, visually reviewed qm35 test."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from autoscout import config
from autoscout.vision.shot_model import load as load_model,predict,features
from autoscout.games.rebuilt2026 import Rebuilt2026
from scripts.eval_2026_visual_review import flights
from scripts.shot_truth import load


def main():
    torch.set_num_threads(2)
    truth=json.loads((config.ROOT/'benchmarks/attribution/2026cc_qm35_test.json').read_text())
    labels=[r for r in truth['labels'] if r['status']=='labeled']
    d,ctx=load('2026cc_qm35');fs=flights(d)
    checked=[fs[int(r['id'].split(':')[1])] for r in labels]
    plugin=Rebuilt2026();hz=plugin.hub_top_zone(ctx)
    baseline=plugin.link_hub_flights(ctx,fs,plugin.history_shooters(ctx,fs,hz),hz)
    reports=[dict(name='current',correct=sum(baseline.get(id(f))==l['slot'] for f,l in zip(checked,labels)),
                  cases=len(labels),unknown=sum(baseline.get(id(f)) is None for f in checked))]
    for relative,digest in truth['frozen_checkpoints_sha256'].items():
        path=config.ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Test checkpoint changed after labels were frozen: {path}')
        name=path.parent.name+'/'+path.stem
        if path.suffix=='.pt':
            outputs=predict(load_model(str(path)),ctx,checked)
            slots=[outputs[id(f)]['slot'] for f in checked]
        else:
            state=joblib.load(path)
            x=np.stack([features(ctx,f,f.target[:-4]) for f in checked])
            with threadpool_limits(limits=2):
                prob=state['model'].predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(x),3)
            slots=[None if p.max()<state['threshold'] else f'{f.target[:-4]}{int(p.argmax())}' for f,p in zip(checked,prob)]
        reports.append(dict(name=name,correct=sum(s==l['slot'] for s,l in zip(slots,labels)),cases=len(labels),
                            unknown=sum(s is None for s in slots),
                            predictions=[dict(id=l['id'],expected=l['slot'],predicted=s) for s,l in zip(slots,labels)]))
    out=config.DATA_DIR/'shot_model_fresh_test.json'
    out.write_text(json.dumps(dict(scope=truth['scope'],results=reports,promoted=False),indent=2))
    for r in reports:print(r['name'],r['correct'],'/',r['cases'],'unknown',r['unknown'])


if __name__=='__main__':main()

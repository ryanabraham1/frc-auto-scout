"""Compare boosted-tree candidate scorers with the neural attribution models."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scripts.train_shot_models import OUT,visual_data


def arrays(rows):
    return np.stack([r[0] for r in rows]),np.array([r[1] for r in rows])


def scores(model,x):
    return model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(x),3)


def evaluate(prob,y,threshold,ids=None):
    ranking=prob.argmax(1)
    pred=np.where(prob.max(1)>=threshold,ranking,3)
    known=y<3
    return dict(examples=len(y),correct=int((pred==y).sum()),
                accuracy=float((pred==y).mean()),unknown_predictions=int((pred==3).sum()),
                known_examples=int(known.sum()),known_ranking_correct=int(((ranking==y)&known).sum()),
                accepted=int((pred<3).sum()),accepted_correct=int(((pred==y)&(pred<3)).sum()),
                predictions=[dict(id=str(k),target=int(a),prediction=int(b),confidence=float(p.max()))
                             for k,a,b,p in zip(ids if ids is not None else range(len(y)),y,pred,prob)])


def main():
    data=np.load(OUT/'dataset.npz',allow_pickle=False)
    visual=visual_data()
    dev_x,dev_y=arrays(visual['development']);test_x,test_y=arrays(visual['holdout'])
    x=np.concatenate([data['train_x'],dev_x])
    target=np.concatenate([data['train_y'],dev_y])
    y=(np.arange(3)[None,:]==target[:,None]).ravel().astype(int)
    weights=np.repeat(np.r_[np.ones(len(data['train_x'])),np.full(len(dev_x),25.)],3)
    reports=[]
    with threadpool_limits(limits=2):
        for leaves in (7,15,31):
            model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=leaves,l2_regularization=1.,
                                                  learning_rate=.08,random_state=7)
            model.fit(x.reshape(-1,x.shape[-1]),y,sample_weight=weights)
            val=scores(model,data['validation_x']);vy=data['validation_y']
            # Select abstention using synthetic validation only, never held-out visual labels.
            threshold=max(np.linspace(.05,.95,19),key=lambda t:evaluate(val,vy,t)['correct'])
            name=f'trees_{leaves}'
            joblib.dump(dict(model=model,threshold=float(threshold),feature_version=1,status='experimental'),
                        OUT/f'{name}.joblib')
            report=dict(name=name,threshold=float(threshold),
                        synthetic_validation=evaluate(val,vy,threshold),
                        development=evaluate(scores(model,dev_x),dev_y,threshold,[r[3] for r in visual['development']]),
                        holdout=evaluate(scores(model,test_x),test_y,threshold,[r[3] for r in visual['holdout']]))
            reports.append(report)
            print(name,'threshold',threshold,{k:{f:v[f] for f in ('correct','examples','known_ranking_correct','unknown_predictions')}
                  for k,v in report.items() if isinstance(v,dict)},flush=True)
    (OUT/'tree_evaluation.json').write_text(json.dumps(dict(experiments=reports,promoted=False,
        scope='Weak synthetic labels + AI-reviewed development shots; held-out match excluded from training.'),indent=2))


if __name__=='__main__':main()

"""Learned, permutation-equivariant ball-to-robot attribution (experimental).

Inputs are relative robot geometry over time and the observed ball path. No team
numbers, OPR, scoreboard values, or rule-based link scores enter the network.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn

OFFSETS = (-1.2, -.8, -.4, 0., .2, .4, .8)
N_POINTS = 8
FEATURES = len(OFFSETS)*6 + N_POINTS*3


def features(ctx, launch, alliance, include_estimates=False):
    slots=[f'{alliance}{i}' for i in range(3)]
    sx,sy=launch.start
    points=np.asarray(launch.pts,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        points=np.array([[launch.t0,sx,sy]],float)
    p=points[np.linspace(0,len(points)-1,N_POINTS).astype(int)]
    rows=[]
    for slot in slots:
        track=ctx.tracks.get(slot,[])
        times=ctx._ts.setdefault(slot,np.asarray([r[0] for r in track]))
        temporal=[]
        reference=None
        for dt in OFFSETS:
            t=launch.t0+dt
            i=int(np.searchsorted(times,t))
            choices=[track[j] for j in (i-1,i) if 0<=j<len(track) and abs(track[j][0]-t)<=.15]
            row=min(choices,key=lambda r:abs(r[0]-t)) if choices else None
            box = row[4] if row is not None and row[3] and row[4] is not None else None
            observed = box is not None
            if box is None and include_estimates and ctx.cal is not None:
                from ..games.base import GamePlugin
                estimated=GamePlugin.robot_box(ctx,slot,t)
                if estimated is not None:box=estimated[0]
            if box is None:
                temporal += [0.]*6
                continue
            x0,y0,x1,y1=map(float,box)
            w,h=max(x1-x0,1.),max(y1-y0,1.)
            temporal += [np.clip((sx-(x0+x1)/2)/w,-20,20),np.clip((sy-y0)/h,-20,20),
                         np.log(w/1920),np.log(h/1080),float(observed) if include_estimates else 1.,
                         1. if include_estimates else float(dt)]
            if reference is None or abs(dt)<abs(reference[0]):
                reference=(dt,w,h)
        _,w,h=reference if reference else (0.,100.,100.)
        ball=np.stack([p[:,0]-launch.t0,np.clip((p[:,1]-sx)/w,-20,20),
                       np.clip((p[:,2]-sy)/h,-20,20)],axis=1).ravel()
        rows.append(np.r_[temporal,ball])
    return np.asarray(rows,np.float32)


class ShotRanker(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden=hidden
        self.encoder=nn.Sequential(nn.Linear(FEATURES,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.score=nn.Linear(hidden,1)
        self.unknown=nn.Linear(hidden,1)

    def forward(self,x):
        z=self.encoder(x)
        return torch.cat([self.score(z).squeeze(-1),self.unknown(z.mean(dim=1))],dim=1)


@lru_cache(maxsize=2)
def load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=ShotRanker(checkpoint['hidden'])
    model.feature_version=checkpoint.get('feature_version',1)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def predict(model,ctx,launches):
    valid=[l for l in launches if l.target in ('red_hub','blue_hub')]
    if not valid:return {}
    x=torch.from_numpy(np.stack([features(ctx,l,l.target[:-4],getattr(model,'feature_version',1)>=2) for l in valid]))
    with torch.inference_mode():
        probabilities=model(x).softmax(dim=1).numpy()
    return {id(l):dict(slot=None if (i:=int(p.argmax()))==3 else f'{l.target[:-4]}{i}',
                       confidence=float(p[i]),probabilities=p.tolist()) for l,p in zip(valid,probabilities)}

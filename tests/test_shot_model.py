from types import SimpleNamespace

import numpy as np
import torch

from autoscout.games.base import MatchContext
from autoscout.vision.shot_model import ShotRanker,FEATURES,features


def test_learned_ranker_reorders_robot_scores_and_keeps_unknown_invariant():
    torch.manual_seed(8)
    model=ShotRanker(16).eval()
    x=torch.randn(4,3,FEATURES)
    order=torch.tensor([2,0,1])
    with torch.inference_mode():
        a=model(x);b=model(x[:,order])
    assert torch.allclose(a[:,:3][:,order],b[:,:3],atol=1e-6)
    assert torch.allclose(a[:,3],b[:,3],atol=1e-6)


def test_features_exclude_unobserved_boxes_and_do_not_use_team_identity():
    tracks={'red0':[(1.,0.,0.,True,[0.,100.,100.,200.])],
            'red1':[(1.,0.,0.,False,[500.,100.,600.,200.])]}
    ctx=MatchContext(None,None,None,None,None,tracks,{'red0':1},{},{})
    launch=SimpleNamespace(t0=1.,start=(50.,100.),pts=[(1.,50.,100.),(1.1,60.,80.)])
    a=features(ctx,launch,'red')
    ctx.identity={'red0':9999}
    assert np.array_equal(a,features(ctx,launch,'red'))
    assert a.shape==(3,FEATURES) and np.isfinite(a).all()
    assert a[1,:42].sum()==0 and a[2,:42].sum()==0


def test_estimate_variant_distinguishes_tracking_from_actual_detection():
    from test_core import houston_cal
    cal=houston_cal()
    box=(900.,500.,960.,560.)
    p=cal.to_field([((box[0]+box[2])/2,box[3])])[0]
    tracks={'red0':[(.5,*p,True,box),(1.,*p,False,None),(1.5,*p,True,box)]}
    ctx=MatchContext(None,None,cal,None,None,tracks,{},{},{})
    launch=SimpleNamespace(t0=1.,start=(930.,500.),pts=[(1.,930.,500.)])
    v1=features(ctx,launch,'red')
    v2=features(ctx,launch,'red',include_estimates=True)
    assert v1[0,18:24].sum()==0
    assert v2[0,22]==0 and v2[0,23]==1  # estimated position, not a detection
    assert v2[1,:42].sum()==0  # no track does not invent a robot
    assert tracks['red0'][1][4] is None

import numpy as np

from autoscout.vision.flight_fragments import join_hub_fragments
from autoscout.vision.trajectories import Launch


def flight(t0,t1,offset=0,target=None):
    ts=np.linspace(t0,t1,10)
    pts=[(float(t),100+100*t+offset,500-400*t+200*t*t) for t in ts]
    return Launch(t0,t1,pts[0][1:],pts[-1][1:],min(pts,key=lambda p:p[2])[1:],1,30,pts,
                  target=target, rising=t0<1)


def zone(x,y):
    return x>200


def test_join_consumes_entry_preserves_release_identity_time_and_inputs():
    a,b=flight(0,.8),flight(1.2,1.4,target='red_hub')
    out,pairs=join_hub_fragments([a,b],zone,{id(a):'red0'})
    assert len(out)==len(pairs)==1
    assert out[0].t0==0 and out[0].t1==1.4 and out[0].target=='red_hub'
    assert out[0].n_pieces==1 and a.target is None and len(a.pts)==10


def test_no_join_for_wrong_alliance_ambiguous_volley_or_bad_geometry():
    a,b=flight(0,.8),flight(1.2,1.4,target='red_hub')
    assert not join_hub_fragments([a,b],zone,{id(a):'blue0'})[1]
    twin=flight(1.2,1.4,offset=1,target='red_hub')
    assert not join_hub_fragments([a,b,twin],zone,{id(a):'red0'})[1]
    wrong=flight(1.2,1.4,offset=40,target='red_hub')
    assert not join_hub_fragments([a,wrong],zone,{id(a):'red0'})[1]


def test_source_cannot_be_reused_and_known_makes_are_not_joined():
    a,b=flight(0,.8),flight(1.2,1.4,target='red_hub')
    c=flight(0,.8,offset=1)
    assert not join_hub_fragments([a,c,b],zone,{id(a):'red0',id(c):'red1'})[1]
    a.target='red_hub'
    assert not join_hub_fragments([a,b],zone,{id(a):'red0'})[1]

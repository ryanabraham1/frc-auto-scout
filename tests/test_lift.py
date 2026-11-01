import numpy as np

from autoscout.vision.lift import lift_score, reach_above


def test_lift_requires_nearby_distinct_frames_and_rejects_a_spike():
    assert lift_score([(1, 1), (3, 1)], 0, 4, 0) is None
    assert lift_score([(1, 1), (1, 1)], 0, 4, 0) is None
    assert lift_score([(1, 0), (1.2, 1), (1.4, 0)], 0, 4, 0) == 0
    assert lift_score([(1, .8), (1.5, 1)], 0, 4, .3) == .5


def test_thin_mast_counts_but_another_robot_is_masked():
    mask = np.zeros((140, 120), np.uint8)
    mask[30:87, 58:63] = 255
    box = (30, 80, 90, 120)
    assert reach_above(mask, box, (0, 0), scale=1) > 1
    assert reach_above(mask, box, (0, 0), scale=1, others=[(55, 20, 65, 90)]) == 0


def test_persistent_motion_is_not_a_new_lift():
    from autoscout import games
    from autoscout.games.base import MatchContext

    ctx = MatchContext(match=None, field=games.get(2025).field, cal=None, clock=None,
                       timeline=None, tracks={}, identity={'blue0': 1, 'blue1': 2},
                       identity_conf={}, launches={})
    early = [(i * .1, 0.) for i in range(80)]
    ctx.lift_obs = {
        'blue0': early + [(9 + i * .2, 1.) for i in range(30)],
        'blue1': early + [(9 + i * .2, 0. if i < 15 else 1.) for i in range(30)],
    }
    assert games.get(2025)._reach(ctx, 1, 12, 15) == 0
    assert games.get(2025)._reach(ctx, 2, 12, 15) == 1

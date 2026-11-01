import numpy as np

from autoscout.games.reefscape2025 import Reefscape2025
from autoscout.vision.element_changes import ElementWatcher


def _jump(points, score, allowed=None):
    allowed = allowed or {"net": 4, "teleop_L3": 4, "teleop_L2": 3, "leave": 3, "foul2": 2}
    return {"points": points, "allowed": allowed, "score": {k: score.get(k, 0.0) for k in allowed}}


def test_joint_decoding_follows_the_official_counts():
    # three +4 jumps with a robot equally near the reef and the barge: each alone would be "L3", but
    # officially there was one L3 and two NET algae
    need = {"net": 2, "teleop_L3": 1, "teleop_L2": 0, "leave": 0}
    jumps = [_jump(4, {"net": 0.5, "teleop_L3": 0.6}), _jump(4, {"net": 0.5, "teleop_L3": 0.6}),
             _jump(4, {"net": 0.9, "teleop_L3": 0.1})]
    out = Reefscape2025.decode_jumps(jumps, need, 0)
    flat = [k for c in out for k in c]
    assert flat.count("net") == 2 and flat.count("teleop_L3") == 1
    assert out[2] == ["net"]  # the clearest net jump stays net


def test_leave_points_are_not_read_as_coral():
    need = {"net": 0, "teleop_L3": 0, "teleop_L2": 0, "leave": 2}
    out = Reefscape2025.decode_jumps([_jump(3, {"teleop_L2": 0.8}), _jump(3, {"teleop_L2": 0.8})], need, 0)
    assert out == [["leave"], ["leave"]]


def test_one_robot_is_not_credited_two_quick_coral():
    # two coral 1 s apart, both robots at the reef, A slightly likelier each time: one each
    acts = [{"t": 10.0, "piece": "coral", "cat": "L4", "p": {1: 0.55, 2: 0.45}},
            {"t": 11.0, "piece": "coral", "cat": "L4", "p": {1: 0.55, 2: 0.45}}]
    cr = Reefscape2025.assign_actions(acts, [1, 2])
    assert abs(sum(c[1] for c in cr) - 1.0) < 0.25  # about one coral for team 1, not two
    far = [dict(a, t=a["t"] + 20 * i) for i, a in enumerate(acts)]
    cr = Reefscape2025.assign_actions(far, [1, 2])
    assert all(c[1] > 0.5 for c in cr)  # far apart: no constraint, each follows its likelihood


def test_watcher_finds_a_piece_that_stays_and_ignores_one_that_passes():
    w = ElementWatcher({"reef": (0, 0, 60, 60)}, (100, 100))
    for i in range(80):
        t = i * 0.25
        f = np.zeros((100, 100, 3), np.uint8)
        if t >= 10:
            f[20:30, 20:35] = 255  # coral placed at t=10 and left there
        if 4 <= t < 5:
            f[40:55, 40:55] = 255  # something white passing by for a second
        w.step(t, f)
    ch = w.changes(min_area=20)
    assert len(ch) == 1 and abs(ch[0]["t"] - 10) < 1.0 and 20 <= ch[0]["x"] <= 35

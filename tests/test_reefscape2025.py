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


def test_algae_counter_decides_net_vs_coral():
    # a robot sits at the barge, so on positions alone this +4 would be NET; the algae counter didn't move
    need = {"net": 0, "teleop_L3": 1, "teleop_L2": 0, "leave": 0}
    j = _jump(4, {"net": 0.9, "teleop_L3": 0.1})
    j["algae"] = 0
    assert Reefscape2025.decode_jumps([j], need, 0) == [["teleop_L3"]]
    j["algae"] = 1
    assert Reefscape2025.decode_jumps([j], {**need, "net": 1, "teleop_L3": 0}, 0) == [["net"]]


def test_overlay_counter_deltas():
    from autoscout.overlay import ScoreSample, ScoreTimeline

    s = [ScoreSample(t, None, 0, 0) for t in (0.0, 0.5, 1.0, 1.5)]
    tl = ScoreTimeline(samples=s, clock=None, blue=np.zeros(4), red=np.zeros(4),
                       counters={"red_algae": np.array([0, 0, 1, 3])})
    assert tl.counter_deltas("red_algae") == [(1.0, 1), (1.5, 2)]
    assert tl.to_dict()["counters"]["red_algae"] == [0, 0, 1, 3]


def _fake_match(H_true, n=60, seed=0):
    """A wide-shot track of team 1 at random spots and a close-up that sees it through H_true."""
    import cv2

    rng = np.random.default_rng(seed)
    Hi = np.linalg.inv(np.asarray(H_true, float))
    rows, boxes = [], []
    for i in range(n):
        t = i * 1.0
        p = rng.uniform([2, 2], [6, 6])
        rows.append([t, float(p[0]), float(p[1]), True, [0, 0, 1, 1]])
        px = cv2.perspectiveTransform(np.array([[p]], float), Hi)[0, 0]
        boxes.append([t, "blue_reef", px[0] - 30, px[1] - 50, px[0] + 30, px[1], "blue", 1, 0.9])
    return {"raw_tracks": {"blue0": rows}, "identity": {"blue0": 1}, "identity_timeline": {}, "aux_boxes": boxes}


def test_closeup_camera_calibrates_from_robots_seen_in_both_views():
    from autoscout.vision.fusion import calibrate_views, to_field

    H = [[0.01, 0.0, 1.0], [0.0, 0.012, 0.5], [0.0, 0.0001, 1.0]]
    r = _fake_match(H)
    r["aux_boxes"][3][2:6] = [900, 900, 960, 950]  # a misread: far from where the track is
    v = calibrate_views([r])["blue_reef"]
    assert v["inliers"] >= 55 and v["err_m"] < 0.05
    b = r["aux_boxes"][10]
    x, y = to_field(v["H"], (b[2] + b[4]) / 2, b[5])
    assert abs(x - r["raw_tracks"]["blue0"][10][1]) < 0.05


def test_closeup_sighting_fills_a_gap_in_the_wide_shot_track():
    from autoscout import games
    from autoscout.games.base import MatchContext
    from autoscout.vision.fusion import fuse_tracks

    H = [[0.01, 0.0, 1.0], [0.0, 0.012, 0.5], [0.0, 0.0001, 1.0]]
    tracks = {"blue0": [(-1.0, 3.0, 3.0, True, [0, 0, 1, 1]), (0.1, 3.0, 3.0, False, None),
                        (0.2, 3.0, 3.0, False, None), (5.0, 3.0, 3.0, True, [0, 0, 1, 1])]}
    ctx = MatchContext(match=None, field=games.get(2025).field, cal=None, clock=None, timeline=None,
                       tracks=tracks, identity={"blue0": 1}, identity_conf={}, launches={})
    # the close-up reads team 1 at pixel (300, 300) at t=0.15, while the wide shot only coasts
    ctx.aux_boxes = [[0.15, "blue_reef", 270, 250, 330, 300, "blue", 1, 0.9],
                     [2.5, "blue_reef", 270, 250, 330, 300, "blue", 1, 0.9]]
    n = fuse_tracks(ctx, {"blue_reef": {"H": H}})
    assert n == 2
    filled = [r for r in ctx.tracks["blue0"] if r[3] and r[4] is None]
    assert len(filled) == 3  # two coasting rows took the position, and one row was added in the gap
    assert all(abs(r[1] - 3.0) > 0.1 for r in filled)
    # a sighting right next to a wide-shot observation leaves the wide shot's position alone
    ctx.aux_boxes = [[4.9, "blue_reef", 270, 250, 330, 300, "blue", 1, 0.9]]
    assert fuse_tracks(ctx, {"blue_reef": {"H": H}}) == 0

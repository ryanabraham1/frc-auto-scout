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

    # Put this sighting on carpet, outside the solid REEF exclusion zone.
    H = [[0.01, 0.0, -0.3], [0.0, 0.012, 0.5], [0.0, 0.0001, 1.0]]
    tracks = {"blue0": [(-1.0, 3.0, 3.0, True, [0, 0, 1, 1]), (0.1, 3.0, 3.0, False, None),
                        (0.2, 3.0, 3.0, False, None), (5.0, 3.0, 3.0, True, [0, 0, 1, 1])]}
    ctx = MatchContext(match=None, field=games.get(2025).field, cal=None, clock=None, timeline=None,
                       tracks=tracks, identity={"blue0": 1}, identity_conf={}, launches={})
    # the close-up reads team 1 at pixel (100, 300) at t=0.15, while the wide shot only coasts
    ctx.aux_boxes = [[0.15, "blue_reef", 70, 250, 130, 300, "blue", 1, 0.9],
                     [2.5, "blue_reef", 70, 250, 130, 300, "blue", 1, 0.9]]
    n = fuse_tracks(ctx, {"blue_reef": {"H": H}})
    assert n == 2
    filled = [r for r in ctx.tracks["blue0"] if r[3] and r[4] is None]
    assert len(filled) == 3  # two coasting rows took the position, and one row was added in the gap
    assert all(abs(r[1] - 3.0) > 0.1 for r in filled)
    # a sighting right next to a wide-shot observation leaves the wide shot's position alone
    ctx.aux_boxes = [[4.9, "blue_reef", 70, 250, 130, 300, "blue", 1, 0.9]]
    assert fuse_tracks(ctx, {"blue_reef": {"H": H}}) == 0


def test_fusion_leaves_the_wide_shot_tracks_untouched():
    from autoscout import games
    from autoscout.games.base import MatchContext
    from autoscout.vision.fusion import fuse_tracks

    H = [[0.01, 0.0, -0.3], [0.0, 0.012, 0.5], [0.0, 0.0001, 1.0]]
    wide = {"blue0": [(-1.0, 3.0, 3.0, True, [0, 0, 1, 1]), (0.1, 3.0, 3.0, False, None)]}
    before = [list(r) for r in wide["blue0"]]
    ctx = MatchContext(match=None, field=games.get(2025).field, cal=None, clock=None, timeline=None,
                       tracks=wide, identity={"blue0": 1}, identity_conf={}, launches={})
    ctx.aux_boxes = [[0.15, "blue_reef", 70, 250, 130, 300, "blue", 1, 0.9]]
    assert fuse_tracks(ctx, {"blue_reef": {"H": H}}) == 1
    assert [list(r) for r in wide["blue0"]] == before  # the caller's tracks (saved as raw_tracks) unchanged
    assert ctx.tracks["blue0"][1][3] is True and ctx.fused_rows


def test_tracks_are_remapped_when_the_event_is_recalibrated():
    from autoscout.field import Calibration
    from autoscout.pipeline import remap_tracks

    cal = Calibration.load("2025casj")
    box = [500, 400, 560, 460]
    p = cal.to_field([(530, 460)])[0]
    tracks = {"blue0": [(0.0, 0.0, 0.0, True, box), (0.1, 0.5, 0.5, False, None)]}
    out = remap_tracks(tracks, cal)["blue0"]
    assert abs(out[0][1] - p[0]) < 1e-6 and abs(out[0][2] - p[1]) < 1e-6
    assert abs(out[1][1] - (0.5 + p[0])) < 1e-6  # a coasting row moves with the sighting before it


def test_closeup_reads_fill_missing_reef_dwell_without_extrapolating():
    from types import SimpleNamespace
    from autoscout.games.base import MatchContext
    from autoscout import games

    ctx = MatchContext(match=None, field=games.get(2025).field, cal=None, clock=None,
                       timeline=None, tracks={}, identity={}, identity_conf={}, launches={})
    ctx.aux_boxes = [[t, 'blue_reef', 0, 0, 50, 50, 'blue', tm, conf]
                     for t, tm, conf in [(8, 1, 1), (8.5, 1, 1), (10, 1, 1),
                                         (9, 2, 1), (9.5, 2, .1)]]
    ctx.cal = SimpleNamespace(aux_watch={'blue_reef': [0, 0, 50, 50]})
    ctx.aux_boxes += [[t, 'blue_reef', 100, 0, 150, 50, 'blue', 3, 1.] for t in (8, 8.5)]
    _, dwell = games.get(2025)._evidence(ctx, 'blue', [1, 2, 3], 10, pre=3, post=0)
    assert dwell[1]['reef'] == .5
    assert dwell[2]['reef'] == 0
    assert dwell[3]['reef'] == 0


def test_missing_reach_does_not_disadvantage_an_unseen_scorer(monkeypatch):
    from autoscout.games.base import MatchContext
    from autoscout import games

    plugin = games.get(2025)
    ctx = MatchContext(match=None, field=plugin.field, cal=None, clock=None, timeline=None,
                       tracks={}, identity={}, identity_conf={}, launches={}, lift_obs={'blue0': []})
    monkeypatch.setattr(plugin, '_evidence', lambda *args: (
        {1: {'reef': .3}, 2: {'reef': .3}}, {1: {'reef': 1.}, 2: {'reef': 1.}}))
    monkeypatch.setattr(plugin, '_reach', lambda ctx, tm, *args: 1. if tm == 1 else None)
    shares, _ = plugin._credit(ctx, 'blue', [1, 2], 10, 'L4')
    assert shares == {1: .5, 2: .5}


def test_l1_credit_does_not_reward_a_raised_elevator(monkeypatch):
    from autoscout.games.base import MatchContext
    from autoscout import games

    plugin = games.get(2025)
    ctx = MatchContext(match=None, field=plugin.field, cal=None, clock=None, timeline=None,
                       tracks={}, identity={}, identity_conf={}, launches={}, lift_obs={'blue0': []})
    monkeypatch.setattr(plugin, '_evidence', lambda *args: (
        {1: {'reef': .3}, 2: {'reef': .3}}, {1: {'reef': 1.}, 2: {'reef': 1.}}))
    def no_reach(*args):
        raise AssertionError('L1 does not require lifting')
    monkeypatch.setattr(plugin, '_reach', no_reach)
    assert plugin._credit(ctx, 'blue', [1, 2], 10, 'L1')[0] == {1: .5, 2: .5}


def test_fused_sightings_do_not_refresh_the_wide_shot_join_anchor():
    from autoscout.vision.fusion import _last_seen

    box = [0, 0, 10, 10]
    rows = [(0, 2, 2, True, box), (4, 2, 2, True, None), (5, 2, 2, False, None)]
    assert _last_seen(rows, 5) == rows[0]


def test_closeup_fit_uses_the_corrected_wide_calibration(tmp_path, monkeypatch):
    import json
    from autoscout import pipeline
    from autoscout.field import Calibration
    from autoscout.vision import fusion

    cal = Calibration.load('2025casj')
    saved = {'calibration': {'view': list(cal.view), 'points': {}},
             'raw_tracks': {'blue0': [[0, 0, 0, True, [500, 400, 560, 460]]]}}
    path = tmp_path / '2025casj_qm1.json'
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(pipeline, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(pipeline, 'CAMS_DIR', tmp_path)
    monkeypatch.setattr(Calibration, 'load', lambda event: cal)
    seen = []
    def fit(results):
        seen.extend(results)
        return {}
    monkeypatch.setattr(fusion, 'calibrate_views', fit)
    pipeline.calibrate_closeups('2025casj')
    assert np.allclose(seen[0]['raw_tracks']['blue0'][0][1:3], cal.to_field([(530, 460)])[0])
    assert json.loads(path.read_text()) == saved


def test_event_prior_favours_the_teams_that_score_and_skips_this_match(monkeypatch):
    from types import SimpleNamespace
    from autoscout import games

    plugin = games.get(2025)
    monkeypatch.setattr(plugin, 'official_counts', lambda bd: {
        'auto': dict.fromkeys(('L1', 'L2', 'L3', 'L4'), 0), 'teleop': {'L1': 0, 'L2': 0, 'L3': 0, 'L4': bd}})
    teams = [1, 2, 3, 4, 5, 6]
    rates = {1: 20, 2: 2, 3: 2, 4: 8, 5: 8, 6: 8}
    rows = []
    for i in range(16):
        red = [teams[i % 6], teams[(i + 1) % 6], teams[(i + 3) % 6]]
        blue = [t for t in teams if t not in red]
        rows.append(SimpleNamespace(key=f"qm{i}", red=red, blue=blue,
                                    breakdown={a: sum(rates[t] for t in tm) for a, tm in (("red", red), ("blue", blue))},
                                    teams=lambda a, red=red, blue=blue: red if a == "red" else blue))
    # this match's own result says team 2 is the star; the prior must not see it
    rows[0].breakdown = {"red": 99, "blue": 99}
    prior = plugin.event_prior("qm0", rows)
    w = {t: v["L4"] for a in prior for t, v in prior[a].items()}
    assert w[1] > 2 * w[rows[0].red[1]] and w[2] < 1
    assert plugin.event_prior("qm0", rows[:5]) is None  # too few other matches

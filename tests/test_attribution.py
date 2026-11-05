"""Tests for the attribution upgrades: pseudo-label filtering, release counting, person veto."""
from types import SimpleNamespace

import numpy as np
import pytest

from autoscout import games
from autoscout.field import FieldModel
from autoscout.games.base import GamePlugin, MatchContext
from autoscout.games.rebuilt2026 import Rebuilt2026
from autoscout.train.pseudo_label import _T, robot_tracks
from autoscout.vision.detect import Det, is_person_box
from autoscout.vision.tracker import RobotTracker, Tracklet

from test_core import houston_cal


def _track(boxes, frames=None, alliance="red", conf=0.8):
    frames = frames or list(range(len(boxes)))
    return _T([(k, *b, conf, alliance) for k, b in zip(frames, boxes)])


def test_pseudo_labels_fill_short_gaps_and_drop_static_or_brief_tracks():
    fps = 10
    moving = [(100 + 5 * k, 300, 160 + 5 * k, 350) for k in range(30)]
    gappy = _track(moving[:10] + moving[15:], list(range(10)) + list(range(15, 30)))
    static = _track([(500, 300, 560, 350)] * 30)  # an alliance-coloured field element
    brief = _track(moving[:5])  # a flicker
    out = robot_tracks([gappy, static, brief], fps)
    assert len(out) == 1
    a, boxes = out[0]
    assert a == "red" and all(k in boxes for k in range(30))  # the 0.5 s gap was filled
    assert abs(boxes[12][0] - (100 + 5 * 12)) < 1e-6


def test_mixed_alliance_track_is_not_a_label():
    rows = [(100 + 5 * k, 300, 160 + 5 * k, 350) for k in range(30)]
    tr = _T([(k, *b, 0.8, "red" if k % 2 else "blue") for k, b in enumerate(rows)])
    assert robot_tracks([tr], 10) == []


def test_person_veto_geometry():
    ref = Det(100, 100, 160, 300, 0.0, "")
    assert is_person_box(Det(105, 150, 158, 300, 0.9, "red"), ref)  # the "robot" is the ref
    robot_behind_ref = Det(60, 230, 240, 310, 0.9, "red")  # only a slice is hidden by the ref
    assert not is_person_box(robot_behind_ref, ref)


def _release_ctx(shooter_box, other_box, blobs):
    cal = houston_cal()
    fm = FieldModel.load(2026)
    times = [0.0, 0.5, 1.0]

    def track(box):
        foot = cal.to_field([((box[0] + box[2]) / 2, box[3])])[0]
        return [(t, float(foot[0]), float(foot[1]), True, box) for t in times]

    tracks = {"red0": track(shooter_box), "red1": track(other_box)}
    ctx = MatchContext(match=None, field=fm, cal=cal, clock=None, timeline=None, tracks=tracks,
                       identity={}, identity_conf={}, launches={})
    ctx.blobs = {"fuel": np.array(blobs, np.float32)}
    ctx.blob_view = (0, 0)
    return ctx


def test_release_counts_fast_upward_fuel_at_the_shooter_only():
    # two red robots in the red alliance zone (right side of the Houston view); red0 fires
    shooter = (1500, 560, 1560, 610)
    other = (1700, 560, 1760, 610)
    blobs = []
    for k in range(12):  # a ball leaving red0's top fast and upward, 60 fps
        t = 0.2 + k / 60
        blobs.append((t, 1530, 555 - 10 * k, 40, 7, 7))  # ~600 px/s: ~10 m/s at this depth
    for k in range(12):  # slow drift above red1 (crowd / a piece settling in a hopper)
        t = 0.2 + k / 60
        blobs.append((t, 1730, 540 - 0.5 * k, 40, 7, 7))
    ctx = _release_ctx(shooter, other, blobs)
    plugin = Rebuilt2026()
    times, csum = plugin.column_activity(ctx)
    assert csum["red0"][-1] >= 3
    assert csum["red1"][-1] == 0


def test_link_cost_cache_matches_raw():
    tr = RobotTracker(houston_cal(), FieldModel.load(2026), {"red": [1, 2, 3], "blue": [4, 5, 6]})
    a = Tracklet(0, obs=[(t, 1.0 + t, 2.0, (0, 0, 1, 1), "red") for t in np.arange(0, 2, 0.1)])
    b = Tracklet(1, obs=[(t, 1.0 + t, 2.0, (0, 0, 1, 1), "red") for t in np.arange(2.5, 4, 0.1)])
    tr._uc, tr._lc = {}, {}
    assert tr._link(a, b) == RobotTracker._link_raw(a, b)


def test_tracklet_with_two_teams_read_is_split():
    tr = RobotTracker(houston_cal(), FieldModel.load(2026), {"red": [1, 2, 3], "blue": [4, 5, 6]})
    t = Tracklet(0, obs=[(k / 10, 10.0, 2.0 + k / 10, (0, 0, 50, 50), "red") for k in range(100)])
    t.read_log = [(1.0, 1, 1.5), (2.0, 1, 1.5), (7.0, 2, 1.5)]
    t.reads = {1: 3.0, 2: 1.5}
    tr.done = [t]
    tr._next = 1
    tr.split_conflicts()
    parts = sorted(tr.done, key=lambda x: x.t0)
    assert len(parts) == 2 and parts[0].t1 < 4.5 <= parts[1].t0  # midway between the reads at 2 s and 7 s
    assert parts[0].reads == {1: 3.0} and parts[1].reads == {2: 1.5}


def test_feed_needs_depth_and_direction_toward_own_zone():
    plugin = Rebuilt2026()
    # red's zone is at high field x: 2 m past the red zone line in the neutral zone is deep enough
    x_line = plugin.field.zones["red_alliance"][0]
    assert plugin._dist_from_own_zone("red", (x_line - 2.0, 4.0)) == 2.0
    assert plugin._dist_from_own_zone("red", (x_line + 1.0, 4.0)) == 0.0
    cal = houston_cal()
    box = (900, 500, 960, 560)
    # red's zone is to the right in the Houston view and blue's to the left
    assert plugin._home_dir(MatchContext(match=None, field=plugin.field, cal=cal, clock=None, timeline=None,
                                         tracks={}, identity={}, identity_conf={}, launches={}), "red", None, box) == 1.0
    assert plugin._home_dir(MatchContext(match=None, field=plugin.field, cal=cal, clock=None, timeline=None,
                                         tracks={}, identity={}, identity_conf={}, launches={}), "blue", None, box) == -1.0


def test_weak_detection_extends_a_tracked_robot_but_never_starts_one(monkeypatch):
    from autoscout.vision.detect import Det

    # This tests association on black frames, not the remote bumper OCR model.
    monkeypatch.setattr("autoscout.vision.identity.read_team_numbers_many", lambda frame, boxes, teams: [{} for _ in boxes])

    cal = houston_cal()
    tr = RobotTracker(cal, FieldModel.load(2026), {"red": [1, 2, 3], "blue": [4, 5, 6]})
    frame = np.zeros((1080, 1920, 3), np.uint8)
    box = (900, 520, 960, 570)
    tr.update(0.0, frame, [Det(*box, 0.9, "red")])
    # next frame: the same robot only weakly detected, plus a weak false positive far away
    tr.update(0.1, frame, [Det(box[0] + 2, box[1], box[2] + 2, box[3], 0.15, "red"),
                           Det(300, 600, 360, 650, 0.15, "blue")])
    assert len(tr.active) == 1 and len(tr.active[0].obs) == 2


def test_unread_robot_is_named_by_how_its_read_teammates_look_in_the_match():
    """Two red robots cross paths unread; each looks like a different read anchor earlier in the
    match, so appearance (not the ambiguous motion) decides who is who."""
    tr = RobotTracker(houston_cal(), FieldModel.load(2026), {"red": [1, 2, 3], "blue": [4, 5, 6]})
    rng = np.random.default_rng(0)
    look = {1: rng.normal(size=16), 2: rng.normal(size=16)}

    def emb(team):
        v = look[team] + 0.05 * rng.normal(size=16)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def trk(i, t0, t1, x, team, read):
        t = Tracklet(i, obs=[(t, x, 4.0, (0, 0, 50, 50), "red") for t in np.arange(t0, t1, 0.1)])
        t.reads, t.emb = ({team: 3.0} if read else {}), emb(team)
        return t

    # anchors: 1 on the left, 2 on the right; then both stand still side by side (a coin flip
    # for motion) and the unread tracklets swap places by appearance only
    tr.done = [trk(0, 0, 3, 5.0, 1, True), trk(1, 0, 3, 5.5, 2, True),
               trk(2, 3.2, 8, 5.25, 2, False), trk(3, 3.2, 8, 5.25, 1, False)]
    tr.times = list(np.arange(0, 8, 0.1))
    tr.relabel()
    assert tr.done[2].team == 2 and tr.done[3].team == 1
    assert tr.done[2].mapp[2] > 0 > tr.done[2].mapp[1]


def test_missing_box_moves_with_robot_and_is_not_a_detection():
    cal = houston_cal()
    p0, p1 = (8., 3.), (10., 3.)
    foot = cal.to_image([p0])[0]
    box = (foot[0]-30, foot[1]-50, foot[0]+30, foot[1])
    rows = [(0., *p0, True, box), (.5, *p1, False, None)]
    ctx = MatchContext(None, None, cal, None, None, {"red0": rows}, {}, {}, {})
    moved, observed = GamePlugin.robot_box(ctx, "red0", .5)
    expected = cal.to_image([p1])[0]
    assert np.allclose([(moved[0]+moved[2])/2, moved[3]], expected)
    assert not observed
    assert moved[0] > box[0] + 50
    # The original actual detection stays exact.
    assert GamePlugin.robot_box(ctx, "red0", 0.) == (box, True)


def _live_ctx():
    from types import SimpleNamespace
    # The production path requires a live (frequent) score timeline.
    ctx = MatchContext(None, FieldModel.load(2026), houston_cal(),
                       SimpleNamespace(auto_end_vt=20., teleop_end_vt=160., phase=lambda t: ("auto", 20.-t)),
                       SimpleNamespace(deltas=lambda alliance: [(10., 1)] * 15),
                       {"red0": [], "red1": []}, {"red0": 1, "red1": 2}, {}, {})
    return ctx


def test_release_credit_uses_identity_at_release_not_score_post():
    ctx = _live_ctx()
    ctx.identity_tl = {"red0": [(0, 9, 1), (9, 20, 2)],
                       "red1": [(0, 9, 2), (9, 20, 1)]}
    times = np.array([7., 8., 9., 10.])
    cols = (times, {"red0": np.array([0., 5., 5., 5.]), "red1": np.zeros(4)})
    weights, _ = Rebuilt2026().live_weights(ctx, "red", [1, 2], [], cols)
    assert weights[1]["auto"] == 15
    assert sum(weights[2].values()) == 0


def test_extended_volley_does_not_count_short_window_twice():
    ctx = _live_ctx()
    times = np.array([3., 6., 8., 10.])
    cols = (times, {"red0": np.array([0., 7., 10., 10.]),
                    "red1": np.array([0., 10., 10., 10.])})
    weights, _ = Rebuilt2026().live_weights(ctx, "red", [1, 2], [], cols)
    assert weights[1]["auto"] == weights[2]["auto"] == 7.5


def test_a_few_stray_blobs_are_not_a_volley():
    ctx = _live_ctx()
    times = np.array([3., 6., 8., 10.])
    cols = (times, {"red0": np.array([0., 3., 3., 3.]), "red1": np.zeros(4)})
    weights, events = Rebuilt2026().live_weights(ctx, "red", [1, 2], [], cols)
    assert not weights and not events


def test_a_release_that_never_climbed_is_not_firing_evidence():
    ctx = SimpleNamespace(cal=SimpleNamespace(px_per_meter_x=lambda p: 50.))
    flat = SimpleNamespace(start=(100., 300.), apex=(130., 295.))
    arc = SimpleNamespace(start=(100., 300.), apex=(130., 200.))
    assert not Rebuilt2026._climbed(ctx, flat, (1., 1.))
    assert Rebuilt2026._climbed(ctx, arc, (1., 1.))


def test_overlapping_release_regions_do_not_favor_first_slot():
    from types import SimpleNamespace
    cal = houston_cal()
    # A first-slot robot's large allowed shooter region overlaps the actual shooter.
    boxes = {"red0": (100., 500., 160., 550.), "red1": (140., 500., 200., 550.)}
    tracks = {}
    for slot, box in boxes.items():
        p = cal.to_field([((box[0]+box[2])/2, box[3])])[0]
        tracks[slot] = [(1., *p, True, box)]
    ctx = MatchContext(None, None, cal, None, None, tracks, {}, {}, {})
    launch = SimpleNamespace(t0=1., start=(155., 495.))
    # Both release rectangles contain the point; the shooter is closer to its center.
    assert GamePlugin.attribute_launch(ctx, launch)[0] == "red1"


def test_official_fuel_without_robot_evidence_is_reported_unattributed():
    from types import SimpleNamespace
    from autoscout.games.rebuilt2026 import PERIODS
    teams = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    counts = {f"{p}Count": (3 if p == "auto" else 0) for p in PERIODS}
    match = SimpleNamespace(teams=lambda a: teams[a], breakdown={a: {"hubScore": counts} for a in teams})
    ctx = MatchContext(match, FieldModel.load(2026), houston_cal(), None, None, {}, {}, {}, {})
    result = Rebuilt2026().analyze(ctx)
    for a in teams:
        assert result["alliances"][a]["unattributed_fuel"] == 3
        assert result["alliances"][a]["unattributed_fuel_by_period"]["auto"] == 3
        assert sum(result["teams"][t]["fuel_scored"] for t in teams[a]) == 0


def test_volley_consensus_preserves_clear_release_from_a_neighbor(monkeypatch):
    from types import SimpleNamespace
    plugin = Rebuilt2026()
    launches = [SimpleNamespace(t0=i*.1,start=(100.,400.),target='red_hub',rising=True,n_pieces=1)
                for i in range(4)]
    monkeypatch.setattr(plugin,'hub_shooter',lambda ctx,l,*args:'red1' if l is launches[2] else 'red0')
    monkeypatch.setattr(plugin,'launch_distances',lambda ctx,l,*args:
                        {'red0':(.8,True),'red1':(.05,True)} if l is launches[2]
                        else {'red0':(.05,True),'red1':(.8,True)})
    links=plugin.link_hub_flights(None,launches,{},lambda x,y:np.zeros(len(x),bool))
    assert links[id(launches[2])]=='red1'


def test_measured_release_at_top_beats_neighbors_firing_history(monkeypatch):
    from types import SimpleNamespace
    plugin=Rebuilt2026()
    early=SimpleNamespace(t0=0.,start=(130.,500.))
    now=SimpleNamespace(t0=1.,start=(150.,480.))
    for l in (early,now): l.rising=True
    monkeypatch.setattr(plugin,'hub_bounce',lambda ctx:lambda l:False)
    monkeypatch.setattr(plugin,'launch_distances',lambda ctx,l:
                        {'red0':(.05,True),'red1':(2.,True)} if l is early else
                        {'red0':(.13,True),'red1':(.04,True)})
    boxes={'red0':(100.,500.,160.,560.),'red1':(140.,480.,200.,540.)}
    monkeypatch.setattr(plugin,'robot_box',lambda ctx,s,t:(boxes[s],True))
    out=plugin.history_shooters(None,[early,now],lambda x,y:np.zeros(len(x),bool))
    assert out[id(now)]=='red1'


@pytest.mark.parametrize("period,arrival", [("auto", 10.), ("shift2", 80.)])
@pytest.mark.parametrize("has_entry_time", [True, False])
def test_hub_entry_survives_an_occluded_shooters_bad_floor_position(monkeypatch, period, arrival, has_entry_time):
    from autoscout.games.rebuilt2026 import PERIODS
    from autoscout.overlay import MatchClock
    from autoscout.vision.trajectories import Launch

    teams = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    breakdown = {a: {"hubScore": {f"{p}Count": int(a == "red" and p == period) for p in PERIODS}}
                 for a in teams}
    match = SimpleNamespace(teams=lambda a: teams[a], breakdown=breakdown)
    # The unseen robot's last box/interpolation puts its foot in neutral.
    # The measured ball nevertheless disappears into its own HUB.
    launch = Launch(arrival - .2, arrival, (1400., 250.), (1410., 296.), (1400., 250.),
                    1., 0., [], target="red_hub", rising=False, entry_t=arrival if has_entry_time else None)
    ctx = MatchContext(match, FieldModel.load(2026), houston_cal(),
                       MatchClock(auto_len=20, teleop_len=140, auto_end_vt=20, teleop_end_vt=164),
                       SimpleNamespace(deltas=lambda a: [(arrival + .5, 1)] + [(120. + i, 0) for i in range(14)]),
                       {"red0": [(arrival - .2, 10., 6., False, None)]},
                       {"red0": 1}, {}, {"fuel": [launch]})
    monkeypatch.setattr(Rebuilt2026, "history_shooters", lambda *a: {})
    monkeypatch.setattr(Rebuilt2026, "link_hub_flights", lambda *a: {id(launch): "red0"})
    monkeypatch.setattr(Rebuilt2026, "release_time", lambda *a: arrival - .2)
    monkeypatch.setattr(Rebuilt2026, "column_activity", lambda *a: (None, {}))
    result = Rebuilt2026().analyze(ctx)
    shot = next(e for e in result["events"] if e["kind"] == "shot")
    assert shot["made"] and shot["zone"] == "neutral"  # retain the estimate for inspection
    assert result["teams"][1]["fuel_scored_by_period"][period] == 1
    assert sum(e["n"] for e in result["events"] if e["kind"] == "scored") == 1
    assert result["alliances"]["red"]["unattributed_fuel_by_period"][period] == 0


def test_hub_shadow_requires_image_occlusion_not_just_a_nearby_dropout():
    from autoscout.field import Calibration

    cal = Calibration.load("2026cosp")
    # Both disappear on the far side and within the old 3.5 m radius. Only
    # the second box reaches the HUB silhouette; the first is in open carpet.
    rows = [(9.337, 9.332, 6.204, True, (1033.5, 323.5, 1117.5, 384.2)),
            (15.049, 11.337, 6.553, True, (1203., 312., 1326., 374.4)),
            (32., 13., 6., True, (1450., 320., 1550., 400.))]
    ctx = MatchContext(None, None, cal, None, None, {"red0": rows}, {}, {}, {})
    plugin = Rebuilt2026()
    assert not plugin._in_hub_shadow(ctx, "red0", 12.)
    assert plugin._in_hub_shadow(ctx, "red0", 20.)

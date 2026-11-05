"""Regression cases with known ball paths and scorer identities, independent of OPR."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autoscout.games.base import GamePlugin, MatchContext
from autoscout.games.rebuilt2026 import Rebuilt2026
from autoscout.games.reefscape2025 import Reefscape2025
from autoscout.vision.launches import LaunchExtractor, fit_flight
from autoscout.vision.trajectories import Launch, LaunchDetector
from test_core import _arc, _blobs, _extractor, houston_cal
from test_attribution import _live_ctx


def test_overlapping_ball_is_recovered_as_whole_component():
    detector = LaunchDetector((18, 90, 110), (38, 255, 255), (0, 0, 200, 150))
    # A slow ball overlaps itself in consecutive frames: the difference is a
    # crescent, but the tracked observation should remain at the actual center.
    for k in range(3):
        frame = np.zeros((150, 200, 3), np.uint8)
        cv2.circle(frame, (50 + 5 * k, 80), 9, (0, 220, 255), -1)
        detector.step(k / 60, frame)
    _, x, y, area, w, h = detector.blobs[-1]
    assert abs(x - 60) <= 1 and abs(y - 80) <= 1
    assert area > 200 and min(w, h) >= 16


def test_static_yellow_does_not_become_a_ball():
    detector = LaunchDetector((18, 90, 110), (38, 255, 255), (0, 0, 100, 100))
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, (50, 50), 9, (0, 220, 255), -1)
    for k in range(8):
        detector.step(k / 60, frame)
    assert detector.blobs == []


def test_tracks_expire_across_frames_without_any_detections():
    ex = LaunchExtractor((0, 0, 1920, 1080))
    ex.max_miss = 3
    B = _blobs(_arc(dur=.2), d=14)
    later = B.copy()
    later[:, 0] += 5
    tracks = ex.link(np.vstack([B, later]))
    assert len(tracks) == 2
    assert all(tr.pts[-1][0] - tr.pts[0][0] < .3 for tr in tracks)


def test_short_fragments_recover_one_occluded_ball():
    ex = _extractor()
    B = _blobs(_arc(dur=.45), ex=ex)
    # Only four observations on each side: neither half alone is a flight.
    B = np.vstack([B[:4], B[9:13]])
    launches = ex.extract(B)
    assert len(launches) == 1
    assert len(launches[0].pts) == 8


def test_rejoin_rejects_a_different_ball_at_the_predicted_position():
    ex = _extractor()
    ex.min_dur = .08
    A = _blobs(_arc(dur=.35), ex=ex)
    dt = .5
    start = (900 + 250 * dt, 600 - 700 * dt + 550 * dt * dt)
    # Same position, opposite horizontal velocity: a different shot.
    B = _blobs(_arc(x0=start[0], y0=start[1], vx=-250, vy=-150, dur=.35), ex=ex)
    B[:, 0] += dt
    assert len(ex.rejoin([(A, fit_flight(A)), (B, fit_flight(B))])) == 2


def test_goal_entry_time_survives_serialization():
    points = _arc(x0=600, vy=-900, dur=1.6)
    target = {"blue_hub": (*points[42], 25)}
    launches = _extractor(target).extract(_blobs(points[:44], ex=_extractor(target)))
    assert len(launches) == 1
    launch = launches[0]
    assert launch.target == "blue_hub" and 1.1 < launch.entry_t < 1.5
    assert launch.to_dict()["entry_t"] == round(launch.entry_t, 3)


def test_inferred_release_precedes_first_sighting(monkeypatch):
    ctx = MatchContext(None, None, None, None, None, {}, {}, {}, {})
    launch = SimpleNamespace(t0=10., rising=False)
    monkeypatch.setattr(GamePlugin, "launch_distances", staticmethod(lambda *a: {}))
    monkeypatch.setattr(GamePlugin, "trace_distances", staticmethod(lambda *a: {"red0": (.1, 8.8)}))
    assert GamePlugin.release_time(ctx, launch, "red0") == 8.8


def test_one_tracked_entry_is_credited_once_in_a_dense_score_stream():
    ctx = _live_ctx()
    ctx.timeline = SimpleNamespace(deltas=lambda a: [(10 + i / 10, 1) for i in range(15)])
    # Team 1 has one observed make; team 2 supplies the other release evidence.
    times = np.array([7., 8., 9., 10., 11.])
    cols = (times, {"red1": np.array([0., 5., 10., 15., 20.])})
    log = [(8., 1, "red", 1., True, 9.8)]
    weights, events = Rebuilt2026().live_weights(ctx, "red", [1, 2], log, cols)
    assert weights[1]["auto"] == 1
    assert weights[2]["auto"] == 14
    assert sum(e["n"] for e in events if e["by"] == "tracked_entry") == 1


def test_legacy_shot_logs_remain_readable():
    ctx = _live_ctx()
    weights, _ = Rebuilt2026().live_weights(ctx, "red", [1, 2], [(8., 1, "red", 1., True)])
    assert weights[1]["auto"] == 15


def test_simultaneous_coral_credit_is_invariant_to_action_order():
    actions = [{"t": 10., "cat": "L4", "piece": "coral", "p": {1: .8, 2: .2}},
               {"t": 10., "cat": "L3", "piece": "coral", "p": {1: .6, 2: .4}},
               {"t": 10., "cat": "L2", "piece": "coral", "p": {1: .4, 2: .6}}]
    expected = Reefscape2025.assign_actions(actions, [1, 2])
    perm = [1, 0, 2]
    got = Reefscape2025.assign_actions([actions[i] for i in perm], [1, 2])
    for k, i in enumerate(perm):
        assert got[k] == pytest.approx(expected[i])


def test_simultaneous_coral_respects_single_piece_capacity():
    actions = [{"t": 10., "cat": "L4", "piece": "coral", "p": {1: .6, 2: .4}} for _ in range(3)]
    credits = Reefscape2025.assign_actions(actions, [1, 2])
    assert all(c == pytest.approx(credits[0]) for c in credits)
    assert credits[0][1] < .58  # repeated ownership is penalized symmetrically


def test_clear_placement_wins_over_event_prior():
    action = {"t": 10., "cat": "L4", "piece": "coral", "p": {1: .95, 2: .05}, "placed": {"t": 9.}}
    credit = Reefscape2025.assign_actions([action], [1, 2], {1: {"L4": .01}, 2: {"L4": 100}})[0]
    assert credit[1] > .85


def test_reef_evidence_follows_identity_at_observation_time():
    plugin = Reefscape2025()
    center = plugin.field.element_center("blue_reef")
    rows = [(t, center[0] + 1.5, center[1], True, [0, 0, 50, 50]) for t in (7., 7.5, 8., 8.5, 9.)]
    ctx = MatchContext(None, plugin.field, houston_cal(), None, None, {"blue0": rows},
                       {"blue0": 2}, {}, {}, identity_tl={"blue0": [(0, 9.5, 1), (9.5, 20, 2)]})
    near, dwell = plugin._evidence(ctx, "blue", [1, 2], 10., pre=3)
    assert near[1]["reef"] < 1 and dwell[1]["reef"] > 1
    assert near[2]["reef"] > 100 and dwell[2]["reef"] == 0
    assert plugin._seen(ctx, 1, 7., 10.) and not plugin._seen(ctx, 2, 7., 10.)


def test_net_launch_cannot_supply_processor_credit(monkeypatch):
    plugin = Reefscape2025()
    launch = SimpleNamespace(t0=9., rising=True)
    ctx = MatchContext(None, plugin.field, None, None, None, {}, {}, {}, {"algae": [launch]})
    monkeypatch.setattr(plugin, "_evidence", lambda *a: ({1: {"processor": .2}, 2: {"processor": .2}}, {}))
    monkeypatch.setattr(plugin, "_seen", lambda *a: True)
    monkeypatch.setattr(plugin, "attribute_launch", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("net evidence leaked")))
    credit, _ = plugin._credit(ctx, "blue", [1, 2], 10., "processor")
    assert credit == {1: .5, 2: .5}


def test_one_algae_launch_supports_only_one_net_post(monkeypatch):
    plugin = Reefscape2025()
    ctx = MatchContext(SimpleNamespace(teams=lambda a: [1, 2, 3]), plugin.field, None, None, None,
                       {}, {"blue0": 1}, {}, {})
    launch = Launch(9., 10., (100, 200), (200, 150), (150, 100), 1., 100., [])
    ctx.launches = {"algae": [launch]}
    net = plugin.field.element_center("blue_net_zone")
    monkeypatch.setattr(plugin, "attribute_launch", lambda *a, **kw: ("blue0", .1))
    monkeypatch.setattr(plugin, "release_time", lambda *a: 9.)
    monkeypatch.setattr(ctx, "robot_at", lambda *a: (*net, True, None))
    actions = [{"t": t, "cat": cat, "p": {1: .1, 2: .9}, "conf": .5}
               for t, cat in [(10., "net"), (11., "net"), (10., "processor")]]
    plugin._match_algae(ctx, "blue", actions)
    assert sum("launched" in A for A in actions) == 1
    assert actions[0]["p"][1] > .85
    assert actions[1]["p"] == {1: .1, 2: .9}
    assert actions[2]["p"] == {1: .1, 2: .9}


def test_analyze_binds_traced_shot_to_release_identity(monkeypatch):
    from autoscout.games.rebuilt2026 import PERIODS

    plugin = Rebuilt2026()
    teams = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    counts = {f"{p}Count": 1 if p == "auto" else 0 for p in PERIODS}
    match = SimpleNamespace(teams=lambda a: teams[a], breakdown={a: {"hubScore": counts} for a in teams})
    clock = SimpleNamespace(phase=lambda t: ("auto", 15 - t))
    launch = Launch(10., 11., (100, 200), (200, 250), (100, 200), 1., 0., [],
                    target="red_hub", rising=False, entry_t=11.)
    ctx = MatchContext(match, plugin.field, houston_cal(), clock, None, {}, {"red0": 2}, {},
                       {"fuel": [launch]}, identity_tl={"red0": [(0, 9, 1), (9, 20, 2)]})
    monkeypatch.setattr(plugin, "history_shooters", lambda *a: {})
    monkeypatch.setattr(plugin, "link_hub_flights", lambda *a: {id(launch): "red0"})
    monkeypatch.setattr(plugin, "release_time", lambda *a: 8.)
    monkeypatch.setattr(plugin, "column_activity", lambda *a: None)
    monkeypatch.setattr(ctx, "robot_at", lambda *a: (15., 4., True, None))
    result = plugin.analyze(ctx)
    assert result["teams"][1]["fuel_scored"] == 1
    assert result["teams"][2]["fuel_scored"] == 0
    assert launch.robot == 1 and launch.release_t == 8.
    assert result["events"][0]["t"] == 8.

    # Live score has 15 balls, but only one can be tied to this shooter. Final
    # reconciliation must not inflate that one tracked make back to 15 balls.
    counts["autoCount"] = 15
    ctx.clock.auto_end_vt, ctx.clock.teleop_end_vt = 20., 160.
    ctx.timeline = SimpleNamespace(deltas=lambda a: [(11 + i / 10, 1) for i in range(15)])
    result = plugin.analyze(ctx)
    assert result["teams"][1]["fuel_scored"] == 1
    assert result["alliances"]["red"]["unattributed_fuel"] == 14


def test_goal_entry_without_shooter_is_kept_for_review(monkeypatch):
    from autoscout.games.rebuilt2026 import PERIODS

    plugin = Rebuilt2026()
    counts = {f"{p}Count": 1 if p == "auto" else 0 for p in PERIODS}
    teams = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    match = SimpleNamespace(teams=lambda a: teams[a], breakdown={a: {"hubScore": counts} for a in teams})
    launch = Launch(10., 11., (100, 200), (200, 250), (100, 200), 1., 0., [],
                    target="red_hub", rising=False, entry_t=11.)
    ctx = MatchContext(match, plugin.field, houston_cal(), SimpleNamespace(phase=lambda t: ("auto", 15-t)),
                       None, {}, {}, {}, {"fuel": [launch]})
    monkeypatch.setattr(plugin, "history_shooters", lambda *a: {})
    monkeypatch.setattr(plugin, "link_hub_flights", lambda *a: {})
    monkeypatch.setattr(plugin, "column_activity", lambda *a: None)
    result = plugin.analyze(ctx)
    assert result["events"][0]["kind"] == "unattributed_shot"
    assert result["events"][0]["made"]


@pytest.mark.parametrize("owners, expected", [
    (["red0", "red0", "red0", "red1", None], ["red0", "red0", "red0", "red1", "red0"]),
    (["red0", None, None, None, None], ["red0", None, None, None, None]),
])
def test_volley_context_preserves_individual_evidence(monkeypatch, owners, expected):
    plugin = Rebuilt2026()
    launches = [Launch(i * .1, i * .1 + 1, (100., 200.), (200., 250.), (150., 150.), 1., 50., [], target="red_hub")
                for i in range(len(owners))]
    owner_map = {id(L): owner for L, owner in zip(launches, owners)}
    monkeypatch.setattr(plugin, "hub_shooter", lambda ctx, L, *args: owner_map[id(L)])
    links = plugin.link_hub_flights(None, launches, {}, lambda xs, ys: np.zeros(len(xs), bool))
    assert [links[id(L)] for L in launches] == expected

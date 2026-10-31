"""Tests for the attribution upgrades: pseudo-label filtering, release counting, person veto."""
import numpy as np

from autoscout import games
from autoscout.field import FieldModel
from autoscout.games.base import MatchContext
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

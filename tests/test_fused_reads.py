import numpy as np

from autoscout.vision import fused_reads
from autoscout.vision.tracker import RobotTracker, Tracklet


class FakeReader:
    """Every window of a view scores the given teams' log-probabilities."""

    def __init__(self, per_view):
        self.per_view = per_view

    def score(self, crops, candidates):
        row = np.array([self.per_view[int(c)] for c in candidates], float)
        return np.tile(row, (len(crops), 1)), np.zeros(len(crops))


def tracker_with(n=20, reads=None, colour="red", width=200):
    tr = RobotTracker(None, None, {"red": [111, 222, 333], "blue": [444, 555, 666]})
    box = (20, 20, 20 + width, 160)
    t = Tracklet(1, obs=[(i * 0.25, 1.0, 1.0, box, colour) for i in range(n)], reads=dict(reads or {}))
    tr.done = [t]
    return tr, t


def frame_at(_t):
    return np.zeros((300, 400, 3), np.uint8)


def test_weak_frames_that_agree_add_up_to_a_read():
    tr, t = tracker_with()
    # Each view leads by only 3: the live reader's confidence would stay low, but ten views agree.
    added = fused_reads.add_fused_reads(tr, frame_at, FakeReader({111: -1.0, 222: -4.0, 333: -4.0}))
    assert added == 1 and t.reads == {111: fused_reads.READ_WEIGHT}


def test_one_overwhelming_view_cannot_make_a_read_alone():
    class OneView:
        def score(self, crops, candidates):
            rows = np.full((len(crops), len(candidates)), -1.0)
            rows[: len(crops) // 2] = [0.0, -40.0, -40.0]  # the first of two views
            return rows, np.zeros(len(crops))

    frames = [(frame_at(0), (20, 20, 220, 160))] * 2
    scores = fused_reads.fused_scores(frames, [111, 222, 333], OneView())
    assert scores[111] - scores[222] == 2 * fused_reads.FRAME_CLIP < fused_reads.MARGIN


def test_a_clear_read_or_a_short_tracklet_is_left_alone():
    strong = FakeReader({111: 0.0, 222: -9.0, 333: -9.0})
    tr, t = tracker_with(reads={222: 1.5})
    assert fused_reads.add_fused_reads(tr, frame_at, strong) == 0 and t.reads == {222: 1.5}
    tr, t = tracker_with(n=fused_reads.MIN_OBS - 1)
    assert fused_reads.add_fused_reads(tr, frame_at, strong) == 0 and t.reads == {}


def test_a_weak_read_can_be_joined_by_a_fused_read():
    tr, t = tracker_with(reads={222: 0.4})
    fused_reads.add_fused_reads(tr, frame_at, FakeReader({111: 0.0, 222: -9.0, 333: -9.0}))
    assert t.reads == {222: 0.4, 111: fused_reads.READ_WEIGHT}


def test_candidates_follow_the_bumper_colour():
    tr, t = tracker_with(colour="blue")
    fused_reads.add_fused_reads(tr, frame_at, FakeReader({444: 0.0, 555: -9.0, 666: -9.0}))
    assert list(t.reads) == [444]


def test_narrow_boxes_are_not_views():
    _, t = tracker_with(width=fused_reads.MIN_BOX_W - 1)
    assert fused_reads.best_views(t) == []


def test_views_are_the_largest_and_spread_out():
    t = Tracklet(1, obs=[(0.0, 0, 0, (0, 0, 50, 50), "red"), (0.1, 0, 0, (0, 0, 40, 40), "red"),
                         (0.5, 0, 0, (0, 0, 30, 30), "red")])
    assert [o[0] for o in fused_reads.best_views(t)] == [0.0, 0.5]

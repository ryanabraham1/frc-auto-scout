"""Robots the detector only ever scores weakly must still be tracked, without inventing robots."""
import numpy as np
import pytest

from autoscout.vision import tracker as T
from autoscout.vision.detect import Det


class FakeTracker(T.RobotTracker):
    def __init__(self, bumper=True):
        self.candidates, self.bumper, self._next = [], bumper, 0
        self.added = []

    def _helper_ok(self, frame, d, foot):
        return self.bumper

    def _add(self, tr, t, d, p):
        tr.obs.append((t, float(p[0]), float(p[1]), [d.x0, d.y0, d.x1, d.y1], d.alliance))


def det(conf=0.1, alliance="blue"):
    return Det(100, 100, 180, 160, conf, alliance)


def sight(tr, t, spot, live, d=None):
    tr._confirm_weak(t, None, [d or det()], np.array([spot], float), set(), live)


def test_repeated_weak_boxes_start_a_robot():
    tr, live = FakeTracker(), []
    for i in range(T.CONFIRM_FRAMES):
        sight(tr, i * 0.1, (3.0 + 0.05 * i, 6.0), live)
    assert len(live) == 1 and len(live[0].obs) == T.CONFIRM_FRAMES


def test_too_few_sightings_start_nothing():
    tr, live = FakeTracker(), []
    for i in range(T.CONFIRM_FRAMES - 1):
        sight(tr, i * 0.1, (3.0, 6.0), live)
    assert live == []


def test_scattered_flickers_start_nothing():
    tr, live = FakeTracker(), []
    for i in range(12):
        sight(tr, i * 0.1, (1.0 + 2.0 * i, 6.0), live)
    assert live == []


def test_a_long_gap_starts_over():
    tr, live = FakeTracker(), []
    for t in (0, 0.1, 0.2, 0.2 + T.CONFIRM_WINDOW + 0.1):
        sight(tr, t, (3.0, 6.0), live)
    assert live == []


def test_a_box_without_bumper_colour_starts_nothing():
    tr, live = FakeTracker(bumper=False), []
    for i in range(10):
        sight(tr, i * 0.1, (3.0, 6.0), live)
    assert live == []


def test_alliances_do_not_merge_into_one_candidate():
    tr, live = FakeTracker(), []
    for i in range(6):
        sight(tr, i * 0.1, (3.0, 6.0), live, det(alliance="blue" if i % 2 else "red"))
    assert live == []


def test_a_weak_box_on_a_tracked_robot_is_not_a_second_robot():
    tr = FakeTracker()
    existing = T.Tracklet(99)
    existing.obs.append((0.0, 3.0, 6.0, [0, 0, 1, 1], "blue"))
    live = [existing]
    for i in range(10):
        sight(tr, i * 0.1, (3.2, 6.0), live)
    assert live == [existing]


def test_strong_boxes_are_left_to_the_normal_path():
    tr, live = FakeTracker(), []
    for i in range(10):
        sight(tr, i * 0.1, (3.0, 6.0), live, det(conf=0.8))
    assert live == [] and tr.candidates == []


@pytest.mark.parametrize("name", ["STRONG_CONF", "WEAK_CONF"])
def test_thresholds_stay_ordered(name):
    assert 0 < T.WEAK_CONF < T.STRONG_CONF < 1

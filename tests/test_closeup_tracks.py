from autoscout.vision.closeup_tracks import label, link


def _box(t, x, team=None, conf=0.0, view="blue_reef", al="blue"):
    return [t, view, x, 800, x + 100, 900, al, team, conf]


def test_unread_boxes_take_the_team_read_on_their_track():
    # a robot read at 10.0, bumper hidden while it lifts at 10.5-11.0 and drifting a little
    boxes = [_box(10.0, 300, 1351, 1.0), _box(10.5, 320), _box(11.0, 340),
             _box(10.0, 700, 6418, 1.0), _box(10.5, 690)]
    assert link(boxes)[0] == link(boxes)[2] != link(boxes)[3]
    assert label(boxes) == [1351, 1351, 1351, 6418, 6418]


def test_a_read_names_earlier_boxes_too():
    boxes = [_box(10.0, 300), _box(10.5, 310), _box(11.0, 320, 1351, 0.9)]
    assert label(boxes) == [1351, 1351, 1351]


def test_disagreeing_reads_either_side_leave_a_box_unnamed():
    boxes = [_box(10.0, 300, 1351, 1.0), _box(10.5, 300), _box(11.0, 300, 6418, 1.0)]
    assert label(boxes)[1] is None


def test_far_jumps_other_views_and_other_colours_do_not_link():
    boxes = [_box(10.0, 100, 1351, 1.0), _box(10.5, 700),
             _box(10.5, 110, view="red_reef"), _box(10.5, 105, al="red")]
    assert label(boxes)[1:] == [None, None, None]


def test_a_track_without_reads_takes_its_fused_wide_shot_robot():
    boxes = [_box(10.0, 300), _box(10.5, 310)]
    assert label(boxes, lambda i: 649 if i == 1 else None) == [649, 649]

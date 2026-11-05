import numpy as np
import pytest

from autoscout.field import Calibration, FieldModel
from autoscout.games.base import GamePlugin
from autoscout.games.rebuilt2026 import hub_active
from autoscout.games.reefscape2025 import Reefscape2025, TELE_VALUES, _decompose
from autoscout.overlay import MatchClock, ScoreSample, clean_scores, fit_clock
from autoscout.vision.identity import IdentityEvidence, digit_similarity
from autoscout.vision.trajectories import LaunchDetector


def houston_cal():
    return Calibration(event_key="t", year=2026, view=(0, 250, 1920, 470), points={
        "corner_blue_yW": (470, 407), "corner_red_yW": (1420, 407), "center_line_yW": (937, 405),
        "corner_blue_y0": (45, 678), "corner_red_y0": (1875, 678), "center_line_y0": (958, 678)})


def test_clean_scores_rejects_single_frame_misreads():
    raw = [0, 0, 4, 4, 4, 423, 4, 4, 9, None, 9, 9, 1, 9, 9]
    out = clean_scores(raw)
    assert out[-1] == 9 and max(out) == 9 and all(np.diff(out) >= 0)


def test_fit_clock():
    samples = []
    for t in np.arange(0, 200, 0.5):
        rem = None
        if 12 <= t <= 32:
            rem = int(np.ceil(32 - t))
        elif 37 <= t <= 177:
            rem = int(np.ceil(177 - t))
        samples.append(ScoreSample(t, rem, 0, 0))
    c = fit_clock(samples, 20, 140)
    assert abs(c.teleop_end_vt - 177) < 1.0 and abs(c.auto_end_vt - 32) < 1.0
    assert c.phase(100)[0] == "teleop" and c.phase(20)[0] == "auto"


def test_calibration_roundtrip():
    cal = houston_cal()
    fm = FieldModel.load(2026)
    p = cal.to_field([(937, 405)])[0]
    assert abs(p[0] - 8.27) < 0.15 and abs(p[1] - fm.width) < 0.15
    back = cal.to_image([(4.0, 2.0)])[0]
    assert np.allclose(cal.to_field([back])[0], (4.0, 2.0), atol=1e-6)


def test_proportional_split_is_exact():
    s = GamePlugin.proportional_split(17, {"a": 3.0, "b": 1.0, "c": 0.0})
    assert sum(s.values()) == 17 and s["a"] > s["b"] > s["c"] == 0
    s = GamePlugin.proportional_split(5, {"a": 0, "b": 0}, {"a": 1, "b": 1})
    assert sum(s.values()) == 5


def test_hub_shifts():
    # the auto winner is inactive in shifts 1 and 3
    assert hub_active("shift1", "red", "red") is False and hub_active("shift2", "red", "red") is True
    assert hub_active("shift1", "blue", "red") is True and hub_active("endgame", "red", "red") is True


def test_reefscape_decompose_and_counts():
    assert ["L4"] in _decompose(5, TELE_VALUES)
    assert ["L1", "L2", "L3"] in _decompose(9, TELE_VALUES)
    row = lambda n: {f"node{c}": i < n for i, c in enumerate("ABCDEFGHIJKL")}
    bd = {"autoReef": {"topRow": row(1), "midRow": row(0), "botRow": row(0), "trough": 1,
                       "tba_topRowCount": 1, "tba_midRowCount": 0, "tba_botRowCount": 0},
          "teleopReef": {"topRow": row(5), "midRow": row(2), "botRow": row(0), "trough": 2,
                         "tba_topRowCount": 5, "tba_midRowCount": 2, "tba_botRowCount": 0},
          "netAlgaeCount": 3, "wallAlgaeCount": 1}
    c = Reefscape2025().official_counts(bd)
    assert c["auto"]["L4"] == 1 and c["teleop"]["L4"] == 4 and c["teleop"]["net"] == 3


def test_identity_solver_uses_evidence_and_overrides():
    ev = IdentityEvidence()
    ev.add("red0", 254, 2.0); ev.add("red1", 1678, 2.0); ev.add("red2", 118, 1.0)
    m, conf = ev.solve(["red0", "red1", "red2"], [118, 254, 1678])
    assert m == {"red0": 254, "red1": 1678, "red2": 118} and min(conf.values()) > 0.5
    ev2 = IdentityEvidence(); ev2.add("red0", 254, 3.0)
    _, conf2 = ev2.solve(["red0", "red1", "red2"], [118, 254, 1678])
    assert conf2["red0"] > 0.9 and conf2["red1"] == 0.0
    m, _ = ev.solve(["red0", "red1", "red2"], [118, 254, 1678], {"red0": 118})
    assert m["red0"] == 118
    assert digit_similarity("254", 254) == 1.0 and digit_similarity("67", 1678) > 0.3


# ---- launches vs. pieces rolling on the carpet ---------------------------------------------
def _detector():
    cal = houston_cal()
    return LaunchDetector((18, 90, 110), (38, 255, 255), cal.view, cal=cal, field_model=FieldModel.load(2026))


def _run(det, pts, fps=30):
    for i, (x, y) in enumerate(pts):
        det._associate(i / fps, [(x - det.view[0], y - det.view[1], 30)])
    return det.finish()


def test_airborne_arc_is_a_launch():
    det = _detector()
    t = np.arange(0, 1.2, 1 / 30)
    # released at (900, 600) moving up/right, pulled down by "gravity" in the image
    pts = [(900 + 250 * s, 600 - 700 * s + 0.5 * 1100 * s * s) for s in t]
    launches = _run(det, pts)
    assert len(launches) == 1 and launches[0].rises > 100


def test_pushed_ball_moving_away_from_camera_is_not_a_launch():
    det = _detector()
    # a ball bulldozed toward the far side: it moves *up* the image at constant speed on the floor
    t = np.arange(0, 1.5, 1 / 30)
    pts = [(1000 + 20 * s, 640 - 120 * s) for s in t]  # ~2-3 m/s on the carpet
    assert _run(det, pts) == []
    assert det.rejected_ground == 1


def test_rolling_ball_sideways_is_not_a_launch():
    det = _detector()
    t = np.arange(0, 1.5, 1 / 30)
    pts = [(700 + 300 * s, 560 + 5 * np.sin(8 * s)) for s in t]
    assert _run(det, pts) == []


def test_fragmented_flight_counts_once():
    det = _detector()
    t = np.arange(0, 1.2, 1 / 30)
    pts = [(900 + 250 * s, 600 - 700 * s + 0.5 * 1100 * s * s) for s in t]
    for i, (x, y) in enumerate(pts):
        if 12 <= i < 18:  # occluded for 0.2 s (e.g. behind the hub)
            det._associate(i / 30, [])
            continue
        det._associate(i / 30, [(x - det.view[0], y - det.view[1], 30)])
    assert len(det.finish()) == 1


def test_score_jump_prefers_single_action():
    near = {1: {"reef": 0.0, "net": 3.0, "processor": 5.0}}
    assert Reefscape2025._explain(4, TELE_VALUES, near, {"foul_minor": 2}) == [(1, "L3")]
    near = {1: {"reef": 4.0, "net": 0.0, "processor": 5.0}}
    assert Reefscape2025._explain(4, TELE_VALUES, near) == [(1, "net")]


# ---- launch extraction v2 (offline, physics-validated) --------------------------------------
from autoscout.vision.launches import LaunchExtractor  # noqa: E402


def _extractor(targets=None):
    cal = houston_cal()
    view = (0, 0, 1920, 720)
    return LaunchExtractor(view, cal, FieldModel.load(2026), targets=targets)


def _blobs(pts, fps=30, d=None, ex=None):
    """Synthetic per-frame blobs for a list of image points (full piece-sized blobs)."""
    rows = []
    for i, (x, y) in enumerate(pts):
        dd = d or float(ex.ball_diam_px(np.array([x]), np.array([y]))[0])
        rows.append((i / fps, x, y, np.pi * (dd / 2) ** 2, dd * 1.4, dd))
    return np.array(rows)


def _arc(x0=900, y0=600, vx=250, vy=-700, g=1100, dur=1.2, fps=30):
    t = np.arange(0, dur, 1 / fps)
    return [(x0 + vx * s, y0 + vy * s + 0.5 * g * s * s) for s in t]


def test_v2_airborne_arc_is_one_launch():
    ex = _extractor()
    L = ex.extract(_blobs(_arc(), ex=ex))
    assert len(L) == 1 and L[0].n_pieces == 1


def test_v2_pushed_or_rolling_pieces_are_not_launches():
    ex = _extractor()
    t = np.arange(0, 1.5, 1 / 30)
    pushed = [(1000 + 20 * s, 640 - 120 * s) for s in t]  # bulldozed away from the camera
    rolling = [(700 + 300 * s, 560 + 3 * np.sin(9 * s)) for s in t]
    B = np.vstack([_blobs(pushed, ex=ex), _blobs(rolling, ex=ex)])
    assert ex.extract(B) == []


def test_v2_slivers_of_slow_pieces_are_filtered():
    ex = _extractor()
    B = _blobs(_arc(), ex=ex)
    B[:, 3] *= 0.15  # crescent-shaped partial blobs (a piece moving less than its size per frame)
    B[:, 5] *= 0.3
    assert ex.extract(B) == []


def test_v2_bounce_is_not_a_new_launch():
    ex = _extractor()
    fall = _arc(x0=700, y0=300, vx=150, vy=200, g=1100, dur=0.5)
    bounce_start = fall[-1]
    bounce = _arc(x0=bounce_start[0] + 3, y0=bounce_start[1], vx=120, vy=-350, g=1100, dur=0.5)
    B = np.vstack([_blobs(fall, ex=ex), _blobs(bounce, ex=ex)])
    B[len(fall):, 0] += len(fall) / 30 + 0.03  # bounce starts right after the landing
    L = ex.extract(B)
    assert len(L) <= 1


def test_v2_made_vs_lob_over_target():
    # a piece that vanishes at the opening is a make; one that flies past it is not
    t = np.arange(0, 2.0, 1 / 30)
    arc = [(600 + 300 * s, 600 - 900 * s + 0.5 * 1100 * s * s) for s in t]
    tgt_i = 45  # descending point on the arc, chosen as the "opening"
    target = {"blue_hub": (arc[tgt_i][0], arc[tgt_i][1], 25.0)}
    ex = _extractor(target)
    made = ex.extract(_blobs(arc[: tgt_i + 3], ex=ex))
    assert len(made) == 1 and made[0].target == "blue_hub"
    ex = _extractor(target)
    lob = ex.extract(_blobs(arc, ex=ex))
    assert len(lob) == 1 and lob[0].target is None


def _tracklet(tid, t0, t1, x0, x1, alliance, reads=None):
    from autoscout.vision.tracker import Tracklet

    ts = np.arange(t0, t1 + 1e-9, 0.25)
    xs = np.linspace(x0, x1, len(ts))
    return Tracklet(tid, obs=[(float(t), float(x), 4.0, (0, 0, 40, 40), alliance) for t, x in zip(ts, xs)],
                    reads=reads or {})


def test_robot_tracker_follows_robots_across_a_crossing():
    """Two red robots cross; tracklets are cut at the crossing. Motion continuity plus one bumper
    read per robot must keep each robot's pieces with its own team (no swap)."""
    from autoscout.vision.tracker import RobotTracker

    tr = RobotTracker(None, None, {"red": [111, 222, 333], "blue": [444, 555, 666]})
    tr.done = [_tracklet(0, 0, 5, 2.0, 7.0, "red", {111: 3.0}),   # A heading right
               _tracklet(1, 0, 5, 12.0, 7.6, "red", {222: 3.0}),  # B heading left
               _tracklet(2, 6, 10, 7.4, 12.0, "red"),             # A after crossing
               _tracklet(3, 6, 10, 7.0, 2.6, "red")]              # B after crossing
    tr.times = [float(t) for t in np.arange(0, 10.01, 0.25)]
    tr.relabel()
    assert [t.team for t in tr.done] == [111, 222, 111, 222]
    assert tr.slots["red0"].track[-1][1] == pytest.approx(12.0)  # team 111's track ends on the right


def _crossed_tracker(after_reads):
    """Two robots cross and the saved labels swap them afterwards."""
    from autoscout.vision.tracker import RobotTracker

    tr = RobotTracker(None, None, {"red": [111, 222, 333], "blue": [444, 555, 666]})
    tr.done = [_tracklet(0, 0, 5, 2.0, 7.0, "red", {111: 3.0}), _tracklet(1, 0, 5, 12.0, 7.6, "red", {222: 3.0}),
               _tracklet(2, 6, 10, 7.4, 12.0, "red", after_reads), _tracklet(3, 6, 10, 7.0, 2.6, "red")]
    for t, team in zip(tr.done, (111, 222, 222, 111)):
        t.team = team
    tr.times = [float(t) for t in np.arange(0, 10.01, 0.25)]
    return tr


def test_anchor_repair_honours_a_unanimous_read_the_labelling_broke():
    tr = _crossed_tracker({111: 2.0})
    tr.repair_saved()
    assert [t.team for t in tr.done] == [111, 222, 111, 222]
    assert tr.slots["red0"].track[-1][1] == pytest.approx(12.0)


def test_anchor_repair_leaves_labels_alone_when_the_read_has_dissent():
    tr = _crossed_tracker({111: 2.0, 222: 0.3})  # clear, but the box may have held both robots
    tr.repair_saved()
    assert [t.team for t in tr.done] == [111, 222, 222, 111]


def test_robot_tracker_manual_label_wins_and_round_trips():
    from autoscout.vision.tracker import NOT_ROBOT, RobotTracker

    tr = RobotTracker(None, None, {"red": [111, 222, 333], "blue": [444, 555, 666]})
    tr.done = [_tracklet(0, 0, 5, 2.0, 7.0, "red", {111: 3.0}), _tracklet(1, 0, 5, 9.0, 9.5, "blue")]
    tr.times = [0.0, 5.0]
    tr.relabel(manual={0: 222, 1: NOT_ROBOT})
    assert tr.done[0].team == 222 and tr.done[1].team is None
    back = RobotTracker.from_dict(tr.to_dict())
    assert [t.team for t in back.done] == [222, None] and back.done[0].reads == {111: 3.0}


def test_clean_scores_follows_a_score_climbing_ball_by_ball():
    """2026 HUB scores rise every read during a volley, so no value repeats; it must still be
    followed live (not held until it settles), while a one-off misread is still rejected."""
    raw = [0, 0, 3, 7, 12, 18, 25, 31, 38, 380, 44, 50, 50, 50]
    out = clean_scores(raw)
    assert out[3] == 7 and out[8] == 38 and out[-1] == 50
    assert max(out) == 50 and all(np.diff(out) >= 0)


def test_a_bumper_box_inside_a_robot_box_is_not_a_second_robot():
    from autoscout.vision.detect import Det
    from autoscout.vision.tracker import drop_part_boxes

    robot = Det(314, 386, 402, 482, .7, "blue")
    bumpers = Det(310, 453, 410, 485, .15, "blue")
    other_alliance = Det(310, 453, 410, 485, .5, "red")
    neighbour = Det(420, 386, 508, 482, .7, "blue")
    assert drop_part_boxes([robot, bumpers, neighbour]) == [robot, neighbour]
    assert drop_part_boxes([bumpers]) == [bumpers]
    assert other_alliance in drop_part_boxes([robot, other_alliance])

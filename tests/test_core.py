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

"""The 2025 benchmark's own arithmetic (scripts/bench2025.py): a wrong grader is worse than none."""
from types import SimpleNamespace

from scripts import bench2025 as b


def _match(red, blue, bd):
    return SimpleNamespace(red=red, blue=blue, breakdown=bd, teams=lambda a: red if a == "red" else blue)


def _team(auto=0, **kw):
    return {"auto_coral": auto, **kw}


def test_auto_coral_to_a_robot_that_never_left_is_counted_as_wrong():
    bd = {"red": {"autoLineRobot1": "Yes", "autoLineRobot2": "No", "autoLineRobot3": "Yes",
                  "endGameRobot1": "None", "endGameRobot2": "None", "endGameRobot3": "None"},
          "blue": {f"autoLineRobot{i}": "Yes" for i in (1, 2, 3)} | {f"endGameRobot{i}": "None" for i in (1, 2, 3)}}
    byk = {"m": _match([1, 2, 3], [4, 5, 6], bd)}
    facts = {t: {"cage": None, "moved": None} for t in range(1, 7)}
    res = {"m": {"teams": {1: _team(2), 2: _team(1), 3: _team(0), 4: _team(1), 5: _team(1), 6: _team(1)}, "facts": facts}}
    x = b.fms(res, byk)["auto_coral_to_non_leavers"]
    assert x["alliances"] == 1  # blue all left: nothing provable there
    assert x["wrong"] == 1 and x["auto_coral"] == 3


def test_barge_agreement_only_counts_alliances_that_can_tell_robots_apart():
    bd = {"red": {"endGameRobot1": "DeepCage", "endGameRobot2": "None", "endGameRobot3": "Parked"}
          | {f"autoLineRobot{i}": "Yes" for i in (1, 2, 3)},
          "blue": {f"endGameRobot{i}": "DeepCage" for i in (1, 2, 3)} | {f"autoLineRobot{i}": "Yes" for i in (1, 2, 3)}}
    byk = {"m": _match([1, 2, 3], [4, 5, 6], bd)}
    facts = {1: {"cage": True, "moved": True}, 2: {"cage": True, "moved": True}, 3: {"cage": None, "moved": True},
             4: {"cage": False, "moved": True}, 5: {"cage": False, "moved": True}, 6: {"cage": False, "moved": True}}
    res = {"m": {"teams": {t: _team() for t in range(1, 7)}, "facts": facts}}
    x = b.fms(res, byk)["barge"]
    assert x["robots"] == 2 and x["not_seen"] == 1  # blue's three identical results say nothing
    assert x["agree"] == 0.5  # team 1 at the cage as FMS says; team 2 seen there though FMS says it was not


def test_a_single_scorer_cell_pins_every_play_of_that_level():
    detail = [{"match_key": "m", "alliance": "red", "phase": "teleop", "kind": "L4", "official": 2,
               "scout_ours": {"1": (2.0, 1), "2": (0.0, 1), "3": (0.0, 0)}},
              {"match_key": "m", "alliance": "red", "phase": "teleop", "kind": "L3", "official": 2,
               "scout_ours": {"1": (1.0, 1), "2": (1.0, 1), "3": (0.0, 0)}}]  # two scorers: pins nothing
    node = {"timed": "coral", "at_spot": {"1": 0.9}}
    events = [{"alliance": "red", "kind": "L4", "phase": "teleop", "t": 50.0, "shares": {"1": 0.8, "2": 0.2}, "node": node},
              {"alliance": "red", "kind": "L4", "phase": "teleop", "t": 60.0, "shares": {"1": 0.3, "2": 0.7}},
              {"alliance": "red", "kind": "L3", "phase": "teleop", "t": 70.0, "shares": {"2": 1.0}},
              {"alliance": "blue", "kind": "L4", "phase": "teleop", "t": 80.0, "shares": {"4": 1.0}}]
    out = b.pinned_plays({"m": {"events": events}}, detail)
    assert out["all"]["plays"] == 2 and out["all"]["right"] == 1
    assert out["by_evidence"]["coral-timed, robot at spot"] == {"plays": 1, "right": 1, "mean_credit": 0.8}
    assert out["by_evidence"]["no branch"]["right"] == 0


def test_paired_bootstrap_sees_a_change_that_holds_in_every_match():
    a = {"per_match": {f"m{i}": {"n_L4": 10.0, "err_L4": 6.0} for i in range(12)}}
    better = {"per_match": {f"m{i}": {"n_L4": 10.0, "err_L4": 4.0} for i in range(12)}}
    d, lo, hi = b.bootstrap_diff(a, better, ("L4",), n=500)
    assert abs(d - 0.1) < 1e-9 and lo > 0  # (6 - 4) / (2 * 10) in every resample

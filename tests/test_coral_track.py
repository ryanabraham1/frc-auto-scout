"""Coral placements: persistence, branch choice and face credit."""
from types import SimpleNamespace

import numpy as np

from autoscout.games.reefscape2025 import Reefscape2025
from autoscout.vision import coral_track as C


def box(x, y, size=40):
    return [x - size / 2, y - size / 2, x + size / 2, y + size / 2, .9]


def frames(n=600, hz=5., extra=()):
    return [[i / hz, [box(*e[:2]) for e in extra if e[2] <= i / hz < e[3]]] for i in range(n)]


def test_a_resting_coral_is_found_when_it_appears_and_a_passing_one_is_not():
    obs = frames(extra=[(300, 800, 10, 200), (500, 800, 5, 12)])  # the second leaves long before the end
    found = C.rests(obs, "blue")
    assert len(found) == 1
    assert abs(found[0]["t"] - 10) < .3


def test_a_flicker_does_not_date_the_placement():
    obs = frames(extra=[(300, 800, 10, 200)])
    obs[10][1].append(box(300, 800))  # one early false box at t = 2
    assert abs(C.rests(obs, "blue")[0]["t"] - 10) < .3


def test_the_faces_of_a_level_share_out_by_fit():
    keys = [("A", "L4"), ("G", "L4"), ("C", "L4"), ("A", "L3")]
    faces = C._faces(keys, np.array([1., 1.2, np.inf, 0.]))["L4"]
    assert set(faces) == {"0", "3"}
    assert abs(sum(faces.values()) - 1) < .01


def plugin_ctx(positions):
    slots = {team: f"blue{i}" for i, team in enumerate(positions)}
    ctx = SimpleNamespace(aux_boxes=[], slot_of=lambda team, t: slots[team],
                          robot_at=lambda slot, t: next((*p, True) for tm, p in positions.items() if slots[tm] == slot))
    return ctx


def test_the_robot_in_front_of_the_coral_face_is_credited():
    plugin = Reefscape2025()
    # K/L face looks at 120 degrees. One robot parked there, one at another face, both next to the REEF.
    ctx = plugin_ctx({1: (4.1, 5.3), 2: (3.4, 2.9)})
    shares, fit = plugin._coral_credit(ctx, "blue", [1, 2], {"t": 10., "face": 5}, "L3")
    assert shares[1] > .9 and fit > .5


def test_an_ambiguous_coral_is_split_between_the_faces_it_could_be_on():
    plugin = Reefscape2025()
    ctx = plugin_ctx({1: (3.2, 4.0), 2: (5.8, 4.0)})  # at faces 0 and 3
    pl = {"t": 10., "face": 0, "faces": {"L4": {"0": .5, "3": .5}}}
    shares, _ = plugin._coral_credit(ctx, "blue", [1, 2], pl, "L4")
    assert abs(shares[1] - .5) < .05


def test_visit_start_follows_an_unbroken_run_of_sightings_only():
    samples = [(t, 1.) for t in (1., 5., 5.5, 6., 6.5)]
    assert Reefscape2025._visit_start(samples, 7.) == 5.


def test_a_robot_the_closeup_shows_elsewhere_loses_credit_for_a_coral():
    plugin = Reefscape2025.__new__(Reefscape2025)
    plugin.field = SimpleNamespace(dist_to_element=lambda p, e: 0.5)
    tracks = {1: (3.0, 4.0, True), 2: (3.0, 4.1, True)}  # both in front of the face in the wide shot
    ctx = SimpleNamespace(slot_of=lambda team, t: team, robot_at=lambda slot, t: tracks[slot],
                          aux_boxes=[(10.0, "blue_reef", 700, 800, 800, 900, "blue", 2, 1.0)], aux_reach=[0.0])
    pl = {"elem": "blue_reef@coral", "t": 10.0, "x": 300, "y": 800, "face": 0, "faces": {"L4": {"0": 1.0}}}
    shares, _ = plugin._coral_credit(ctx, "blue", [1, 2], pl, "L4")
    assert shares[1] > 2 * shares[2]

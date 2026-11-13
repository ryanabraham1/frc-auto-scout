"""3D camera solve from AprilTags, ballistic flight fit and REEF node timing: geometry checks on made-up scenes."""
import numpy as np
import pytest

from autoscout.vision import camera3d as c3
from autoscout.vision import flight3d as f3
from autoscout.vision import reef3d


def _camera(k1=0.0):
    """A broadcast-like camera: 20 m behind the y=W side at mid field, 6 m up, looking at the field centre."""
    import cv2

    centre = np.array([8.0, 20.0, 6.0])
    look = np.array([8.3, 4.0, 0.5]) - centre
    z = look / np.linalg.norm(look)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return c3.Camera(f=1330.0, cx=960.0, cy=330.0, k1=k1, rvec=cv2.Rodrigues(R)[0].ravel(), tvec=-R @ centre,
                     size=(1920, 1080))


def test_tag_layouts_match_the_field():
    tags26, size = c3.tag_layout(2026)
    assert len(tags26) == 32 and size == pytest.approx(0.1651)
    # tag 26 is on the blue HUB (centre x 4.63), tag 10 on the red one: blue-origin frame, not WPILib's centred one
    assert 3.0 < tags26[26][0] < 5.0 and 11.0 < tags26[10][0] < 13.0
    tags25, _ = c3.tag_layout(2025)
    assert tags25[18][:3] == pytest.approx((3.6576, 4.0259, 0.3081), abs=1e-3)  # blue REEF, facing its driver station


def test_solve_recovers_a_camera_from_tags_seen_across_the_field():
    cam = _camera()
    ids = [2, 11, 21, 24, 13, 14, 15, 16, 31, 32]
    obs = {i: cam.project(c3.tag_corners(2026, i)) for i in ids}
    got = c3.solve_best(obs, (1920, 1080), 2026, centre=(960, 330))
    assert got.rms < 0.05
    assert np.linalg.norm(got.centre - cam.centre) < 0.05
    assert got.f == pytest.approx(cam.f, rel=0.01)


def test_floor_inverts_project_and_shift_moves_the_picture():
    cam = _camera()
    p = np.array([[3.0, 2.5, 0.0], [12.0, 6.0, 0.0]])
    assert cam.floor(cam.project(p)) == pytest.approx(p[:, :2], abs=1e-6)
    high = cam.floor(cam.project([[3.0, 2.5, 1.2]]), z=1.2)
    assert high[0] == pytest.approx([3.0, 2.5], abs=1e-6)
    moved = cam.shifted(-30.0, 12.0)
    assert moved.project(p) == pytest.approx(cam.project(p) + [-30.0, 12.0], abs=1e-6)


def test_hub_front_tags_alone_do_not_pin_the_camera_down():
    """Why wall tags are needed at events whose wide shot only decodes the HUB fronts: those tags all lie
    in one plane, and a pixel of noise then moves the solved camera by metres."""
    cam = _camera()
    rng = np.random.default_rng(1)
    far = []
    for _ in range(4):
        obs = {i: cam.project(c3.tag_corners(2026, i)) + rng.normal(0, 0.7, (4, 2)) for i in (2, 11, 21, 24)}
        got = c3.solve_from_tags(obs, (1920, 1080), 2026, fit_k1=False, centre=(960, 330))
        far.append(np.linalg.norm(got.centre - cam.centre))
    assert max(far) > 1.0


def _shot(cam, src, hub, T=1.1, fps=60, noise=1.0, seed=0, first=0.05):
    v = (hub - src) / T
    v[2] += 0.5 * f3.G * T
    v *= 1.06  # make up for drag
    ts = np.arange(first, T * 1.05, 1 / fps)
    P, V = f3.integrate(src, v, ts, f3.DRAG["fuel"])
    end = np.nonzero((V[:, 2] < 0) & (P[:, 2] < hub[2]))[0]
    n = end[0] + 1 if len(end) else len(ts)
    uv = cam.project(P[:n]) + np.random.default_rng(seed).normal(0, noise, (n, 2))
    return np.c_[50.0 + ts[:n], uv], P[n - 1]


def test_flight_fit_finds_where_a_ball_left_the_floor():
    cam = _camera()
    hub = np.array([4.625, 4.035, 1.83])
    for seed, src in enumerate(([1.5, 2.0, f3.RELEASE_Z], [3.0, 7.0, f3.RELEASE_Z], [0.8, 4.5, f3.RELEASE_Z])):
        pts, last = _shot(cam, np.array(src), hub, seed=seed)
        f = f3.fit(cam, pts, anchor=last[:2])
        assert f is not None and f.ok
        assert np.linalg.norm(f.release_xy - src[:2]) < 0.35
        assert f.release_t == pytest.approx(50.0, abs=0.08)
        assert f3.release_distance(f, src[:2]) < 3.0  # the true spot is inside the error ellipse
        assert f3.release_distance(f, np.array(src[:2]) + [0.0, 2.5]) > 3.0  # a robot 2.5 m away is not


def test_two_robots_in_line_with_the_camera_are_told_apart():
    """The case flat image linking cannot solve: the ball rises past the nearer robot's box on screen,
    but it left the floor at the farther one."""
    cam = _camera()
    hub = np.array([4.625, 4.035, 1.83])
    far_robot = np.array([2.0, 2.0])
    ray = (far_robot - cam.centre[:2]) / np.linalg.norm(far_robot - cam.centre[:2])
    near_robot = far_robot - 2.2 * ray  # same line of sight, 2.2 m nearer the camera
    pts, last = _shot(cam, np.r_[far_robot, f3.RELEASE_Z], hub)
    f = f3.fit(cam, pts, anchor=last[:2])
    assert f3.release_distance(f, far_robot) < f3.release_distance(f, near_robot) - 2.0


def test_integrate_runs_backward_to_where_it_came_from():
    p0, v0 = np.array([2.0, 3.0, 0.6]), np.array([3.0, 1.0, 7.0])
    P, V = f3.integrate(p0, v0, [0.5], f3.DRAG["fuel"])
    back, _ = f3.integrate(P[0], V[0], [-0.5], f3.DRAG["fuel"])
    assert back[0] == pytest.approx(p0, abs=1e-4)


def _breakdown(auto, tele):
    def rows(nodes):
        out = {row: {f"node{b}": False for b in reef3d.LETTERS} for row in reef3d.ROWS.values()}
        for b, level in nodes:
            out[reef3d.ROWS[level]][f"node{b}"] = True
        return out

    return {"autoReef": rows(auto), "teleopReef": rows(auto + tele)}


def test_official_nodes_split_auto_from_teleop():
    nodes = reef3d.official_nodes(_breakdown([("J", "L4")], [("A", "L4"), ("B", "L2")]))
    assert {(n["branch"], n["level"], n["phase"]) for n in nodes} == {
        ("J", "L4", "auto"), ("A", "L4", "teleop"), ("B", "L2", "teleop")}


def test_scores_pair_with_the_branch_seen_filling_just_before():
    nodes = [{"branch": "A", "level": "L4", "phase": "teleop", "t": 60.0},
             {"branch": "B", "level": "L4", "phase": "teleop", "t": 90.0},
             {"branch": "E", "level": "L4", "phase": "teleop"},  # nobody saw it fill
             {"branch": "C", "level": "L3", "phase": "teleop", "t": 61.0}]
    jumps = [{"t": 91.2, "level": "L4", "phase": "teleop"}, {"t": 61.5, "level": "L4", "phase": "teleop"},
             {"t": 120.0, "level": "L4", "phase": "teleop"}, {"t": 62.5, "level": "L3", "phase": "teleop"}]
    pairs = reef3d.pair_jumps(nodes, jumps)
    assert pairs == {0: 1, 1: 0, 2: 2, 3: 3}


def test_a_coral_that_comes_to_rest_and_stays_is_timed():
    obs = []
    for k in range(200):
        t = k * 0.2
        boxes = [[100, 100, 110, 130, 0.9]] if t >= 12.0 else []
        if 5.0 <= t < 5.6:
            boxes.append([300, 300, 310, 330, 0.9])  # carried past: never rests
        obs.append([t, boxes])
    got = reef3d.rests(obs, lambda x, y: True)
    assert len(got) == 1 and got[0]["t"] == pytest.approx(12.0, abs=0.01) and got[0]["size"] == 30


def test_scoring_spot_is_in_front_of_the_branch_face():
    from autoscout.vision import field3d as F

    centre = np.array(F.REEF_CENTRE["blue"])
    a, b = reef3d.node_pose("blue", "A"), reef3d.node_pose("blue", "B")
    assert a[0] < centre[0] and b[0] < centre[0]  # face 0 looks at the blue driver stations (x = 0)
    assert np.linalg.norm(a - b) == pytest.approx(2 * F.BRANCH_HALF, abs=1e-6)
    assert np.linalg.norm((a + b) / 2 - centre) == pytest.approx(reef3d.ROBOT_REACH, abs=1e-6)

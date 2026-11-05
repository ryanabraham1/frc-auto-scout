"""Score the REEFs from vision alone at the end of a 2025 match and compare with the official breakdown.

For each REEF close-up: solve its camera from the AprilTags it sees (vision/field3d.py), detect CORAL with
the CORAL detector, put each one on the nearest branch tip (A-L, L2-L4) or the L1 trough, then compare
with FMS: which branches hold CORAL (per level), the L1 count, and the CORAL points they add up to
(TELEOP values; penalties, ALGAE and AUTO bonuses ignored). Far faces are hidden from a close-up, so
branches it can't see are reported separately rather than counted as misses.

usage: python scripts/reef_state_check.py WEIGHTS KEY [KEY ...] [--out DIR] [--conf 0.3]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout.overlay import MatchClock  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, VIDEO_DIR  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402
from autoscout.video import Video  # noqa: E402
from autoscout.vision import field3d as F  # noqa: E402

PTS = {"L1": 2, "L2": 3, "L3": 4, "L4": 5}
CORAL_LEN = 0.30  # m
CROP_DETECT = False
ROWS = {"L4": "topRow", "L3": "midRow", "L2": "botRow"}
LETTERS = "ABCDEFGHIJKL"


def detect_tags(view_img):
    d = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                cv2.aruco.DetectorParameters())
    g = cv2.resize(cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cs, ids, _ = d.detectMarkers(g)
    return {int(i): c[0] / 2 for i, c in zip(ids.ravel(), cs) if int(i) in F.TAGS} if ids is not None else {}


RED_TAGS, BLUE_TAGS = {1, 2, 6, 7, 8, 9, 10, 11}, {12, 13, 17, 18, 19, 20, 21, 22}


def panels(frame, y0=645, y1=1015):
    """Close-up panels in the bottom band, split at the broadcast's coloured separators (two panels during
    the match, three around the end: blue REEF, barge, red REEF)."""
    band = cv2.cvtColor(frame[y0 + 15:y1 - 15], cv2.COLOR_BGR2HSV)
    sat = ((band[..., 1] > 120) & (band[..., 2] > 120)).mean(0)
    cols = np.nonzero(sat > 0.85)[0]
    seps = [int(g.mean()) for g in np.split(cols, np.nonzero(np.diff(cols) > 3)[0] + 1) if 0 < len(g) <= 12] if len(cols) else []
    edges = [0] + seps + [frame.shape[1]]
    return [(a + 4, y0, b - a - 8, y1 - y0) for a, b in zip(edges, edges[1:]) if b - a > 200]


def focal_lengths(v, t0, t1, step=3.0) -> dict[str, float]:
    """Each close-up's focal length (two-panel layout px), from frames where it shows 3+ tags (median)."""
    fs = {"blue": [], "red": []}
    for t in np.arange(t0, t1, step):
        f = v.frame_at(float(t))
        ps = panels(f) if f is not None else []
        if len(ps) != 2:
            continue
        for al, (x, y, w, h) in zip(("blue", "red"), ps):
            obs = detect_tags(f[y:y + h, x:x + w])
            if len(obs) >= 3:
                cam = F.solve_camera(obs, (w, h))
                if cam and cam[3] < 3.0:
                    fs[al].append(cam[0][0, 0])
    out = {al: float(np.median(v)) for al, v in fs.items() if v}
    if len(out) == 1:  # the two REEF close-ups are the same camera model, set up symmetrically
        out = dict.fromkeys(("blue", "red"), next(iter(out.values())))
    return out


def saved_cameras(event: str) -> dict:
    """Event close-up cameras solved once by hand (configs/events/EVENT_reef_cams.json): {alliance: camera}."""
    from autoscout.config import ROOT

    p = ROOT / "configs" / "events" / f"{event}_reef_cams.json"
    out = {}
    if p.exists():
        for al, c in json.loads(p.read_text()).items():
            if al in ("blue", "red"):
                w, h = c["size"]
                K = np.array([[c["f"], 0, w / 2], [0, c["f"], h / 2], [0, 0, 1.0]])
                out[al] = (K, np.array(c["rvec"]).reshape(3, 1), np.array(c["tvec"]).reshape(3, 1), 0.0)
    return out


def last_two_panel(v, t_end, span=40.0, step=0.5):
    """(time, {alliance: panel rect}) of the last normal two-panel frame before t_end."""
    for t in np.arange(t_end, t_end - span, -step):
        f = v.frame_at(float(t))
        ps = panels(f) if f is not None else []
        if len(ps) == 2:
            return float(t), dict(zip(("blue", "red"), ps))
    return None


def cameras_near(v, t_end, focal, span=8.0, step=0.5):
    """{alliance: (camera, eval time, tags, panel)} from the normal two-panel layout (blue REEF left, red
    right) in the span before t_end: each tag's corners are pooled over those frames (the close-up cameras
    hold still for long stretches, and one frame rarely shows enough tags), then the camera is solved."""
    pooled = {"blue": {}, "red": {}}
    last = {}
    for t in np.arange(t_end, t_end - span, -step):
        f = v.frame_at(float(t))
        if f is None:
            continue
        ps = panels(f)
        if len(ps) != 2:
            continue
        for al, rect in zip(("blue", "red"), ps):
            x, y, w, h = rect
            last.setdefault(al, (float(t), rect))
            if rect != last[al][1]:
                continue
            for i, c in detect_tags(f[y:y + h, x:x + w]).items():
                pooled[al].setdefault(i, []).append(c)
    out = {}
    for al, tags in pooled.items():
        mine = RED_TAGS if al == "red" else BLUE_TAGS
        obs = {i: np.median(np.array(c), 0) for i, c in tags.items() if i in mine or len(c) >= 2}
        if al not in last or len(obs) < 2:
            continue
        t, rect = last[al]
        if len(obs) < 3 and al not in focal:
            continue
        cam = F.solve_camera(obs, rect[2:], f=focal.get(al) if len(obs) < 3 else None)
        if cam is not None and cam[3] <= 6.0:
            out[al] = (cam, t, sorted(obs), rect, None)
    return out


def match_view(src: np.ndarray, dst: np.ndarray):
    """Homography taking src pixels to dst pixels for two framings of the same camera feed (the end-of-match
    panel is the close-up shrunk and cropped), from ORB matches; None if they don't agree."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 20:
        return None
    pa = np.float32([ka[i.queryIdx].pt for i in m])
    pb = np.float32([kb[i.trainIdx].pt for i in m])
    # the panel is the same feed scaled and cropped: a similarity, far steadier than a full homography
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return np.vstack([M, [0, 0, 1.0]]) if M is not None and inl.sum() >= 15 else None


def reef_state(model, frame, rect, cam, alliance, conf, to_cam=None, cam_rect=None):
    """{(letter, level): seen} for visible tips, the L1 count, and the drawn view. to_cam maps this panel's
    pixels into the pixels the camera was solved in (cam_size) when the broadcast reframed the close-up."""
    x, y, w, h = rect
    cw, ch = cam_rect[2:] if cam_rect else (w, h)
    covered = np.ones((ch, cw), bool)
    if to_cam is not None:
        # warp the reframed panel back to the solved close-up framing, so CORAL show at the size the
        # detector was trained on, then work in that framing
        warped = cv2.warpPerspective(frame[y:y + h, x:x + w], to_cam, (cw, ch))
        covered = cv2.warpPerspective(np.ones((h, w), np.uint8), to_cam, (cw, ch)) > 0
        covered = cv2.erode(covered.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        # paste it where that close-up sits in the two-panel layout (it fits there)
        x, y = cam_rect[:2]
        canvas = np.zeros_like(frame)
        canvas[y:y + ch, x:x + cw] = warped
        frame, (w, h) = canvas, (cw, ch)
    to_cam = np.eye(3)
    view = frame[y:y + h, x:x + w].copy()
    tips = F.branch_tips(alliance)
    keys = list(tips)
    P = F.project(cam, [tips[k] for k in keys])  # camera view pixels
    # which tips face the camera (a far-side branch is hidden behind the REEF)
    K, rv, tv, _ = cam
    R, _ = cv2.Rodrigues(rv)
    cpos = (-R.T @ tv).ravel()
    vis = {}
    for k, p in zip(keys, P):
        a = np.radians(F.face_normal_deg(alliance, LETTERS.index(k[0]) // 2))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        # the REEF is an open frame of pipes: a far face's branches still show through and over it (its L4
        # CORAL plainly), so only L1 (the trough sits behind the solid base) needs the face toward the camera
        vis[k] = bool(0 <= p[0] < cw and 0 <= p[1] < ch and (k[1] != "L1" or n @ (cpos - tips[k]) > 0))
    # detect on the full frame (the scale it was trained at), keep this panel's boxes in panel pixels
    if CROP_DETECT:  # the close-up alone, at the training scale (a 948 px panel at imgsz 960)
        boxes = model.predict(np.ascontiguousarray(frame[y:y + h, x:x + w]), imgsz=960, conf=conf, verbose=False)[0].boxes.xyxy.tolist()
    else:
        boxes = [(bx0 - x, by0 - y, bx1 - x, by1 - y)
                 for bx0, by0, bx1, by1 in model.predict(frame, imgsz=1920, conf=conf, verbose=False)[0].boxes.xyxy.tolist()
                 if x <= (bx0 + bx1) / 2 < x + w and y <= (by0 + by1) / 2 < y + h]
    scale = float(np.sqrt(abs(np.linalg.det(to_cam[:2, :2]))))
    depth = {k: float(np.linalg.norm(tips[k] - cpos)) for k in keys}
    f_px = float(K[0, 0])
    # Near and far faces' tips can land on the same pixels (a camera about level with L4): a CORAL's size
    # says how far it is (CORAL_LEN at that depth), and each branch holds one, so assign jointly.
    dets = []
    for bx0, by0, bx1, by1 in boxes:
        c = cv2.perspectiveTransform(np.float64([[[(bx0 + bx1) / 2, (by0 + by1) / 2]]]), to_cam)[0, 0]
        dets.append((c, max(bx1 - bx0, by1 - by0) * scale, (bx0, by0, bx1, by1)))
    cand = [k for k in keys if vis[k]]
    occupied, l1 = set(), 0
    if dets and cand:
        from scipy.optimize import linear_sum_assignment

        big = 1e6
        # L1 (the trough) holds many: give it one column per detection
        cols = [k for k in cand if k[1] != "L1"] + [k for k in cand if k[1] == "L1" for _ in dets]
        C = np.full((len(dets), len(cols)), big)
        for i, (c, size, _) in enumerate(dets):
            for j, k in enumerate(cols):
                d = float(np.hypot(*(P[keys.index(k)] - c)))
                if d > max(30.0, 1.2 * size):
                    continue
                exp = f_px * CORAL_LEN / depth[k]
                C[i, j] = (d / 15.0) ** 2 + (np.log(max(size, 1.0) / exp) / 0.3) ** 2
        rows, cs = linear_sum_assignment(C)
        for i, j in zip(rows, cs):
            if C[i, j] >= big:
                continue
            k = cols[j]
            if k[1] == "L1":
                l1 += 1
            else:
                occupied.add(k)
            bx0, by0, bx1, by1 = dets[i][2]
            cv2.rectangle(view, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 255, 0), 2)
            cv2.putText(view, f"{k[0]}{k[1][1]}", (int(bx0), int(by0) - 3), 0, 0.5, (0, 255, 0), 1)
    Pv = cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(to_cam)).reshape(-1, 2)
    for k, p in zip(keys, Pv):
        if vis[k]:
            cv2.circle(view, (int(p[0]), int(p[1])), 3, (0, 0, 255) if k[1] == "L4" else (255, 255, 0), -1)
    cov = {k for k, p in zip(keys, P) if 0 <= p[0] < cw and 0 <= p[1] < ch and covered[int(p[1]), int(p[0])]}
    return occupied, l1, vis, view, cov


class HostedModel:
    """A Roboflow Universe model run through Roboflow's hosted API (weights "roboflow:PROJECT/VERSION"), with
    the same predict() shape as an ultralytics model; only its CORAL class is kept. Needs ROBOFLOW_API_KEY."""

    def __init__(self, model_id: str):
        import os

        from inference_sdk import InferenceHTTPClient

        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                                          api_key=os.environ["ROBOFLOW_API_KEY"])

    def predict(self, img, imgsz=None, conf=0.25, verbose=False):
        from types import SimpleNamespace

        res = self.client.infer(img, model_id=self.model_id)
        xyxy = [[p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, p["x"] + p["width"] / 2, p["y"] + p["height"] / 2]
                for p in res.get("predictions", []) if p["class"].lower() == "coral" and p["confidence"] >= conf]
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=SimpleNamespace(tolist=lambda b=xyxy: b)))]


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights")
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("data/reef_state_check"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--at", choices=("auto", "end"), default="end", help="end of AUTO or end of the match")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = HostedModel(a.weights.split(":", 1)[1]) if a.weights.startswith("roboflow:") else YOLO(a.weights)
    tba = TBA()
    tot = {"tp": 0, "fp": 0, "fn": 0, "hidden": 0, "pts_vis": 0, "pts_off": 0}
    for key in a.keys:
        r = json.loads((RESULTS_DIR / f"{key}.json").read_text())
        clk = MatchClock(**r["clock"])
        v = Video(str(next(Path(VIDEO_DIR).glob(f"{r['video']}.*"))))
        bd = MatchInfo.from_tba(tba.match(key)).breakdown
        focal = focal_lengths(v, clk.start_vt, clk.teleop_end_vt)
        # the close-up cameras hold still for a whole match: pool every tag sighting, evaluate at the target
        # (end of AUTO, or the last two-panel frame before the buzzer)
        t_eval = clk.auto_end_vt + 1.5 if a.at == "auto" else clk.teleop_end_vt - 0.3
        whole = cameras_near(v, clk.teleop_end_vt, focal, span=clk.teleop_end_vt - clk.start_vt, step=2.0)
        saved = saved_cameras(r["event_key"])
        last = last_two_panel(v, t_eval)
        cams = {}
        for al in ("blue", "red"):
            cam = whole[al][0] if al in whole else saved.get(al)
            if cam is not None and last is not None:
                cams[al] = (cam, last[0], whole[al][2] if al in whole else "saved", last[1][al], None)
        for al in ("blue", "red"):
            if al not in cams:
                print(f"{key} {al}: no camera solve (too few tags)")
                continue
            cam, t, tags, rect, _ = cams[al]
            frame, to_cam, prect = v.frame_at(t), None, rect
            if a.at == "end":
                # score the very end: the broadcast's end-of-match layout shrinks the close-ups, so map that
                # panel onto the two-panel framing the camera was solved in
                fe = v.frame_at(t_eval)
                ps = panels(fe)
                pe = (ps[0] if al == "blue" else ps[-1]) if len(ps) >= 2 else None
                if pe is not None:
                    ref = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    H = match_view(fe[pe[1]:pe[1] + pe[3], pe[0]:pe[0] + pe[2]], ref)
                    if H is not None:
                        frame, to_cam, prect, t = fe, H, pe, t_eval
            occ, l1, vis, img, cov = reef_state(model, frame, prect, cam, al, a.conf, to_cam, rect)
            if to_cam is not None and len(cov) < len(vis):
                # the end-of-match panel crops the close-up: branches it can't show keep their state from the
                # last full close-up frame (scored CORAL stay put)
                occ0, l10, _, _, _ = reef_state(model, v.frame_at(last[0]), rect, cam, al, a.conf)
                occ = {k for k in occ if k in cov} | {k for k in occ0 if k not in cov}
                l1 = max(l1, l10)
            cv2.imwrite(str(a.out / f"{key}_{al}_{a.at}.jpg"), img)
            tr = bd[al]["autoReef" if a.at == "auto" else "teleopReef"]
            truth = {(L, lvl) for lvl, row in ROWS.items() for L in LETTERS if tr[row][f"node{L}"]}
            seen_keys = {(k[0], k[1]) for k in vis if vis[k] and k[1] != "L1"}
            tp = len(occ & truth)
            fp = len(occ - truth)
            fn = len((truth & seen_keys) - occ)
            hidden = len(truth - seen_keys)
            pts_v = sum(PTS[k[1]] for k in occ) + PTS["L1"] * l1
            pts_o = sum(PTS[k[1]] for k in truth) + PTS["L1"] * tr["trough"]
            print(f"{key} {al} ({a.at}, frame {t - clk.teleop_end_vt if a.at == 'end' else t - clk.auto_end_vt:+.1f}s): camera {cam[3]:.1f} px from tags {tags} | branches right {tp}, wrong {fp}, "
                  f"missed {fn}, hidden {hidden} of {len(truth)} | L1 seen {l1} vs official {tr['trough']} | "
                  f"coral points vision {pts_v} vs official {pts_o}")
            for k, n in (("tp", tp), ("fp", fp), ("fn", fn), ("hidden", hidden), ("pts_vis", pts_v), ("pts_off", pts_o)):
                tot[k] += n
    if tot["tp"] + tot["fp"]:
        print(f"TOTAL branch precision {tot['tp'] / (tot['tp'] + tot['fp']):.2f}, recall of visible "
              f"{tot['tp'] / max(tot['tp'] + tot['fn'], 1):.2f}, hidden {tot['hidden']}; "
              f"points vision {tot['pts_vis']} vs official {tot['pts_off']}")


if __name__ == "__main__":
    main()

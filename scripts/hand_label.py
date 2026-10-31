"""Hand labels for the robot detector, from one click per robot.

A person marks the centre of every robot in a frame as [x, y, "r"|"b"] (wide-shot crop pixels, like
scripts/det_eval.py). Each mark becomes a box: the current detector's box if one contains the mark
(tight already), else SAM2 prompted with a robot-sized box around the mark (robot width from the
event calibration), so marking is quick and the boxes stay tight.

  python scripts/hand_label.py extract MATCH T1,T2,...    # crops + gridded sheets in data/hand/
  python scripts/hand_label.py build                      # marks -> YOLO dataset data/datasets/hand
  python scripts/hand_label.py check OUT.jpg [glob]       # boxes drawn on frames, for review

Marks live in data/hand/points.json as {stem: [[x, y, "r"|"b"], ...]}.
"""
import json
import sys

import cv2
import numpy as np

from autoscout import config
from autoscout.field import Calibration
from autoscout.tba import TBA, MatchInfo
from autoscout.video import Video, download

DIR = config.DATA_DIR / "hand"
OUT = config.DATA_DIR / "datasets" / "hand"
CLASSES = ["red_robot", "blue_robot"]


def extract(match_key, times):
    DIR.mkdir(parents=True, exist_ok=True)
    m = MatchInfo.from_tba(TBA().match(match_key))
    v = Video(download(m.videos[0], height=1080))
    cal = Calibration.load(m.event_key).scaled((v.width, v.height))
    x, y, w, h = cal.view
    for t in times:
        f = v.frame_at(t)
        if f is None:
            continue
        crop = f[y: y + h, x: x + w]
        stem = f"{match_key}_{t:g}".replace(".", "p")
        cv2.imwrite(str(DIR / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = crop.copy()
        for gx in range(0, w, 100):
            cv2.line(g, (gx, 0), (gx, h), (0, 220, 220), 1)
            cv2.putText(g, str(gx), (gx + 2, 14), 0, 0.45, (0, 255, 255), 1)
        for gy in range(0, h, 50):
            cv2.line(g, (0, gy), (w, gy), (0, 150, 150) if gy % 100 else (0, 220, 220), 1)
            cv2.putText(g, str(gy), (2, gy + 14), 0, 0.45, (0, 255, 255), 1)
        cv2.putText(g, stem, (w - 420, h - 12), 0, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(DIR / f"{stem}_grid.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(stem, crop.shape)


def _robot_px(cal_by_event, stem, x, y):
    """Expected robot width (px) at crop point (x, y), from the event calibration."""
    ev = stem.split("_")[0]
    cal = cal_by_event.get(ev)
    if cal is None:
        return 90.0
    X, Y = x + cal.view[0], y + cal.view[1]
    f = cal.to_field([(X, Y)])[0]
    return float(np.clip(1.0 * cal.px_per_meter_x(f), 30, 260))


def build():
    from ultralytics import SAM

    from autoscout.vision.detect import alliance_from_pixels, default_detector

    pts = json.loads((DIR / "points.json").read_text())
    det = default_detector(conf=0.15)
    sam = SAM("sam2.1_b.pt")
    cals = {}
    for stem in pts:
        ev = stem.split("_")[0]
        if ev not in cals:
            c = Calibration.load(ev)
            cals[ev] = c.scaled((1920, 1080)) if c else None
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    boxes_all = {}
    for stem, P in pts.items():
        img = cv2.imread(str(DIR / f"{stem}.jpg"))
        H, W = img.shape[:2]
        dets = det(img)
        boxes = []
        for x, y, a in P:
            inside = [d for d in dets if d.x0 <= x <= d.x1 and d.y0 <= y <= d.y1]
            if inside:  # the detector's own box around this mark is already tight
                d = min(inside, key=lambda d: d.w * d.h)
                boxes.append((a, d.x0, d.y0, d.x1, d.y1, "det"))
                continue
            rw = _robot_px(cals, stem, x, y)
            prompt = [max(0, x - 0.75 * rw), max(0, y - 0.65 * rw), min(W, x + 0.75 * rw), min(H, y + 0.55 * rw)]
            r = sam(img, bboxes=[prompt], verbose=False)[0]
            bx = None
            if r.masks is not None and len(r.masks.data):
                m = r.masks.data[0].cpu().numpy() > 0
                ys, xs = np.nonzero(m)
                if len(xs) > 30:
                    sy, sx = H / m.shape[0], W / m.shape[1]
                    bx = (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)
            ok = bx is not None and 0.35 * rw < bx[2] - bx[0] < 1.8 * rw and bx[0] <= x <= bx[2] and bx[1] <= y <= bx[3]
            if not ok:  # SAM grabbed a ball pile or a field element: fall back to a robot-sized box
                bx = (x - 0.5 * rw, y - 0.45 * rw, x + 0.5 * rw, y + 0.4 * rw)
            boxes.append((a, *bx, "sam" if ok else "prior"))
        boxes_all[stem] = boxes
        cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for a, x0, y0, x1, y1, _src in boxes:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            c = 0 if a == "r" else 1
            lines.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    (DIR / "boxes.json").write_text(json.dumps(boxes_all))
    n = sum(len(b) for b in boxes_all.values())
    src = {}
    for b in boxes_all.values():
        for x in b:
            src[x[-1]] = src.get(x[-1], 0) + 1
    print(f"{len(boxes_all)} frames, {n} robots, box sources {src}")


def check(out, pattern="*"):
    import fnmatch

    boxes = json.loads((DIR / "boxes.json").read_text())
    tiles = []
    for stem, B in boxes.items():
        if not fnmatch.fnmatch(stem, pattern):
            continue
        img = cv2.imread(str(DIR / f"{stem}.jpg"))
        for a, x0, y0, x1, y1, src in B:
            col = (0, 0, 255) if a == "r" else (255, 0, 0)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 3)
            cv2.putText(img, src[0], (int(x0) + 3, int(y0) + 20), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(img, stem, (10, 30), 0, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1]))))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i: i + 2]) for i in range(0, len(tiles), 2)]))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], [float(t) for t in sys.argv[3].split(",")])
    elif sys.argv[1] == "build":
        build()
    else:
        check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "*")

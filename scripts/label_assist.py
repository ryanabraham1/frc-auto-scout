"""Generate numbered robot-box proposals for fast manual labeling.

Usage: python scripts/label_assist.py EVENT VIDEO OUT_DIR t1 t2 ...
Writes OUT_DIR/<vid>_<t>.jpg (view crop with numbered proposals) and a .json of the boxes.
Proposals come from the current robot detector at low confidence plus foreground blobs vs.
a median background (non game-piece colored, robot-sized).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout.field import Calibration, FieldModel  # noqa: E402
from autoscout.video import Video  # noqa: E402
from autoscout.vision.background import foreground_mask, median_background  # noqa: E402
from autoscout.vision.detect import Det, RobotDetector, nms_merge  # noqa: E402

event, video_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
times = [float(t) for t in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
cal = Calibration.load(event)
fm = FieldModel.load(cal.year)
v = Video(video_path)
cal = cal.scaled((v.width, v.height))
x, y, w, h = cal.view
bg = median_background(v, cal.view, 10, v.duration - 20, n=40)
det = RobotDetector(conf=0.05)
poly = cal.field_polygon_image(fm, margin=0.4) - [x, y]
fmask = np.zeros((h, w), np.uint8)
cv2.fillPoly(fmask, [poly], 1)
for t in times:
    f = v.frame_at(t)[y: y + h, x: x + w]
    props = [d for d in det(f)]
    fg = foreground_mask(f, bg) & fmask
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    piece = np.zeros_like(fg)
    for gp in fm.game_pieces.values():
        lo, hi = gp["color_hsv"]
        piece |= (cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0).astype(np.uint8)
    fg = fg & (1 - piece)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(fg)
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        ppm = cal.px_per_meter((bx + bw / 2 + x, by + bh + y))
        if 0.35 * ppm < bw < 2.2 * ppm and 0.25 * ppm < bh < 2.0 * ppm and a > 0.15 * bw * bh:
            props.append(Det(bx, by, bx + bw, by + bh, 0.01, "?"))
    props = nms_merge(props, 0.6)
    vis = f.copy()
    for k, d in enumerate(props):
        col = (0, 0, 255) if d.alliance == "red" else (255, 0, 0) if d.alliance == "blue" else (0, 255, 0)
        cv2.rectangle(vis, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), col, 2)
        cv2.putText(vis, str(k), (int(d.x0), int(d.y0) - 3), 0, 0.7, (255, 255, 255), 2)
    stem = f"{Path(video_path).stem}_{t:.1f}"
    cv2.imwrite(str(out / f"{stem}.jpg"), vis)
    cv2.imwrite(str(out / f"{stem}_raw.jpg"), f)
    json.dump([[float(d.x0), float(d.y0), float(d.x1), float(d.y1), d.alliance] for d in props], open(out / f"{stem}.json", "w"))
    print(stem, len(props))

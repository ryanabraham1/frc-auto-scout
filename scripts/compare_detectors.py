"""Side-by-side robot detections of two weight files on frames (for eyeballing generalization)."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout.field import Calibration  # noqa: E402
from autoscout.video import Video  # noqa: E402
from autoscout.vision.detect import RobotDetector, alliance_from_pixels  # noqa: E402

out, w1, w2, *items = sys.argv[1:]
dets = [RobotDetector(w1, conf=0.3), RobotDetector(w2, conf=0.3)]
rows = []
for it in items:
    video, event, t = it.split(":")
    cal = Calibration.load(event)
    x, y, w, h = cal.view
    im = Video(video).frame_at(float(t))[y: y + h, x: x + w]
    pair = []
    for d in dets:
        vis = im.copy()
        for b in d(im):
            a = b.alliance or alliance_from_pixels(im, b)[0]
            col = (0, 0, 255) if a == "red" else (255, 0, 0) if a == "blue" else (0, 255, 0)
            cv2.rectangle(vis, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), col, 3)
        pair.append(cv2.resize(vis, (960, int(960 * h / w))))
    rows.append(np.hstack(pair))
cv2.imwrite(out, np.vstack(rows))

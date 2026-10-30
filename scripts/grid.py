"""Render a video frame with a labelled pixel grid (helper for manual calibration/labeling)."""
import sys

import cv2

sys.path.insert(0, ".")
from autoscout.video import Video  # noqa: E402

video, t, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
x0, y0, x1, y1 = (map(int, sys.argv[4].split(",")) if len(sys.argv) > 4 else (0, 0, 1920, 1080))
step = int(sys.argv[5]) if len(sys.argv) > 5 else 100
im = Video(video).frame_at(t)[y0:y1, x0:x1].copy()
for x in range((x0 // step + 1) * step, x1, step):
    cv2.line(im, (x - x0, 0), (x - x0, im.shape[0]), (0, 255, 0), 1)
    cv2.putText(im, str(x), (x - x0 + 2, 14), 0, 0.45, (0, 255, 0), 1)
for y in range((y0 // step + 1) * step, y1, step):
    cv2.line(im, (0, y - y0), (im.shape[1], y - y0), (0, 255, 255), 1)
    cv2.putText(im, str(y), (2, y - y0 - 2), 0, 0.45, (0, 255, 255), 1)
cv2.imwrite(out, im)

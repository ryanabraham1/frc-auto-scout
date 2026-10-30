"""Overlay a calibration's projected field lines/zones on a frame for visual checking."""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout.field import Calibration, FieldModel  # noqa: E402
from autoscout.video import Video  # noqa: E402

event, video, t, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
cal = Calibration.load(event)
fm = FieldModel.load(cal.year)
im = Video(video).frame_at(t)
err = cal.fit(fm)
for x in np.arange(0, fm.length + 0.01, 1.0):
    a, b = cal.to_image([(x, 0), (x, fm.width)]).astype(int)
    cv2.line(im, tuple(a), tuple(b), (80, 80, 80), 1)
for name, (x0, y0, x1, y1) in fm.zones.items():
    poly = cal.to_image([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).astype(np.int32)
    col = (255, 80, 0) if "blue" in name else (0, 0, 255) if "red" in name else (0, 255, 0)
    cv2.polylines(im, [poly], True, col, 2)
for name, e in fm.elements.items():
    c = cal.to_image([e["center"]])[0].astype(int)
    cv2.circle(im, tuple(c), 6, (0, 255, 255), -1)
    cv2.putText(im, name, (c[0] + 5, c[1]), 0, 0.5, (0, 255, 255), 1)
for n, p in cal.points.items():
    cv2.drawMarker(im, (int(p[0]), int(p[1])), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
x, y, w, h = cal.view
cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 1)
print("reprojection error (m):", round(err, 3))
cv2.imwrite(out, im[y: y + h, x: x + w])

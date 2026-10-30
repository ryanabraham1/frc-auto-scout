"""Write hand-labelled robot boxes (full-frame pixels) from a YAML file into the YOLO dataset.

YAML: list of {video, event, t, robots: [[alliance, x0, y0, x1, y1], ...]}
Each frame is cropped to the event's wide-shot view, like at inference time.
"""
import sys

import yaml

sys.path.insert(0, ".")
from autoscout.field import Calibration  # noqa: E402
from autoscout.train.sam_label import write_label  # noqa: E402
from autoscout.video import Video  # noqa: E402

for fr in yaml.safe_load(open(sys.argv[1])):
    cal = Calibration.load(fr["event"])
    x, y, w, h = cal.view
    img = Video(fr["video"]).frame_at(fr["t"])[y: y + h, x: x + w]
    boxes = [(a, x0 - x, y0 - y, x1 - x, y1 - y) for a, x0, y0, x1, y1 in fr["robots"]]
    stem = f"{fr['video'].split('/')[-1].split('.')[0]}_m{fr['t']:.0f}"
    write_label(img, boxes, stem)
    print(stem, len(boxes))

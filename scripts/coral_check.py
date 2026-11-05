"""Is the CORAL detector good enough? Check a checkpoint on matches it never trained on.

1. Frames with its detections drawn (wide shot and both REEF close-ups) -> OUT/KEY_T.jpg
2. CORAL on each REEF at the end of the match, counted in that REEF's close-up, vs the official
   breakdown (every CORAL scored is still there: branches L2-L4 plus the L1 trough).

usage: python scripts/coral_check.py WEIGHTS OUT KEY [KEY ...] [--conf 0.25]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")
from autoscout.field import Calibration  # noqa: E402
from autoscout.games.reefscape2025 import Reefscape2025  # noqa: E402
from autoscout.overlay import MatchClock  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, VIDEO_DIR  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402
from autoscout.video import Video  # noqa: E402


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights")
    ap.add_argument("out", type=Path)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    tba = TBA()
    for key in a.keys:
        r = json.loads((RESULTS_DIR / f"{key}.json").read_text())
        base = Calibration.load(r["event_key"])
        clk = MatchClock(**r["clock"])
        v = Video(str(next(Path(VIDEO_DIR).glob(f"{r['video']}.*"))))
        bd = MatchInfo.from_tba(tba.match(key)).breakdown
        for t in (clk.start_vt + 30, clk.start_vt + 80, clk.teleop_end_vt + 1.0):
            f = v.frame_at(t)
            boxes = model.predict(f, imgsz=1920, conf=a.conf, verbose=False)[0].boxes
            img = f.copy()
            for (x0, y0, x1, y1), c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.imwrite(str(a.out / f"{key}_{t:.0f}.jpg"), img[200:1080])
            if t >= clk.teleop_end_vt:
                for name, (x0, y0, x1, y1) in (base.aux_watch or {}).items():
                    al = name.split("_")[0]
                    n = sum(1 for (bx0, by0, bx1, by1) in boxes.xyxy.tolist()
                            if x0 <= (bx0 + bx1) / 2 <= x1 and y0 <= (by0 + by1) / 2 <= y1)
                    c = Reefscape2025.official_counts(None, bd[al])
                    off = sum(c[ph][k] for ph in c for k in ("L1", "L2", "L3", "L4"))
                    print(f"{key} {al} reef at end: detected {n} coral in its close-up, official {off}")
    print("frames in", a.out)


if __name__ == "__main__":
    main()

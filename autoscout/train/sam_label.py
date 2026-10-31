"""Data engine: turn a few seed boxes into many labeled frames with SAM2 video tracking.

Seed file (YAML) format, one entry per segment:
  - video: data/videos1080/OTOfavQlQXc.mp4
    event: 2026txhou          # for the wide-shot view crop
    t: 40.0                   # seed time (s)
    duration: 40              # seconds to propagate forward
    fps: 2
    robots:                   # full-frame pixel boxes at time t
      - [blue, 388, 465, 468, 578]
      - [red, 1135, 455, 1195, 510]

Output: YOLO dataset under data/datasets/robots/{images,labels}. Class 0 = red_robot, 1 = blue_robot.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .. import config
from ..field import Calibration

DATASET = config.DATA_DIR / "datasets" / "robots"
CLASSES = ["red_robot", "blue_robot"]


def write_label(img: np.ndarray, boxes: list[tuple[str, float, float, float, float]], stem: str,
                dataset: Path | None = None) -> None:
    dataset = DATASET if dataset is None else Path(dataset)
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "labels").mkdir(parents=True, exist_ok=True)
    H, W = img.shape[:2]
    cv2.imwrite(str(dataset / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for a, x0, y0, x1, y1 in boxes:
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        lines.append(f"{CLASSES.index(a + '_robot')} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    (dataset / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def propagate(seg: dict, model: str = "sam2.1_t.pt") -> int:
    from ultralytics.models.sam import SAM2VideoPredictor

    cal = Calibration.load(seg["event"])
    x, y, w, h = cal.view if cal else (0, 0, 1920, 1080)
    stem0 = f"{Path(seg['video']).stem}_{seg['t']:.0f}"
    tmp = config.DATA_DIR / "tmp" / f"{stem0}.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fps = seg.get("fps", 2)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(seg["t"]), "-t", str(seg.get("duration", 30)),
                    "-i", seg["video"], "-vf", f"crop={w}:{h}:{x}:{y},fps={fps}", "-an", str(tmp)], check=True)
    alliances = [r[0] for r in seg["robots"]]
    seeds = [[r[1] - x, r[2] - y, r[3] - x, r[4] - y] for r in seg["robots"]]
    pred = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                             model=model, verbose=False, save=False))
    n = 0
    cap = cv2.VideoCapture(str(tmp))
    for i, r in enumerate(pred(source=str(tmp), bboxes=seeds, stream=True)):
        ok, img = cap.read()
        if not ok:
            break
        boxes = []
        if r.masks is not None:
            for k, m in enumerate(r.masks.data.cpu().numpy()):
                if k >= len(alliances):
                    break
                ys, xs = np.nonzero(m)
                if len(xs) < 30:
                    continue
                sy, sx = img.shape[0] / m.shape[0], img.shape[1] / m.shape[1]
                boxes.append((alliances[k], xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy))
        write_label(img, boxes, f"{stem0}_{i:04d}")
        n += 1
    return n


def main(seed_file: str) -> None:
    segs = yaml.safe_load(Path(seed_file).read_text())
    for seg in segs:
        n = propagate(seg)
        print(f"{seg['video']} @ {seg['t']}: {n} frames labeled", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

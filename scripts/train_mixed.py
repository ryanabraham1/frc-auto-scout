"""Train the robot detector on hand labels (data/datasets/hand, repeated) plus pseudo labels.

usage: python scripts/train_mixed.py NAME [--base models/autoscout-robots.pt] [--repeat 6] [--epochs 40]
Writes data/models/experiments/NAME.pt; evaluate with scripts/det_eval.py before using it.
"""
import argparse
import shutil
from pathlib import Path

from autoscout import config
from autoscout.train.sam_label import CLASSES, DATASET
from autoscout.train.train_robots import held_out

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--base", default=str(config.ROOT / "models" / "autoscout-robots.pt"))
ap.add_argument("--repeat", type=int, default=6)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=1280)
a = ap.parse_args()

hold = held_out()
hand = sorted((config.DATA_DIR / "datasets" / "hand" / "images").glob("*.jpg"))
auto = [p for p in sorted((DATASET / "images").glob("*.jpg")) if not any(p.stem.startswith(m + "_") for m in hold)]
hand = [p for p in hand if not any(p.stem.startswith(m + "_") for m in hold)]
work = config.DATA_DIR / "datasets" / "mixed"
work.mkdir(parents=True, exist_ok=True)
train = [str(p) for p in auto] + [str(p) for p in hand] * a.repeat
(work / "train.txt").write_text("\n".join(train))
(work / "val.txt").write_text("\n".join(str(p) for p in hand))  # monitored only; det_eval decides
(work / "data.yaml").write_text(f"path: {work}\ntrain: train.txt\nval: val.txt\nnames: {dict(enumerate(CLASSES))}\n")
print(f"{len(auto)} pseudo-labelled + {len(hand)} hand frames x{a.repeat}")

from ultralytics import YOLO  # noqa: E402

dev = "0" if config.device() == "cuda" else config.device()
res = YOLO(a.base).train(data=str(work / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=8, device=dev,
                         project=str(config.DATA_DIR / "runs"), name=a.name, exist_ok=True, mosaic=0.5, fliplr=0.5,
                         hsv_h=0.0, scale=0.4, close_mosaic=8, cos_lr=True, patience=15, workers=4, plots=False,
                         verbose=False)
out = config.DATA_DIR / "models" / "experiments" / f"{a.name}.pt"
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(Path(res.save_dir) / "weights" / "best.pt", out)
print("weights:", out)

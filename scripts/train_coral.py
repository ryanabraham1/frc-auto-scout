"""Train the 2025 CORAL detector on data/datasets/coral (autoscout/train/coral_label.py).

Splits by match (never a match in both train and val), trains one class at full broadcast width so a
CORAL in the wide shot (~10 x 28 px) survives, and saves to models/autoscout-coral.pt.

usage: python scripts/train_coral.py [--base yolo11s.pt] [--epochs 40] [--val-matches 4] [--batch 4]
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402
from autoscout.train.coral_label import DATASET  # noqa: E402


def split(val_n: int, seed: int = 0, dataset: Path = DATASET) -> Path:
    imgs = sorted((dataset / "images").glob("*.jpg"))
    match = lambda p: re.match(r"(.+_qm\d+)_", p.stem)[1]  # noqa: E731
    keys = sorted({match(p) for p in imgs})
    random.Random(seed).shuffle(keys)
    val = set(keys[:val_n])
    root = dataset / "split"
    shutil.rmtree(root, ignore_errors=True)
    for part in ("train", "val"):
        (root / part).mkdir(parents=True)
        (root / part / "list.txt").write_text("\n".join(str(p.resolve()) for p in imgs if (match(p) in val) == (part == "val")))
    y = root / "coral.yaml"
    y.write_text(yaml.safe_dump({"path": str(dataset.resolve()), "train": str((root / "train" / "list.txt").resolve()),
                                 "val": str((root / "val" / "list.txt").resolve()), "names": {0: "coral"}}))
    print("val matches:", sorted(val), "train images:", sum(1 for p in imgs if match(p) not in val))
    return y


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(config.ROOT / "yolo11s.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-matches", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--name", default="run", help="run folder under data/models/coral")
    ap.add_argument("--out", type=Path, default=config.ROOT / "models" / "autoscout-coral.pt")
    a = ap.parse_args()
    data = split(a.val_matches, dataset=a.dataset)
    res = YOLO(a.base).train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, rect=False,
                             project=str(config.DATA_DIR / "models" / "coral"), name=a.name, exist_ok=True,
                             mosaic=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.3, workers=2, patience=15, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, a.out)
    print("weights:", a.out)


if __name__ == "__main__":
    main()

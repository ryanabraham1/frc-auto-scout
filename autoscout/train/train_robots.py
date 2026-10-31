"""Train / fine-tune the robot detector (YOLO) on data/datasets/robots."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .. import config
from .sam_label import CLASSES, DATASET


def held_out() -> set[str]:
    """Matches used by scripts/det_eval.py: never train on them, or the evaluation flatters."""
    p = config.DATA_DIR / "det_eval" / "points.json"
    if not p.exists():
        return set()
    return {k.rsplit("_", 1)[0] for k in json.loads(p.read_text())}


def split(val_frac: float = 0.15, seed: int = 0) -> Path:
    hold = held_out()
    imgs = [p for p in sorted((DATASET / "images").glob("*.jpg"))
            if not any(p.stem.startswith(m + "_") for m in hold)]
    # split by source segment so validation frames are not near-duplicates of training frames
    groups: dict[str, list[Path]] = {}
    for p in imgs:
        groups.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # tiny datasets (bootstrapping): validate on the training set rather than starve training
    n_val = max(1, int(len(keys) * val_frac)) if len(imgs) >= 60 and len(keys) > 2 else 0
    val = {p for k in keys[:n_val] for p in groups[k]}
    for name in ("train.txt", "val.txt"):
        (DATASET / name).unlink(missing_ok=True)
    (DATASET / "train.txt").write_text("\n".join(str(p) for p in imgs if p not in val or not val))
    (DATASET / "val.txt").write_text("\n".join(str(p) for p in (val or imgs[:10])))
    y = DATASET / "data.yaml"
    y.write_text(f"path: {DATASET}\ntrain: train.txt\nval: val.txt\nnames: {dict(enumerate(CLASSES))}\n")
    return y


def train(base: str | None = None, epochs: int = 60, imgsz: int = 1280, batch: int = 8, device: str | None = None) -> Path:
    """Fine-tune from COCO-pretrained YOLO11s (a bigger, better-initialised backbone than the
    Spy Glass nano model) on the pseudo-labelled broadcast crops (see train/pseudo_label.py)."""
    from ultralytics import YOLO

    data = split()
    base = base or "yolo11s.pt"
    model = YOLO(base)
    dev = device or {"cuda": "0", "mps": "mps"}.get(config.device(), "cpu")
    res = model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=dev,
                      project=str(config.DATA_DIR / "runs"), name="robots", exist_ok=True,
                      mosaic=0.5, fliplr=0.5, hsv_h=0.0,  # keep hue: red vs blue matters
                      scale=0.4, close_mosaic=10, cos_lr=True,
                      patience=20, workers=4, plots=False, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    out = config.DATA_DIR / "models" / "autoscout-robots.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    if config.device() == "cuda":  # TensorRT FP16 for the RTX 3060
        try:
            eng = YOLO(str(out)).export(format="engine", half=True, imgsz=imgsz)
            shutil.copy(eng, config.DATA_DIR / "models" / "autoscout-robots.engine")
        except Exception as e:  # TensorRT optional
            print("TensorRT export skipped:", e)
    return out

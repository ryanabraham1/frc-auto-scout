"""Train / fine-tune the robot detector (YOLO) on data/datasets/robots."""
from __future__ import annotations

import json
import random
import shutil
import re
from datetime import datetime
from pathlib import Path

import yaml

from .. import config
from .sam_label import CLASSES, DATASET


def held_out() -> set[str]:
    """Matches used by scripts/det_eval.py: never train on them, or the evaluation flatters."""
    p = config.DATA_DIR / "det_eval" / "points.json"
    if not p.exists():
        return set()
    return {k.rsplit("_", 1)[0] for k in json.loads(p.read_text())}


def source_group(path: Path) -> str:
    """All clips and hand labels from a match belong in the same partition."""
    match = re.match(r"(20\d{2}[a-z0-9]+_(?:qm\d+|[qs]f\d+m\d+|f\d+m\d+))(?:_|$)", path.stem)
    if match:
        return match[1]
    # Dashboard labels use the video id; keep all timestamps from it together.
    dashboard = re.match(r"^(.*)_u\d+(?:_\d+)?$", path.stem)
    if dashboard:
        return dashboard[1]
    return path.stem.rsplit("_", 1)[0]


def source_aliases() -> dict[str, str]:
    """Map dashboard video IDs to matches using local API cache, without network access."""
    by_video: dict[str, set[str]] = {}
    for path in (config.DATA_DIR / "tba_cache").glob("*.json"):
        if not (path.name.startswith("match__") or path.name.endswith("__matches.json")):
            continue
        try:
            raw = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        for match in raw if isinstance(raw, list) else [raw]:
            if not isinstance(match, dict) or "key" not in match:
                continue
            for video in match.get("videos") or []:
                if video.get("type") == "youtube":
                    by_video.setdefault(video["key"], set()).add(match["key"])
    aliases = {}
    for video, matches in by_video.items():
        canonical = next(iter(matches)) if len(matches) == 1 else video
        aliases[video] = canonical
        aliases.update({m: canonical for m in matches})
    return aliases


def split(val_frac: float = 0.15, seed: int = 0, repeat: int = 6, hand_only: bool = False) -> Path:
    if not 0 < val_frac < 1 or repeat < 1:
        raise ValueError("validation fraction must be between 0 and 1; repeat must be positive")
    hold = held_out()
    aliases = source_aliases()
    held_groups = {aliases.get(m, m) for m in hold}
    hand_dir = config.DATA_DIR / "datasets" / "hand"
    imgs = []
    for directory in ((hand_dir,) if hand_only else (DATASET, hand_dir)):
        for p in sorted((directory / "images").glob("*.jpg")):
            if aliases.get(source_group(p), source_group(p)) in held_groups:
                continue
            if not (directory / "labels" / f"{p.stem}.txt").exists():
                raise ValueError(f"Missing label file for {p.name}")
            imgs.append(p.resolve())
    if not imgs:
        raise ValueError("No training frames. Save hand labels in the dashboard first.")
    groups: dict[str, list[Path]] = {}
    for p in imgs:
        group = aliases.get(source_group(p), source_group(p))
        groups.setdefault(group, []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("Label at least two different matches/videos for independent validation.")
    # Prefer independently corrected frames for validation where available.
    hand_groups = [k for k in keys if any(p.parent.parent == hand_dir.resolve() for p in groups[k])]
    keys = hand_groups + [k for k in keys if k not in hand_groups]
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_frac)))
    if hand_groups:
        n_val = min(n_val, max(1, len(hand_groups) // 3))
    val = {p for k in keys[:n_val] for p in groups[k]}
    # A validation match's pseudo labels must also stay out of training.
    train_imgs = [p for p in imgs if p not in val]
    train = [p for p in train_imgs for _ in range(repeat if p.parent.parent == hand_dir.resolve() else 1)]
    val_hand = [p for p in val if p.parent.parent == hand_dir.resolve()]
    work = config.DATA_DIR / "datasets" / "mixed"
    work.mkdir(parents=True, exist_ok=True)
    (work / "train.txt").write_text("\n".join(p.as_posix() for p in train))
    (work / "val.txt").write_text("\n".join(p.as_posix() for p in sorted(val_hand or val)))
    (work / "split.json").write_text(json.dumps(dict(train_groups=sorted(set(groups) - set(keys[:n_val])),
        val_groups=keys[:n_val], held_out=sorted(hold), unique_train_frames=len(train_imgs),
        train_examples=len(train), validation_frames=len(val_hand or val), hand_repeat=repeat), indent=2))
    y = work / "data.yaml"
    y.write_text(yaml.safe_dump(dict(path=work.resolve().as_posix(), train="train.txt", val="val.txt",
                                    names=dict(enumerate(CLASSES)))))
    return y


def promote(candidate: Path) -> dict:
    """Benchmark a candidate against the deployed weights before replacing them."""
    from ..vision.benchmark import evaluate_model, improves
    from ..vision.detect import RobotDetector, PersonVeto, find_weights

    directory = config.DATA_DIR / "det_eval"
    report = {"candidate": str(candidate), "promoted": False}
    if not (directory / "points.json").exists():
        report["reason"] = "No held-out benchmark; candidate saved for evaluation."
        return report
    # Compare portable weights, even if deployment currently uses TensorRT.
    baseline = find_weights()
    if baseline and baseline.suffix == ".engine":
        baseline = baseline.with_suffix(".pt")
    veto = PersonVeto()
    report["candidate_score"] = evaluate_model(RobotDetector(candidate, conf=0.3), directory, veto)
    if baseline and baseline.exists():
        report["baseline"] = str(baseline)
        report["baseline_score"] = evaluate_model(RobotDetector(baseline, conf=0.3), directory, veto)
        if not improves(report["candidate_score"], report["baseline_score"]):
            report["reason"] = "No gain without precision/recall regression; deployed weights retained."
            return report
    else:
        report["reason"] = "No portable baseline to compare; candidate saved for evaluation."
        return report
    out = config.DATA_DIR / "models" / "autoscout-robots.pt"
    if out.exists():
        shutil.copy2(out, candidate.parent / "previous-deployed.pt")
    temp = out.with_suffix(".tmp")
    shutil.copy2(candidate, temp)
    temp.replace(out)
    # An old engine otherwise takes priority over the newly deployed model.
    engine = out.with_suffix(".engine")
    if engine.exists():
        engine.replace(candidate.parent / "previous-deployed.engine")
    report.update(promoted=True, reason="Improved held-out F1 with no precision/recall regression.")
    return report


def train(base: str | None = None, epochs: int = 60, imgsz: int = 1280, batch: int = 8,
          device: str | None = None, name: str | None = None, repeat: int = 6,
          hand_only: bool = False, freeze: int = 0, lr0: float = 0.0003) -> Path:
    """Serialize GPU training across dashboard and CLI processes, including Windows."""
    from filelock import FileLock, Timeout

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(config.DATA_DIR / "detector-training.lock", timeout=0):
            return _train(base, epochs, imgsz, batch, device, name, repeat, hand_only, freeze, lr0)
    except Timeout as e:
        raise RuntimeError("Detector training is already running in another process.") from e


def _train(base, epochs, imgsz, batch, device, name, repeat, hand_only, freeze, lr0):
    """Fine-tune deployed weights with corrected frames and independent match validation."""
    from ultralytics import YOLO

    data = split(repeat=repeat, hand_only=hand_only)
    from ..vision.detect import DEFAULT_WEIGHTS
    base = base or next((str(p) for p in DEFAULT_WEIGHTS[1:3] if p.exists()), "yolo11s.pt")
    model = YOLO(base)
    dev = device or {"cuda": "0", "mps": "mps"}.get(config.device(), "cpu")
    res = model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=dev,
                      project=str(config.DATA_DIR / "runs"), name=name or "robots_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                      exist_ok=False,
                      optimizer="AdamW", lr0=lr0, freeze=freeze,
                      mosaic=0.5, fliplr=0.5, hsv_h=0.0,  # keep hue: red vs blue matters
                      scale=0.4, close_mosaic=10, cos_lr=True,
                      patience=20, workers=4, plots=False, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    out = config.DATA_DIR / "models" / "experiments" / Path(res.save_dir).name / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    report = promote(out)
    (out.parent / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(report["reason"])
    return out

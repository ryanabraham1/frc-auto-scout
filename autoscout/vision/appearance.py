"""Robot appearance: features of robot crops, and a per-event bank of what each team looks like.

A robot looks the same in every match of an event, so every robot a person labels (and every
robot confirmed by a bumper read) is remembered per event. Unlabelled tracklets in later matches
are then compared with the bank, which turns "who is this?" from guesswork into a lookup.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import config

BANK_DIR = config.DATA_DIR / "appearance"
MAX_PER_TEAM = 400


def _hist(crop: np.ndarray) -> np.ndarray:
    """Colour of the robot body (top 65% of the box, above the bumper), hue x saturation, plus
    a coarse brightness histogram for black / silver robots."""
    body = crop[: int(0.65 * crop.shape[0])]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 6], [0, 180, 0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    f = np.concatenate([h / max(h.sum(), 1), 0.5 * v / max(v.sum(), 1)])
    return np.sqrt(f)  # Hellinger: cosine of sqrt-histograms ~ Bhattacharyya similarity


def features(crops: list[np.ndarray], model=None) -> np.ndarray:
    """(n, D) unit-length features: detector-backbone embedding and colour histogram, concatenated."""
    if not crops:
        return np.zeros((0, 0), np.float32)
    if model is None:
        from ultralytics import YOLO

        from .detect import find_weights

        model = YOLO(str(find_weights()))
    embs = []
    for i in range(0, len(crops), 64):
        for e in model.embed(crops[i: i + 64], imgsz=crops[0].shape[0], verbose=False):
            e = e.float().cpu().numpy()
            embs.append(e / (np.linalg.norm(e) + 1e-9))
    hists = [_hist(c) for c in crops]
    hists = [h / (np.linalg.norm(h) + 1e-9) for h in hists]
    f = np.hstack([np.array(embs), np.array(hists)]).astype(np.float32)
    return f / np.sqrt(2.0)


def mean_unit(f: np.ndarray) -> np.ndarray:
    m = f.mean(0)
    return m / (np.linalg.norm(m) + 1e-9)


# ---- per-event bank ------------------------------------------------------------------------------
def _path(event_key: str) -> Path:
    return BANK_DIR / f"{event_key}.npz"


def _load(event_key: str) -> dict[int, tuple[np.ndarray, list[str]]]:
    p = _path(event_key)
    if not p.exists():
        return {}
    z = np.load(p)
    return {int(k[1:]): (z[k], list(z["k" + k[1:]])) for k in z.files if k.startswith("t")}


def load_bank(event_key: str) -> dict[int, np.ndarray]:
    """{team: (n, D) features} of the event's identified robots."""
    return {t: f for t, (f, _) in _load(event_key).items() if len(f)}


def add_to_bank(event_key: str, examples: dict[int, list[tuple[str, np.ndarray]]]) -> None:
    """Remember identified tracklets as {team: [(key, feature)]}, key = "match:tracklet". A key seen
    before replaces its old entry under any team (a robot relabelled moves), and each team keeps its
    most recent MAX_PER_TEAM."""
    from filelock import FileLock

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    # matches of one event can process in parallel: read-modify-write the bank one at a time
    with FileLock(str(_path(event_key)) + ".lock"):
        _add_to_bank(event_key, examples)


def _add_to_bank(event_key: str, examples: dict[int, list[tuple[str, np.ndarray]]]) -> None:
    bank = _load(event_key)
    new_keys = {k for ex in examples.values() for k, _ in ex}
    out = {}
    for team in set(bank) | set(examples):
        f, keys = bank.get(team, (np.zeros((0, 0), np.float32), []))
        keep = [i for i, k in enumerate(keys) if k not in new_keys]
        rows = [f[i] for i in keep] + [e for _, e in examples.get(team, [])]
        ks = [keys[i] for i in keep] + [k for k, _ in examples.get(team, [])]
        if rows:
            out[team] = (np.array(rows[-MAX_PER_TEAM:], np.float32), ks[-MAX_PER_TEAM:])
    arrays = {}
    for t, (f, ks) in out.items():
        arrays[f"t{t}"], arrays[f"k{t}"] = f, np.array(ks)
    tmp = _path(event_key).with_suffix(".tmp.npz")  # replace in one step so readers never see half a file
    np.savez_compressed(tmp, **arrays)
    tmp.replace(_path(event_key))


def similarity(emb: np.ndarray, bank: dict[int, np.ndarray], teams: list[int], k: int = 5) -> dict[int, float]:
    """Mean cosine similarity of `emb` to its k nearest bank examples of each team."""
    out = {}
    for t in teams:
        B = bank.get(t)
        if B is None or not len(B) or B.shape[1] != len(emb):
            continue
        s = np.sort(B @ emb)[::-1][:k]
        out[t] = float(s.mean())
    return out

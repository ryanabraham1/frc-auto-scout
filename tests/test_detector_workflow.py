"""Accuracy measurement and safe local detector-training workflow."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from autoscout import config
from autoscout.train import train_robots as training
from autoscout.vision import detect
from autoscout.vision.benchmark import score_points, improves
from autoscout.vision.detect import Det
from autoscout.web import app as web


def test_duplicate_and_wrong_alliance_predictions_are_false_positives():
    ds = [Det(0, 0, 20, 20, .9, "red"), Det(1, 1, 21, 21, .8, "red"),
          Det(30, 0, 50, 20, .9, "red")]
    score = score_points(ds, [[10, 10, "r"], [40, 10, "b"]])
    assert (score["tp"], score["fp"], score["fn"]) == (1, 2, 1)


def test_matching_maximizes_robots_without_double_counting():
    ds = [Det(0, 0, 50, 20, .9, "blue"), Det(0, 0, 20, 20, .8, "blue")]
    score = score_points(ds, [[10, 10, "b"], [40, 10, "b"]], margin=0)
    assert score["tp"] == 2
    assert score_points(ds[:1], [[10, 10, "b"], [40, 10, "b"]])["tp"] == 1
    assert score_points([], [ [10, 10, "r"] ])["fn"] == 1
    assert score_points(ds, [])["fp"] == 2


def test_precision_regression_blocks_even_a_higher_f1():
    assert not improves(dict(f1=.85, precision=.79, recall=.92),
                        dict(f1=.80, precision=.85, recall=.76))


def _frame(directory, stem):
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    path = directory / "images" / f"{stem}.jpg"
    path.touch()
    (directory / "labels" / f"{stem}.txt").write_text("")
    return path


def test_split_separates_entire_matches_and_excludes_benchmark_video_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    robots = tmp_path / "datasets" / "robots"
    hand = tmp_path / "datasets" / "hand"
    monkeypatch.setattr(training, "DATASET", robots)
    monkeypatch.setattr(training, "held_out", lambda: {"2026test_qm40"})
    monkeypatch.setattr(training, "source_aliases", lambda: {"youtube_id": "2026test_qm40"})
    for match in (1, 2, 3, 4, 40):
        _frame(hand, f"2026test_qm{match}_40")
        _frame(robots, f"2026test_qm{match}_12_0000")
        _frame(robots, f"2026test_qm{match}_120_0000")
    _frame(hand, "youtube_id_u70_0")
    out = training.split()
    train = (out.parent / "train.txt").read_text().splitlines()
    val = (out.parent / "val.txt").read_text().splitlines()
    assert train and val
    assert not any("qm40_" in p or "youtube_id" in p for p in train + val)
    train_groups = {training.source_group(Path(p)) for p in train}
    val_groups = {training.source_group(Path(p)) for p in val}
    assert not train_groups & val_groups
    assert all("hand" in p for p in val)
    assert len(train) > len(set(train))  # corrected training examples are weighted


def test_dashboard_video_names_keep_underscores():
    assert training.source_group(Path("abc_def_u70_0.jpg")) == "abc_def"


def test_engine_skipped_on_cpu_or_when_older_than_pt(tmp_path, monkeypatch):
    import os
    engine = tmp_path / "model.engine"
    portable = engine.with_suffix(".pt")
    engine.touch()
    portable.touch()
    monkeypatch.setattr(detect, "DEFAULT_WEIGHTS", [engine, portable])
    monkeypatch.setattr(config, "device", lambda: "cpu")
    assert detect.find_weights() == portable
    monkeypatch.setattr(config, "device", lambda: "cuda")
    os.utime(engine, (100, 100))
    os.utime(portable, (200, 200))
    assert detect.find_weights() == portable


def test_promotion_preserves_existing_model_on_regression(tmp_path, monkeypatch):
    from autoscout.vision import benchmark
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    directory = tmp_path / "det_eval"
    directory.mkdir()
    (directory / "points.json").write_text('{"frame": [[1, 1, "r"]]}')
    baseline = tmp_path / "models" / "autoscout-robots.pt"
    baseline.parent.mkdir()
    baseline.write_bytes(b"deployed")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(detect, "find_weights", lambda: baseline)
    monkeypatch.setattr(detect, "RobotDetector", lambda path, **kw: path)
    monkeypatch.setattr(detect, "PersonVeto", lambda: None)
    monkeypatch.setattr(benchmark, "evaluate_model", lambda path, *args:
                        dict(f1=.8, precision=.8, recall=.8) if path == candidate
                        else dict(f1=.81, precision=.81, recall=.81))
    assert not training.promote(candidate)["promoted"]
    assert baseline.read_bytes() == b"deployed"
    monkeypatch.setattr(benchmark, "evaluate_model", lambda path, *args:
                        dict(f1=.9, precision=.9, recall=.9) if path == candidate
                        else dict(f1=.81, precision=.81, recall=.81))
    engine = baseline.with_suffix(".engine")
    engine.write_bytes(b"old engine")
    assert training.promote(candidate)["promoted"]
    assert baseline.read_bytes() == b"candidate"
    assert (tmp_path / "previous-deployed.pt").read_bytes() == b"deployed"
    assert not engine.exists()


def test_dashboard_writes_corrected_frames_to_hand_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    image = np.zeros((100, 200, 3), np.uint8)
    monkeypatch.setattr(web, "_label_frame", lambda *a: (Path("video.mp4"), image, (10, 10, 100, 80)))
    result = web.save_labels(dict(video="video.mp4", t=5, robots=[["red", 20, 20, 40, 40]]))
    label = tmp_path / "datasets" / "hand" / "labels" / f'{result["stem"]}.txt'
    assert label.exists()
    assert label.read_text().startswith("0 0.200000 0.250000")
    with pytest.raises(HTTPException) as error:
        web.save_labels(dict(video="video.mp4", t=5, robots=[["red", -1, 20, 40, 40]]))
    assert error.value.status_code == 400


def test_label_frame_rejects_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    with pytest.raises(HTTPException) as error:
        web._label_frame(str(tmp_path / "outside.mp4"), 1)
    assert error.value.status_code == 400


def test_label_frame_scales_calibration_and_releases_video(tmp_path, monkeypatch):
    from autoscout import video
    from types import SimpleNamespace
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = tmp_path / "video.mp4"
    path.touch()
    called = {}
    class FakeVideo:
        duration = 60
        cap = SimpleNamespace(release=lambda: called.update(released=True))
        def frame_at(self, t):
            return np.zeros((540, 960, 3), np.uint8)
    monkeypatch.setattr(video, "Video", lambda p: FakeVideo())
    def scaled(size):
        called["size"] = size
        return SimpleNamespace(view=(10, 20, 900, 400))
    monkeypatch.setattr(web.Calibration, "load", lambda key: SimpleNamespace(scaled=scaled))
    _, im, rect = web._label_frame(str(path), 20, "2026test")
    assert called == {"released": True, "size": (960, 540)}
    assert rect == (10, 20, 900, 400)


def test_training_lock_rejects_duplicate_process(tmp_path, monkeypatch):
    from filelock import FileLock
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with FileLock(tmp_path / "detector-training.lock"):
        with pytest.raises(RuntimeError, match="already running"):
            training.train()

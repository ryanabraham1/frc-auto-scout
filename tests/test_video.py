"""Prefetch must preserve frames and safely relinquish the shared capture."""
import threading

import cv2
import numpy as np
import pytest

from autoscout.video import Video
from autoscout.vision.trajectories import LaunchDetector


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    for _ in range(30):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    v = Video(path)
    yield v
    v.close()


@pytest.mark.parametrize("t0,t1,step", [(0, None, 1), (.13, .8, 3), (.9, .2, 1), (0, 5, 7)])
def test_prefetch_preserves_every_pixel_and_timestamp(video, t0, t1, step):
    expected = list(video.iter_frames(t0, t1, step))
    with video.prefetched_frames(t0, t1, step) as frames:
        actual = list(frames)
    assert len(actual) == len(expected)
    for (ta, a), (tb, b) in zip(actual, expected):
        assert ta == tb
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fail", [False, True])
def test_prefetch_joins_on_early_exit_and_consumer_error(video, fail):
    expected = video.frame_at(.4)
    try:
        with video.prefetched_frames() as frames:
            next(frames)
            if fail:
                raise ValueError("processing failed")
    except ValueError:
        assert fail
    np.testing.assert_array_equal(video.frame_at(.4), expected)


def test_prefetch_overlaps_decode_and_propagates_decoder_errors(monkeypatch, video):
    ahead = threading.Event()
    released = threading.Event()

    def source(*args):
        try:
            yield 0, np.zeros((1, 1, 3), np.uint8)
            ahead.set()
            raise RuntimeError("decode failed")
        finally:
            released.set()

    monkeypatch.setattr(video, "iter_frames", source)
    with video.prefetched_frames() as frames:
        next(frames)
        assert ahead.wait(2), "next frame should decode while consumer holds the current frame"
        with pytest.raises(RuntimeError, match="decode failed"):
            next(frames)
    assert released.is_set()


def test_blob_collection_is_identical_without_live_tracking():
    live = LaunchDetector((40, 100, 100), (80, 255, 255), (0, 0, 160, 100), lag=2)
    offline = LaunchDetector((40, 100, 100), (80, 255, 255), (0, 0, 160, 100), lag=2)
    offline.online = False
    for i in range(24):
        frame = np.zeros((100, 160, 3), np.uint8)
        cv2.circle(frame, (10 + i * 5, 60 - i), 5, (0, 255, 0), -1)
        if i == 12:  # camera cuts reset the reference mask
            live.prev = offline.prev = None
        live.step(i / 60, frame)
        offline.step(i / 60, frame)
    assert live.blobs
    assert live.blobs == offline.blobs
    assert not offline.tracks


def test_download_supports_browser_session_and_prefers_explicit_cookie_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoscout import config
    from autoscout import video as module

    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / 'video.mp4').write_bytes(b'cached video')
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module, 'ensure_decodable', lambda p:p)
    monkeypatch.setattr(config, 'yt_cookies', lambda:None)
    monkeypatch.setenv('YT_COOKIES_BROWSER', 'firefox:C:/Profile With Spaces')
    assert module.download('video',tmp_path).name == 'video.mp4'
    assert calls[-1][calls[-1].index('--cookies-from-browser')+1] == 'firefox:C:/Profile With Spaces'
    (tmp_path/'video.mp4').unlink()
    cookies=tmp_path/'cookies.txt'
    monkeypatch.setattr(config, 'yt_cookies', lambda:cookies)
    module.download('video',tmp_path)
    assert '--cookies-from-browser' not in calls[-1]
    assert calls[-1][calls[-1].index('--cookies')+1] == str(cookies)

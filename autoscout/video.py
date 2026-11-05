"""Video download (yt-dlp) and frame access."""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import config

VIDEO_DIR = config.DATA_DIR / "videos"


def match_video(match, height: int = 1080) -> Path:
    """Resolve a TBA video or a documented local broadcast excerpt."""
    if match.local_video:
        path = Path(match.local_video)
        if not path.is_file():
            raise FileNotFoundError(f"Local match video is missing: {path}")
        return ensure_decodable(path)
    if not match.videos:
        raise RuntimeError(f"No video for {match.key}")
    return download(match.videos[0], height=height)


def download(youtube_id: str, out_dir: Path | None = None, height: int = 720) -> Path:
    """Download a YouTube video (H.264 preferred so OpenCV can decode it everywhere)."""
    out_dir = out_dir or VIDEO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("mp4", "mkv", "webm"):
        p = out_dir / f"{youtube_id}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return ensure_decodable(p)
    fmt = (
        f"bv*[height<={height}][vcodec^=avc1]+ba/bv*[height<={height}][vcodec^=avc1]/"
        f"bv*[height<={height}]+ba/b[height<={height}]/b"
    )
    cmd = [
        sys.executable, "-m", "yt_dlp", "-q", "--no-progress", "--no-playlist",
        "-f", fmt, "--merge-output-format", "mp4",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--js-runtimes", "node", "--remote-components", "ejs:github",
    ]
    cookies = config.yt_cookies()
    if cookies:
        cmd += ["--cookies", str(cookies)]
    elif browser := config.yt_cookies_browser():
        cmd += ["--cookies-from-browser", browser]
    cmd.append(f"https://www.youtube.com/watch?v={youtube_id}")
    last_err = ""
    # When YouTube answers the default client with "confirm you're not a bot" (it does after many
    # downloads from one address), other player clients usually still work: web_embedded serves
    # 1080p H.264, mweb/android_vr are further fallbacks.
    clients = [None, "web_embedded", "mweb", "android_vr"]
    for attempt in range(len(clients)):
        args = list(cmd)
        if clients[attempt]:
            args[-1:-1] = ["--extractor-args", f"youtube:player_client={clients[attempt]}"]
        res = subprocess.run(args, capture_output=True, text=True)
        for ext in ("mp4", "mkv", "webm"):
            p = out_dir / f"{youtube_id}.{ext}"
            if p.exists() and p.stat().st_size > 0:
                return ensure_decodable(p)
        last_err = res.stderr[-2000:]
        # yt-dlp's --js-runtimes flag is unknown to old versions; retry without it.
        if "no such option" in last_err and "--js-runtimes" in cmd:
            i = cmd.index("--js-runtimes")
            del cmd[i : i + 4]
    raise RuntimeError(f"download failed for {youtube_id}: {last_err}")


def _codec(path: Path) -> str:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return res.stdout.strip()


def ensure_decodable(path: Path) -> Path:
    """OpenCV's bundled FFmpeg can't decode AV1/VP9 everywhere; transcode those to H.264 once."""
    path = Path(path)
    if _codec(path) in ("h264", "hevc", "mpeg4", ""):
        return path
    out = path.with_name(path.stem + ".h264.mp4")
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-c:a", "aac", str(out)], check=True)
    path.unlink(missing_ok=True)
    final = path.with_suffix(".mp4") if path.suffix != ".mp4" else path
    out.rename(final)
    return final


class Video:
    """Random + sequential frame access with time in seconds."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.n_frames / self.fps if self.fps else 0.0

    def frame_at(self, t: float) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None

    def iter_frames(self, t0: float = 0.0, t1: float | None = None, step: int = 1) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (time_s, frame) for every `step`-th frame between t0 and t1 (decodes sequentially)."""
        t1 = self.duration if t1 is None else min(t1, self.duration)
        idx0 = int(round(t0 * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
        idx = idx0
        while True:
            t = idx / self.fps
            if t > t1:
                break
            if (idx - idx0) % step == 0:
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield t, frame
            else:
                if not self.cap.grab():
                    break
            idx += 1

    @contextmanager
    def prefetched_frames(self, t0: float = 0.0, t1: float | None = None, step: int = 1):
        """Decode one frame ahead while the caller processes the current frame.

        Uses the same sequential decoder, pixels, timestamps and sampling as iter_frames.
        The context joins the decoder on errors or early exit, before the capture can be reused.
        Only one worker touches the capture; callers must not seek it inside this context.
        """
        source = self.iter_frames(t0, t1, step)
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-decode") as pool:
                pending = pool.submit(next, source, None)

                def frames():
                    nonlocal pending
                    while True:
                        item = pending.result()
                        if item is None:
                            return
                        pending = pool.submit(next, source, None)
                        yield item

                stream = frames()
                try:
                    yield stream
                finally:
                    stream.close()
                    pending.cancel()
        finally:
            source.close()

    def close(self) -> None:
        self.cap.release()

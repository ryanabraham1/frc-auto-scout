"""Get one match's footage out of an event livestream.

Two sources, because YouTube will not cut a window out of a stream that is still live
("This format cannot be partially downloaded"):

* ended stream: ask yt-dlp for just the window, at `actual_time - stream start`.
* ongoing stream: `Recorder` keeps short wall-clock-named segments from the live edge, and
  `cut_clip` joins the ones covering a match. Works on Twitch too, but only from the moment
  recording starts: nothing can be recovered from before that.

Sources are allow-listed and rebuilt from their parsed parts. The URL arrives from a shared
no-login app, so nothing it says may reach yt-dlp as an option or send this computer anywhere else.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import config

# FMS actual_time is the start of the match. The overlay clock then finds the real start, so a
# window only has to contain the match: pre-roll for clock error, tail for the 2:40 match plus
# a stream-start estimate that can be a little off.
VOD_BEFORE_S, VOD_AFTER_S = 30, 230
# A live viewer is behind real time, so a segment's wall clock is later than what it shows.
LIVE_BEFORE_S, LIVE_AFTER_S = 15, 240
SEGMENT_S = 20
SEGMENT_TIME = "%Y%m%dT%H%M%S"
FORMAT = "bv*[height<=1080][vcodec^=avc1]+ba/b[height<=1080]/bv*[height<=1080]+ba/b"
LIVE_FORMAT = "b[protocol^=m3u8][height<=1080]/b[height<=1080]/b"

YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{6,20}\Z")
YOUTUBE_HANDLE = re.compile(r"@[A-Za-z0-9._-]{1,60}\Z")
YOUTUBE_CHANNEL = re.compile(r"UC[A-Za-z0-9_-]{10,40}\Z")
TWITCH_CHANNEL = re.compile(r"[A-Za-z0-9_]{2,40}\Z")
TWITCH_VOD = re.compile(r"[0-9]{5,15}\Z")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
TWITCH_HOSTS = {"twitch.tv", "www.twitch.tv", "m.twitch.tv"}


class StreamError(RuntimeError):
    """A failure whose message is safe and useful to show on the team lead page."""


def parse_source(url: str) -> str:
    """Return the canonical URL for an allowed stream, or raise StreamError."""
    parts = urlsplit(url.strip()) if isinstance(url, str) and len(url) <= 300 else None
    try:
        port = parts.port if parts else None
    except ValueError:
        port = -1
    if parts is None or parts.scheme != "https" or parts.username or parts.password or port:
        raise StreamError("Use an https YouTube or Twitch stream link")
    host = (parts.hostname or "").lower()
    path = [p for p in parts.path.split("/") if p]
    if host in YOUTUBE_HOSTS:
        vid = ""
        if path == ["watch"]:
            vid = parse_qs(parts.query).get("v", [""])[0]
        elif len(path) == 2 and path[0] == "live":
            vid = path[1]
        elif len(path) == 2 and path[1] == "live" and YOUTUBE_HANDLE.fullmatch(path[0]):
            return f"https://www.youtube.com/{path[0]}/live"
        elif len(path) == 3 and path[0] == "channel" and path[2] == "live" and YOUTUBE_CHANNEL.fullmatch(path[1]):
            return f"https://www.youtube.com/channel/{path[1]}/live"
        if YOUTUBE_ID.fullmatch(vid):
            return f"https://www.youtube.com/watch?v={vid}"
    elif host == "youtu.be" and len(path) == 1 and YOUTUBE_ID.fullmatch(path[0]):
        return f"https://www.youtube.com/watch?v={path[0]}"
    elif host in TWITCH_HOSTS:
        if len(path) == 2 and path[0] == "videos" and TWITCH_VOD.fullmatch(path[1]):
            return f"https://www.twitch.tv/videos/{path[1]}"
        if len(path) == 1 and TWITCH_CHANNEL.fullmatch(path[0]):
            return f"https://www.twitch.tv/{path[0].lower()}"
    raise StreamError("Use a YouTube or Twitch stream link")


def video_id(url: str) -> str | None:
    return parse_qs(urlsplit(url).query).get("v", [None])[0]


def webcast_url(webcast: dict) -> str | None:
    """A TBA webcast as a link we accept. YouTube's id is in `file` when `channel` is the channel."""
    kind, channel, file = webcast.get("type"), webcast.get("channel"), webcast.get("file")
    try:
        if kind == "youtube" and (file or channel):
            return parse_source(f"https://www.youtube.com/watch?v={file or channel}")
        if kind == "twitch" and channel:
            return parse_source(f"https://www.twitch.tv/{channel}")
    except StreamError:
        pass
    return None


def stream_candidates(webcasts: list[dict], day: str, vod: bool = False) -> list[str]:
    """Streams that could cover `day` (YYYY-MM-DD), best first: that day's, whole-event, other days'.

    Mirrors the app's pickWebcast. A Twitch channel has no history, so it can't serve an ended match.
    """
    ranked = []
    for order, webcast in enumerate(webcasts or []):
        url = webcast_url(webcast)
        if not url or (vod and "twitch.tv" in url):
            continue
        date = webcast.get("date")
        ranked.append((0 if date == day else 1 if not date else 2, order, url))
    return [url for _, _, url in sorted(ranked)]


def event_stream(webcasts: list[dict], day: str, vod: bool = False) -> str | None:
    found = stream_candidates(webcasts, day, vod)
    return found[0] if found else None


def live_stream(webcasts: list[dict], day: str, check=None) -> str | None:
    """The first candidate actually on air. TBA's link for today can be dead: MadTown's YouTube
    stream stopped after 6 minutes and the event carried on on Twitch."""
    check = check or probe
    for url in stream_candidates(webcasts, day):
        try:
            if check(url).live_status == "is_live":
                return url
        except (StreamError, subprocess.TimeoutExpired):
            continue
    return None


def twitch_archives(webcasts: list[dict], limit: int = 6) -> list[str]:
    """Past broadcasts of the event's Twitch channels, newest first. A channel link has no history
    but its archive does, including the broadcast still in progress."""
    found = []
    for webcast in webcasts or []:
        channel = webcast.get("channel")
        if webcast.get("type") != "twitch" or not channel or not TWITCH_CHANNEL.fullmatch(channel):
            continue
        command = [c for c in _ytdlp("--flat-playlist", "--yes-playlist", "--playlist-end", str(limit), "--print", "id",
                                     "--", f"https://www.twitch.tv/{channel.lower()}/videos?filter=archives&sort=time")
                   if c != "--no-playlist"]
        res = subprocess.run(command, capture_output=True, text=True, timeout=90)
        for line in res.stdout.split():
            vod = line.lstrip("v")
            if TWITCH_VOD.fullmatch(vod):
                found.append(f"https://www.twitch.tv/videos/{vod}")
    return found


def grab_frame(source: str | Path, out: Path) -> Path:
    """One frame: the end of a recorded segment, or the current picture of a stream link."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, Path):
        command = ["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", str(source), "-frames:v", "1", str(out)]
    else:
        res = subprocess.run(_ytdlp("-g", "-f", LIVE_FORMAT, "--", source), capture_output=True, text=True, timeout=90)
        lines = res.stdout.split()
        if res.returncode or not lines:
            raise StreamError(_explain(res.stderr))
        command = ["ffmpeg", "-v", "error", "-y", "-i", lines[0], "-frames:v", "1", str(out)]
    out.unlink(missing_ok=True)
    subprocess.run(command, capture_output=True, timeout=60)
    if not out.exists():
        raise StreamError("Could not grab a frame from that stream")
    return out


def event_day(epoch: float, timezone: str | None) -> str:
    """Calendar day at the event: TBA dates webcasts in event-local time."""
    from datetime import timedelta, timezone as tz
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        zone = ZoneInfo(timezone) if timezone else tz(timedelta(hours=-6))
    except (ZoneInfoNotFoundError, ValueError):
        zone = tz(timedelta(hours=-6))  # no tzdata: still the right day for a daytime US match
    return datetime.fromtimestamp(epoch, zone).strftime("%Y-%m-%d")


def _ytdlp(*args: str) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
           "--js-runtimes", "node", "--remote-components", "ejs:github"]
    cookies = config.yt_cookies()
    browser = config.yt_cookies_browser()
    if cookies:
        cmd += ["--cookies", str(cookies)]
    elif browser:
        cmd += ["--cookies-from-browser", browser]
    return [*cmd, *args]


def _explain(stderr: str) -> str:
    text = stderr.lower()
    if "not a bot" in text or "sign in" in text:
        return "YouTube wants a login: refresh the cookies on the Auto-Scout computer"
    if "private" in text or "unavailable" in text or "does not exist" in text or "404" in text:
        return "That stream is private, removed or not available"
    if "not currently live" in text or "offline" in text or "will begin" in text:
        return "That stream is not live right now"
    return "Could not read that stream"


@dataclass
class StreamInfo:
    live_status: str  # is_live | was_live | not_live | is_upcoming | post_live
    start: float | None  # epoch seconds the stream began
    duration: float | None


def probe(url: str, timeout: float = 90) -> StreamInfo:
    res = subprocess.run(_ytdlp("--skip-download", "--dump-single-json", "--", url),
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode:
        raise StreamError(_explain(res.stderr))
    data = json.loads(res.stdout)
    start = data.get("release_timestamp") or data.get("timestamp")
    return StreamInfo(data.get("live_status") or "not_live", float(start) if start else None,
                      data.get("duration"))


def vod_window(actual_time: float, stream_start: float, duration: float | None) -> tuple[int, int]:
    """Seconds into an ended stream to fetch; StreamError if the match is not in that stream."""
    t0 = actual_time - stream_start
    if t0 < 0 or (duration and t0 > duration):
        raise StreamError("This match is not in that stream (a different day's stream?)")
    end = t0 + VOD_AFTER_S
    return max(0, int(t0 - VOD_BEFORE_S)), int(min(end, duration) if duration else end)


def _duration(path: Path) -> float:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def clip_vod(url: str, start: int, end: int, out: Path, log: Path | None = None, timeout: float = 1200) -> Path:
    """Fetch [start, end] seconds of an ended stream to `out`."""
    if out.exists() and out.stat().st_size:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    tmp.unlink(missing_ok=True)
    res = subprocess.run(_ytdlp("-f", FORMAT, "--merge-output-format", "mp4", "--download-sections",
                                f"*{start}-{end}", "-o", str(tmp), "--", url),
                         capture_output=True, text=True, timeout=timeout)
    if log:
        with log.open("a", encoding="utf-8") as f:
            f.write(res.stdout + res.stderr)
    if res.returncode or not tmp.exists():
        raise StreamError(_explain(res.stderr))
    # A short file is a stream that had not published this part yet, never a usable match.
    if _duration(tmp) < 0.6 * (end - start):
        tmp.unlink(missing_ok=True)
        raise StreamError("The stream did not have the whole match")
    tmp.replace(out)
    return out


def segment_time(path: Path) -> float | None:
    try:
        return datetime.strptime(path.stem, SEGMENT_TIME).timestamp()
    except ValueError:
        return None


def segments(folder: Path) -> list[tuple[float, Path]]:
    found = [(t, p) for p in folder.glob("*.ts") if (t := segment_time(p)) is not None]
    return sorted(found)


def covered_until(folder: Path) -> float | None:
    """Wall-clock time the recording reaches. The newest segment may still be open, so its start."""
    found = segments(folder)
    return found[-1][0] if found else None


def cut_clip(folder: Path, start: float, end: float, out: Path) -> Path:
    """Join the recorded segments overlapping wall-clock [start, end] into `out`."""
    found = segments(folder)
    picked = [(t, p) for i, (t, p) in enumerate(found)
              if t < end and (found[i + 1][0] if i + 1 < len(found) else t + SEGMENT_S) > start]
    if not picked:
        raise StreamError("The recording does not cover this match")
    times = [t for t, _ in picked]
    # A reconnect mid-match leaves a hole that the clock reader would silently stretch over.
    if any(b - a > SEGMENT_S * 2.5 for a, b in zip(times, times[1:])):
        raise StreamError("The stream dropped during this match")
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for _, p in picked), encoding="utf-8")
    tmp = out.with_name(out.stem + ".part.mp4")
    res = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                          "-c", "copy", "-bsf:a", "aac_adtstoasc", str(tmp)], capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    if res.returncode or not tmp.exists():
        raise StreamError("Could not cut this match from the recording")
    tmp.replace(out)
    return out


def prune(folder: Path, keep_s: float, now: float | None = None) -> None:
    now = time.time() if now is None else now
    for t, p in segments(folder):
        if now - t > keep_s:
            p.unlink(missing_ok=True)


class Recorder:
    """Keep recording a live stream into `folder`, reconnecting until stopped.

    Stream URLs expire and event streams drop, so ffmpeg is expected to exit: each pass resolves
    a fresh URL. `status` is what the lead sees; it never contains a URL.
    """

    def __init__(self, url, folder: Path, log: Path, retry_s: float = 20):
        """`url` is a link, or a callable asked on every reconnect so a new day's stream is picked up."""
        self.url, self.folder, self.log, self.retry_s = url, folder, log, retry_s
        self.status = "starting"
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._thread.join(timeout=15)

    def _resolve(self) -> str:
        url = self.url() if callable(self.url) else self.url
        if not url:
            raise StreamError("TBA lists no stream for this event yet")
        res = subprocess.run(_ytdlp("-g", "-f", LIVE_FORMAT, "--", url), capture_output=True,
                             text=True, timeout=90)
        lines = res.stdout.split()
        if res.returncode or not lines:
            raise StreamError(_explain(res.stderr))
        return lines[0]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                source = self._resolve()
            except (StreamError, subprocess.TimeoutExpired) as e:
                self.status = str(e) if isinstance(e, StreamError) else "Could not read that stream"
                self._stop.wait(self.retry_s)
                continue
            self.status = "recording"
            pattern = str(self.folder / f"{SEGMENT_TIME}.ts")
            with self.log.open("a", encoding="utf-8") as out:
                self._proc = subprocess.Popen(
                    ["ffmpeg", "-v", "warning", "-nostdin", "-i", source, "-map", "0:v:0", "-map", "0:a:0?",
                     "-c", "copy", "-f", "segment", "-segment_time", str(SEGMENT_S),
                     "-segment_format", "mpegts", "-reset_timestamps", "1", "-strftime", "1", pattern],
                    stdout=out, stderr=subprocess.STDOUT)
                self._proc.wait()
            if not self._stop.is_set():
                self.status = "reconnecting"
                self._stop.wait(self.retry_s)

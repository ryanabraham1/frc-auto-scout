"""Take jobs from the scouting app by calling out to it.

This computer is not reachable from the internet and needs no tunnel: it asks the app's
`autoscout-worker` Edge Function for work, and pushes status and results back. Jobs wait in the
app while the computer is off, and results land with every browser closed. It holds one
worker token, never a Supabase key.

Remote job kinds:
  matches  TBA's per-match videos
  vod      an ended livestream: each match is clipped out by time
  live     an ongoing livestream: recorded from now on, each match cut as TBA posts it,
           then scouted unless the lead asked for clips only

With no link, the stream comes from TBA's webcasts: the right day's for each match, and for a
live job whichever is current each time the recorder reconnects.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import requests

from . import config, streams
from .integration import MATCH, MatchSkipped, Queue, cached_result, export_result, process_match
from .tba import TBA

log = logging.getLogger("autoscout.bridge")

EVENT = re.compile(r"(?:2025|2026)[a-z0-9]+\Z")
MIN_FREE_GB = 10  # below this, refuse new work: a full disk takes the whole queue down with it
CLIP_CACHE_GB = 15
LIVE_KEEP_S = 3 * 3600  # long enough for TBA to post a slow match
IDLE_S, BUSY_S = 15, 5
OFFLINE_END_S = 2 * 3600  # a live job with nothing to record for this long has finished


def calibrated(event_key: str) -> bool:
    return (config.EVENTS_DIR / f"{event_key}.yaml").exists()


def calibrated_events() -> list[str]:
    return sorted(p.stem for p in config.EVENTS_DIR.glob("*.yaml") if EVENT.fullmatch(p.stem))


def free_gb() -> float:
    return shutil.disk_usage(config.DATA_DIR).free / 1e9


def prune_clips(folder: Path, keep_gb: float = CLIP_CACHE_GB) -> None:
    """Clips only matter until processed; keep the newest for re-runs and drop the rest."""
    clips = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for p in clips:
        total += p.stat().st_size
        if total > keep_gb * 1e9:
            p.unlink(missing_ok=True)


class Backend:
    def __init__(self, url: str, token: str):
        self.url, self.session = url, requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def call(self, action: str, **body) -> dict:
        # Generous: when the app's API is under load a sync still lands, just slowly.
        r = self.session.post(self.url, json={"action": action, **body}, timeout=90)
        r.raise_for_status()
        return r.json()


class Runner:
    """Turns a job's source into footage, then runs the normal per-match pipeline."""

    def __init__(self, clips: Path, tba: TBA | None = None, process=process_match):
        self.clips, self.process, self._tba = clips, process, tba

    @property
    def tba(self) -> TBA:
        self._tba = self._tba or TBA()
        return self._tba

    def __call__(self, match_key: str, force: bool, log_path: Path, source: dict | None = None):
        try:
            return self._run(match_key, force, log_path, source or {})
        except MatchSkipped:
            raise
        except streams.StreamError as e:
            raise MatchSkipped(str(e)) from e
        except subprocess.CalledProcessError as e:
            raise MatchSkipped("Processing failed on the computer; see its job log") from e
        except subprocess.TimeoutExpired as e:
            raise MatchSkipped("Timed out on the computer") from e
        except requests.RequestException as e:
            raise MatchSkipped("Could not reach The Blue Alliance") from e

    def _run(self, match_key, force, log_path, source):
        if source.get("clip"):
            return self.process(match_key, force, log_path, Path(source["clip"]))
        if not source.get("stream"):
            # A clip kept from a clips-only live job beats waiting for TBA's video.
            kept = [p for p in self.clips.glob(f"{match_key}_*.mp4") if not p.stem.endswith(".part")]
            if kept:
                return self.process(match_key, force, log_path, max(kept, key=lambda p: p.stat().st_mtime))
            if not self.tba.match(match_key).get("videos") and not (not force and cached_result(match_key)):
                raise MatchSkipped("TBA has no video for this match yet: use the stream instead")
            return self.process(match_key, force, log_path)
        if not force and (done := cached_result(match_key)):
            return done
        # Fresh, and into the cache run-match reads: a stale entry has no actual_time yet.
        actual = self.tba.get(f"/match/{match_key}", max_age_s=30).get("actual_time")
        if not actual:
            raise MatchSkipped("TBA has no start time for this match yet")
        if clip := self._recorded(match_key, actual):
            return self.process(match_key, force, log_path, clip)
        if source.get("url"):  # a link the lead pasted is what they asked for: no fallback
            return self.process(match_key, force, log_path, self._clip(source["url"], match_key, actual, source, log_path))
        failure: Exception | None = None
        for url in self._event_streams(match_key, actual):
            try:
                return self.process(match_key, force, log_path, self._clip(url, match_key, actual, source, log_path))
            except (MatchSkipped, streams.StreamError) as e:
                failure = e  # a dead day's stream or another broadcast: try the next one
        if self.tba.match(match_key).get("videos"):
            return self.process(match_key, force, log_path)
        raise failure or MatchSkipped("TBA lists no stream for this match: paste its link")

    def _recorded(self, match_key: str, actual: float) -> Path | None:
        """Our own recording has exact wall-clock times, which a platform's start time doesn't."""
        start, end = actual - streams.LIVE_BEFORE_S, actual + streams.LIVE_AFTER_S
        for folder in sorted(self.clips.parent.glob("live/*")) + sorted(self.clips.parent.glob("safety/*")):
            if folder.is_dir() and streams.covers(folder, start, end):
                return streams.cut_clip(folder, start, end, self.clips / f"{match_key}_rec.mp4")
        return None

    def _clip(self, url, match_key, actual, source, log_path) -> Path:
        start = (source.get("stream_starts") or {}).get(streams.video_id(url))
        info = streams.probe(url)
        start = start or info.start
        if not start:
            raise MatchSkipped("Could not tell when that stream started")
        window = streams.vod_window(actual, start, info.duration)
        tag = hashlib.sha256(f"{url}{window}".encode()).hexdigest()[:8]
        return streams.clip_vod(url, *window, self.clips / f"{match_key}_{tag}.mp4", log_path)

    def _event_streams(self, match_key: str, actual: float):
        event = self.tba.get(f"/event/{match_key.split('_')[0]}", max_age_s=3600)
        webcasts = event.get("webcasts") or []
        day = streams.event_day(actual, event.get("timezone"))
        yield from streams.stream_candidates(webcasts, day, vod=True)
        yield from streams.twitch_archives(webcasts)


class LiveSession:
    """One ongoing stream: keep recording, and hand each finished match to the queue."""

    def __init__(self, job: dict, root: Path, enqueue, tba: TBA | None = None, recorder=None,
                 cut=streams.cut_clip, covered=streams.covered_until):
        self.job, self.enqueue, self.cut, self.covered = job, enqueue, cut, covered
        self.folder = root / "live" / job["id"]
        self.clips = root / "clips"
        self.state_path = self.folder / "state.json"
        self.folder.mkdir(parents=True, exist_ok=True)
        saved = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.started = saved.get("started", time.time())
        self.seen: dict[str, str] = saved.get("seen", {})  # match key -> local job id, "" = clip only
        self.errors: dict[str, str] = saved.get("errors", {})
        self._save()
        self._tba = tba
        self.recorder = recorder or streams.Recorder(job.get("source_url") or self.todays_stream, self.folder,
                                                     self.folder / "ffmpeg.log")
        self.last_recorded = time.time()
        self.next_poll = 0.0

    def _save(self):
        self.state_path.write_text(json.dumps({"started": self.started, "seen": self.seen, "errors": self.errors}))

    @property
    def tba(self) -> TBA:
        self._tba = self._tba or TBA()
        return self._tba

    def todays_stream(self) -> str | None:
        try:
            event = self.tba.get(f"/event/{self.job['event_key']}", max_age_s=600)
        except requests.RequestException:
            return None
        day = streams.event_day(time.time(), event.get("timezone"))
        return streams.live_stream(event.get("webcasts") or [], day)

    def start(self):
        self.recorder.start()

    def stop(self):
        self.recorder.stop()

    def finished_recording(self, now: float) -> bool:
        return now - self.last_recorded > OFFLINE_END_S

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.recorder.status == "recording":
            self.last_recorded = now
        streams.prune(self.folder, LIVE_KEEP_S, now)
        if now < self.next_poll:
            return
        self.next_poll = now + 20
        reach = self.covered(self.folder)
        if reach is None:
            return
        matches = self.tba.get(f"/event/{self.job['event_key']}/matches", max_age_s=15)
        for m in sorted(matches, key=lambda x: x.get("actual_time") or 0):
            key, actual = m["key"], m.get("actual_time")
            # A match that began before recording can't be recovered, so it was never in scope.
            if not actual or key in self.seen or key in self.errors or actual - streams.LIVE_BEFORE_S < self.started:
                continue
            end = actual + streams.LIVE_AFTER_S
            if end > reach:
                continue  # still recording it, or waiting for the stream to catch up
            clip = self.clips / f"{key}_live_{self.job['id'][:8]}.mp4"
            try:
                self.cut(self.folder, actual - streams.LIVE_BEFORE_S, end, clip)
            except streams.StreamError as e:
                self.errors[key] = str(e)
                self._save()
                continue
            if self.job.get("scout", True):
                self.tba.get(f"/match/{key}", max_age_s=30)
                self.seen[key] = self.enqueue(key, str(clip))
            else:
                self.seen[key] = ""
            self._save()


COMMAND_OPS = {"reanalyze": "reanalyze", "rescore": "rescore", "identify": "identify"}


def field_points(event_key: str, payload: dict) -> tuple[dict, tuple]:
    """Validated calibration input from the app: known landmark names, numbers inside the frame."""
    from .field import FieldModel

    year = int(event_key[:4])
    size = payload.get("frame_size") or [1920, 1080]
    width, height = int(size[0]), int(size[1])
    known = set(FieldModel.load(year).landmarks) | {"blue_hub_top", "red_hub_top"}
    points = {}
    for name, xy in (payload.get("points") or {}).items():
        if name not in known or not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError(f"Unknown or malformed point {name}")
        x, y = float(xy[0]), float(xy[1])
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"{name} is outside the frame")
        points[name] = (x, y)
    view = tuple(int(v) for v in payload.get("view") or ())
    if len(view) != 4 or view[2] <= 0 or view[3] <= 0 or view[0] < 0 or view[1] < 0 \
            or view[0] + view[2] > width or view[1] + view[3] > height:
        raise ValueError("The field area must sit inside the frame")
    return points, view


class Commands:
    """Things a lead does from the app that aren't match jobs. Runs apart from the sync loop because
    grabbing a frame takes seconds and an API stall must never hold one up."""

    def __init__(self, bridge: "Bridge", run=subprocess.run):
        self.bridge, self.run = bridge, run
        self.inbox: queue.Queue = queue.Queue()
        self.done: list[dict] = []
        self.seen: set[str] = set()
        self.lock = threading.Lock()

    def offer(self, command: dict) -> None:
        if command["id"] not in self.seen:
            self.seen.add(command["id"])
            self.inbox.put(command)

    def take_reports(self) -> list[dict]:
        with self.lock:
            reports, self.done = self.done, []
        return reports

    def put_back(self, reports: list[dict]) -> None:
        with self.lock:
            self.done = reports + self.done

    def work(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                command = self.inbox.get(timeout=1)
            except queue.Empty:
                continue
            self.handle(command)

    def handle(self, command: dict) -> None:
        report = {"id": command["id"], "status": "done", "result": {}, "error": None}
        try:
            if not EVENT.fullmatch(command.get("event_key") or ""):
                raise ValueError("Auto-Scout supports 2025 and 2026 events")
            action = {"capture_frame": self.capture_frame, "save_calibration": self.save_calibration}.get(command["kind"])
            report["result"] = action(command) if action else self.match_op(command)
        except (ValueError, TypeError, streams.StreamError, subprocess.SubprocessError, requests.RequestException) as e:
            report.update(status="failed", error=str(e)[:300] if isinstance(e, (ValueError, streams.StreamError))
                          else "That failed on the computer; see its log")
            log.warning("command %s failed: %r", command.get("kind"), e)
        with self.lock:
            self.done.append(report)

    def _source(self, command: dict):
        """Newest recording if this event is being followed, else a pasted link, else what's on air."""
        event = command["event_key"]
        link = (command.get("payload") or {}).get("source_url")
        if link:
            return streams.parse_source(link)
        for session in self.bridge.live.values():
            if session.job["event_key"] == event:
                found = streams.segments(session.folder)
                if len(found) > 1:
                    return found[-2][1]  # the newest may still be open
        info = TBA().get(f"/event/{event}", max_age_s=600)
        day = streams.event_day(time.time(), info.get("timezone"))
        url = streams.live_stream(info.get("webcasts") or [], day)
        if not url:
            raise streams.StreamError("Nothing is live for this event: paste a stream link")
        return url

    def capture_frame(self, command: dict) -> dict:
        import cv2

        from .field import Calibration

        event = command["event_key"]
        path = streams.grab_frame(self._source(command), config.EVENTS_DIR / f"{event}_ref.jpg")
        image = cv2.imread(str(path))
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise streams.StreamError("Could not read that frame")
        cal = Calibration.load(event)
        height, width = image.shape[:2]
        self.bridge.backend.call(
            "calibration", event_key=event, frame_jpeg=base64.b64encode(jpeg.tobytes()).decode(),
            frame_size=[width, height], view=list(cal.view) if cal else None,
            points={k: list(v) for k, v in cal.points.items()} if cal else None)
        return {"frame_size": [width, height]}

    def save_calibration(self, command: dict) -> dict:
        from .field import Calibration, FieldModel

        event = command["event_key"]
        payload = command.get("payload") or {}
        points, view = field_points(event, payload)
        fm = FieldModel.load(int(event[:4]))
        if sum(name in fm.landmarks for name in points) < 4:
            raise ValueError("Place at least 4 field landmarks")
        size = tuple(int(v) for v in payload.get("frame_size") or (1920, 1080))
        cal = Calibration.load(event) or Calibration(event_key=event, year=int(event[:4]), view=view, points={}, frame_size=size)
        cal.view, cal.points, cal.frame_size = view, points, size
        reference = config.EVENTS_DIR / f"{event}_ref.jpg"
        if reference.exists():
            cal.reference = str(reference.relative_to(config.ROOT))
        error = cal.fit(fm)
        cal.save()
        self.bridge.backend.call("calibration", event_key=event, view=list(view),
                                 points={k: list(v) for k, v in points.items()}, error_m=round(error, 3))
        return {"error_m": round(error, 3)}

    def match_op(self, command: dict) -> dict:
        """Recompute saved matches, then replace their results in the app."""
        from . import db

        op = COMMAND_OPS.get(command["kind"])
        keys = (command.get("payload") or {}).get("match_keys") or []
        event = command["event_key"]
        if not op or not keys or len(keys) > 100 or any(not MATCH.fullmatch(k) or not k.startswith(f"{event}_") for k in keys):
            raise ValueError("Pick up to 100 matches from this event")
        done, failed = [], []
        for key in keys:
            res = self.run([sys.executable, "-m", "autoscout.cli", op, key], cwd=config.ROOT, capture_output=True,
                           text=True, timeout=1800)
            with closing(db.connect()) as c:
                result = db.load_result(c, key)
            if res.returncode or result is None:
                failed.append(key)
                continue
            self.bridge.backend.call("refresh", match_key=key, envelope=export_result(result))
            done.append(key)
        return {"done": done, "failed": failed}


class Bridge:
    def __init__(self, backend: Backend, root: Path | None = None, runner=None, session_factory=LiveSession):
        self.backend = backend
        self.queue = Queue(root or config.DATA_DIR / "integration")
        self.root = self.queue.root
        self.clips = self.root / "clips"
        self.clips.mkdir(exist_ok=True)
        self.runner = runner or Runner(self.clips)
        self.session_factory = session_factory
        self.live: dict[str, LiveSession] = {}
        self.unknown_cancels: list[str] = []
        self.commands = Commands(self)
        with closing(self.db()) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS remote_jobs (
                id TEXT PRIMARY KEY, kind TEXT, job TEXT, local_id TEXT, state TEXT, error TEXT)""")
            c.execute("CREATE TABLE IF NOT EXISTS uploaded (remote_id TEXT, match_key TEXT, PRIMARY KEY (remote_id, match_key))")
            c.commit()

    def db(self):
        c = sqlite3.connect(self.root / "jobs.db", timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---- taking a job ------------------------------------------------------------------
    def reject_reason(self, job: dict) -> str | None:
        event = job.get("event_key") or ""
        if not EVENT.fullmatch(event):
            return "Auto-Scout supports 2025 and 2026 events"
        if not calibrated(event):
            return f"{event} is not calibrated yet: calibrate it on the Auto-Scout computer first"
        if free_gb() < MIN_FREE_GB:
            return "The Auto-Scout computer is nearly out of disk space"
        if job["kind"] != "live":
            keys = job.get("match_keys") or []
            if not keys or len(keys) > 100 or any(not MATCH.fullmatch(k) or not k.startswith(f"{event}_") for k in keys):
                return "Pick up to 100 matches from this event"
        if job.get("source_url"):  # none = use the event's stream from TBA
            try:
                streams.parse_source(job["source_url"])
            except streams.StreamError as e:
                return str(e)
        return None

    def accept(self, job: dict) -> None:
        with closing(self.db()) as c:
            if c.execute("SELECT 1 FROM remote_jobs WHERE id=?", (job["id"],)).fetchone():
                return
        state, local_id, error = "active", None, self.reject_reason(job)
        if error:
            state = "failed"
        elif job["kind"] == "live":
            if job.get("source_url"):
                job = {**job, "source_url": streams.parse_source(job["source_url"])}
        else:
            local_id = self._submit(job)
        with closing(self.db()) as c:
            c.execute("INSERT INTO remote_jobs VALUES (?,?,?,?,?,?)",
                      (job["id"], job["kind"], json.dumps(job), local_id, state, error))
            c.commit()
        if job["kind"] == "live" and not error:
            self._open_live(job)

    def _submit(self, job: dict) -> str:
        request = {"match_keys": job["match_keys"], "force": bool(job.get("force"))}
        if job["kind"] == "vod":
            url = streams.parse_source(job["source_url"]) if job.get("source_url") else None
            request["source"] = {"stream": True, "url": url, "stream_starts": self._stream_starts(job)}
        return self.queue.enqueue(json.dumps(request), job["id"], limit=None)["id"]

    @staticmethod
    def _stream_starts(job: dict) -> dict[str, float]:
        """A lead's calibrations in the app, by video id, beat the platform's start time."""
        from datetime import datetime

        return {vid: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                for vid, value in (job.get("stream_starts") or {}).items()}

    def _open_live(self, job: dict) -> None:
        def enqueue(match_key: str, clip: str) -> str:
            request = {"match_keys": [match_key], "force": bool(job.get("force")), "source": {"clip": clip}}
            return self.queue.enqueue(json.dumps(request), f"{job['id']}:{match_key}", limit=None)["id"]

        session = self.session_factory(job, self.root, enqueue)
        session.start()
        self.live[job["id"]] = session

    def resume(self) -> None:
        with closing(self.db()) as c:
            rows = c.execute("SELECT job FROM remote_jobs WHERE kind='live' AND state='active'").fetchall()
        for row in rows:
            self._open_live(json.loads(row["job"]))

    # ---- stopping a job ----------------------------------------------------------------
    def cancel(self, remote_id: str) -> None:
        with closing(self.db()) as c:
            row = c.execute("SELECT * FROM remote_jobs WHERE id=?", (remote_id,)).fetchone()
        if row is None:
            # Handed out, but the reply never reached us. Only this computer can confirm a stop.
            self.unknown_cancels.append(remote_id)
            return
        if row["local_id"]:
            self.queue.cancel(row["local_id"])
        elif row["kind"] == "live" and remote_id in self.live:
            self.live.pop(remote_id).stop()
            self._set_state(remote_id, "draining")  # matches already cut still get processed

    def _set_state(self, remote_id: str, state: str) -> None:
        with closing(self.db()) as c:
            c.execute("UPDATE remote_jobs SET state=? WHERE id=?", (state, remote_id))
            c.commit()

    # ---- reporting ---------------------------------------------------------------------
    def _local_ids(self, row, job: dict) -> list[str]:
        if row["local_id"]:
            return [row["local_id"]]
        return [i for i in self._live_state(job).get("seen", {}).values() if i]

    def _live_state(self, job: dict) -> dict:
        path = self.root / "live" / job["id"] / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def report(self, row) -> dict:
        job = json.loads(row["job"])
        if row["state"] == "failed":
            return {"id": row["id"], "status": "failed", "error": row["error"], "completed_matches": [],
                    "match_errors": {}, "progress": {}}
        locals_ = [self.queue.get(i) for i in self._local_ids(row, job)]
        completed = [k for j in locals_ for k in j["completed_matches"]]
        errors: dict[str, str] = {}
        for j in locals_:
            errors.update(self.queue.match_errors(j["id"]))
        progress: dict = {}
        statuses = {j["status"] for j in locals_}
        session = self.live.get(row["id"])
        if row["kind"] == "live":
            state = self._live_state(job)
            progress["clipped"] = sorted(k for k, i in state.get("seen", {}).items() if not i)
            progress["queued"] = [k for j in locals_ if j["status"] == "queued" for k in j["match_keys"]]
        if session:
            errors.update(session.errors)
            progress["recorder"] = session.recorder.status
            status = "running"
        elif row["kind"] == "live":
            errors.update(state.get("errors", {}))
            status = "running" if statuses & {"queued", "running", "cancelling"} else (
                "cancelled" if row["state"] == "draining" else "succeeded")
        else:
            status = locals_[0]["status"]
            if status == "cancelling":
                status = "running"
        running = next((j for j in locals_ if j["status"] in ("running", "cancelling")), None)
        if running:
            current = next((k for k in running["match_keys"]
                            if k not in running["completed_matches"] and k not in errors), None)
            if current:
                progress.update(self._match_progress(current))
        return {"id": row["id"], "status": status, "completed_matches": completed, "match_errors": errors,
                "error": locals_[0]["error"] if len(locals_) == 1 else None, "progress": progress}

    @staticmethod
    def _match_progress(match_key: str) -> dict:
        path = config.DATA_DIR / "progress" / f"{match_key}.json"
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return {"match_key": match_key}
        if d.get("stage") in ("done", "error"):
            return {"match_key": match_key}  # an earlier run's file: this one is still fetching footage
        return {"match_key": match_key, "stage": d.get("stage"), "pct": d.get("pct"), "eta_s": d.get("eta_s")}

    def upload_results(self, row) -> None:
        job = json.loads(row["job"])
        for local_id in self._local_ids(row, job):
            for mk in self.queue.get(local_id)["completed_matches"]:
                with closing(self.db()) as c:
                    done = c.execute("SELECT 1 FROM uploaded WHERE remote_id=? AND match_key=?",
                                     (row["id"], mk)).fetchone()
                if done:
                    continue
                envelope = json.loads((self.root / f"{local_id}_{mk}.json").read_text(encoding="utf-8"))
                try:
                    self.backend.call("result", job_id=row["id"], match_key=mk, envelope=envelope)
                except requests.HTTPError as e:
                    # A refusal will never change, and retrying would stop every heartbeat behind it.
                    # An outage or bad token is not one: those retry, so nothing is lost.
                    status = e.response.status_code if e.response is not None else 500
                    if status not in (413, 422):
                        raise
                    log.error("app refused %s (%s)", mk, status)
                    self.queue.skip(local_id, mk, "The app refused this result; see the computer's log")
                with closing(self.db()) as c:
                    c.execute("INSERT OR IGNORE INTO uploaded VALUES (?,?)", (row["id"], mk))
                    c.commit()

    def worker_info(self) -> dict:
        return {"calibrated_events": calibrated_events(), "free_gb": round(free_gb(), 1),
                "busy": bool(self.live) or self._any_active()}

    def _any_active(self) -> bool:
        with closing(self.queue.connect()) as c:
            return c.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running','cancelling') LIMIT 1").fetchone() is not None

    def tick(self) -> bool:
        """One exchange with the app. True while there is work in hand."""
        for job_id, session in list(self.live.items()):
            try:
                session.tick()
            except requests.RequestException:
                log.warning("TBA unreachable while following %s", job_id)
            if session.finished_recording(time.time()):
                self.live.pop(job_id).stop()
                self._set_state(job_id, "draining_done")
        with closing(self.db()) as c:
            rows = c.execute("SELECT * FROM remote_jobs WHERE state IN ('active','draining','draining_done','failed')").fetchall()
        reports = []
        for row in rows:
            self.upload_results(row)
            reports.append(self.report(row))
        stopped, self.unknown_cancels = self.unknown_cancels, []
        reports += [{"id": i, "status": "cancelled", "completed_matches": [], "match_errors": {},
                     "progress": {}, "error": None} for i in stopped]
        finished = self.commands.take_reports()
        try:
            reply = self.backend.call("sync", worker=self.worker_info(), jobs=reports, commands=finished)
        except (requests.RequestException, ValueError):
            self.unknown_cancels += stopped
            self.commands.put_back(finished)
            raise
        with closing(self.db()) as c:
            for r in reports:  # the app has the final word now, so stop reporting this one
                if r["status"] in ("succeeded", "failed", "cancelled"):
                    c.execute("UPDATE remote_jobs SET state='done' WHERE id=?", (r["id"],))
            c.commit()
        for job in reply.get("jobs", []):
            self.accept(job)
        for remote_id in reply.get("cancel", []):
            self.cancel(remote_id)
        for command in reply.get("commands", []):
            self.commands.offer(command)
        prune_clips(self.clips)
        return bool(reports or self.live or reply.get("jobs"))


def run() -> None:
    from filelock import FileLock

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    url = os.environ.get("AUTOSCOUT_BRIDGE_URL", "")
    token = os.environ.get("AUTOSCOUT_WORKER_TOKEN", "")
    if not url.startswith("https://") or len(token) < 32:
        raise SystemExit("Set AUTOSCOUT_BRIDGE_URL and AUTOSCOUT_WORKER_TOKEN (32+ characters) in .secrets/.env")
    bridge = Bridge(Backend(url, token))
    stop = threading.Event()

    def work():
        while not stop.is_set():
            if not bridge.queue.run_next(bridge.runner):
                stop.wait(1)

    with FileLock(str(bridge.root / "worker.lock"), timeout=0):
        bridge.queue.recover()
        bridge.resume()
        threading.Thread(target=work, daemon=True).start()
        threading.Thread(target=bridge.commands.work, args=(stop,), daemon=True).start()
        log.info("bridge up; %d calibrated events", len(calibrated_events()))
        failures = 0
        try:
            while True:
                try:
                    busy = bridge.tick()
                    failures = 0
                except (requests.RequestException, ValueError) as e:
                    # Venue and home networks drop. Only the wait changes: jobs stay safe in the app.
                    failures += 1
                    busy = False
                    log.warning("app unreachable (%s)", type(e).__name__)
                time.sleep(min(300, IDLE_S * 2 ** min(failures, 5)) if failures else BUSY_S if busy else IDLE_S)
        finally:
            stop.set()
            for session in bridge.live.values():
                session.stop()

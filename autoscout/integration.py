"""Dedicated remote API. Never expose the unrestricted calibration dashboard remotely."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config

MATCH = re.compile(r"(?:2025|2026)[a-z0-9]+_(?:qm[1-9][0-9]*|(?:ef|qf|sf|f)[1-9][0-9]*m[1-9][0-9]*)\Z")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_keys: list[str] = Field(min_length=1, max_length=100)
    force: bool = False

    @field_validator("match_keys")
    @classmethod
    def validate_matches(cls, keys):
        if any(not MATCH.fullmatch(k) or len(k) > 100 for k in keys):
            raise ValueError("Use full 2025/2026 TBA match keys, e.g. 2026txhou_qm11")
        return list(dict.fromkeys(keys))


class MatchSkipped(Exception):
    """One match cannot be done; the job carries on. The message is shown to the team lead."""


def export_result(result: dict) -> dict:
    """Versioned evidence envelope; do not fabricate human-report fields."""
    keys = ("match_key", "event_key", "year", "label", "comp_level", "match_number",
            "set_number", "red", "blue", "red_score", "blue_score", "video", "video_offset",
            "clock", "teams", "alliances", "events", "identity", "identity_conf",
            "identity_notes", "count_source", "auto_winner", "processing_s", "detector")
    return {"schema_version": 1, "source": "autoscout", "match": {k: result[k] for k in keys if k in result}}


class Queue:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE, fingerprint TEXT,
                request TEXT, status TEXT, completed TEXT, error TEXT,
                created_at REAL, updated_at REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS match_errors (
                job_id TEXT, match_key TEXT, error TEXT, PRIMARY KEY (job_id, match_key))""")
            c.commit()

    def connect(self):
        c = sqlite3.connect(self.root / "jobs.db", timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def submit(self, body: JobRequest, key: str):
        return self.enqueue(body.model_dump_json(), key)

    def enqueue(self, request: str, key: str, limit: int | None = 20):
        """`request` may carry a `source` the public API can't: only the bridge builds those.

        The bridge passes limit=None: the app caps its own queue, and a live event that outruns
        the GPU must back up here rather than lose matches.
        """
        fingerprint = hashlib.sha256(request.encode()).hexdigest()
        with closing(self.connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute("SELECT * FROM jobs WHERE request_key=?", (key,)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise HTTPException(409, "Idempotency-Key already used with different matches/options")
                return self.public(old)
            # Bound disk/GPU backlog independently of request size.
            if limit is not None and c.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= limit:
                raise HTTPException(429, "Queue full; retry later")
            jid = str(uuid.uuid4())
            now = time.time()
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                      (jid, key, fingerprint, request, "queued", "[]", None, now, now))
            c.commit()
        return self.get(jid)

    @staticmethod
    def public(row):
        return {"id": row["id"], "status": row["status"],
                "match_keys": json.loads(row["request"])["match_keys"],
                "completed_matches": json.loads(row["completed"]), "error": row["error"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def get(self, jid):
        with closing(self.connect()) as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown job")
        return self.public(row)

    def update(self, jid, status, completed, error=None):
        with closing(self.connect()) as c:
            c.execute("UPDATE jobs SET status=?,completed=?,error=?,updated_at=? WHERE id=?",
                      (status, json.dumps(completed), error, time.time(), jid))
            c.commit()

    def recover(self):
        with closing(self.connect()) as c:
            c.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            c.execute("UPDATE jobs SET status='cancelled' WHERE status='cancelling'")
            c.commit()

    def cancel(self, jid):
        """A queued job stops now; a running one stops after the match it is on."""
        with closing(self.connect()) as c:
            c.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE id=? AND status='queued'",
                      (time.time(), jid))
            c.execute("UPDATE jobs SET status='cancelling',updated_at=? WHERE id=? AND status='running'",
                      (time.time(), jid))
            c.commit()

    def status_of(self, jid):
        with closing(self.connect()) as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
        return row["status"] if row else None

    def skip(self, jid, match_key, error):
        with closing(self.connect()) as c:
            c.execute("INSERT OR REPLACE INTO match_errors VALUES (?,?,?)", (jid, match_key, error))
            c.commit()

    def match_errors(self, jid):
        with closing(self.connect()) as c:
            rows = c.execute("SELECT match_key,error FROM match_errors WHERE job_id=?", (jid,)).fetchall()
        return {r["match_key"]: r["error"] for r in rows}

    def run_next(self, runner):
        with closing(self.connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return False
            c.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (time.time(), row["id"]))
            c.commit()
        jid = row["id"]
        request = json.loads(row["request"])
        completed = json.loads(row["completed"])
        extra = (request["source"],) if request.get("source") else ()
        try:
            for mk in request["match_keys"]:
                if self.status_of(jid) == "cancelling":
                    self.update(jid, "cancelled", completed)
                    return True
                target = self.root / f"{jid}_{mk}.json"
                # Resume the exact job's snapshots, including force jobs, after a restart.
                if mk not in completed or not target.exists():
                    try:
                        raw = runner(mk, request["force"], self.root / f"{jid}.log", *extra)
                    except MatchSkipped as e:
                        self.skip(jid, mk, str(e))
                        continue
                    result = export_result(raw)
                    if result["match"].get("match_key") != mk:
                        raise ValueError("Processor returned the wrong match")
                    tmp = target.with_suffix(".tmp")
                    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
                    tmp.replace(target)
                    if mk not in completed:
                        completed.append(mk)
                    # A cancel that landed mid-match must survive this progress write.
                    self.update(jid, "cancelling" if self.status_of(jid) == "cancelling" else "running", completed)
            if self.status_of(jid) == "cancelling":
                self.update(jid, "cancelled", completed)
            elif completed or not self.match_errors(jid):
                self.update(jid, "succeeded", completed)
            else:
                self.update(jid, "failed", completed, "None of the matches could be processed")
        except Exception:
            # Full processing output is local; remote errors never contain paths/secrets.
            with (self.root / f"{jid}.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            self.update(jid, "failed", completed,
                        "Processing failed; check the computer's job log, calibration, TBA key and video availability")
        return True


def cached_result(match_key: str):
    from . import db

    with closing(db.connect()) as c:
        row = c.execute("SELECT status FROM matches WHERE match_key=?", (match_key,)).fetchone()
        return db.load_result(c, match_key) if row and row["status"] == "done" else None


def process_match(match_key: str, force: bool, log: Path, video: Path | None = None):
    from . import db

    if not force and (cached := cached_result(match_key)):
        return cached
    command = [sys.executable, "-m", "autoscout.cli", "run-match", match_key]
    if video:
        command += ["--video", str(video)]
    with log.open("a", encoding="utf-8") as out:
        subprocess.run(command, cwd=config.ROOT, stdout=out, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)
    with closing(db.connect()) as c:
        result = db.load_result(c, match_key)
    if result is None:
        raise RuntimeError("No saved result")
    return result


def create_app(*, root: Path | None = None, token: str | None = None,
               runner=process_match, start_worker: bool = True):
    secret = token if token is not None else os.environ.get("AUTOSCOUT_API_TOKEN", "")
    if len(secret) < 32:
        raise RuntimeError("Set AUTOSCOUT_API_TOKEN to a random secret of at least 32 characters")
    queue = Queue(root or config.DATA_DIR / "integration")

    @asynccontextmanager
    async def lifespan(app):
        from filelock import FileLock

        stop = threading.Event()
        def work():
            while not stop.is_set():
                if not queue.run_next(runner):
                    stop.wait(1)
        # A single service instance owns recovery and the serial GPU worker.
        with FileLock(str(queue.root / "worker.lock"), timeout=0):
            queue.recover()
            thread = threading.Thread(target=work, daemon=True)
            if start_worker:
                thread.start()
            try:
                yield
            finally:
                stop.set()
                if start_worker:
                    thread.join()  # Finish current job before releasing the worker lock.

    def authorize(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.encode(), f"Bearer {secret}".encode()):
            raise HTTPException(401, "Invalid API token")

    app = FastAPI(title="Auto-Scout Integration", version="1", lifespan=lifespan,
                  dependencies=[Depends(authorize)], docs_url=None, redoc_url=None, openapi_url=None)
    app.state.queue = queue

    @app.get("/v1/health")
    def health():
        return {"status": "online", "schema_version": 1, "supported_years": [2025, 2026]}

    @app.post("/v1/jobs", status_code=202)
    def submit(body: JobRequest, idempotency_key: str = Header(min_length=1, max_length=128)):
        return queue.submit(body, idempotency_key)

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: uuid.UUID):
        return queue.get(str(job_id))

    @app.get("/v1/jobs/{job_id}/results")
    def results(job_id: uuid.UUID):
        job = queue.get(str(job_id))
        return {"schema_version": 1, "job": job, "results": [
            json.loads((queue.root / f"{job_id}_{mk}.json").read_text(encoding="utf-8"))
            for mk in job["completed_matches"]]}

    return app

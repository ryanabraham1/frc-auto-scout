"""Lightweight per-match progress reporting (read by the dashboard's Processing page)."""
from __future__ import annotations

import json
import time

from . import config

DIR = config.DATA_DIR / "progress"


class Progress:
    # rough share of total processing time per stage, for an overall percentage
    STAGES = {"download": 0.05, "scoreboard": 0.08, "frames": 0.62, "identity": 0.15,
              "close-ups": 0.05, "analysis": 0.05}

    def __init__(self, match_key: str):
        self.key = match_key
        self.t0 = time.time()
        self.stage = "starting"
        self.frac = 0.0
        self._last = 0.0
        DIR.mkdir(parents=True, exist_ok=True)
        self.write(force=True)

    def update(self, stage: str, frac: float = 0.0, detail: str = "", force: bool = False):
        self.stage, self.frac, self.detail = stage, max(0.0, min(frac, 1.0)), detail
        self.write(force)

    def overall(self) -> float:
        done = 0.0
        for s, w in self.STAGES.items():
            if s == self.stage:
                return done + w * self.frac
            done += w
        return 1.0 if self.stage == "done" else done

    def write(self, force: bool = False, **extra):
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        pct = self.overall()
        el = now - self.t0
        d = {"match_key": self.key, "stage": self.stage, "stage_pct": round(100 * self.frac, 1),
             "pct": round(100 * pct, 1), "elapsed_s": round(el), "updated": now,
             "eta_s": round(el / pct - el) if 0.03 < pct < 1 else None,
             "detail": getattr(self, "detail", ""), **extra}
        (DIR / f"{self.key}.json").write_text(json.dumps(d))

    def done(self, error: str | None = None):
        self.stage = "error" if error else "done"
        self.frac = 1.0
        self.write(force=True, error=error)


def all_progress(max_age_s: float = 6 * 3600) -> list[dict]:
    out = []
    if DIR.exists():
        for p in DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if time.time() - d.get("updated", 0) < max_age_s:
                d["stale"] = d["stage"] not in ("done", "error") and time.time() - d["updated"] > 120
                out.append(d)
    return sorted(out, key=lambda d: -d["updated"])

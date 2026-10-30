"""The Blue Alliance API v3 client with an on-disk cache."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import config

BASE = "https://www.thebluealliance.com/api/v3"


class TBA:
    def __init__(self, key: str | None = None, cache_dir: Path | None = None, max_age_s: float = 3600):
        self.key = key or config.tba_key()
        self.cache_dir = cache_dir or (config.DATA_DIR / "tba_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_s = max_age_s
        self.session = requests.Session()
        self.session.headers["X-TBA-Auth-Key"] = self.key

    def get(self, path: str, max_age_s: float | None = None):
        max_age = self.max_age_s if max_age_s is None else max_age_s
        cache = self.cache_dir / (path.strip("/").replace("/", "__") + ".json")
        if cache.exists() and time.time() - cache.stat().st_mtime < max_age:
            return json.loads(cache.read_text())
        r = self.session.get(BASE + path, timeout=30)
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data))
        return data

    def event_matches(self, event_key: str) -> list[dict]:
        return self.get(f"/event/{event_key}/matches")

    def match(self, match_key: str) -> dict:
        return self.get(f"/match/{match_key}")

    def event(self, event_key: str) -> dict:
        return self.get(f"/event/{event_key}", max_age_s=86400)


@dataclass
class MatchInfo:
    key: str
    year: int
    event_key: str
    comp_level: str
    match_number: int
    set_number: int
    red: list[int]
    blue: list[int]
    videos: list[str]
    breakdown: dict = field(default_factory=dict)  # {"red": {...}, "blue": {...}}
    red_score: int | None = None
    blue_score: int | None = None

    @classmethod
    def from_tba(cls, m: dict) -> "MatchInfo":
        teams = lambda c: [int(t[3:]) for t in m["alliances"][c]["team_keys"]]
        vids = [v["key"] for v in m.get("videos") or [] if v.get("type") == "youtube"]
        return cls(
            key=m["key"],
            year=int(m["key"][:4]),
            event_key=m["event_key"],
            comp_level=m["comp_level"],
            match_number=m["match_number"],
            set_number=m.get("set_number", 1),
            red=teams("red"),
            blue=teams("blue"),
            videos=vids,
            breakdown=m.get("score_breakdown") or {},
            red_score=m["alliances"]["red"].get("score"),
            blue_score=m["alliances"]["blue"].get("score"),
        )

    def teams(self, alliance: str) -> list[int]:
        return self.red if alliance == "red" else self.blue

    @property
    def label(self) -> str:
        lvl = {"qm": "Q", "ef": "EF", "qf": "QF", "sf": "SF", "f": "F"}.get(self.comp_level, self.comp_level)
        return f"{lvl}{self.match_number}" if self.comp_level == "qm" else f"{lvl}{self.set_number}-{self.match_number}"


def sort_key(m: MatchInfo):
    order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
    return (order.get(m.comp_level, 9), m.set_number, m.match_number)

"""The Blue Alliance API v3 client with an on-disk cache."""
from __future__ import annotations

import json
import re
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


def team_number(team_key: str, event_key: str, event=None) -> int:
    """Offseason B teams have keys like frc2073B. Their FMS number (9990) is what the broadcast
    shows, and the rest of the pipeline needs one integer per robot, so use the event's remap.
    Without a remap the digits are still better than a crash that loses the match."""
    digits = team_key[3:]
    if digits.isdigit():
        return int(digits)
    try:
        remap = (event if event is not None else TBA().event(event_key)).get("remap_teams") or {}
    except (requests.RequestException, RuntimeError):
        remap = {}
    for fms, shown in remap.items():
        if shown == team_key and fms[3:].isdigit():
            return int(fms[3:])
    found = re.match(r"\d+", digits)
    if not found:
        raise ValueError(f"Unreadable team key {team_key}")
    return int(found.group())


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
    local_video: str | None = None
    video_source: dict = field(default_factory=dict)

    @classmethod
    def from_tba(cls, m: dict) -> "MatchInfo":
        teams = lambda c: [team_number(t, m["event_key"]) for t in m["alliances"][c]["team_keys"]]
        vids = [v["key"] for v in m.get("videos") or [] if v.get("type") == "youtube"]
        source = {}
        local_video = None
        if not vids:
            sources = config.EVENTS_DIR / f"{m['event_key']}_videos.json"
            if sources.exists():
                source = json.loads(sources.read_text(encoding="utf-8")).get(m["key"], {})
                if source.get("youtube_id"):
                    vids = [source["youtube_id"]]
                elif source.get("local_path"):
                    path = Path(source["local_path"])
                    local_video = str(path if path.is_absolute() else config.ROOT / path)
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
            local_video=local_video,
            video_source=source,
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

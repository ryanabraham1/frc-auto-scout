"""SQLite storage of processed matches (results JSON) and per-team-match rows for fast queries."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import config

DB_PATH = config.DATA_DIR / "autoscout.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  match_key TEXT PRIMARY KEY, event_key TEXT, year INT, label TEXT, comp_level TEXT,
  set_number INT, match_number INT, red TEXT, blue TEXT, red_score INT, blue_score INT,
  status TEXT, error TEXT, processed_at REAL, result_path TEXT
);
CREATE TABLE IF NOT EXISTS team_matches (
  match_key TEXT, team INT, event_key TEXT, year INT, alliance TEXT, station INT,
  metrics TEXT, PRIMARY KEY (match_key, team)
);
CREATE INDEX IF NOT EXISTS tm_team ON team_matches(team, event_key);
CREATE TABLE IF NOT EXISTS overrides (
  match_key TEXT, slot TEXT, team INT, PRIMARY KEY (match_key, slot)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_result(con: sqlite3.Connection, result: dict, result_path: str) -> None:
    mk = result["match_key"]
    con.execute(
        "REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mk, result["event_key"], result["year"], result["label"], result["comp_level"], result["set_number"],
         result["match_number"], json.dumps(result["red"]), json.dumps(result["blue"]), result.get("red_score"),
         result.get("blue_score"), "done", None, time.time(), result_path))
    con.execute("DELETE FROM team_matches WHERE match_key=?", (mk,))
    for team, m in result["teams"].items():
        con.execute("INSERT INTO team_matches VALUES (?,?,?,?,?,?,?)",
                    (mk, int(team), result["event_key"], result["year"], m["alliance"], m["station"], json.dumps(m)))
    con.commit()


def save_error(con: sqlite3.Connection, match_key: str, event_key: str, year: int, err: str) -> None:
    con.execute("INSERT OR IGNORE INTO matches (match_key, event_key, year) VALUES (?,?,?)", (match_key, event_key, year))
    con.execute("UPDATE matches SET status='error', error=?, processed_at=? WHERE match_key=?", (err, time.time(), match_key))
    con.commit()


def load_result(con: sqlite3.Connection, match_key: str) -> dict | None:
    r = con.execute("SELECT result_path FROM matches WHERE match_key=?", (match_key,)).fetchone()
    if not r or not r["result_path"] or not Path(r["result_path"]).exists():
        return None
    return json.loads(Path(r["result_path"]).read_text())


def set_override(con, match_key: str, slot: str, team: int) -> None:
    con.execute("REPLACE INTO overrides VALUES (?,?,?)", (match_key, slot, team))
    con.commit()


def get_overrides(con, match_key: str) -> dict[str, int]:
    return {r["slot"]: r["team"] for r in con.execute("SELECT slot, team FROM overrides WHERE match_key=?", (match_key,))}

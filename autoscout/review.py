"""Human review of scoring credit: the ground truth that attribution accuracy is measured against.

The dashboard's Review page shows a short clip around each scoring action the system credited and
asks a person which robot did it, without showing the system's answer. Labels are saved per match
in ``benchmarks/attribution/<match>.human.json`` (same layout as the AI-reviewed pilot file, so
both evaluate the same way) and can be committed.

Two kinds of action are reviewed:
  * ``score``: a score post (2025 score jumps, 2026 live-score increases), credited as shares
    among the alliance's robots. Correct when the reviewed team has the single largest share.
  * ``launch`` (2026): one tracked FUEL flight, credited to one robot or to the human player.

Candidates are put in a fixed pseudo-random order per match, so whatever prefix of the queue a
person gets through is a random sample of that match's actions, not the easy ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import config

LABEL_DIR = config.ROOT / "benchmarks" / "attribution"
SCHEMA_VERSION = 1
HP = "HP"  # a 2026 launch thrown by the human player, not a robot
STATUSES = ("accepted", "uncertain", "rejected")  # rejected: no such action happened
LAUNCH_KINDS = ("shot", "shot_inactive", "feed", "knocked", "human_player")
NOT_SCORES = ("coral_seen", "unexplained", "net_shot_seen", "scored", *LAUNCH_KINDS)
T_TOL = 0.05  # s: a label and a prediction at the same action


def label_path(match_key: str) -> Path:
    return LABEL_DIR / f"{match_key}.human.json"


def _order_key(match_key: str, cid: str) -> str:
    return hashlib.sha1(f"{match_key}|{cid}".encode()).hexdigest()


def candidates(result: dict) -> list[dict]:
    """Reviewable actions in a result, with the system's credit, in review order.

    Each: id, action, kind, t (video s), alliance, phase, points/n, start/end (launch pixels,
    full frame), predicted (top team, "HP", or None when tied/missing) and shares.
    """
    out = []
    seen = Counter()

    def add(c):
        base = f"{c['action']}:{c['alliance']}@{c['t']}"
        seen[base] += 1
        c["id"] = base if seen[base] == 1 else f"{base}#{seen[base]}"
        out.append(c)

    scored = defaultdict(dict)  # 2026: one credit row per team per score increase
    for e in result.get("events", []):
        a, kind = e.get("alliance"), e.get("kind")
        if a not in ("red", "blue") or e.get("t") is None:
            continue
        t = float(e["t"])
        if kind == "scored":
            if e.get("team") is not None:
                key = (a, t)
                scored[key][str(e["team"])] = scored[key].get(str(e["team"]), 0.0) + float(e.get("n") or 0)
                scored[key].setdefault("_period", e.get("period"))
            continue
        if kind in LAUNCH_KINDS:
            team = HP if kind == "human_player" else e.get("team")
            add({"action": "launch", "kind": kind, "t": t, "alliance": a, "phase": e.get("period"),
                 "n": e.get("n"), "start": e.get("start"), "end": e.get("end"), "made": bool(e.get("made")),
                 "shares": {str(team): 1.0} if team is not None else {}})
            continue
        if kind in NOT_SCORES or not e.get("shares"):
            continue
        add({"action": "score", "kind": kind, "t": t, "alliance": a, "phase": e.get("phase"),
             "points": e.get("points"), "shares": {str(k): float(v) for k, v in e["shares"].items()}})
    for (a, t), sh in scored.items():
        period = sh.pop("_period", None)
        tot = sum(sh.values())
        add({"action": "score", "kind": "fuel", "t": t, "alliance": a, "phase": period,
             "shares": {k: round(v / tot, 3) for k, v in sh.items()} if tot > 0 else {}})
    for c in out:
        c["predicted"] = top_credit(c["shares"])
    return sorted(out, key=lambda c: _order_key(result.get("match_key", ""), c["id"]))


def top_credit(shares: dict) -> int | str | None:
    """The single team (or "HP") with the largest share; None when empty or tied."""
    if not shares:
        return None
    best = max(shares.values())
    winners = [k for k, v in shares.items() if abs(v - best) < 1e-9]
    if len(winners) != 1:
        return None
    return _team(winners[0])


def _team(v):
    if v is None or v == HP:
        return v
    return int(v)


def blind(c: dict) -> dict:
    """A candidate without the system's answer, for showing while a person labels it."""
    return {k: v for k, v in c.items() if k not in ("predicted", "shares")}


# ---- label file -------------------------------------------------------------------------------
def load_labels(match_key: str) -> dict | None:
    p = label_path(match_key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def new_label_file(result: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "match_key": result["match_key"], "video_id": result.get("video"),
        "reviewer_type": "human_dashboard",
        "scope": "Scorer identity for system-detected candidate actions, reviewed blind in the dashboard; "
                 "cannot measure actions the system never detected",
        "protocol": {"prediction_visibility": "The system's credit is hidden until a label is saved.",
                     "sampling": "Candidates are reviewed in a fixed pseudo-random order per match."},
        "teams": {"red": list(result.get("red", [])), "blue": list(result.get("blue", []))},
        "labels": [],
    }


def save_label(result: dict, cand: dict, team, status: str, reviewer: str = "", note: str = "") -> dict:
    """Add or replace one label; returns the saved label."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    team = _team(team) if team not in ("", None) else None
    teams = result.get(cand["alliance"], [])
    if status == "accepted":
        allowed = list(teams) + ([HP] if cand["action"] == "launch" else [])
        if team not in allowed:
            raise ValueError(f"team must be one of {allowed}")
    else:
        team = None
    from filelock import FileLock

    p = label_path(result["match_key"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(p) + ".lock"):
        doc = load_labels(result["match_key"]) or new_label_file(result)
        label = {"id": cand["id"], "action": cand["action"], "kind": cand.get("kind"), "alliance": cand["alliance"],
                 "post_t": cand["t"], "team": team, "status": status,
                 "reviewer": reviewer[:60], "evidence": note[:500], "saved_at": round(time.time())}
        doc["labels"] = [x for x in doc["labels"] if x["id"] != cand["id"]] + [label]
        doc["labels"].sort(key=lambda x: (x["alliance"], x["post_t"], x["id"]))
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    return label


def delete_label(match_key: str, cid: str) -> None:
    from filelock import FileLock

    p = label_path(match_key)
    if not p.exists():
        return
    with FileLock(str(p) + ".lock"):
        doc = load_labels(match_key)
        doc["labels"] = [x for x in doc["labels"] if x["id"] != cid]
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)


# ---- evaluation -------------------------------------------------------------------------------
def evaluate(truth: dict, result: dict) -> dict:
    """Score labels against a result. Missing and tied predictions count as errors."""
    if truth["match_key"] != result["match_key"]:
        raise ValueError("Label and result match keys differ")
    cands = candidates(result)
    rows = []
    for label in truth["labels"]:
        if label["status"] != "accepted":
            continue
        action = label.get("action", "score")
        team = _team(label["team"])
        hits = [c for c in cands if c["action"] == action and c["alliance"] == label["alliance"]
                and abs(c["t"] - float(label["post_t"])) < T_TOL]
        if len(hits) > 1:  # several actions at one time: the id says which
            hits = [c for c in hits if c["id"] == label["id"]] or hits
        pred, mass, reason = None, 0.0, "missing_prediction"
        if len(hits) > 1:
            reason = "ambiguous_prediction_alignment"
        elif hits:
            c = hits[0]
            mass = float(c["shares"].get(str(team), 0.0))
            pred = c["predicted"]
            reason = "tied_top_credit" if pred is None and c["shares"] else (
                "missing_prediction" if pred is None else "correct" if pred == team else "wrong_team")
        rows.append({"id": label["id"], "action": action, "kind": label.get("kind"), "alliance": label["alliance"],
                     "t": label["post_t"], "reviewed_team": team, "predicted_team": pred,
                     "correct": reason == "correct", "credit_to_reviewed_team": round(mass, 3), "reason": reason})
    statuses = Counter(x["status"] for x in truth["labels"])
    return {"match_key": truth["match_key"], "reviewer_type": truth.get("reviewer_type"),
            "labels": len(truth["labels"]), "label_statuses": dict(statuses),
            "candidates": len(cands), "primary": summarize(rows),
            "by_action": {a: summarize([r for r in rows if r["action"] == a]) for a in sorted({r["action"] for r in rows})},
            "alliances": {a: summarize([r for r in rows if r["alliance"] == a]) for a in ("red", "blue")},
            "rows": rows}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {"accepted": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None,
            "mean_credit_to_reviewed_team": round(sum(r["credit_to_reviewed_team"] for r in rows) / n, 4) if n else None,
            "reasons": dict(Counter(r["reason"] for r in rows))}


def evaluate_all(results_dir: Path, event_key: str | None = None, label_dir: Path | None = None,
                 include_ai: bool = False) -> dict:
    """Every labelled match with a saved result, pooled. AI-reviewed pilot files only with include_ai."""
    label_dir = label_dir or LABEL_DIR
    files = sorted(label_dir.glob("*.human.json"))
    if include_ai:
        files += [p for p in sorted(label_dir.glob("*.json"))
                  if p.name.count(".") == 1 and json.loads(p.read_text(encoding="utf-8")).get("labels") is not None]
    matches, rows, rejected, uncertain = [], [], 0, 0
    for p in files:
        truth = json.loads(p.read_text(encoding="utf-8"))
        mk = truth["match_key"]
        if event_key and not mk.startswith(event_key + "_"):
            continue
        rp = Path(results_dir) / f"{mk}.json"
        if not rp.exists():
            matches.append({"match_key": mk, "label_file": p.name, "error": "no saved result"})
            continue
        rep = evaluate(truth, json.loads(rp.read_text(encoding="utf-8")))
        rejected += rep["label_statuses"].get("rejected", 0)
        uncertain += rep["label_statuses"].get("uncertain", 0)
        rows += [dict(r, match_key=mk) for r in rep["rows"]]
        matches.append({"match_key": mk, "label_file": p.name, "reviewer_type": rep["reviewer_type"],
                        "candidates": rep["candidates"], "labels": rep["labels"], **rep["primary"]})
    reviewed = len(rows) + rejected + uncertain
    return {"event_key": event_key, "matches": matches, "primary": summarize(rows),
            "by_action": {a: summarize([r for r in rows if r["action"] == a]) for a in sorted({r["action"] for r in rows})},
            "by_kind": {k: summarize([r for r in rows if str(r["kind"]) == k]) for k in sorted({str(r["kind"]) for r in rows})},
            "rejected": rejected, "uncertain": uncertain,
            "false_candidate_rate": round(rejected / reviewed, 4) if reviewed else None,
            "rows": rows}

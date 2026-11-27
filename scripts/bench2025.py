"""2025 accuracy benchmark that does not grade the scouter against its own inputs.

The scouter decodes the broadcast score's jumps and splits the official FMS counts among robots, so its
totals always equal the final score: comparing the two says nothing. Human scouting data is not used
either (scouts miscount). Every number here is measured against something FMS recorded exactly, with
the part being tested kept away from that record:

  reef     The REEF read from vision alone (score and breakdown hidden): which branches A-L x L2-L4 hold
           CORAL at the end of AUTO and of the match, against the official node map. Precision, recall,
           and the REEF points those branches add up to against the official REEF points. This is the
           "does the score we see match the real score" check, and it grades the 3D cameras and the
           CORAL detector directly.
  timing   With the node map known: how many official branches vision timed, and how many of those times
           line up with a score jump of that level (the scoreboard clock is a second witness).
  fms      Per-robot facts FMS records: AUTO coral credited to a robot that never left its start line
           (provably wrong), and whether the robot we call team X is where FMS says team X ended the
           match (BARGE) and whether it moved in AUTO (LEAVE). These test identity and tracking.
  plays    Scores a person checked on footage (benchmarks/attribution/*.user_verified.json), then the
           AI-read footage checks, reported separately.
  opr      Each alliance's official count split by component OPR fitted on every OTHER qualification
           match (leave-one-match-out). Noisy for one match, unbiased over an event: use it to compare
           two versions (paired bootstrap), not as a per-match grade.
  decode   Decoded score-jump actions against official counts.
  scouts   Optional (data/labels/scouting_<event>.json, not in the repo): a team's public scout sheets,
           used ONLY where the three scouts' counts for an alliance, phase and level add up to exactly
           what FMS counted. Scouts miscount; a cell whose sum is right is unlikely to hide two errors
           that cancel. Reported apart from the FMS checks, with how many cells passed.

usage:
  python scripts/bench2025.py EVENT [--matches FILE.json] [--only reef,fms,...] [--out FILE.json]
  python scripts/bench2025.py EVENT --set autoscout.games.reefscape2025.NODE_CREDIT=False   # A/B a constant
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from autoscout import config  # noqa: E402

RESULTS = config.DATA_DIR / "results"
LABELS = Path("benchmarks/attribution")
CORAL = ("L1", "L2", "L3", "L4")
COMPONENTS = ("L1", "L2", "L3", "L4", "net", "processor")
AUTO_PTS = {"L2": 4, "L3": 6, "L4": 7}
TELE_PTS = {"L2": 3, "L3": 4, "L4": 5}
CAGE_DX, MOVED_M = 0.9, 0.6  # reefscape2025.identity_evidence's own thresholds


# ---- inputs -------------------------------------------------------------------------------------
def event_matches(event: str) -> dict:
    from autoscout.tba import TBA, MatchInfo

    ms = [MatchInfo.from_tba(m) for m in TBA().event_matches(event) if m["comp_level"] == "qm"]
    return {m.key: m for m in ms if m.breakdown}


def default_keys(event: str, byk: dict) -> list[str]:
    """Processed qualification matches that also have the CORAL detector's boxes."""
    return sorted((k for k in byk if (RESULTS / f"{k}.json").exists() and (RESULTS / f"{k}_coral.json").exists()),
                  key=lambda k: byk[k].match_number)


def _apply(patches: dict) -> None:
    for name, value in patches.items():
        mod, attr = name.rsplit(".", 1)
        setattr(importlib.import_module(mod), attr, value)


def _one(args) -> dict:
    """Reanalyse one match (nothing saved) and keep what the checks read, plus the track-based facts."""
    key, patches, keys = args
    _apply(patches)
    from autoscout import pipeline
    from autoscout.games import get

    pipeline.PRIOR_MATCH_KEYS = set(keys)
    r = pipeline.reanalyze(key, save=False)
    fm = get(r["year"]).field
    return {"match_key": key, "teams": {int(t): v for t, v in r["teams"].items()}, "events": r["events"],
            "red": r["red"], "blue": r["blue"], "clock": r["clock"], "facts": track_facts(r, fm)}


def track_facts(r: dict, fm) -> dict:
    """{team: {"cage": bool|None, "moved": bool|None}} from where the robot we call that team was seen:
    within CAGE_DX of the BARGE line in the last seconds, and how far it went during AUTO. None: not seen."""
    clock, tl = r["clock"], r.get("identity_timeline") or {}
    out = {}
    end, a0, a1 = clock["teleop_end_vt"], clock["auto_end_vt"] - clock["auto_len"], clock["auto_end_vt"]
    rows = defaultdict(list)
    for slot, tr in r["raw_tracks"].items():
        spans = tl.get(slot, [])
        for row in tr:
            if not row[3]:
                continue  # coasted positions are guesses
            team = next((tm for s, e, tm in spans if s <= row[0] < e), r["identity"].get(slot))
            if team is not None:
                rows[int(team)].append((row[0], row[1], row[2]))
    for team in r["red"] + r["blue"]:
        P = np.array(sorted(rows.get(team, []))).reshape(-1, 3)
        late = P[(P[:, 0] >= end - 4) & (P[:, 0] <= end + 1)]
        auto = P[(P[:, 0] >= a0) & (P[:, 0] <= a1)]
        out[team] = {"cage": bool(np.abs(late[:, 1] - fm.length / 2).min() < CAGE_DX) if len(late) else None,
                     "moved": bool(np.linalg.norm(auto[-1, 1:] - auto[0, 1:]) > MOVED_M) if len(auto) > 3 else None}
    return out


def reanalyzed(keys: list[str], patches: dict | None = None, workers: int = 6) -> dict:
    patches = patches or {}
    with ProcessPoolExecutor(workers) as ex:
        return {r["match_key"]: r for r in ex.map(_one, [(k, patches, keys) for k in keys])}


# ---- reef: vision alone against the official node map -------------------------------------------
def _cams(event: str, key: str, alliance: str) -> dict:
    from autoscout.vision import camera3d, reef3d

    cams = reef3d.views(event, alliance)
    wide = camera3d.load_event_camera(event)
    p = config.DATA_DIR / "cam3d_shift" / f"{key}.json"
    if wide is not None and p.exists():
        dx, dy, n = json.loads(p.read_text())
        if n:
            wide = wide.shifted(dx, dy)
    if wide is not None:
        cams["wide"] = wide
    return cams


def _reef_one(args) -> list[dict]:
    key, event, patches = args
    _apply(patches)
    from autoscout.tba import TBA, MatchInfo
    from autoscout.vision import reef3d

    m = MatchInfo.from_tba(TBA().match(key))
    obs = json.loads((RESULTS / f"{key}_coral.json").read_text())
    clock = json.loads((RESULTS / f"{key}.json").read_text())["clock"]
    rows = []
    for a in ("red", "blue"):
        seen = reef3d.read_reef(obs, _cams(event, key, a), a)
        end = {(n["branch"], n["level"]) for n in seen if n["level"] != "L1"}
        auto = {(n["branch"], n["level"]) for n in seen if n["level"] != "L1" and n["t"] <= clock["auto_end_vt"] + 2.0}
        off = reef3d.official_nodes(m.breakdown[a])
        o_end = {(n["branch"], n["level"]) for n in off}
        o_auto = {(n["branch"], n["level"]) for n in off if n["phase"] == "auto"}
        pts = lambda full, au: sum(AUTO_PTS[n[1]] if n in au else TELE_PTS[n[1]] for n in full)  # noqa: E731
        rows.append({"match_key": key, "alliance": a, "tp": len(end & o_end), "fp": len(end - o_end),
                     "fn": len(o_end - end), "auto_tp": len(auto & o_auto), "auto_fp": len(auto - o_auto),
                     "auto_fn": len(o_auto - auto), "points": pts(end, auto & end), "official_points": pts(o_end, o_auto),
                     "trough": sum(n["level"] == "L1" for n in seen),
                     "official_trough": m.breakdown[a]["teleopReef"]["trough"] + m.breakdown[a]["autoReef"]["trough"],
                     "wrong": sorted(f"{b}{L[1]}" for b, L in end - o_end), "missed": sorted(f"{b}{L[1]}" for b, L in o_end - end)})
    return rows


def reef(keys: list[str], event: str, patches: dict | None = None, workers: int = 6) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_reef_one, [(k, event, patches or {}) for k in keys]) for r in rs]
    s = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "auto_tp", "auto_fp", "auto_fn", "points", "official_points")}
    pr = lambda tp, fp, fn: (tp / max(tp + fp, 1), tp / max(tp + fn, 1))  # noqa: E731
    p, rc = pr(s["tp"], s["fp"], s["fn"])
    ap, ar = pr(s["auto_tp"], s["auto_fp"], s["auto_fn"])
    got, off = np.array([r["points"] for r in rows], float), np.array([r["official_points"] for r in rows], float)
    tg, to = np.array([r["trough"] for r in rows], float), np.array([r["official_trough"] for r in rows], float)
    return {"alliances": len(rows), "branches": s["tp"] + s["fn"], "precision": p, "recall": rc,
            "f1": 2 * p * rc / max(p + rc, 1e-9), "auto_precision": ap, "auto_recall": ar,
            "auto_f1": 2 * ap * ar / max(ap + ar, 1e-9), "auto_branches": s["auto_tp"] + s["auto_fn"],
            "exact_alliances": float(np.mean([r["fp"] == 0 and r["fn"] == 0 for r in rows])),
            "points_ratio": float(got.sum() / max(off.sum(), 1)), "points_mae": float(np.abs(got - off).mean()),
            "points_r": float(np.corrcoef(got, off)[0, 1]) if got.std() > 0 else 0.0,
            "trough_mae": float(np.abs(tg - to).mean()), "trough_ratio": float(tg.sum() / max(to.sum(), 1)),
            "rows": rows}


# ---- timing: vision's fill times against the scoreboard's ---------------------------------------
def timing(results: dict) -> dict:
    """Of the scored L2-L4 actions, how many were paired with a branch, timed by its own coral, and
    found a robot at the spot (the three steps node credit needs)."""
    n = paired = timed = spot = 0
    for r in results.values():
        for e in r["events"]:
            if e.get("kind") in ("L2", "L3", "L4") and "shares" in e:
                n += 1
                node = e.get("node")
                if node:
                    paired += 1
                    timed += node["timed"] == "coral"
                    spot += max(node["at_spot"].values(), default=0.0) >= 0.5
    return {"scores": n, "paired": paired / max(n, 1), "timed_by_coral": timed / max(n, 1),
            "robot_at_spot": spot / max(n, 1)}


# ---- fms: per-robot facts the field recorded ----------------------------------------------------
def fms(results: dict, byk: dict) -> dict:
    out = {}
    wrong = tot = eq = 0.0
    n_all = 0
    for key, r in results.items():
        m = byk[key]
        for a in ("red", "blue"):
            bd, teams = m.breakdown[a], m.teams(a)
            left = [bd.get(f"autoLineRobot{i}") == "Yes" for i in (1, 2, 3)]
            auto = [r["teams"][t]["auto_coral"] for t in teams]
            if all(left) or sum(auto) == 0:
                continue
            n_all += 1
            tot += sum(auto)
            wrong += sum(c for c, ok in zip(auto, left) if not ok)
            eq += sum(auto) * (3 - sum(left)) / 3
    out["auto_coral_to_non_leavers"] = {"alliances": n_all, "auto_coral": tot, "wrong": wrong,
                                        "share": wrong / max(tot, 1), "equal_split_share": eq / max(tot, 1)}
    for name, fact, official in (("barge", "cage", lambda bd, i: bd.get(f"endGameRobot{i}") in ("DeepCage", "ShallowCage")),
                                 ("leave", "moved", lambda bd, i: bd.get(f"autoLineRobot{i}") == "Yes")):
        ok = n = unseen = 0
        ok_al = n_al = 0
        for key, r in results.items():
            m = byk[key]
            for a in ("red", "blue"):
                truth = [official(m.breakdown[a], i) for i in (1, 2, 3)]
                if all(v == truth[0] for v in truth):
                    continue  # all three the same: says nothing about who is who
                got = [r["facts"][t][fact] for t in m.teams(a)]
                seen = [(g, v) for g, v in zip(got, truth) if g is not None]
                unseen += 3 - len(seen)
                n += len(seen)
                ok += sum(g == v for g, v in seen)
                if len(seen) == 3:
                    n_al += 1
                    ok_al += all(g == v for g, v in seen)
        out[name] = {"robots": n, "agree": ok / max(n, 1), "not_seen": unseen,
                     "alliances_all_seen": n_al, "alliances_all_agree": ok_al / max(n_al, 1)}
    return out


# ---- plays: scores checked on footage -----------------------------------------------------------
def plays(results: dict) -> dict:
    out = {}
    for kind in ("user_verified", "footage_checks"):
        rows = []
        for p in sorted(LABELS.glob(f"2025*.{kind}.json")):
            d = json.loads(p.read_text())
            r = results.get(d["match_key"])
            if r is None:
                continue
            for lab in d["labels"]:
                if lab.get("status") != "accepted":
                    continue
                ev = [e for e in r["events"] if e.get("shares") and e.get("alliance") == lab["alliance"]
                      and e.get("kind") == lab["kind"] and abs(e["t"] - lab["post_t"]) < 0.3]
                shares = {int(k): v for k, v in ev[0]["shares"].items()} if len(ev) == 1 else {}
                top = max(shares, key=shares.get) if shares else None
                rows.append({"id": f"{d['match_key']}:{lab['id']}", "team": lab["team"], "got": top,
                             "credit": shares.get(lab["team"], 0.0)})
        out[kind] = {"plays": len(rows), "right": sum(x["got"] == x["team"] for x in rows),
                     "mean_credit": float(np.mean([x["credit"] for x in rows])) if rows else None,
                     "wrong": [x for x in rows if x["got"] != x["team"]]}
    return out


# ---- opr: leave-one-match-out component OPR split -----------------------------------------------
def _official(bd_a: dict) -> dict:
    ar, tr = bd_a["autoReef"], bd_a["teleopReef"]
    rows = {"L4": "tba_topRowCount", "L3": "tba_midRowCount", "L2": "tba_botRowCount"}
    out = {L: tr[k] for L, k in rows.items()}  # teleop rows are the end state, auto included
    out["L1"] = ar["trough"] + tr["trough"]
    out["net"], out["processor"] = bd_a.get("netAlgaeCount", 0), bd_a.get("wallAlgaeCount", 0)
    return out


def opr_expected(byk: dict, keys: list[str], ridge: float = 0.5) -> dict:
    """{match: {alliance: {component: {team: expected count}}}}: the official count split by each team's
    component OPR over all the event's other matches (ridge toward the event mean, clipped at 0)."""
    teams = sorted({t for m in byk.values() for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    rows, vals, owner = [], [], []
    for k, m in byk.items():
        for a in ("red", "blue"):
            row = np.zeros(len(teams))
            row[[idx[t] for t in m.teams(a)]] = 1
            rows.append(row)
            vals.append([_official(m.breakdown[a])[c] for c in COMPONENTS])
            owner.append(k)
    A, B, owner = np.array(rows), np.array(vals, float), np.array(owner)
    out = {}
    for key in keys:
        keep = owner != key
        Ak, Bk = A[keep], B[keep]
        mean = Bk.mean(0) / 3
        x = np.linalg.solve(Ak.T @ Ak + ridge * np.eye(len(teams)), Ak.T @ (Bk - Ak.sum(1, keepdims=True) * mean)) + mean
        x = np.clip(x, 0.0, None)
        out[key] = {}
        m = byk[key]
        for a in ("red", "blue"):
            off = _official(m.breakdown[a])
            out[key][a] = {}
            for ci, c in enumerate(COMPONENTS):
                w = np.array([x[idx[t], ci] for t in m.teams(a)])
                w = w / w.sum() if w.sum() > 1e-6 else np.full(3, 1 / 3)
                out[key][a][c] = dict(zip(m.teams(a), off[c] * w))
    return out


def _ours(r: dict, team: int, c: str) -> float:
    x = r["teams"][team]
    return float(x[f"coral_{c}"] if c in CORAL else x[f"algae_{c}"])


def opr(results: dict, byk: dict, expected: dict | None = None) -> dict:
    """agree = 1 - sum|ours - expected| / (2 * official), per component and for coral as a whole; equal
    = the same for a three-way split. per_match: the numerators, for paired bootstrap between versions."""
    expected = expected or opr_expected(byk, list(results))
    per_match = {}
    for key, r in results.items():
        m = byk[key]
        row = defaultdict(float)
        for a in ("red", "blue"):
            for c in COMPONENTS:
                e = expected[key][a][c]
                n = sum(e.values())
                row[f"n_{c}"] += n
                row[f"err_{c}"] += sum(abs(_ours(r, t, c) - e[t]) for t in m.teams(a))
                row[f"eq_{c}"] += sum(abs(n / 3 - e[t]) for t in m.teams(a))
        per_match[key] = dict(row)
    out = {"per_match": per_match}
    for name, comps in (("coral", CORAL), ("L1", ("L1",)), ("L2-L4", ("L2", "L3", "L4")), ("L4", ("L4",)),
                        ("net", ("net",)), ("processor", ("processor",))):
        n = sum(v[f"n_{c}"] for v in per_match.values() for c in comps)
        out[name] = {"official": n,
                     "agree": 1 - sum(v[f"err_{c}"] for v in per_match.values() for c in comps) / max(2 * n, 1),
                     "equal": 1 - sum(v[f"eq_{c}"] for v in per_match.values() for c in comps) / max(2 * n, 1)}
    return out


def bootstrap_diff(a: dict, b: dict, comps=CORAL, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in OPR agreement (b - a) with a 95% range, resampling matches (paired)."""
    keys = sorted(set(a["per_match"]) & set(b["per_match"]))
    N = np.array([sum(a["per_match"][k][f"n_{c}"] for c in comps) for k in keys])
    Ea = np.array([sum(a["per_match"][k][f"err_{c}"] for c in comps) for k in keys])
    Eb = np.array([sum(b["per_match"][k][f"err_{c}"] for c in comps) for k in keys])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(keys), (n, len(keys)))
    d = (Ea[i].sum(1) - Eb[i].sum(1)) / np.maximum(2 * N[i].sum(1), 1)
    return float((Ea.sum() - Eb.sum()) / max(2 * N.sum(), 1)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---- scouts: human counts, kept only where FMS confirms their sum ---------------------------------
def scouts(results: dict, byk: dict, event: str) -> dict | None:
    from autoscout.games.reefscape2025 import Reefscape2025

    p = config.DATA_DIR / "labels" / f"scouting_{event}.json"
    if not p.exists():
        return None
    rows = defaultdict(dict)
    for x in json.loads(p.read_text())["rows"]:
        reps = x["reports"]
        rows[(x["match_key"], x["alliance"])][x["team"]] = {
            **{ph: {L: float(np.mean([q[ph][L] for q in reps])) for L in CORAL} for ph in ("auto", "teleop")},
            "algae": {c: float(np.mean([q[c] for q in reps])) for c in ("net", "processor")}}
    tot = defaultdict(float)
    cells = passed = 0
    detail = []
    per_match = defaultdict(lambda: defaultdict(float))
    for key, r in results.items():
        m = byk[key]
        for a in ("red", "blue"):
            sc = rows.get((key, a), {})
            if set(sc) != set(m.teams(a)):
                continue
            off = Reefscape2025.official_counts(None, m.breakdown[a])
            todo = [(ph, L, off[ph][L], ("L1" if L == "L1" else "L2-L4", ph, "coral")) for ph in ("auto", "teleop") for L in CORAL]
            todo += [("algae", c, off["teleop"][c], (c,)) for c in ("net", "processor")]
            for ph, L, n, groups in todo:
                if n == 0:
                    continue
                cells += 1
                if abs(sum(sc[t][ph][L] for t in sc) - n) > 1e-6:
                    continue
                passed += 1
                cell = {}
                for t in m.teams(a):
                    x = r["teams"][t]
                    if ph == "algae":
                        ours = x[f"algae_{L}"]
                    else:
                        au = x["auto_coral_by_level"][L]
                        ours = au if ph == "auto" else x[f"coral_{L}"] - au
                    cell[t] = (sc[t][ph][L], ours)
                    for g in groups:
                        tot[f"err_{g}"] += abs(ours - sc[t][ph][L])
                        tot[f"eq_{g}"] += abs(n / 3 - sc[t][ph][L])
                        per_match[key][f"err_{g}"] += abs(ours - sc[t][ph][L])
                for g in groups:
                    tot[f"n_{g}"] += n
                    per_match[key][f"n_{g}"] += n
                detail.append({"match_key": key, "alliance": a, "phase": ph, "kind": L, "official": n,
                               "scout_ours": {str(t): v for t, v in cell.items()},
                               "moved": sum(abs(o - w) for w, o in cell.values()) / 2})
    out = {"cells": cells, "cells_fms_confirmed": passed, "per_match": {k: dict(v) for k, v in per_match.items()},
           "detail": sorted(detail, key=lambda d: -d["moved"]), "pinned": pinned_plays(results, detail)}
    for g in ("coral", "L2-L4", "L1", "auto", "teleop", "net", "processor"):
        n = tot[f"n_{g}"]
        out[g] = {"official": n, "right": 1 - tot[f"err_{g}"] / max(2 * n, 1), "equal": 1 - tot[f"eq_{g}"] / max(2 * n, 1)}
    return out


def pinned_plays(results: dict, detail: list[dict]) -> dict:
    """Single plays whose scorer is known: in a confirmed cell where one robot scored everything at that
    level and phase, each of the alliance's scores there is that robot's. Grouped by the evidence the
    credit rested on, so the weak kind shows."""
    rows = []
    for d in detail:
        who = [int(t) for t, (w, _o) in d["scout_ours"].items() if w > 0]
        if d["phase"] == "algae" or len(who) != 1:
            continue
        for e in results[d["match_key"]]["events"]:
            if (e.get("alliance"), e.get("kind"), e.get("phase")) != (d["alliance"], d["kind"], d["phase"]) or "shares" not in e:
                continue
            node = e.get("node")
            if d["kind"] == "L1":
                how = "L1"
            elif not node:
                how = "no branch"
            else:
                best = max(node["at_spot"].values(), default=0.0)
                how = (("coral-timed" if node["timed"] == "coral" else "score-timed")
                       + (", robot at spot" if best >= 0.5 else ", robot near spot" if best >= 0.1 else ", nobody at spot"))
            sh = {int(k): v for k, v in e["shares"].items()}
            rows.append({"match_key": d["match_key"], "alliance": d["alliance"], "t": e["t"], "kind": d["kind"],
                         "phase": d["phase"], "scorer": who[0], "got": max(sh, key=sh.get), "credit": sh.get(who[0], 0.0),
                         "how": how, "shares": sh, "node": node})
    groups = defaultdict(list)
    for x in rows:
        groups[x["how"]].append(x)
    summ = lambda xs: {"plays": len(xs), "right": sum(x["got"] == x["scorer"] for x in xs),  # noqa: E731
                       "mean_credit": float(np.mean([x["credit"] for x in xs])) if xs else None}
    return {"all": summ(rows), "L2-L4": summ([x for x in rows if x["kind"] != "L1"]),
            "by_evidence": {k: summ(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}, "rows": rows}


def scouts_diff(a: dict, b: dict, g: str = "coral", n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in scout agreement (b - a) with a 95% range, resampling matches (paired)."""
    keys = sorted(set(a["per_match"]) & set(b["per_match"]))
    N = np.array([a["per_match"][k].get(f"n_{g}", 0.0) for k in keys])
    Ea = np.array([a["per_match"][k].get(f"err_{g}", 0.0) for k in keys])
    Eb = np.array([b["per_match"][k].get(f"err_{g}", 0.0) for k in keys])
    i = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
    d = (Ea[i].sum(1) - Eb[i].sum(1)) / np.maximum(2 * N[i].sum(1), 1)
    return float((Ea.sum() - Eb.sum()) / max(2 * N.sum(), 1)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---- decode -------------------------------------------------------------------------------------
def decode(results: dict, byk: dict) -> dict:
    from autoscout.games.reefscape2025 import CATS, Reefscape2025

    diff = off_n = 0
    for key, r in results.items():
        for a in ("red", "blue"):
            off = Reefscape2025.official_counts(None, byk[key].breakdown[a])
            got = defaultdict(int)
            for e in r["events"]:
                if e.get("alliance") == a and e.get("kind") in CATS and "shares" in e:
                    got[(e["phase"], e["kind"])] += 1
            for ph in ("auto", "teleop"):
                for c in CATS:
                    diff += abs(got[(ph, c)] - off[ph][c])
                    off_n += off[ph][c]
    return {"official_actions": off_n, "abs_difference": diff, "share": diff / max(off_n, 1)}


# ---- report -------------------------------------------------------------------------------------
def run(event: str, keys: list[str], patches: dict | None = None, only=None, workers: int = 6) -> dict:
    byk = event_matches(event)
    want = lambda s: only is None or s in only  # noqa: E731
    rep = {"event": event, "matches": keys, "patches": {k: repr(v) for k, v in (patches or {}).items()}}
    if want("reef"):
        rep["reef"] = reef(keys, event, patches, workers)
    if any(want(s) for s in ("timing", "fms", "plays", "opr", "decode", "scouts")):
        res = reanalyzed(keys, patches, workers)
        if want("timing"):
            rep["timing"] = timing(res)
        if want("fms"):
            rep["fms"] = fms(res, byk)
        if want("plays"):
            rep["plays"] = plays(res)
        if want("opr"):
            rep["opr"] = opr(res, byk)
        if want("decode"):
            rep["decode"] = decode(res, byk)
        if want("scouts") and (sc := scouts(res, byk, event)) is not None:
            rep["scouts"] = sc
    return rep


def show(rep: dict) -> None:
    print(f"{rep['event']}: {len(rep['matches'])} matches" + (f"  with {rep['patches']}" if rep["patches"] else ""))
    if "reef" in rep:
        r = rep["reef"]
        print(f"  REEF read blind ({r['alliances']} alliances, {r['branches']} official L2-L4 branches)")
        print(f"    end of match   precision {r['precision']:.1%}  recall {r['recall']:.1%}  F1 {r['f1']:.1%}"
              f"   alliances read exactly {r['exact_alliances']:.0%}")
        print(f"    end of AUTO    precision {r['auto_precision']:.1%}  recall {r['auto_recall']:.1%}  "
              f"F1 {r['auto_f1']:.1%}   ({r['auto_branches']} branches)")
        print(f"    REEF points    {r['points_ratio']:.0%} of official, off by {r['points_mae']:.1f} pts an alliance, r={r['points_r']:.3f}")
        print(f"    L1 trough      {r['trough_ratio']:.0%} of official count, off by {r['trough_mae']:.1f} an alliance")
    if "timing" in rep:
        t = rep["timing"]
        print(f"  L2-L4 scores ({t['scores']}): paired with a branch {t['paired']:.0%}, timed by its coral "
              f"{t['timed_by_coral']:.0%}, robot seen at the spot {t['robot_at_spot']:.0%}")
    if "fms" in rep:
        f = rep["fms"]
        x = f["auto_coral_to_non_leavers"]
        print(f"  AUTO coral credited to robots that never left: {x['wrong']:.0f} of {x['auto_coral']:.0f} "
              f"({x['share']:.1%}; equal split {x['equal_split_share']:.1%}) in {x['alliances']} alliances")
        for k, label in (("barge", "at the BARGE at the end as FMS says"), ("leave", "moved in AUTO as FMS says")):
            x = f[k]
            print(f"  robot {label}: {x['agree']:.1%} of {x['robots']} seen robots ({x['not_seen']} not seen); "
                  f"whole alliance right {x['alliances_all_agree']:.0%} of {x['alliances_all_seen']}")
    if "plays" in rep:
        for kind, label in (("user_verified", "checked by a person"), ("footage_checks", "AI-read footage checks")):
            x = rep["plays"][kind]
            if x["plays"]:
                print(f"  plays {label}: {x['right']}/{x['plays']} right, mean credit to the scorer {x['mean_credit']:.2f}"
                      + (f"   wrong: {[w['id'] for w in x['wrong']]}" if x["wrong"] else ""))
    if "opr" in rep:
        print("  split vs leave-one-match-out OPR (agreement; equal split in brackets):")
        print("    " + "   ".join(f"{k} {rep['opr'][k]['agree']:.1%} ({rep['opr'][k]['equal']:.1%})"
                                  for k in ("coral", "L1", "L2-L4", "L4", "net", "processor")))
    if "scouts" in rep:
        x = rep["scouts"]
        print(f"  scout sheets, only the {x['cells_fms_confirmed']} of {x['cells']} alliance counts (per phase and level) FMS confirms "
              f"(credited to the scouted robot; equal split in brackets):")
        print("    " + "   ".join(f"{k} {x[k]['right']:.1%} ({x[k]['equal']:.1%}) n={x[k]['official']:.0f}"
                                  for k in ("coral", "L2-L4", "L1", "auto", "teleop", "net", "processor")))
        pp = x["pinned"]
        print(f"    plays with a known scorer (one robot scored the whole confirmed count): L2-L4 "
              f"{pp['L2-L4']['right']}/{pp['L2-L4']['plays']} top-credited right, mean credit {pp['L2-L4']['mean_credit']:.2f}")
        for how, v in pp["by_evidence"].items():
            print(f"      {how:34s} {v['right']:3d}/{v['plays']:<3d} right, mean credit {v['mean_credit']:.2f}")
    if "decode" in rep:
        d = rep["decode"]
        print(f"  decoded vs official action counts: off by {d['abs_difference']} of {d['official_actions']} ({d['share']:.1%})")


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"True": True, "False": False, "None": None}.get(text, text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event")
    ap.add_argument("--matches", type=Path, help="JSON list of match keys (default: every processed match with coral boxes)")
    ap.add_argument("--only", help="comma-separated checks: reef,timing,fms,plays,opr,decode,scouts")
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.CONST=VALUE",
                    help="also run with this constant changed and print the difference (repeatable)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    byk = event_matches(a.event)
    keys = json.loads(a.matches.read_text()) if a.matches else default_keys(a.event, byk)
    only = set(a.only.split(",")) if a.only else None
    base = run(a.event, keys, None, only, a.workers)
    show(base)
    reports = {"base": base}
    if a.set:
        patches = {k: _value(v) for k, v in (s.split("=", 1) for s in a.set)}
        var = run(a.event, keys, patches, only, a.workers)
        show(var)
        reports["variant"] = var
        if "opr" in base:
            for name, comps in (("coral", CORAL), ("L2-L4", ("L2", "L3", "L4")), ("L1", ("L1",)), ("net", ("net",))):
                d, lo, hi = bootstrap_diff(base["opr"], var["opr"], comps)
                print(f"  OPR agreement, variant - base, {name}: {d:+.2%}  (95% {lo:+.2%} to {hi:+.2%})")
        if "scouts" in base:
            for g in ("coral", "L2-L4", "L1", "net", "processor"):
                d, lo, hi = scouts_diff(base["scouts"], var["scouts"], g)
                print(f"  scout agreement, variant - base, {g}: {d:+.2%}  (95% {lo:+.2%} to {hi:+.2%})")
    if a.out:
        a.out.write_text(json.dumps(reports, indent=1, default=float))

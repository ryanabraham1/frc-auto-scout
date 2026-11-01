"""Compare attribution before/after these fixes on frozen cached observations.

The baseline methods come from git HEAD. No videos are downloaded or models trained.
--apply refreshes unchanged source results and the dashboard database after checking
team membership and official count conservation. Originals remain in the run folder.
"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from autoscout import db, pipeline
from autoscout.games.base import GamePlugin
from autoscout.games.rebuilt2026 import Rebuilt2026
from autoscout.games.reefscape2025 import Reefscape2025
from autoscout.games import base, rebuilt2026
from autoscout.tba import TBA
from scripts.association_eval import legacy_box


def head_methods(path, cls_name, names, globals_):
    src = subprocess.run(["git", "show", f"HEAD:{path}"], check=True, capture_output=True, text=True).stdout
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), globals_, ns)
            out[node.name] = ns[node.name]
    if set(out) != set(names):
        raise ValueError("Baseline methods missing from HEAD")
    return out


def totals(r):
    metrics = ["fuel_scored"] if r["year"] == 2026 else ["coral_total", "coral_L4", "algae_net", "algae_processor"]
    return {f"{a}/{m}": sum(v[m] for v in r["teams"].values() if v["alliance"] == a)
            + (r["alliances"][a]["hp_fuel_scored"] + r["alliances"][a].get("unattributed_fuel", 0)
               if r["year"] == 2026 else 0)
            for a in ("red", "blue") for m in metrics}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--current-only", action="store_true", help="Refresh current logic without replaying HEAD")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = pipeline.RESULTS_DIR
    snapshots = a.out / "before"
    candidates = a.out / "after"
    baseline = a.out / "baseline"
    snapshots.mkdir()
    candidates.mkdir()
    baseline.mkdir()
    keys = a.keys or [p.stem for p in sorted(source.glob("*_qm*.json")) if p.stem.count("_") == 1]
    hashes = {}
    for key in keys:
        path = source / f"{key}.json"
        shutil.copy2(path, snapshots / path.name)
        hashes[key] = digest(snapshots / path.name)
        for suffix in ("_blobs.npz", "_labels.json"):
            src = source / f"{key}{suffix}"
            if src.exists():
                # Blobs are immutable observations; a hard link avoids copying gigabytes.
                if suffix.endswith("npz"):
                    (snapshots / src.name).hardlink_to(src)
                else:
                    shutil.copy2(src, snapshots / src.name)
    shared_names = ["launch_distances", "attribute_launch", "trace_back"]
    season_names = ["live_weights", "history_shooters"]
    old_shared = head_methods("autoscout/games/base.py", "GamePlugin", shared_names, vars(base))
    old_season = head_methods("autoscout/games/rebuilt2026.py", "Rebuilt2026", season_names, vars(rebuilt2026))
    current_shared = {n: getattr(GamePlugin, n) for n in shared_names + ["robot_box"]}
    current_season = {n: getattr(Rebuilt2026, n) for n in season_names}
    pipeline.RESULTS_DIR = snapshots
    tba = TBA(max_age_s=float("inf"))
    con = db.connect() if a.apply else None
    report = {"scope": "pipeline replay, same observations; count conservation is not attribution ground truth",
              "inputs_sha256": hashes,
              "baseline_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
              "matches": []}
    try:
        for key in keys:
            start = time.monotonic()
            if a.current_only:
                before = json.loads((snapshots / f"{key}.json").read_text())
            else:
                try:
                    for n, fn in {**old_shared, "robot_box": legacy_box}.items():
                        setattr(GamePlugin, n, staticmethod(fn))
                    for n, fn in old_season.items():
                        setattr(Rebuilt2026, n, fn)
                    before = pipeline.reanalyze(key, tba, save=False)
                    (baseline / f"{key}.json").write_text(json.dumps(before, default=pipeline._json_default))
                finally:
                    for n, fn in current_shared.items():
                        setattr(GamePlugin, n, staticmethod(fn))
                    for n, fn in current_season.items():
                        setattr(Rebuilt2026, n, fn)
            after = pipeline.reanalyze(key, tba, save=False)
            assert set(map(str, before["teams"])) == set(map(str, after["teams"])), key
            shortfalls = {}
            if after["year"] != 2026 and not a.current_only:
                assert totals(before) == totals(after), (key, totals(before), totals(after))
            if after["year"] == 2025:
                match = tba.match(key)
                for alliance in ("red", "blue"):
                    counts = Reefscape2025.official_counts(None, match["score_breakdown"][alliance])
                    for cat, metric in [("L1", "coral_L1"), ("L2", "coral_L2"), ("L3", "coral_L3"),
                                        ("L4", "coral_L4"), ("net", "algae_net"), ("processor", "algae_processor")]:
                        expected = sum(counts[p][cat] for p in counts)
                        allocated = sum(v[metric] for v in after["teams"].values() if v["alliance"] == alliance)
                        assert 0 <= allocated <= expected, (key, alliance, cat, allocated, expected)
                        if allocated < expected:
                            shortfalls[f"{alliance}/{cat}"] = expected - allocated
            # Every official 2026 period is conserved, including human-player credit.
            if after["year"] == 2026:
                for alliance in ("red", "blue"):
                    team_total = sum(v["fuel_scored"] for v in after["teams"].values() if v["alliance"] == alliance)
                    summary = after["alliances"][alliance]
                    assert team_total + summary["hp_fuel_scored"] + summary["unattributed_fuel"] == summary["official_fuel"], (
                        key, alliance, team_total, summary["hp_fuel_scored"], summary["official_fuel"])
            candidate = candidates / f"{key}.json"
            candidate.write_text(json.dumps(after, default=pipeline._json_default))
            before_rows = {str(k): v for k, v in before["teams"].items()}
            changed = sum(before_rows[str(team)] != values for team, values in after["teams"].items())
            status = "verified"
            if a.apply:
                path = source / candidate.name
                if digest(path) != hashes[key]:
                    status = "skipped_concurrent_update"
                else:
                    shutil.copy2(candidate, path)
                    db.save_result(con, after, str(path))
                    status = "applied"
            report["matches"].append({"key": key, "teams_changed": changed, "status": status,
                                      "seconds": round(time.monotonic()-start, 2), "totals": totals(after),
                                      "unattributed_counts_2025": shortfalls})
            (a.out / "report.json").write_text(json.dumps(report, indent=2))
            print(key, status, "changed teams", changed, flush=True)
    finally:
        pipeline.RESULTS_DIR = source
        if con:
            con.close()


if __name__ == "__main__":
    main()

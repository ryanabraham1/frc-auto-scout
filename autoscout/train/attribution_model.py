"""Learned 2026 FUEL attribution: which of an alliance's three robots shot each tracked flight.

The hand-tuned linking rules become features. Each flight into a HUB gets one row per candidate robot, and
a softmax over the three candidates gives P(robot | flight). There are no per-flight labels to train on, so
the model learns from label proportions: a robot's predicted share of its alliance's flights in a period
should match its scouted share of that alliance's fuel. Shares, not counts, because scouts undercount.

Scouted numbers are noisy, so results are always reported on matches the model never trained on, next to
the rules and an equal split.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config

FEATURES = ["launch_near", "trace_near", "observed", "rule_link", "hidden", "own_zone", "hub_near", "unseen_s",
            "at_hub_x_hidden", "rising_x_launch", "falling_x_trace", "fired_recently"]
NEAR_SCALE = 1.0  # box widths: distance d becomes exp(-d / NEAR_SCALE)
RIDGE = 0.01
FEATURE_DIR = config.DATA_DIR / "labels" / "features"


def load_scouting(path: Path, event: str) -> dict:
    """{match_key: {alliance: {team: (auto, teleop)}}} from 'match,team,a,auto,active,inactive,endgame,n;...'."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for row in path.read_text().strip().split(";"):
        m, team, a, auto, active, _inactive, endgame, _n = row.split(",")
        out[f"{event}_{m}"]["red" if a == "r" else "blue"][int(team)] = (int(auto), int(active) + int(endgame))
    return out


def candidate_features(plugin, ctx, L, slot: str, at_hub: bool, linked: str | None, fired: dict) -> list[float]:
    rising = float(getattr(L, "rising", True))
    near = lambda d: float(np.exp(-d / NEAR_SCALE))  # noqa: E731
    launch = plugin.launch_distances(ctx, L, (slot[:-1],)).get(slot)
    trace = plugin.trace_distances(ctx, L, (slot[:-1],)).get(slot)
    ln, tn = near(launch[0]) if launch else 0.0, near(trace[0]) if trace else 0.0
    hidden = float(plugin._in_hub_shadow(ctx, slot, L.t0))
    r = ctx.robot_at(slot, L.t0)
    own = float(r is not None and plugin.zone_for(slot[:-1], r[:2]) == "own")
    hub = plugin.field.elements[f"{slot[:-1]}_hub"]["center"]
    hub_near = float(np.exp(-np.hypot(r[0] - hub[0], r[1] - hub[1]) / 3.0)) if r is not None else 0.0
    ctx.robot_at(slot, L.t0)
    tr, ts = ctx.tracks[slot], ctx._ts[slot]
    i = int(np.searchsorted(ts, L.t0))
    last = next((x[0] for x in reversed(tr[max(0, i - 300):i]) if x[3]), None)
    unseen = min((L.t0 - last) / 10.0, 1.0) if last is not None else 1.0
    ts_f = fired.get(slot, np.empty(0))
    recent = float(np.count_nonzero(np.abs(ts_f - L.t0) < 3.0) > 0)
    return [ln, tn, float(bool(launch and launch[1])), float(linked == slot), hidden, own, hub_near, unseen,
            float(at_hub) * hidden, rising * ln, (1 - rising) * tn, recent]


def extract(match_key: str) -> Path:
    """Reanalyze a saved match and write its per-flight candidate features."""
    from .. import pipeline
    from ..games.rebuilt2026 import COUNT_LAG, Rebuilt2026, period_of

    got = {}
    original = Rebuilt2026.link_hub_flights

    def capture(self, ctx, launches, shooter_of, hub_zone):
        out = original(self, ctx, launches, shooter_of, hub_zone)
        got.update(self=self, ctx=ctx, launches=launches, out=out, zone=hub_zone)
        return out

    Rebuilt2026.link_hub_flights = capture
    try:
        result = pipeline.reanalyze(match_key, save=False)
    finally:
        Rebuilt2026.link_hub_flights = original
    plugin, ctx, out = got["self"], got["ctx"], got["out"]
    fired = defaultdict(list)
    flights = [L for L in got["launches"] if L.target and L.target.endswith("_hub")]
    for L in flights:
        if out.get(id(L)):
            fired[out[id(L)]].append(L.t0)
    fired = {k: np.sort(v) for k, v in fired.items()}
    X, meta = [], []
    for L in flights:
        a = L.target[:-4]
        slots = [s for s in ctx.tracks if s[:-1] == a]
        period = period_of(ctx, L.t0 + COUNT_LAG)
        if len(slots) != 3 or period is None:
            continue
        at_hub = bool(got["zone"](np.array([L.start[0]]), np.array([L.start[1]]))[0])
        X.append([candidate_features(plugin, ctx, L, s, at_hub, out.get(id(L)), fired) for s in slots])
        meta.append((a, 0 if period == "auto" else 1, float(L.n_pieces or 1),
                     *[ctx.team_at(s, L.t0) for s in slots]))
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_DIR / f"{match_key}.npz"
    rules = {int(k): (v["fuel_scored_auto"], v["fuel_scored_teleop"]) for k, v in result["teams"].items()}
    np.savez(path, X=np.array(X, float), meta=np.array(meta, dtype=object),
             rules=np.array([[k, *v] for k, v in rules.items()], float))
    return path


def groups(match_key: str, scouting: dict) -> list[dict]:
    """One training group per alliance and period: its flights and the scouted share of each robot."""
    d = np.load(FEATURE_DIR / f"{match_key}.npz", allow_pickle=True)
    X, meta, rules = d["X"], d["meta"], {int(r[0]): r[1:] for r in d["rules"]}
    out = []
    for a, teams in scouting.get(match_key, {}).items():
        if len(teams) != 3:
            continue
        for phase in (0, 1):
            total = sum(v[phase] for v in teams.values())
            rows = [i for i, m in enumerate(meta) if m[0] == a and m[1] == phase]
            if total < 5 or not rows:
                continue
            order = [int(t) for t in meta[rows[0]][3:6]]
            if set(order) != set(teams):
                continue
            rule = np.array([rules.get(t, np.zeros(2))[phase] for t in order], float)
            out.append({"match": match_key, "X": X[rows], "n": np.array([meta[i][2] for i in rows], float),
                        "share": np.array([teams[t][phase] / total for t in order]), "total": total,
                        "rule": rule / rule.sum() if rule.sum() else np.full(3, 1 / 3), "phase": phase})
    return out


def predict(w: np.ndarray, g: dict) -> np.ndarray:
    z = g["X"] @ w
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    share = (g["n"][:, None] * p).sum(axis=0)
    return share / share.sum()


def fit(train: list[dict]) -> np.ndarray:
    from scipy.optimize import minimize

    def loss(w):
        total = RIDGE * float(w @ w)
        for g in train:
            total -= float(g["share"] @ np.log(predict(w, g) + 1e-6)) / len(train)
        return total

    return minimize(loss, np.zeros(len(FEATURES)), method="L-BFGS-B", options={"maxiter": 200}).x


def cross_validate(all_groups: list[dict], folds: int = 5) -> dict:
    """Mean absolute share error on matches held out of training, for the model, the rules and an equal split."""
    matches = sorted({g["match"] for g in all_groups})
    err = {"model": [], "rules": [], "equal": []}
    for f in range(folds):
        held = set(matches[f::folds])
        w = fit([g for g in all_groups if g["match"] not in held])
        for g in all_groups:
            if g["match"] in held:
                err["model"].append(np.abs(predict(w, g) - g["share"]).mean())
                err["rules"].append(np.abs(g["rule"] - g["share"]).mean())
                err["equal"].append(np.abs(1 / 3 - g["share"]).mean())
    return {k: float(np.mean(v)) for k, v in err.items()} | {"groups": len(err["model"]), "matches": len(matches)}

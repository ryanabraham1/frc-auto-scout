"""Compare analysis variants on an event's processed matches without touching the saved results.

Each variant patches module attributes for the duration of a reanalysis of every match, and the
per-team averages are scored against OPR like scripts/attribution_check.py.

usage: python scripts/ablate.py EVENT
"""
import contextlib
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from attribution_check import METRICS  # noqa: E402
from opr import opr  # noqa: E402

from autoscout import pipeline  # noqa: E402
from autoscout.games import reefscape2025 as R25  # noqa: E402
from autoscout.games import base as B  # noqa: E402
from autoscout.tba import TBA, MatchInfo  # noqa: E402


@contextlib.contextmanager
def patched(changes):
    old = [(obj, k, getattr(obj, k)) for obj, k, _ in changes]
    for obj, k, v in changes:
        setattr(obj, k, v)
    try:
        yield
    finally:
        for obj, k, v in old:
            setattr(obj, k, v)


def no_aux(ctx, evidence, **kw):
    return None


def original_2026():
    """The 2026 plugin's attribution methods as committed (git HEAD), on today's observations."""
    import subprocess
    import types

    from autoscout import games
    from autoscout.games import rebuilt2026 as R26

    src = subprocess.run(["git", "show", "HEAD:autoscout/games/rebuilt2026.py"], capture_output=True, text=True).stdout
    mod = types.ModuleType("rebuilt2026_head")
    mod.__package__ = "autoscout.games"
    keep = dict(games.base.REGISTRY)
    exec(compile(src, "rebuilt2026_head.py", "exec"), mod.__dict__)
    games.base.REGISTRY.clear()
    games.base.REGISTRY.update(keep)
    old = mod.Rebuilt2026
    return [(R26.Rebuilt2026, k, getattr(old, k)) for k in ("analyze", "column_activity", "live_weights")]


VARIANTS_2026 = {
    "original attribution (git HEAD)": original_2026,
    "current": lambda: [],
}

VARIANTS = {
    "nearest robot gets each action": [(R25, "SHARE_MODE", "nearest")],
    "soft share 0.5 m": [(R25, "SHARE_MODE", "soft"), (R25, "SOFT_SCALE", 0.5)],
    "soft share 1.0 m": [(R25, "SHARE_MODE", "soft"), (R25, "SOFT_SCALE", 1.0)],
    "soft share 2.0 m": [(R25, "SHARE_MODE", "soft"), (R25, "SOFT_SCALE", 2.0)],
}


def score(event, results):
    tba = TBA()
    ms = {m["key"]: MatchInfo.from_tba(m) for m in tba.event_matches(event)}
    out = []
    for key, field, _ in METRICS[int(event[:4])]:
        o = opr(event, field, tba)
        ours = {}
        for r in results:
            m = ms[r["match_key"]]
            for t in m.red + m.blue:
                tm = r["teams"].get(t) or r["teams"][str(t)]  # int keys before a JSON round trip
                ours.setdefault(t, []).append(tm[key])
        teams = sorted(ours)
        x = np.array([np.mean(ours[t]) for t in teams])
        y = np.array([o[t] for t in teams])
        out.append(f"{key}={np.corrcoef(x, y)[0, 1]:.2f}")
    return " ".join(out)


def main(event):
    keys = sorted(p.stem for p in pipeline.RESULTS_DIR.glob(f"{event}_qm*.json") if p.stem.count("_") == 1)
    tba = TBA()
    variants = VARIANTS_2026 if event.startswith("2026") else {k: (lambda v=v: v) for k, v in VARIANTS.items()}
    for name, make in variants.items():
        with patched(make()):
            res = [pipeline.reanalyze(k, tba, save=False) for k in keys]
        print(f"{name:40s} {score(event, res)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

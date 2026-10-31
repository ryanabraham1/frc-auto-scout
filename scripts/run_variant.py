"""Re-process matches with detection settings changed, for A/B tests (overwrites their results).

usage: python scripts/run_variant.py [--ensemble] [--static-frac F] MATCH [MATCH ...]
"""
import argparse

from autoscout import pipeline
from autoscout.pipeline import Options, process_match
from autoscout.vision import detect

ap = argparse.ArgumentParser()
ap.add_argument("matches", nargs="+")
ap.add_argument("--ensemble", action="store_true")
ap.add_argument("--static-frac", type=float, default=None)
a = ap.parse_args()
if a.static_frac is not None:
    orig = detect.StaticSuppressor.__init__
    detect.StaticSuppressor.__init__ = lambda self, cell=24, min_frames=40, frac=a.static_frac: orig(self, cell, min_frames, frac)
det = detect.default_detector(ensemble=a.ensemble)
for m in a.matches:
    try:
        r = process_match(m, Options(log=False), detector=det)
        print(m, "ok", flush=True)
    except Exception as e:
        print(m, "failed", e, flush=True)

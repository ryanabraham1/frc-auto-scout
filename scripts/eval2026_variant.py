"""usage: python scripts/eval2026_variant.py EVENT NAME=VALUE ... (patches rebuilt2026 constants, runs eval2026)"""
import sys, runpy
sys.path.insert(0, ".")
from autoscout.games import rebuilt2026 as R
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    setattr(R, k, type(getattr(R, k))(v) if not isinstance(getattr(R, k), bool) else v == "True")
sys.argv = ["eval2026.py", sys.argv[1]]
runpy.run_path("scripts/eval2026.py", run_name="__main__")

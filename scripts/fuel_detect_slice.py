"""Ball detection experiment on a slice of a processed 2026 match: tracked makes per hub vs live score rise.

usage: python detect_slice.py MATCH VIDEO T0 T1 MODE[=value ...]  e.g. bg=30 lag=2 kernel=3
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from autoscout import games  # noqa: E402
from autoscout.field import Calibration  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, launch_targets, launch_view  # noqa: E402
from autoscout.video import Video  # noqa: E402
from autoscout.vision.launches import LaunchExtractor  # noqa: E402
from autoscout.vision.trajectories import LaunchDetector  # noqa: E402

key, vpath, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
opts = dict(a.split("=") for a in sys.argv[5:])
old = json.loads((RESULTS_DIR / f"{key}.json").read_text())
plugin = games.get(old["year"])
c = old["calibration"]
base = Calibration.load(old["event_key"])
cal = Calibration(event_key=old["event_key"], year=old["year"], view=tuple(c["view"]),
                  points={**{k: tuple(v) for k, v in c["points"].items()},
                          **{k: tuple(v) for k, v in base.points.items() if k.endswith("_top")}})
video = Video(vpath)
lview = launch_view(cal)
lo, hi = plugin.launch_pieces["fuel"]
d = LaunchDetector(lo, hi, lview, lag=int(opts.get("lag", 2)))
d.online = False
d.bg_frames = int(opts.get("bg", 0))
if "kernel" in opts:
    k = int(opts["kernel"])
    d.kernel = np.ones((k, k), np.uint8)
X, Y, W, H = lview
for t, frame in video.iter_frames(t0, t1, step=1):
    d.step(t, frame[Y: Y + H, X: X + W])
B = np.array(d.blobs, np.float32).reshape(-1, 6)
if "save" in opts:
    np.save(opts["save"], B)
ex = LaunchExtractor(lview, cal, plugin.field, targets=launch_targets(cal, plugin.field))
L = ex.extract(B)
tl = old["timeline"]
ts = np.array(tl["t"])
lag = 1.0
print(f"{key} {t0}-{t1}s {opts}: blobs {len(B)} ({len(B) / max(len(np.unique(B[:, 0])), 1):.0f}/frame) "
      f"flights {len(L)} stats {ex.stats}")
for a in ("red", "blue"):
    s = np.array(tl[a], float)
    i0, i1 = np.searchsorted(ts, t0 + lag), np.searchsorted(ts, t1 + lag) - 1
    rise = s[i1] - s[i0]
    made = sum(l.n_pieces for l in L if l.target == f"{a}_hub")
    print(f"  {a}: score rose {rise:.0f}, tracked makes {made:.0f} ({made / max(rise, 1):.0%})")

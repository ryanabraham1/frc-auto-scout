"""Run the CORAL detector through processed 2025 matches and keep every box (data/results/<match>_coral.json).

The boxes cover the whole broadcast frame (wide shot and both REEF close-ups) at 5 Hz. Nothing is
reanalysed: this only collects the detections that placement tracking reads.

usage: python scripts/coral_scan.py MATCH_KEY... [--conf 0.1] [--hz 5] [--force]
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, ".")
from autoscout.field import Calibration  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, _scaled_views, download  # noqa: E402
from autoscout.video import Video  # noqa: E402
from autoscout.vision import coral_track  # noqa: E402
from autoscout.vision.aux_views import AuxViewNormalizer  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matches", nargs="+")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for mk in a.matches:
        side = RESULTS_DIR / f"{mk}_coral.json"
        if side.exists() and not a.force:
            print(mk, "cached", flush=True)
            continue
        t = time.time()
        try:
            old = json.loads((RESULTS_DIR / f"{mk}.json").read_text())
            video = Video(str(download(old["video"])))
            base = Calibration.load(old["event_key"])
            clk = SimpleNamespace(**old["clock"])
            normalize = AuxViewNormalizer(video.frame_at(clk.teleop_end_vt - clk.teleop_len + 20),
                                          _scaled_views(base, video) if base else None)
            obs = coral_track.scan(video, clk.auto_end_vt - clk.auto_len - 1, clk.teleop_end_vt + 3, normalize,
                                   hz=a.hz, conf=a.conf)
            side.write_text(json.dumps(obs))
            print(mk, len(obs), "frames", sum(len(b) for _, b in obs), "boxes", f"{time.time() - t:.0f}s", flush=True)
        except Exception as e:  # keep going: one bad video must not stop the batch
            print(mk, "FAILED", repr(e), flush=True)

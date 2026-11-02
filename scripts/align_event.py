"""Align each processed match's wide shot to its event calibration's reference frame and re-map robot
positions from their saved boxes (no video re-processing).

The broadcast camera's framing can move between matches (2025casj qm2 sat ~80 px left of qm10, ~0.7 m of
position at the REEFs); without a reference frame every match used the one calibration. --ref-match saves
that match's wide shot as the reference first (it must be framed like the calibration was drawn).

usage: python scripts/align_event.py EVENT [--ref-match KEY] [--dry-run]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout import pipeline  # noqa: E402
from autoscout.config import ROOT  # noqa: E402
from autoscout.field import Calibration, align_to_reference  # noqa: E402
from autoscout.overlay import MatchClock  # noqa: E402
from autoscout.video import Video  # noqa: E402


def mid_frame(r: dict) -> np.ndarray | None:
    v = Video(str(next(Path(pipeline.VIDEO_DIR).glob(f"{r['video']}.*"))))
    return v.frame_at(MatchClock(**r["clock"]).teleop_start_vt + 20)


def set_reference(event: str, key: str) -> Path:
    r = json.loads((pipeline.RESULTS_DIR / f"{key}.json").read_text())
    out = ROOT / "configs" / "events" / f"{event}_ref.jpg"
    cv2.imwrite(str(out), mid_frame(r))
    y = ROOT / "configs" / "events" / f"{event}.yaml"
    text = y.read_text()
    rel = f"configs/events/{event}_ref.jpg"
    text, n = re.subn(r"^reference:.*$", f"reference: {rel}", text, count=1, flags=re.M)
    if not n:
        text += f"\nreference: {rel}\n"
    y.write_text(text)
    return out


def main(event: str, ref_match: str | None, dry: bool) -> None:
    if ref_match:
        print("reference frame:", set_reference(event, ref_match))
    base = Calibration.load(event)
    ref = pipeline.reference_frame(base)
    if ref is None:
        raise SystemExit(f"{event} has no reference frame: pass --ref-match")
    for p in sorted(pipeline.RESULTS_DIR.glob(f"{event}_qm*.json")):
        if p.stem.count("_") != 1:
            continue
        r = json.loads(p.read_text())
        frame = mid_frame(r)
        if frame is None:
            print(p.stem, "no frame")
            continue
        cal = align_to_reference(base, frame, ref)
        moved = max(float(np.hypot(*(np.subtract(cal.points[k], base.points[k])))) for k in base.points)
        print(f"{p.stem}: landmarks moved up to {moved:.0f} px")
        if dry:
            continue
        r["raw_tracks"] = pipeline.remap_tracks(r["raw_tracks"], cal)
        r.update(pipeline._track_fields(r["raw_tracks"]))
        r["calibration"] = pipeline._cal_dict(cal, base)
        p.write_text(json.dumps(r, default=pipeline._json_default))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event")
    ap.add_argument("--ref-match")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.event, a.ref_match, a.dry_run)

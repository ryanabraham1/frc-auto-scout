"""Robot-detector evaluation against hand-marked robot points.

Marking a point on every robot is much quicker than drawing boxes, and enough to score a detector:
  recall    = marked robots covered by a detection box of the right alliance,
  precision = detection boxes that contain a marked robot (any alliance).

  python scripts/det_eval.py extract MATCH T1,T2,...   # saves wide-shot crops + gridded previews
  python scripts/det_eval.py eval [weights ...]        # scores detectors (default: the pipeline's)

Points live in data/det_eval/points.json as {stem: [[x, y, "r"|"b"], ...]} in crop pixels.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from autoscout import config
from autoscout.field import Calibration
from autoscout.tba import TBA, MatchInfo
from autoscout.video import Video, download

DIR = config.DATA_DIR / "det_eval"


def extract(match_key: str, times: list[float]):
    DIR.mkdir(parents=True, exist_ok=True)
    m = MatchInfo.from_tba(TBA().match(match_key))
    v = Video(download(m.videos[0], height=1080))
    cal = Calibration.load(m.event_key).scaled((v.width, v.height))
    x, y, w, h = cal.view
    for t in times:
        f = v.frame_at(t)
        if f is None:
            continue
        crop = f[y: y + h, x: x + w]
        stem = f"{match_key}_{int(t)}"
        cv2.imwrite(str(DIR / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = crop.copy()
        for gx in range(0, w, 100):
            cv2.line(g, (gx, 0), (gx, h), (0, 255, 255) if gx % 500 == 0 else (0, 160, 160), 1)
            cv2.putText(g, str(gx), (gx + 2, 14), 0, 0.45, (0, 255, 255), 1)
        for gy in range(0, h, 100):
            cv2.line(g, (0, gy), (w, gy), (0, 255, 255) if gy % 500 == 0 else (0, 160, 160), 1)
            cv2.putText(g, str(gy), (2, gy + 14), 0, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(DIR / f"{stem}_grid.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(stem, crop.shape)


def evaluate(weights: list[str], conf=0.3, imgsz=1280):
    from autoscout.vision.detect import PersonVeto, RobotDetector, alliance_from_pixels, default_detector
    from autoscout.vision.benchmark import score_points

    pts = json.loads((DIR / "points.json").read_text())
    dets = {"pipeline default": default_detector()} if not weights else {}
    for detector in dets.values():
        detector.conf, detector.imgsz = conf, imgsz
    for wt in weights:
        dets[wt] = RobotDetector(wt, conf=conf, imgsz=imgsz)
    veto = PersonVeto()
    summary = []
    for name, det in dets.items():
        hit = tot = tp = nd = 0
        per = []
        frames = []
        started = time.perf_counter()
        for stem, P in pts.items():
            img = cv2.imread(str(DIR / f"{stem}.jpg"))
            if img is None:
                raise FileNotFoundError(DIR / f"{stem}.jpg")
            ds = veto(img, det(img))
            for d in ds:
                if not d.alliance:
                    d.alliance, _ = alliance_from_pixels(img, d)
            score = score_points(ds, P)
            h = t = score["tp"]
            hit, tot, tp, nd = hit + h, tot + len(P), tp + t, nd + len(ds)
            per.append(f"{stem}:{h}/{len(P)}")
            frames.append(dict(frame=stem, **score))
        seconds = time.perf_counter() - started
        print(f"{name:40s} recall {hit}/{tot} = {hit / max(tot, 1):.0%}   precision {tp}/{nd} = {tp / max(nd, 1):.0%}")
        print("   ", " ".join(per))
        f1 = 2 * tp / max(tot + nd, 1)
        print(f"    F1 {f1:.3f}; {seconds:.1f}s for {len(pts)} frames")
        summary.append({"model": str(name), "recall": hit / max(tot, 1), "precision": tp / max(nd, 1),
                        "f1": f1, "tp": tp, "fp": nd - tp, "fn": tot - tp,
                        "robots": tot, "frames": len(pts), "scoring": "one-to-one-alliance-v2",
                        "conf": conf, "imgsz": imgsz, "seconds": seconds, "per_frame": frames,
                        "dataset_sha256": hashlib.sha256((DIR / "points.json").read_bytes()).hexdigest()})
    # merged into data/det_eval/results.json (shown on the dashboard's Runs page)
    out = DIR / "results.json"
    old = json.loads(out.read_text()) if out.exists() else []
    # Old scores used many-to-many matching and are not comparable.
    if any(r.get("scoring") != "one-to-one-alliance-v2" for r in old):
        legacy = DIR / "results_legacy.json"
        if not legacy.exists():
            legacy.write_text(json.dumps(old, indent=1))
        old = []
    key = lambda r: (r["model"], r.get("conf"), r.get("imgsz"))
    merged = {key(r): r for r in old}
    merged.update({key(r): r for r in summary})
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(list(merged.values()), indent=1))
    temp.replace(out)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract(sys.argv[2], [float(t) for t in sys.argv[3].split(",")])
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("command", choices=["eval"])
        ap.add_argument("weights", nargs="*")
        ap.add_argument("--conf", type=float, default=0.3)
        ap.add_argument("--imgsz", type=int, default=1280)
        args = ap.parse_args()
        evaluate(args.weights, conf=args.conf, imgsz=args.imgsz)

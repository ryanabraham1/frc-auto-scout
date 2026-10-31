"""Robot-detector evaluation against hand-marked robot points.

Marking a point on every robot is much quicker than drawing boxes, and enough to score a detector:
  recall    = marked robots covered by a detection box of the right alliance,
  precision = detection boxes that contain a marked robot (any alliance).

  python scripts/det_eval.py extract MATCH T1,T2,...   # saves wide-shot crops + gridded previews
  python scripts/det_eval.py eval [weights ...]        # scores detectors (default: the pipeline's)

Points live in data/det_eval/points.json as {stem: [[x, y, "r"|"b"], ...]} in crop pixels.
"""
import json
import sys
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


def evaluate(weights: list[str]):
    from autoscout.vision.detect import PersonVeto, RobotDetector, alliance_from_pixels, default_detector

    pts = json.loads((DIR / "points.json").read_text())
    dets = {"pipeline default": default_detector()} if not weights else {}
    for wt in weights:
        dets[wt] = RobotDetector(wt, conf=0.3)
    veto = PersonVeto()
    summary = []
    for name, det in dets.items():
        hit = tot = tp = nd = 0
        per = []
        for stem, P in pts.items():
            img = cv2.imread(str(DIR / f"{stem}.jpg"))
            ds = veto(img, det(img))
            for d in ds:
                if not d.alliance:
                    d.alliance, _ = alliance_from_pixels(img, d)
            inside = lambda d, p: d.x0 - 4 <= p[0] <= d.x1 + 4 and d.y0 - 4 <= p[1] <= d.y1 + 4  # noqa: E731
            h = sum(any(inside(d, p) and d.alliance[:1] == p[2] for d in ds) for p in P)
            t = sum(any(inside(d, p) for p in P) for d in ds)
            hit, tot, tp, nd = hit + h, tot + len(P), tp + t, nd + len(ds)
            per.append(f"{stem}:{h}/{len(P)}")
        print(f"{name:40s} recall {hit}/{tot} = {hit / max(tot, 1):.0%}   precision {tp}/{nd} = {tp / max(nd, 1):.0%}")
        print("   ", " ".join(per))
        summary.append({"model": str(name), "recall": round(hit / max(tot, 1), 3), "precision": round(tp / max(nd, 1), 3),
                        "robots": tot, "frames": len(pts)})
    # merged into data/det_eval/results.json (shown on the dashboard's Runs page)
    out = DIR / "results.json"
    old = {r["model"]: r for r in (json.loads(out.read_text()) if out.exists() else [])}
    old.update({r["model"]: r for r in summary})
    out.write_text(json.dumps(list(old.values()), indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], [float(t) for t in sys.argv[3].split(",")])
    else:
        evaluate(sys.argv[2:])

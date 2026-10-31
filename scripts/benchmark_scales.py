"""Compare single-scale and cross-scale robot detections on the local point benchmark.

This is an experiment, not an automatic model switch. All variants use identical
one-to-one alliance-aware scoring; output includes each event to expose regressions.
"""
import json
from collections import defaultdict

import cv2

from autoscout import config
from autoscout.vision.benchmark import score_points
from autoscout.vision.detect import RobotDetector, PersonVeto, iou, nms_merge


def main():
    directory = config.DATA_DIR / "det_eval"
    labels = json.loads((directory / "points.json").read_text())
    weights = config.ROOT / "models" / "autoscout-robots.pt"
    detectors = [RobotDetector(weights, imgsz=s, conf=.1) for s in (1280, 1600)]
    veto = PersonVeto()
    scores = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for stem, points in labels.items():
        image = cv2.imread(str(directory / f"{stem}.jpg"))
        if image is None:
            raise FileNotFoundError(stem)
        people = veto.people(image)
        from autoscout.vision.detect import is_person_box
        a, b = [[d for d in detector(image) if not any(is_person_box(d, p) for p in people)]
                for detector in detectors]
        variants = {"1280": [d for d in a if d.conf >= .3], "1600": [d for d in b if d.conf >= .3]}
        variants["union"] = nms_merge(variants["1280"] + variants["1600"], thr=.4)
        for high in (.4, .5, .6):
            combined = []
            for own, other in ((a, b), (b, a)):
                for d in own:
                    if d.conf >= high or (d.conf >= .3 and any(
                            o.conf >= .2 and o.alliance == d.alliance and iou(d, o) >= .3 for o in other)):
                        combined.append(d)
            variants[f"agree-or-{high}"] = nms_merge(combined, thr=.4)
        for name, detections in variants.items():
            score = score_points(detections, points)
            for event in ("all", stem.split("_")[0]):
                for j, key in enumerate(("tp", "fp", "fn")):
                    scores[name][event][j] += score[key]
    out = []
    for name, events in scores.items():
        result = {"variant": name, "events": {}}
        for event, (tp, fp, fn) in events.items():
            result["events"][event] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))
        print(name, json.dumps(result["events"]))
        out.append(result)
    (directory / "scales.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

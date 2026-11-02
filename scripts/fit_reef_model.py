"""Fit where each REEF's branches are in an event's wide shot (vision/reef_model.py) from the coral already
seen placed in its processed matches, and save it next to the event's calibration.

usage: python scripts/fit_reef_model.py EVENT
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
from autoscout.evaluation import load_results  # noqa: E402
from autoscout.field import Calibration  # noqa: E402
from autoscout.games import get  # noqa: E402
from autoscout.pipeline import RESULTS_DIR  # noqa: E402
from autoscout.vision.reef_model import LEVEL_Z, fit_reef_model, model_path  # noqa: E402


def main(event: str) -> dict:
    fm = get(2025).field
    centres = {k: tuple(fm.elements[k]["center"]) for k in ("blue_reef", "red_reef")}
    pts = {k: [] for k in centres}
    for r in load_results(event, RESULTS_DIR):
        c = r["calibration"]  # aligned to this match's framing
        cal = Calibration(event_key=event, year=2025, view=tuple(c["view"]), points={k: tuple(v) for k, v in c["points"].items()})
        for e in r["events"]:
            pl = e.get("placed")
            if pl and pl["elem"] in centres and e.get("kind") in LEVEL_Z:
                pts[pl["elem"]].append((pl["x"], pl["y"], e["kind"], cal))
    model = fit_reef_model(centres, pts)
    model_path(event).write_text(json.dumps(model, indent=1))
    print(json.dumps(model))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event")
    main(ap.parse_args().event)

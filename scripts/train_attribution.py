"""usage: python scripts/train_attribution.py EVENT SCOUTING_FILE [--refresh]  (extract features, cross-validate, fit)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from autoscout.pipeline import RESULTS_DIR  # noqa: E402
from autoscout.train import attribution_model as A  # noqa: E402

event, path = sys.argv[1], Path(sys.argv[2])
scouting = A.load_scouting(path, event)
groups = []
for key in sorted(scouting):
    if not (RESULTS_DIR / f"{key}.json").exists():
        continue
    if "--refresh" in sys.argv or not (A.FEATURE_DIR / f"{key}.npz").exists():
        A.extract(key)
    groups += A.groups(key, scouting)
print(json.dumps(A.cross_validate(groups), indent=1))
w = A.fit(groups)
print(json.dumps(dict(zip(A.FEATURES, (round(float(v), 2) for v in w)))))
np.save(A.FEATURE_DIR / f"{event}_weights.npy", w)

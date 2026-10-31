# Local detector comparison — October 31, 2026

Measured on this Windows PC's RTX 3060 (12 GB), using the existing 20 marked broadcast crops:
84 robots across 2025casj, 2026dal, 2026txhou, 2026cosp and 2026cc.

| Shipped weights, inference settings | True positives | False positives | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Previous 1280 input | 64 | 16 | 76.2% | 80.0% | 78.0% |
| Single 1600 input | 65 | 12 | 77.4% | 84.4% | 80.7% |
| New CUDA default, 1280 + 1600 consensus | 67 | 10 | 79.8% | 87.0% | 83.2% |

Matching is maximum-cardinality, one to one, alliance-correct, with a 4-pixel box margin.
Wrong-colour boxes and duplicates count as false positives. The previous evaluator reported 85% precision
for the first row because it allowed duplicates and ignored alliance colour when computing precision.

The two-scale default uses one loaded model. A box with confidence at least 0.5 stands alone; a box
between 0.3 and 0.5 needs an overlapping same-alliance box at the other scale (confidence at least 0.2,
IoU at least 0.3). Cross-scale boxes are merged with class-agnostic NMS at IoU 0.4. Unsupported moderate
boxes remain weak evidence when tracking, allowing continuation but preventing new robot tracks.

The new default improves aggregate precision and recall, but not every event: 2025casj recall falls
from 15/18 to 14/18. There are only four benchmark frames per event. Settings were compared on this
same benchmark, so these gains need confirmation on additional independently marked matches. They are
detector measurements, not measurements of per-team attribution accuracy.

A 30-epoch fine-tune on corrected frames (22 training frames weighted six times, six validation frames,
with whole matches separated and all benchmark matches excluded) was also tested. Its two-scale result
was 69 true positives / 15 false positives: 82.1% recall, 82.1% precision, 82.1% F1. It was retained as an
experiment rather than deployed because its precision and F1 were worse than the new default.

Reproduce locally:

```powershell
.venv/Scripts/python.exe scripts/det_eval.py eval models/autoscout-robots.pt --imgsz 1280
.venv/Scripts/python.exe scripts/det_eval.py eval models/autoscout-robots.pt --imgsz 1600
.venv/Scripts/python.exe scripts/det_eval.py eval
.venv/Scripts/python.exe scripts/benchmark_scales.py
```

`data/det_eval/results.json` contains per-frame counts, confidence thresholds, scales and the annotation
file's SHA-256. Legacy results are archived separately. `AUTOSCOUT_MULTISCALE=0` selects single-scale
inference; `AUTOSCOUT_IMGSZ=1280` also reduces the CUDA input size. CPU/MPS keep single-scale inference.

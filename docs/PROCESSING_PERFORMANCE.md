# Processing performance

The processing optimizations preserve the detector, inference precision, image
sizes, frame rates, OCR windows, team candidates, and scoring rules.

* Identity interval search uses indexed slices and cached overlap checks instead
  of repeatedly filtering and sorting whole chains. It evaluates the same
  intervals in the same order and uses the original cost calculations.
* Bumper OCR reuses the encoder output from its greedy recognition pass for
  candidate scoring, eliminating a second encoder pass. A thread-specific hook
  keeps simultaneous wide-shot and close-up reads separate.
* Greedy OCR confidence uses the same scalar reductions, transferring token IDs
  and results together instead of synchronizing CUDA separately for every crop.
* OCR crop preparation applies identical float32 normalization to one batch,
  avoiding separate tensor operations for every window. Median CPU preparation
  times were 0.99 → 0.33 ms for 8 crops, 9.36 → 4.66 ms for 64, and
  36.41 → 23.27 ms for 256 (12 repetitions; bit-identical input tensors).

Measured on cached tracklets, including complete relabelling and slot tracks:

| Match | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 2025casj_qm1 | 9.65 s | 3.27 s | 2.95× |
| 2026dal_qm103 | 11.18 s | 3.49 s | 3.20× |

Both comparisons produced exactly equal assignments, tracker statistics, and
slot tracks. On eight real bumper windows, the actual PARSeq model also produced
bit-identical candidate and greedy scores with encoder reuse (CPU verification).
GPU OCR speed and full-match speedup have not been measured: concurrent model
training and match workers were saturating the RTX 3060 and its memory during
verification. These component measurements do not establish a 3× full-match
speedup.

New results include `processing_stages_s`, also printed in the processing log,
to distinguish decoding/tracking, identity, scoring, and other elapsed time.
`closeup_wait` measures only the remaining wait after the main pass; close-up
work overlaps that pass. `video_and_calibration` includes any video download.
For comparable latency measurements, use one match worker while GPU training
and other inference jobs are idle.

Run the equivalence and cancellation regressions with:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_processing_speed.py tests/test_video.py tests/test_dashboard.py -q
```

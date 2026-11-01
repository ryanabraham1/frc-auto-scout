# 2025 (REEFSCAPE) accuracy work: progress and what's left

Last updated: 2026-11-01. Event used throughout: Silicon Valley Regional (`2025casj`).

## How accuracy is measured

- `python scripts/eval2025.py 2025casj`: per-team averages vs OPR computed only from matches the scouter
  never processed (no shared data), plus how many score jumps were decoded into the wrong category.
- Variant comparisons pin a fixed match list, because background processing keeps adding matches.
- `scripts/credit_review.py sheet MATCH OUT --alliance blue`: one review row per credited score (wide shot +
  reef close-up at the moment the coral went on, robots labelled, credit split) for checking by eye.
- Correlation with OPR can't separate small effects on a handful of matches; changes are kept only when
  they help across many matches, and spot-checked against the video.

## Done (pushed to `main`)

| Commit | What |
|---|---|
| `93c7b3b` | Score jumps decoded jointly against the official FMS counts (net vs L3 coral, leave vs L2); coral credited from coral seen appearing on the reef (wide shot); elevator box-shape clue; one-piece-at-a-time rule; debug video overlays; dashboard credit splits. Held-out coral r 0.58 -> 0.67, wrong-category decodes 44% -> 4%. |
| `979b1e9` | Overlay algae counter read per alliance (exact algae per jump); 2025 detection uses our own model + field checks (precision 61% -> 90% on 12 hand-marked 2025 frames). |
| `83e34d3` | Dashboard video gallery uses still thumbnails (videos stalled in Zen/Firefox). |
| `9611a2c`, `31254bf` | Reef close-up cameras: every robot seen there is recorded, each camera self-calibrates against the wide shot, close-up sightings fill wide-shot track gaps, coral placements watched in the close-ups too. |
| `a2bc600` | Detector loading no longer crashes after new weights retire the TensorRT engine. |
| `e50b9c0` | Wrong-credit fixes from checking qm2 by eye: close-up bumper reads label the exact tracklet (fixes 1323/2854 swaps), better close-up bumper reading (reads 198 -> 657 in qm2), placements only credit robots actually at that spot. |

Also deployed (not in git, data/): retrained robot detector, benchmark F1 0.817 -> 0.857
(recall 81.6%, precision 90.3%).

## Accuracy completion changes (2026-11-01)

- Unseen track positions expire after 3 s and require three real sightings. Unread close-ups
  join only a recent **wide-shot** observation; a fused sighting cannot refresh that anchor.
  Fusion rejects positions inside a REEF. Removed the even-split AUTO prior.
- Saved track positions are re-mapped after calibration changes. Camera fitting also re-maps
  its input observations, so old positions cannot train a new close-up homography.
- Repeated, confident close-up reads overlapping the reef horizontally provide dwell time
  when the wide shot maps a robot too far away. Isolated reads, long gaps, loading-wall
  robots, and duplicate timestamps do not invent dwell; simultaneous views are not added.
- Reach above the chassis is collected in both views (`autoscout lift-pass EVENT`). Only
  L2-L4 use it. A held lift needs distinct observations no more than 0.75 s apart and must
  exceed both the match baseline and the recent background. Missing measurements stay
  unknown, rather than losing to robots whose mechanisms happened to be measurable.
- Retained conservative reach weighting: `REACH_BOOST=1.0`, `REACH_ADD=0.3`. The original
  stronger weight (`3.0`, `1.0`) was worse on the fixed comparison below.
- qm2's disputed blue L4 posts now both select 1323: **11.51 s: 61%**, **41.54 s: 52%** on
  the fixed snapshot. These remain probabilistic credits, not certainty or team overrides.
  Unannotated frames at 10/11/12 s and 40 s confirm the scorer and the raised mechanism.
- Added regression coverage for sparse lift samples, background motion, missing reach,
  L1, close-up dwell, fusion anchors, and fitting with corrected coordinates. **89 tests pass**.

## Fixed comparison completed

All **35 cached qualification matches** were freshly measured for reach with the thin-mast
collector. Every variant uses identical raw observations, identities, camera fits, evaluated
matches and OPR reference matches. The selected keys are committed in
[`2025_ACCURACY_MATCHES.json`](2025_ACCURACY_MATCHES.json); numerical reports and settings are
in [`2025_ACCURACY_EVALUATION.json`](2025_ACCURACY_EVALUATION.json).

OPR uses the **39 other qualification matches**, with no shared evaluated match. Once the
whole event is processed these are still disjoint sets, but the reference matches can no
longer be described as "never processed". This is a development diagnostic, not an
independent human scorer-accuracy benchmark. The gains are small.

| Lift evidence | Coral r | L4 r | Auto coral r |
|---|---:|---:|---:|
| No lift clue | 0.899 | 0.603 | 0.883 |
| Box shape alone | 0.905 | 0.602 | 0.883 |
| Strong pixel reach | 0.902 | 0.586 | 0.890 |
| **Conservative pixel reach (kept)** | **0.908** | **0.603** | **0.890** |
| Equal split | 0.855 | 0.437 | 0.821 |

Selected MAE: coral 1.98 (equal split 2.91), L4 1.47 (1.59), auto 0.36 (0.49).
NET r=0.702; processor r=0.462. Decoded category-count absolute difference: **70 / 1128
(6.2%)**. This is a count discrepancy, not a measured percentage of individually wrong
category assignments. Processor attribution remains weak.

The separate AI-reviewed qm10 benchmark committed by another session has 15 accepted
scorer labels: strong reach picks 6/15 versus 5/15 for box shape alone. This tiny, AI-reviewed
set is not independent human ground truth and does not establish general accuracy.

Commands to reproduce the fixed comparison:

```powershell
.venv/Scripts/python.exe scripts/lift_eval2025.py 2025casj --matches docs/2025_ACCURACY_MATCHES.json --results-dir data/accuracy_2025/with_lifts --out-dir data/accuracy_2025/reproduce
.venv/Scripts/python.exe scripts/eval2025.py 2025casj --saved --matches docs/2025_ACCURACY_MATCHES.json --out data/accuracy_2025/current.json
```

The immutable input snapshot is local, git-ignored data. Reproduce before changing the
camera fit; camera homographies also affect attribution. Review sheets for qm2, qm11 and
qm12 use the conservative snapshot in `data/accuracy_2025/review/`: 53, 35 and 33 coral actions
(27, 18 and 17 sheets). Visually spot-checked qm2 blue 11.51/16.02, 39.04/41.54 and
52.05/54.55 s, qm11 red 16.02/58.06 s, and qm12 red 13.51/58.56 s. The disputed
qm2 corrections agree with the video; several other allocations remain ambiguous or
low-confidence. These sheets do not establish that every coral is credited correctly.

## Full event data pass still running

The old `full2` process had imported a pipeline predating the lift collector's tuple return
and failed on qm6. Its orphan GPU workers were removed. A current-code run is processing
the remaining matches in `data/run_2025casj_full3.log`; qm7 and qm9 have completed. The run
omits qm6, so the finalizer explicitly retries qm6 and failed DB entries before calibration.
**qm19 is blocked by YouTube's copyright restriction**, not silently counted as processed.

`scripts/finish2025.py --wait-pid PID --publish` waits for the existing event process, retries
failed available matches, refits cameras, fills missing reach observations, reanalyzes all
available matches, evaluates the pinned subset, regenerates review sheets, renders qm2,
qm11, qm12 and qm20, checks that each video decodes, and publishes only the camera fit,
its final data report and this progress document. It records failure rather than claiming completion if a stage fails.
A detached finalizer is running, waiting on event PID 4336. Its launcher PID is recorded
in `data/accuracy_2025/completion/launcher.json`.
Status and logs: `data/accuracy_2025/completion/`. A completed pass writes
`docs/2025_ACCURACY_FINAL_DATA.md`; that report is not present until the data work finishes.

The full event pass and final camera-fit metrics must not be called complete while this
section says running. Generated review sheets are review artifacts, not a substitute for
visual checking. Initial snapshot review and renders do not replace the final rerender.

## Deferred work

Recheck `2025cc` floor calibration; improve far-wall robot detection and processor
attribution; obtain additional independently reviewed scorer labels. Preserve unrelated
2026 and dashboard work in this shared tree.

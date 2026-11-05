# Ball tracking and scoring attribution — November 4, 2026

This change addresses lost ball fragments, incorrect identities at release, and repeated use of one
observed shot as evidence for several scores. It does not establish that every ball in real match
footage can be tracked. No 2026 match video or saved 2026 observations were available in this workspace;
the cloud proxy rejected the YouTube download with HTTP 403 before authentication.

## Behavior changes

- Moving-color pixels seed whole-ball components. Overlapping consecutive frames no longer reduce a
  ball to a small leading crescent with a biased centroid. Static yellow pieces remain excluded.
- Ball prediction fits recent observations to reduce acceleration noise. Tracks expire by elapsed
  time even when an entire frame has no detections. Rejoining requires position, velocity, and a
  valid joint flight; a ball that entered a goal cannot be joined to the next shot. Short fragments
  can be recovered together before the minimum-length rejection.
- Every observed trajectory point is saved, with millisecond timestamps and subpixel coordinates.
  Reanalysis retains the attributed shooter and inferred release/goal-entry times. Demo trails use
  the observation timestamps, including gaps, and can label individual balls and shooters.
- 2026 identity and zone checks use the inferred release time. Goal entries are assigned with finite
  capacity to live score increases before residual evidence is shared. Final official reconciliation
  leaves unsupported live-score fuel explicitly unattributed instead of inflating a single make.
  Stream context fills unassigned shots only when the whole stream supports a majority; it preserves
  individually linked shots. Goal entries without a shooter remain available for review.
- 2025 proximity, dwell, visibility, placement and reach evidence follow the team occupying a slot
  when each observation occurred. Each algae launch can support one net post and cannot boost
  processor credit. Simultaneous coral posts use joint batch assignments rather than an arbitrary
  action order. Strong placement/launch evidence reduces the influence of the event prior.

## Validation

`python -m pytest -q`: **138 passed**. The existing association test now stubs bumper OCR on its blank
frames instead of downloading a recognizer unrelated to the association behavior being tested.

The known-path synthetic benchmark compares the same generated observations with baseline commit
`05a34de` and this implementation:

| Case | Known balls | Baseline recovered | Updated recovered |
| --- | ---: | ---: | ---: |
| Clean flights | 5 | 5 | 5 |
| Centroid noise | 5 | 5 | 5 |
| Occlusion | 5 | 5 | 5 |
| Short occluded fragments | 5 | 0 | 5 |
| Dense volley | 9 | 9 | 9 |
| Crossing flights | 5 | 5 | 5 |

These are synthetic tracking results, not real-match scoring accuracy. A prediction matches at most
one known ball and must follow its time-dependent path for at least 80% of the observed points.

Reanalysis of 11 frozen `2025casj` matches preserves the official totals of every scoring category
for both alliances. Eight team rows across four matches change. The frozen, AI-reviewed qm10 pilot
remains **5/15** accepted scorer labels correct; there is no measured improvement on that pilot.
Counts matching FMS do not establish correct attribution. Input hashes, exact counts, and both
synthetic reports are in [the numerical report](TRACKING_ACCURACY_20261104.json).

## Videos and reproduction

- [2025casj Q11 saved-observation replay](demos/2025casj-qm11-observations.mp4): actual saved robot
  boxes, algae paths, and scoring shares replayed on a calibrated diagram. **No match footage.**
- [Synthetic ball tracking diagnostic](demos/ball-tracking-synthetic.mp4): all six generated scenarios,
  labeled throughout as synthetic. **No match footage.**

```bash
python -m scripts.benchmark_ball_tracking --out data/ball_benchmark.json --video data/ball_benchmark.mp4
python scripts/render_demo.py 2025casj_qm11 - data/q11-replay.mp4 \
  --results-dir data/accuracy_upgrade/after --observations-only --flight-labels \
  --width 960 --fps 20 --start 30 --duration 35
```

For a full-footage demo once a match MP4 is available:

```bash
autoscout recollect 2026cc_qm62 /path/to/match.mp4
autoscout relaunch 2026cc_qm62
python scripts/render_demo.py 2026cc_qm62 /path/to/match.mp4 data/q62-demo.mp4 --flight-labels
```

Existing saved launches remain readable. Recollection is necessary to obtain whole-ball components;
relaunching applies the new association to existing blobs. Reanalysis alone updates scoring logic.
Full-footage demos and independent scorer/shot labels for both years remain necessary before
claiming a large real-match accuracy improvement.

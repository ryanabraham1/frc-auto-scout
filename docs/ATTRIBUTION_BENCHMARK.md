# Robot attribution changes — November 1, 2026

Short detector dropouts previously left a moving robot's image box at its old position for up to
one second, while presenting that old box as an observation. Release regions also had a flat
zero distance over a large rectangle above each robot: overlapping regions were resolved by slot
iteration order. Correcting the box position alone made association worse because the existing
confidence penalty then preferred nearby observed robots over the corrected estimate.

The complete change moves a recent box with the estimated field position, preserves its measured
shape with perspective scaling, distinguishes estimates from observations, and uses a small
center preference to distinguish overlapping release regions. Short dropouts get a smaller
uncertainty penalty than positions without a recent detection. Both direct launch assignment and
2026 firing-history assignment use the same geometry and uncertainty rules.

Two separate 2026 credit errors are fixed: releases follow the team occupying the slot at release
time, rather than when the score posts; extending a volley window adds only older evidence instead
of counting recent evidence twice. Official fuel without any usable attribution or fallback
evidence is explicitly reported as `unattributed_fuel`, with a per-period breakdown. It is not
silently lost or invented as credit to an arbitrary robot.

## Measured association result

`python -m scripts.association_eval --out data/association_final.json` removes known robot boxes
for 0.2, 0.5 and 0.9 seconds before a sample. The missing field positions are interpolated using
the remaining observations. The hidden box supplies only a localization reference and a simulated
release point, randomized across the upper part of that robot. All same-alliance robots compete
for the release using the production attribution method. Samples are at least eight seconds apart
within each slot. Reports include source hashes and per-sample outcomes.

| Event | Samples | Previous association | Updated association |
|---|---:|---:|---:|
| 2025casj | 6,222 | 92.8% | 96.3% |
| 2025cc | 373 | 87.9% | 96.2% |
| 2026cc | 1,804 | 92.2% | 98.2% |
| 2026cosp | 6,384 | 89.6% | 97.8% |
| 2026dal | 148 | 80.4% | 93.2% |
| All 93 matches | 14,931 | **91.1%** | **97.1%** |

Errors fell from 1,325 to 427 (67.8% fewer). Mean box localization error fell from 0.304 to
0.115 robot widths (62.2% lower). For the longest simulated dropouts, association rose from
87.8% to 97.0%.

Candidate center weights of 0.25, 0.5 and 1.0 were explored on Silicon Valley and Pikes Peak data;
0.25 was selected. The final production rule also retains the larger uncertainty penalty when
there is no recent actual wide-shot box. Across the 18 validation matches from 2025cc, 2026cc and
Daly, which were excluded from weight selection, association improved from 2,110/2,325 (90.8%) to
2,269/2,325 (97.6%): 74.0% fewer errors. Validation reports are
`data/association_validation_cc.json` and `data/association_validation_dal.json`.

This is a **simulated-occlusion, physical-robot association benchmark**, not measured end-to-end
scoring accuracy. It does not verify bumper OCR, whether a real ball scored, calibration accuracy,
or long occlusions. The synthetic release distribution is an assumption, and existing detector
boxes serve as references rather than independent human labels. The three gap lengths share
sampled robots, so the 14,931 trials are not statistically independent. Do not describe 97.1% as
per-team scoring accuracy. A broad human-labeled scoring-action benchmark is still needed.

## Pipeline replay and cached results

`scripts/replay_attribution.py` snapshots source observations and can compare attribution methods
from git HEAD with the updated methods on identical inputs. It checks team membership and count
conservation, including human-player and explicitly unattributed 2026 fuel. These checks establish
consistency, not correctness of the scorer's identity. `--apply --current-only` refreshes saved
results and database rows, keeps original JSON in the run's `before` directory, and skips files
changed by another processing job after the snapshot. Runs and per-match reports are saved in
`data/attribution_refresh_20261101_*`.

For 2025 refreshes, each allocated category is checked against the official breakdown and any
shortfall is recorded in the replay report. An old cached result's missing allocation is not treated
as the reference total. Event-average comparisons against OPR from other qualification matches
showed small, mixed changes; they do not establish a large improvement in real scoring accuracy.

Regression tests cover moving missing boxes, overlapping release regions, identity switches
between release and score post, nonduplicated volley evidence, and explicit unassigned counts.

# Validation status (development build)

This is the historical initial-pipeline report. The detector and identity implementations have changed since
these runs. For current detector results and the evaluation protocol, see the README's robot-detector section
and `data/det_eval/results.json`. Detector precision now requires one-to-one, alliance-correct matches; the old
many-to-many precision numbers are not comparable. Official alliance totals remain a reconciliation constraint,
not an independent measurement of per-team accuracy.

Tested on 4 real broadcast matches with very different camera setups. All processing was done on a 4-core CPU
with no GPU.

| Match | Broadcast | Overlay score read (end of teleop) vs official | Shot detection ratio (detected / official fuel) | Notes |
|---|---|---|---|---|
| 2026dal_qm103 (Champs, Daly) | close, clear | R596/B293 vs R606/B304 (difference = post-buzzer counting and fouls) | red 1.59, blue 1.59 | bumper OCR identified 2 robots with confidence ≥ 0.86 |
| 2026txhou_qm11 (FIT District Houston) | far, refs block the near side | R41/B41 vs R44/B41 | red 2.0, blue 2.6 | robots small; detector weakest here |
| 2026cosp_qm14 (Pikes Peak) | medium | R171/B94 vs R181/B98 | red 1.38, blue 2.82 | 1 climb (8044, Level 1) copied exactly from FMS |
| 2025casj_qm11 (Silicon Valley) | medium | R52/B133 vs R68/B147 (difference = endgame, which posts after the buzzer) | 2025: per-robot points sum to 68/145 vs official 68/147 | score jumps decoded into coral levels, net and processor |

What is solid:
- **Match clock and live score reading.** The overlay layout is detected automatically for every broadcast tested,
  and digits are read with templates learned per broadcast.
- **Alliance totals.** These are exact by construction, because the official breakdown is split, not re-counted.
- **Per-station results:** tower and climb levels, leave and barge. These are exact.
- **Rejecting game pieces rolling or pushed on the carpet.** About 60% of moving-fuel tracks are rejected this way
  in 2026 matches. Unit tests cover pushed-away, rolling-sideways and occluded-arc cases.

What still needs your event's data:
- **Robot detection.** The current model was trained on only 9 hand-labeled frames, ensembled with FRC 578's Spy
  Glass. It finds about 35% of robot-frames in the Daly match. Label about 150 frames for your own event in the
  dashboard and retrain. This is the biggest accuracy lever: better tracks give better attribution, zone time and
  identity.
- **Team identity.** Bumper OCR works on close broadcasts (Champs). On far broadcasts most slots need a quick manual
  check, a few seconds per match on the match page. Climb and leave results help separate robots automatically when
  they differ.
- **Crowded moments.** A launch is credited to the nearest robot. In a crowd it can go to a neighbor; one of four
  random spot checks was likely misattributed. Every event links to the video so you can audit it.
- **Detection ratio above 1.** Some non-fuel motion and unmerged track fragments are still counted. Per-robot
  *scored* numbers are unaffected (they are shares of the official count), and feeds are scaled by the same ratio.

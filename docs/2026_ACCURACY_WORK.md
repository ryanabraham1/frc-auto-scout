# 2026cc measurement and identity review

Start the review page with `.venv/Scripts/python.exe scripts/review_2026_shots.py`,
then visit http://127.0.0.1:8001. The prepared set contains 100 single-piece hub
flights across qm62 (34), qm10 (33), and qm28 (33). Click the robot box, choose
the bumper team, and save. Scrub frames or replay the original video when needed.
Use Unclear or Not a make rather than guessing. All six teams are available so
wrong-alliance links can be recorded too.

The set and predictions are frozen in `data/shot_review_2026cc/manifest.json`;
manual labels are saved separately in `labels.json`. The summary reports physical
box agreement separately from team agreement. Unknowns are excluded from accuracy,
and the reviewed count remains visible. This benchmark measures attribution of
tracked single-piece flights; it does not measure missed-ball recall.

Codex visually reviewed all 100 using blinded three-frame sequences, close-up
bumper views, and full-resolution wide-shot crops. There are 35 attributable shots
and 65 uncertain origins. These are **AI visual-review labels, not human-verified
ground truth**. Decisions, evidence references, and uncertainty notes are committed
in `benchmarks/attribution/2026cc_video_review.json`; the original predictions and
source checksums are in `2026cc_frozen_predictions.json`. The qm28 labels are kept
separate from the qm62/qm10 development set.

Run `scripts/eval_2026_visual_review.py` to compare the frozen baseline with current
code. On the 35 labeled shots, team agreement improved from **29/35 to 32/35**.
Development agreement improved from 19/24 to 22/24; qm28 stayed at 10/11. Physical
box agreement improved from 27/31 to 30/31. Uncertain cases are excluded rather
than counted as correct. This small, visibility-selected sample does not establish
full-match accuracy or an 80% linking threshold.

Two fixes produced the attribution gains: a unique measured release near a robot's
top takes precedence over a neighbor's firing history, and stream-majority votes
preserve clear individual release associations instead of overwriting them.
Regression tests cover both neighboring-shooter failures.

A conservative fragment joiner now connects an untargeted rising flight to a later
hub-entry fragment. It fits the longer rising segment, requires a same-alliance
shooter and an unambiguous one-to-one ballistic continuation, and consumes both
original fragments so a ball counts only once. Three joins were found in qm28;
none changed the labeled sample, so no attribution-accuracy gain is claimed for
this feature yet. Short, noisy and ambiguous streams are rejected.

Run `scripts/audit_2026_identity.py` to regenerate
`data/2026cc_identity_audit.json`. The initial 16-match audit found 100 disagreements
among 284 close-up proximity comparisons and 132 strong-read tracklet conflicts.
These are flags for video review, not error-rate estimates: close-up association
uses proximity to the hub, and tracklet OCR participated in the original labeling.
No identities are changed by the audit.

`scripts/complete_2026cc.py` resumes unfinished qualification processing with one
worker, retries once, then writes completion, fuel OPR, and refreshed identity
reports. It can wait for an existing GPU PID with `--wait-pid`. A file lock prevents
duplicate instances. Output is redirected to `data/complete_2026cc.log` and
`data/complete_2026cc.err.log` for the running background job.

The original training job finished and event processing started. Three pending
matches already have cached videos (qm35, qm42, qm70). Other downloads are currently
blocked by YouTube's sign-in/bot check. No YouTube cookies are configured; set
`YT_COOKIES` in `.secrets/.env` to an authenticated Netscape cookies file before
rerunning the completion script. TBA also lists no video for qm16, qm27, qm40,
qm50, and qm56; those need a supplied video or corrected TBA video entry to reach
all 70. Missing matches are explicitly reported, never counted as completed.

`scripts/refresh_2026cc_accuracy.py` applies these fixes to cached qualification
results and the dashboard database, first backing up every input under
`data/2026cc_before_accuracy_fixes`. It asserts conservation of official alliance
fuel totals before saving each match. The background refresh also waits for the
completion job and refreshes newly available matches afterward. Its progress is
in `data/2026cc_accuracy_refresh.log` and `2026cc_accuracy_refresh.json`.

All 131 tests under `tests/` passed. Running pytest against the repository root
also collects old staging copies under `data/`, which causes duplicate-module
collection errors; run `python -m pytest tests -q` for the actual suite.

Further detector training and make-based credit still need broader validation.
Make-based credit remains off: the reviewed sample does not establish the required
end-to-end coverage, and the detector still misses visible robots.

## Learned linker and detector experiments

Full-event processing was stopped by request at 22 of 70 qualification matches, so
the event-level comparisons below use those 22. All cached matches now carry the
accuracy fixes, and a replay of current code reproduces production on all 132
team-match rows.

**Learned linkers stay experimental.** `scripts/train_shot_models.py` and
`train_shot_trees.py` train MLP and boosted-tree linkers under
`data/models/experiments/`. On a fresh qm35 set excluded from linker training, the
current linker scored 19/19, v1 and v2 `mlp_16` and v2 `mlp_32` scored 19/19, and
the rest 18/19. That set was AI-reviewed and selected for visible rising releases,
so it cannot separate the leaders. Nothing here shows a learned linker beating the
current code.

**Detector fine-tunes were rejected.** Neither the hard-frames run nor the mixed-data
run improved the independent benchmark, so the deployed weights are unchanged.

**Event-level replay.** `scripts/replay_learned_scout.py` writes isolated results to
`data/learned_scout_replay/`; `scripts/compare_scout_replays.py` compares them.
Every variant is forced to official alliance totals and OPR comes from those same
totals, so an all-match OPR flatters an equal split: it scored 0.75 team-mean
correlation, above every video variant. The comparison therefore fits OPR on the
48 matches outside the replay set and reports `within`, the correlation of each
team's deviation from its alliance mean.

| Variant | Within | Team mean | Team MAE | Within vs current (95% CI) |
| --- | --- | --- | --- | --- |
| current, live (production) | 0.314 | 0.51 | 66.8 | - |
| current, made | 0.327 | 0.52 | 69.9 | +0.015 [-0.047, +0.080] |
| v2 mlp_16, live | 0.300 | 0.49 | 68.6 | -0.014 [-0.057, +0.029] |
| v2 mlp_16, made | 0.340 | 0.51 | 72.1 | +0.026 [-0.041, +0.097] |
| v2 mlp_32, live | 0.309 | 0.50 | 66.9 | -0.005 [-0.062, +0.057] |
| v2 mlp_32, made | 0.339 | 0.52 | 69.4 | +0.026 [-0.041, +0.098] |
| equal split | 0.000 | 0.51 | 65.7 | - |

Production's within-alliance signal is real (95% CI 0.138 to 0.457, bootstrapped over
matches), so video does help divide an alliance's total. No variant differs from
production beyond noise, and team means do not beat an equal split yet: each team has
only about three replayed matches. Production stays on the current linker with
`SPLIT_BY = "live"`. Made-based credit has a small positive trend on `within` but
worse team MAE, so it is not enabled. OPR is itself a noisy proxy; these numbers rank
variants and are not per-match accuracy.

The identity audit flags are reviewed in the next section.

## Identity flag review

The audit's flag counts overstate the problem. Of 158 tracklet flags, 151 were tiny
unassigned fragments totalling 13 seconds. Clear reads agreed with the assigned team
for 4,430 seconds and disagreed for 21. Only 44 of the 201 close-up flags had the read
team away from the hub, and a close-up routinely shows the second-nearest robot.

`benchmarks/attribution/2026cc_identity_review.json` records an AI visual review of a
stratified sample: all 7 assigned tracklet conflicts, 22 repeated close-up conflicts,
4 far and 12 mid-range flags, and 12 random frames. Of the 45 sampled flags, 7 were
real, 37 false alarms and 1 not checkable. Those 7 are five distinct errors; a sixth
was spotted in passing. In the random frames, 44 of 45 labelled boxes showed no error
and 1 pair was unresolved, so no error was confirmed. That bounds the box-level error
rate at roughly 7%, and the sample is too small to go lower. Only 45 of 72 possible
robots had a box in those frames, so coverage is the larger gap.

Where a wrong tracklet had reads, they named the right team: motion links (up to 30)
outweighed a clear read (about 9.5 for a read of 2.1). Raising the weight in a full
re-solve looked good on the known errors, but a random sample of its changes showed
correct labels being overwritten, so re-solving is unsafe. `RobotTracker._repair_anchors`
instead starts from the finished labelling and only exchanges stretches that fix a
contradicted unanimous read. Reads with any dissent are ignored: one such read, from
a close-up proximity match, broke two correct qm15 labels until this rule was added.

`scripts/repair_2026cc_identity.py` applied this to the 22 cached matches. Five changed
(qm1, qm11, qm13, qm35, qm70): 51 tracklets, 93 seconds. Originals are in
`data/2026cc_before_identity_repair`, and alliance totals stayed conserved. The verified
tracklet set went from 42/63 to 50/63; it is enriched for errors and was used for
tuning, so it is not an accuracy estimate. Fuel moved most in qm11 (25) and qm13 (16).
The shot benchmarks contain no repaired match apart from one qm35 tracklet.

On held-out OPR the within-alliance correlation went from 0.314 to 0.318 (+0.003, 95%
CI 0.000 to +0.008). Identity errors are real but brief, and they do not explain flat
per-team accuracy. Detection coverage remains the leading suspect.

Known remaining errors: the yellow-frame robot in qm13 (846) is labelled 359 or 6017
for long stretches, a three-way mix-up that a pairwise exchange cannot fix; qm11 2046
and 6036 stay swapped around 74 to 96 seconds; qm10 973 and 6647 stay swapped around
18 to 21 seconds, and a box labelled 6800 near 100 seconds is 841; a qm15 box labelled
5940 near 37 seconds is another robot.

## Why full re-solves change labels

An earlier version of this document said re-solving damages correct labels and that a
larger appearance bank might do the same to later matches. Neither claim holds. The
sample behind it came from the heavier-read sweep, not from a plain re-solve.

The cause is a code version gap. Fifteen cached matches (qm1 to qm13, qm28, qm57) were
labelled on 31 October before commit `ad14cdf` added within-match appearance and
multi-start labelling. They account for 700 of the 707 tracklets a re-solve changes.
The seven matches labelled by current code change by 7 tracklets in total, so the
solver is stable on its own output. The re-solved labelling had a lower cost than the
saved one in every changed match, and almost all of that gain came from match appearance.
The bank is not the cause: re-solving with no bank changed 787 tracklets, and
without the match's own rows 770.

With 1 in 4 clear reads hidden (`scripts/identity_holdout.py`), match appearance names
93.8% of observations correctly, against 84.8% with `MATCH_APP_WEIGHT=0`. That test only
covers tracklets that have a read, so it does not show what happens on the rest.

Re-solving the old matches was not shown to be better, so production keeps its labels.
`scripts/replay_resolved_identity.py` replayed all 22 matches in scratch copies. Held-out
within-alliance correlation against the 0.318 baseline: earlier-matches bank 0.314, bank
without the match 0.301, no bank 0.330. Every 95% range spans zero, roughly -0.09 to
+0.09, so OPR cannot separate them.

A blind AI review took 32 of the 160 team-to-team changes (15+ observations) in the old
matches, with the A/B order hidden until the readings were recorded. The re-solved label
was right 14 times, the saved label 13, neither 3, and 2 were unreadable. One "neither"
(qm7 id113) rests on a partial read. The result depends on the match: qm13 favoured the
re-solve 6 to 1, the other matches favoured the saved labels 12 to 8. Several re-solve
errors are one confusion, qm1 971 labelled 2910. Where the two labellings disagree,
each is wrong close to half the time.

Saving a dashboard label on an old match therefore also replaces its other labels with
a different, about equally accurate set. Current-code matches are not affected, and
nothing here blocks processing the remaining 48 matches.

**Bumper reading is the next lead.** 29 of the 32 sampled tracklets had no read at all,
yet the review could read the full number on 25 of them, 23 of those with no app read.
The sample is drawn from disagreements and the review used each tracklet's largest box,
so this does not give an event-wide miss rate.

Run `scripts/identity_weight_sweep.py` to reproduce the sweep. The working scripts and
verdicts are in `data/identity_review/` (`resolve_diag.py`, `blind.py`,
`blind_verdicts.json`).

## Fused bumper reads

The live reader judges one frame at a time and needs that frame to pass its thresholds
alone. `data/identity_review/strong_read.py` instead scores the alliance's teams with
the same recognizer on up to 10 of a tracklet's largest views, 3x enlarged, and adds the
evidence up, clipped per frame. The margin is the lead of the best team over the next.

It ran on the 1,379 unread or weakly read tracklets of 15+ observations in the 22 cached
matches, plus 318 clearly read controls. At margin 15 or more, all 292 controls were
right, and 762 targets qualified. Controls are easy, so that overstates precision. On a
blind sample of 30 confident disagreements with the saved label, 23 were readable: fused
right 17, saved right 3, neither 3. All three wrong calls against the saved label were
qm70 9408 read as 359, and 846's bumper is also hard for the recognizer.

`scripts/replay_fused_reads.py` tested the reads in scratch copies, weight 2.5:

| Variant | Changed | Seconds | Truth /63 | First blind set /27 | Held-out within (vs 0.318) |
|---|---|---|---|---|---|
| Saved | 0 | 0 | 50 | 13 | 0.318 |
| Margin 15, repair | 303 | 595 | 55 | 21 | 0.336 (+0.018, CI -0.026 to +0.065) |
| Margin 30, repair | 197 | 393 | 55 | 19 | 0.311 (-0.005, CI -0.035 to +0.019) |
| Margin 15, re-solve | 749 | 954 | 57 | 24 | 0.321 (+0.006, CI -0.055 to +0.071) |
| Margin 30, re-solve | 711 | 868 | 56 | 22 | 0.257 (-0.058, CI -0.149 to +0.025) |

The truth and first blind sets helped shape this work, so they flatter it. The fair test
is a fresh blind sample of 32 random team-to-team changes by margin 15 repair, from tracklets
not reviewed before. 18 were readable: new label right 14, saved right 0, neither 4;
14 were unreadable. Of the 27 knock-on changes in that sample, 14 were readable: 10 new
right, 4 neither. One verdict (qm4 id294) rests on the robot's look rather than its number.

Margin 15 repair is the candidate. Held-out OPR does not confirm it, and all reviews are
AI readings. It fixes labels; whether fuel attribution improves is not shown.

`scripts/apply_fused_reads.py` applied it to all 22 cached matches: 762 reads added, 303
labels changed, 701 fuel moved between teams. Results match the tested replay and the
dashboard database exactly, and official alliance totals hold. Originals and a copy of
the database are in `data/2026cc_before_fused_reads`; the appearance bank was not updated.
The fused reader is not in the pipeline, so the remaining 48 matches would not get it.

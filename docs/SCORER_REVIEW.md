# Scorer review: ground truth for attribution accuracy

The simulated-dropout benchmark ([ATTRIBUTION_BENCHMARK.md](ATTRIBUTION_BENCHMARK.md)) reports 97% physical
association. The one video-reviewed match ([VIDEO_ATTRIBUTION_BENCHMARK.md](VIDEO_ATTRIBUTION_BENCHMARK.md))
found the right scorer for only 5 of 15 actions. Per-team correlation with OPR is too noisy to tune against.
The dashboard's **Review credit** page provides the missing measurement: people label who actually scored.

## Labelling

`autoscout serve`, then **Review credit** → pick a match (or **Review credit** on a match page). Each action
shows a short 720p clip cut from the local match video:

| Action | Season | Clip | Question |
|---|---|---|---|
| `score` | 2025 score jumps, 2026 live-score increases | 4 s before the post to 1 s after | Which robot scored this? |
| `launch` | 2026 tracked FUEL flights | 2 s before release to 2 s after; a ring marks the release, a dashed line marks the flight | Which robot (or the human player) did this come out of? |

Keys: `1`/`2`/`3` choose the alliance's teams in station order, `h` for human player, `u` for can't tell,
`x` for didn't happen (not a real action), `space` to replay, `s` for half speed, and the arrow keys
to move. The system's credit stays hidden until the label is saved. After that, the next action shows whether
the previous answer agreed.

Actions are queued in a fixed pseudo-random order per match. Any completed prefix is therefore a random
sample of that match's actions, rather than only the easy ones. Aim for **300–500 labelled actions across
several events and both seasons**. With fewer, a few percent improvement is not distinguishable from
sampling noise.

## Storage

Labels are saved in `benchmarks/attribution/<match>.human.json` (`reviewer_type: human_dashboard`). The
format is the AI-reviewed pilot `2025casj_qm10.json` format plus `action`, `kind`, `reviewer` and
`saved_at`. Commit these files: they are the benchmark. Clips are cached in `data/review_clips/` and can
be deleted.

## Scoring

```bash
autoscout review-eval                 # every labelled match with a saved result
autoscout review-eval 2026cc --out data/review_eval_2026cc.json
autoscout review-eval --include-ai    # also the AI-reviewed pilot files
```

This reports correct-scorer accuracy (the reviewed team must have the single largest share), the mean
share given to the reviewed team, and accuracy by action and kind. It also reports the fraction of
system actions people marked as didn't happen. Missing or tied predictions count as errors. Can't-tell
and didn't-happen labels are excluded from accuracy. Labels are matched to predictions by action,
alliance and time (within 0.05 s), so a result can be reanalyzed and scored again against the same
labels. A `relaunch` or `rescore` that moves event times leaves those labels unmatched; they report as
`missing_prediction`.

**Limits:** only actions the system produced can be reviewed, so missed scoring actions (recall) are not
measured. After changes are tuned on labelled matches, label new matches for an honest held-out number.

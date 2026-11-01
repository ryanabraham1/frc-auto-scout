# Video-reviewed scoring attribution pilot

On November 1, 2026, the assistant visually reviewed Silicon Valley qualification match 10
(`2025casj_qm10`), [the original broadcast](https://www.youtube.com/watch?v=u-AQDAo-7qY).
These are **AI-reviewed video labels, not independently human-verified ground truth**.

The review used the full broadcast sampled every two seconds, both unannotated reef close-ups
around every candidate score post, and denser one-second sequences for ambiguous intervals.
Predicted scorer identity and tracker boxes were hidden during labeling. Candidate timestamps
came from the system; candidate scoring categories were seen during setup. The evidence is
therefore independent of the predicted team credit, but this is not a fully blind experiment.
Labels were written and hashed before examining the system's team-credit output. The review did
not use the system's per-team totals to decide labels, and no parameters were tuned to this pilot.

## Result

Of 25 candidate coral actions, 15 had accepted scorer labels, two had provisional identities and
eight remained unresolved. Only the 15 accepted labels contribute to primary accuracy: **60%
coverage of candidate actions**. The current saved output and a clean-source replay of commit
`3a17553` each selected the reviewed scorer for **5/15 actions (33.3%)**.

| Alliance | Accepted actions | Correct top scorer |
|---|---:|---:|
| Red | 5 | 3 (60%) |
| Blue | 10 | 2 (20%) |
| Total | 15 | 5 (33.3%) |

This is much weaker than the simulated-dropout association result. That earlier test checks
geometric association to a physical robot; it does not validate identity tracking or end-to-end
scoring attribution. The pilot identifies repeated wrong team credit, especially placements by
3045 credited to 1700 or 670. It does not establish which stage caused each error.

## Accepted labels and frozen main predictions

Times are video seconds at the candidate score post, not match-clock seconds or exact release times.
Each link starts approximately three seconds earlier for review.

| Post | Visually reviewed scorer | Main top credit | Correct |
|---|---:|---:|---|
| [red 48.05](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=45s) | 6814 | Tied | No |
| [red 77.08](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=74s) | 10644 | 10644 | Yes |
| [red 96.10](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=93s) | 6814 | 6814 | Yes |
| [red 117.12](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=114s) | 6814 | 6814 | Yes |
| [red 143.64](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=140s) | 6814 | 649 | No |
| [blue 12.51](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=9s) | 3045 | 3045 | Yes |
| [blue 35.54](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=32s) | 3045 | 1700 | No |
| [blue 46.05](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=43s) | 3045 | 1700 | No |
| [blue 65.06](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=62s) | 3045 | 1700 | No |
| [blue 70.57](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=67s) | 3045 | 1700 | No |
| [blue 75.58](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=72s) | 3045 | 3045 | Yes |
| [blue 82.08](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=79s) | 3045 | 670 | No |
| [blue 92.59](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=89s) | 670 | 1700 | No |
| [blue 143.14](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=140s) | 670 | 3045 | No |
| [blue 146.15](https://www.youtube.com/watch?v=u-AQDAo-7qY&t=143s) | 3045 | 670 | No |

The labels include evidence notes and intervals for all 25 candidates, including the unresolved
ones. Their accepted per-team counts are lower bounds for reviewed actions, not full scouting totals.
In particular, unresolved low placements, overlapping posts around 119–122 seconds, and late
endgame updates prevent a complete per-team count. Scoring levels, missed actions, and failed
attempts are not evaluated by this pilot.

## Reproduce

From the repository root:

```powershell
python -m scripts.manual_attribution_eval --result-file benchmarks/attribution/2025casj_qm10.predictions.json
python -m scripts.manual_attribution_eval --result-file data/results/2025casj_qm10.json --out data/manual_attribution_latest.json
```

The committed prediction extract is frozen from the clean-source main replay; it includes its
source commit and original full-result hash. The replay reuses cached detections and existing
match calibration/identity evidence rather than rerunning video inference. The corresponding
report includes hashes of the label and prediction files. The evaluator counts missing, ambiguous
and tied predictions as errors instead of silently excluding them. Provisional and unresolved
review labels are excluded from accuracy and remain visible in coverage.

An independent human reviewer should verify these labels before treating them as a gold benchmark.
After tuning against this match, it must become a development set; a new match should be reviewed
before evaluating the tuned system. The committed evidence intervals and video links make that
verification concrete.

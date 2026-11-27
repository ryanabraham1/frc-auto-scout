# 2025 accuracy benchmark

`scripts/bench2025.py` measures how accurate the 2025 scouter is without grading it against its own
inputs.

```
.venv/Scripts/python.exe scripts/bench2025.py 2025casj --matches docs/2025_BENCH_MATCHES.json
.venv/Scripts/python.exe scripts/bench2025.py 2025casj --matches docs/2025_BENCH_MATCHES.json \
    --set autoscout.games.reefscape2025.NODE_CREDIT=False        # compare a changed constant
```

## Why not compare our score with the final score

The scouter decodes the broadcast score's jumps and splits the official FMS counts among the robots,
so its totals equal the final score by construction. The comparison only means something when the
score is hidden from the part being tested. That is the `reef` check below.

## What each check measures

| Check | Truth | What it tests |
| --- | --- | --- |
| `reef` | FMS node map (branches A-L x L2-L4, end of AUTO and end of match) | The REEF read from vision alone: 3D cameras and CORAL detector. Precision, recall, and the REEF points the branches add up to against the official REEF points. |
| `fms` | FMS per-robot LEAVE and BARGE | AUTO coral credited to a robot that never left its line (provably wrong). Whether the robot we call team X moved in AUTO and ended at the BARGE as FMS says: identity and tracking. |
| `plays` | Scores a person checked on footage | `benchmarks/attribution/*.user_verified.json`, then AI-read footage checks, reported apart. |
| `scouts` | Scout sheets, only where FMS confirms them | See below. |
| `opr` | Official counts split by leave-one-match-out component OPR | Comparing two versions (paired bootstrap). Too noisy to grade one version. |
| `timing` | none (a count) | Of L2-L4 scores: paired with a branch, timed by its own coral, robot seen at the spot. |
| `decode` | Official counts | Decoded score-jump actions against the official counts. |

### Scout sheets, and how far they are trusted

Team 1351 published its scout sheets for all 74 qualification matches of 2025casj. Scouts miscount,
so the sheets are never used as they are. For each alliance, phase and level, the three scouts' counts
are added up and compared with what FMS counted. Only cells where the sum is exactly right are kept
(201 of 357 on the 43 benchmark matches): two errors that cancel inside one cell are possible but
unlikely. The sheets are not in the repository (no licence, and the raw files carry student names);
`data/labels/scouting_2025casj.json` holds the numbers only.

A confirmed cell in which one robot scored everything pins the scorer of every play in it. That gives
about 320 L2-L4 plays with a known scorer, grouped by the evidence the credit rested on.

## What it found

**REEF scores are entered by people and can post late.** Most post within a second or two of the
coral landing. Some trail by 10 to 20 s, and a few arrive in a burst at the end of the match (qm23:
coral landed at 118.9 s and 128.7 s, both posted after 139 s). Looking for the scoring robot around
the post then finds whoever is at the REEF later. Credit now follows the moment vision saw the coral
come to rest.

**The 3D branch credit is worth 10 to 20 points.** On the 43 benchmark matches, coral credited to the
scouted robot (an equal three-way split scores 44%):

| Version | Coral | L2-L4 | L1 |
| --- | --- | --- | --- |
| No 3D branch credit | 67.6% | 69.2% | 57.6% |
| 3D branch credit as of 2026-11-12 | 77.8% | 80.5% | 61.2% |
| Shares sharpened | 84.2% | 88.0% | 61.2% |
| Late posts keep the coral's time, new REEF reader | 86.3% | 90.6% | 60.0% |
| Hidden robots, L1 trough credit | 89.0% | 91.2% | 75.3% |
| Close-up cameras refined, AUTO branches take AUTO coral | 88.8% | 91.2% | 74.1% |

Fourteen matches that were never used to tune anything (qm45 to qm58) score 88.0% (L2-L4 89.1%, L1
80.0%), and 88 of their 90 plays with a known scorer go to the right robot. Without the 3D branch
credit the same matches score 79.5%.

The last row is within noise of the one above it on this measure; those two changes were kept for what
they fixed directly (camera error at the branches from 15 and 21 px to 6 and 5 px, and the
user-verified qm1 L4 at 0:53).

Per team, the average coral a match now sits 0.45 from the scouts' average (0.62 before), and the top
scorers are no longer short: 1323 gets 22.9 a match (scouts 23.2, OPR 24.4; 21.1 before), 581 gets
15.3 (scouts 16.0; 13.9 before).

**Where a robot is seen at the branch as the coral lands, the call is right about 94% of the time**
(203 of 216). Where the coral was not timed by vision and the score's own time had to be used, 89%.
L1 is 74%.

**The old half-event OPR check misleads.** It flagged 766 as a robot that scores no L4, because one
dead match fell in the reference half. Leave-one-match-out OPR barely moves when credit clearly
improves, so it is kept only for paired comparisons.

**The REEF read with no node map** finds 74% of filled branches and 83% of what it reports is real;
the REEF points it adds up to track the official ones at r = 0.94 (before this work: 94% found, 36%
real, r = 0.72). Coral in the trough or on the carpet next to a branch, and branches that share pixels
in a view, are what it still gets wrong. The scouter itself is given the node map, so this is a
check of the geometry, not of the credit.

## What is still wrong

- **Identity when robots are packed at the REEF.** Their tracklets are cut short there and the
  labelling has least to go on; several of the remaining wrong calls are a partner standing in for the
  scorer (qm4 blue: a box labelled 2643 whose bumper reads 581).
- **Coral that vision did not time** (about a third of L2-L4 scores): a new coral lands against one
  already boxed and never gets its own track.
- **AUTO tracking.** Robots are often lost as they first drive off: of robots FMS says left their
  line, many are only seen standing at it.
- **Algae** is untouched by any of this: NET and PROCESSOR credit still follows the score's time.

## Tried and not kept

- **More bumper reading does not fix the identity errors.** Reading bumpers across each unread
  tracklet's best views (the 2026 fused reads) added 228 reads on the 43 matches and relabelled 124
  tracklets: no measurable change. Reading the bumper of the robot standing at the branch as the coral
  lands, and trusting that over the tracker's label, would be worse: where the two disagree the label
  is right about four times in five.
- **Constants.** The windows and spreads around the scoring spot, the trust in a score-timed pairing,
  the arrival step level and the trough geometry were all swept; every variant moved the result by
  less than a point.

## Limits of the benchmark

One event, one camera layout. The scout check covers the 56% of cells FMS confirms, which leans
toward alliances that are easier to scout. Differences under about one point are noise: the paired
bootstrap range printed with `--set` says how much a change can be trusted.

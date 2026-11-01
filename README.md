# FRC Auto-Scout

Automated per-robot scouting from FRC match broadcast video. Works for **2026 REBUILT** and **2025 REEFSCAPE**,
with a plugin template for **2027**.

For every match it produces a per-team breakdown, for example in 2026:

| Team | Fuel scored (auto / per shift / endgame) | Shots at inactive hub | Fuel fed (from neutral / opponent zone) | Tower (auto + endgame) | Zone time |
|---|---|---|---|---|---|

and in 2025: coral per level (L1–L4, auto/teleop), net and processor algae, leave, barge result, defense time.

## How it works

```
TBA match (teams, video, official score breakdown)
  -> download video (yt-dlp, 1080p)
  -> read the broadcast overlay: match clock + live alliance scores over time   (overlay.py, digits.py)
  -> map the wide shot to field meters with a per-event calibration             (field.py)
  -> detect robots (YOLO) + track exactly 3 slots per alliance in field coords (vision/detect.py, tracker.py)
  -> track airborne game pieces (launches), rejecting pieces rolling or pushed
     along the carpet                                                            (vision/trajectories.py)
  -> assign slots to team numbers: bumper OCR + per-station official results + manual fixes (vision/identity.py)
  -> season plugin: launches/score jumps/positions -> per-robot evidence          (games/rebuilt2026.py, reefscape2025.py)
  -> split the *official* alliance totals (FMS breakdown) among robots by that evidence
  -> SQLite + dashboard (team tables, match breakdowns, events linked to the video) (db.py, web/)
```

**Accuracy model.** Alliance totals are exact because they come from the official FMS breakdown, e.g. 2026
`hubScore` FUEL counts per period, or 2025 coral per level. Video decides *who* did it: each robot's share of an
official count is proportional to what the system saw that robot do in that period. Per-station results are copied
exactly: 2026 tower levels, 2025 leave and barge. Feeding has no official count, so feeds are the detected count
scaled by the alliance's detection ratio.

Short-gap robot association now uses motion-compensated boxes and resolves overlapping launch regions by
release location. On 14,931 simulated detection dropouts from 93 cached matches, physical robot association
improved from 91.1% to 97.1%; this is not an end-to-end scoring-accuracy measurement. See
[the attribution benchmark](docs/ATTRIBUTION_BENCHMARK.md) for validation splits and limitations. In 2026,
official fuel with no usable robot or fallback evidence is reported separately as `unattributed_fuel`.

**2026 rules encoded.** A launch from the robot's own Alliance Zone is a *shot*, because scoring is only legal from
there. A launch from the Neutral Zone or the opponent's zone is a *feed*. A launch from off-field at an Outpost is a
*human player* throw. The Hub schedule follows the rules: both Hubs are active in auto, the transition shift and
endgame; the auto winner is inactive in shifts 1 and 3. Shots at an inactive Hub are counted separately.

**2025.** The live overlay score says *what* was scored and when; the video says *who*. All of an alliance's score
jumps are decoded together so they add up to the official FMS counts per category (L1–L4 auto/teleop, net,
processor, leave, barge, foul points): a +4 that could be teleop L3 or NET goes to whichever the official counts still
need and the robots' positions fit, and the +3s of LEAVE posted at the end of auto stay LEAVE. Where the broadcast
shows each alliance's algae count (configure `overlay.counters` in the event YAML, as for 2025casj), every jump's
number of algae is read from it directly. Each robot action is
then credited from several clues: coral seen *appearing on the reef* in the wide shot (where on the reef → which
robot was there; `vision/element_changes.py`), a robot's elevator going up (its box getting taller), time spent at
the element around the post, and algae seen flying from a robot to the NET. A robot carries one coral at a time,
so it is unlikely to be credited two coral within 3 s. NET shots happen on the far side, where robots are often
not detected, so a robot the tracker can't see at that moment stays a NET candidate. Repeated close-up bumper reads
overlapping the reef supply dwell evidence when the wide-shot floor mapping is unreliable. Guessed positions expire
after 3 s and need at least three prior sightings. AUTO uses confirmed close-up reads, without an even-split preload
prior. Reach above the chassis supports L2-L4 credit conservatively, relative to the robot's recent background;
missing reach measurements stay unknown. See [the 2025 progress and evaluation notes](docs/2025_ACCURACY_PROGRESS.md).

2025 accuracy, Silicon Valley (per-team averages vs OPR from matches the scouter never saw; 11 held-out matches never
used for tuning): coral r 0.58 → 0.67, L4 0.54 → 0.59, net algae 0.79 → 0.80; score jumps decoded into the wrong
category 44% → 4%. Check with `python scripts/eval2025.py 2025casj`. Matches processed before the reef watcher
existed get it with `autoscout watch-elements EVENT` (seconds per match, no robot detection), and the algae counter
with `autoscout rescore EVENT`.

2025 robot detection uses our own model only (no Spy Glass), with field checks: a robot's ground point must map
inside the field line (2025 `robot_margin`), not inside a REEF, and no box as wide as a REEF over its base counts.
On 12 hand-marked 2025 frames (48 robots): recall 75%, precision 90% (the ensemble without the checks: 83% / 61%).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
# secrets (git-ignored): .secrets/.env with
#   TBA_API_KEY=...            (thebluealliance.com/account)
#   YT_COOKIES=.secrets/cookies.txt   (only needed if YouTube blocks downloads)
autoscout calibrate 2026txhou_qm1        # saves a reference frame for the event
autoscout serve                           # http://127.0.0.1:8000 -> Calibrate -> click >= 4 landmarks
autoscout run-event 2026txhou            # processes every match with a video
autoscout export 2026txhou --out txhou.csv
```

Useful commands:

| Command | What it does |
|---|---|
| `autoscout run-match 2026txhou_qm11 --debug-video` | one match, and writes an annotated video to `data/results/` |
| `autoscout reanalyze 2026txhou` | recompute stats from saved observations (seconds; after identity fixes or logic changes) |
| `autoscout report 2026txhou` | validation summary vs. official results |
| `autoscout watch-elements 2025casj` | find coral placed on the reefs in already processed 2025 matches, then reanalyze |
| `autoscout train-robots` | (re)train the robot detector on `data/datasets/robots` |

## Robot detector: the one thing to build per season

Robot detection is the weakest link. Broadcast robots are small, and referees and piles of game pieces hide them.
The detector used is the first of these that exists:

1. `data/models/autoscout-robots.engine`, an optional TensorRT export (CUDA only; skipped when stale)
2. `data/models/autoscout-robots.pt`, a model you trained (`autoscout train-robots`)
3. `models/autoscout-robots.pt`, the shipped model
4. `data/models/spyglass/model11n.pt`, FRC Team 578's "Spy Glass" bumper model, used only when there is no model of our own

On CUDA, portable models use 1600-pixel inputs by default; CPU/MPS use 1280. Override with
`AUTOSCOUT_IMGSZ=1280` when throughput matters. The CUDA default checks the same model at 1280 and 1600,
requiring moderate-confidence boxes to agree across scales; high-confidence boxes stand alone. This reuses
one model's weights. Set `AUTOSCOUT_MULTISCALE=0` for single-scale inference. Stale TensorRT engines are skipped, and CPU deployments
use portable weights. People (referees, field staff, people at the rail) are removed with a COCO person detector.
Spy Glass is no longer ensembled by default. On hand-marked frames it added 3 points of recall but cost 11 points
of precision (see below).

**Measure before you switch models.** `scripts/det_eval.py` scores any weights against hand-marked robot points:

```bash
python scripts/det_eval.py extract 2026cosp_qm40 45,80,115,150   # wide-shot crops + gridded previews
# mark every robot as [x, y, "r"|"b"] in data/det_eval/points.json
python scripts/det_eval.py eval data/models/autoscout-robots.pt models/autoscout-robots.pt
```

The benchmark now matches robots and predictions **one to one**, with matching alliance colours required for
both precision and recall. Duplicate boxes and wrong-alliance predictions count as false positives. Older
many-to-many precision scores are archived in `data/det_eval/results_legacy.json` and should not be compared.
On the existing 20 frames (84 robots, five events), the shipped weights score **76.2% recall / 80.0% precision**
at 1280, **77.4% / 84.4%** at 1600, and **79.8% / 87.0%** with the CUDA two-scale default on this RTX 3060.
These settings were compared on this benchmark; confirmation on additional matches is still needed.
This small benchmark measures detection, not end-to-end
team attribution or scoring accuracy. Per-frame errors, F1, input size and timing are saved with each result.

To label frames by hand, use the dashboard's **Label robots** page:
- Click **Suggest**: the current model pre-fills boxes.
- Fix the boxes. Label *every* visible robot.
- Click **Save frame**, or **Save & +5 s** to move on.

Corrected frames are saved separately in `data/datasets/hand`. The video picker shows cached match names and
selects the event automatically; calibrations scale to the video's actual resolution.
Click **Train detector**, or run `autoscout train-robots`. Training uses corrected frames with extra weight,
keeps whole matches/videos separate for validation, and excludes benchmark matches (including cached video-ID
aliases). The split manifest is saved in `data/datasets/mixed/split.json`. Only one training process may run.
The dashboard shows CLI training progress too.

For a short fine-tune using only corrected frames:

```bash
autoscout train-robots --hand-only --freeze 10 --epochs 30 --batch 4 --imgsz 1600
```

Candidates and evaluation reports stay in `data/models/experiments/<run>/`. Candidates are compared using the
same inference settings as deployment. Training replaces deployed weights
only when held-out F1 improves without reducing precision or recall. Previous deployed weights are backed up,
and an obsolete TensorRT engine is retired. Without a benchmark, the candidate is saved for review.

## Team identity (bumper numbers)

Every robot fragment is checked against the six team numbers in the match. A scene-text recognizer (PARSeq,
downloaded once through torch.hub) scores how likely each number is on the robot's bumper, using windows slid along
the bumper. A read only counts when one team clearly beats both the others and the recognizer's own free read. On
Windows and Linux this replaced EasyOCR, which rarely read a full bumper number, and is what made 2025 identity
correct. macOS still uses Apple Vision. Close-up camera panes (reef cams, hub cams) are read the same way.

Check identity by eye with `python scripts/identity_sheet.py MATCH out.jpg`: one row of robot thumbnails per team.

## Mac (Apple silicon)

See [docs/SETUP_MAC.md](docs/SETUP_MAC.md). Robot detection and training use the Apple GPU automatically.

## Hardware (i9-12900K + RTX 3060)

See [docs/SETUP_PC.md](docs/SETUP_PC.md). With CUDA:
- robot detection runs at 10 Hz with FP16 (TensorRT after training)
- OCR runs on the GPU
- a 2:40 match processes in about 1–2 minutes

On CPU only it takes about 4–5 minutes per match.

The main frame loop decodes one frame ahead while processing the current frame. This preserves
all frame pixels, timestamps, sampling rates and detection settings, with a bounded frame buffer.
The blob-only collection command also skips live trajectory linking, whose output it does not use.
These changes apply to newly started processing jobs.

## New season (2027)

See [docs/NEW_SEASON.md](docs/NEW_SEASON.md). You write one field YAML and one plugin file (`games/template_new_season.py`).
Everything else is season-independent: video ingest, overlay clock/score reading, calibration, tracking, identity,
launches, reconciliation, database and dashboard.

## Repository layout

```
autoscout/            package
  overlay.py digits.py   scoreboard reading (self-teaching digit templates)
  field.py               field models + calibration (homography) + per-video alignment
  vision/                detection, slot tracker, team identity, launch trajectories
  games/                 season plugins (2025, 2026, template)
  pipeline.py            per-match processing; reanalyze()
  train/                 labeling data engine + detector training
  web/                   FastAPI dashboard
configs/fields/<year>.yaml   field geometry, zones, elements, piece colors
configs/events/<event>.yaml  per-event calibration (+ <event>_overrides.yaml for identity fixes)
tests/                       unit tests (pytest)
```

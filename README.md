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

**2026 rules encoded.** A launch from the robot's own Alliance Zone is a *shot*, because scoring is only legal from
there. A launch from the Neutral Zone or the opponent's zone is a *feed*. A launch from off-field at an Outpost is a
*human player* throw. The Hub schedule follows the rules: both Hubs are active in auto, the transition shift and
endgame; the auto winner is inactive in shifts 1 and 3. Shots at an inactive Hub are counted separately.

**2025.** Every jump in the live overlay score is matched to the scoring element a robot of that alliance was
working at in the preceding seconds (Reef, Net side of the Barge, Processor). The point value identifies the action:
for example, +5 at the Reef in teleop is an L4 coral. Algae seen flying into the Net adds evidence.

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
| `autoscout train-robots` | (re)train the robot detector on `data/datasets/robots` |

## Robot detector: the one thing to build per season

Robot detection is the weakest link. Broadcast robots are small, and referees and piles of game pieces hide them.
The detector used is the first of these that exists:

1. `data/models/autoscout-robots.engine`, a TensorRT export made automatically on a CUDA machine
2. `data/models/autoscout-robots.pt`, a model you trained (`autoscout train-robots`)
3. `models/autoscout-robots.pt`, the shipped model
4. `data/models/spyglass/model11n.pt`, FRC Team 578's "Spy Glass" bumper model, used only when there is no model of our own

People (referees, field staff, people at the rail) are removed from robot detections with a COCO person detector.
Spy Glass is no longer ensembled by default. On hand-marked frames it added 3 points of recall but cost 11 points
of precision (see below).

**Measure before you switch models.** `scripts/det_eval.py` scores any weights against hand-marked robot points:

```bash
python scripts/det_eval.py extract 2026cosp_qm40 45,80,115,150   # wide-shot crops + gridded previews
# mark every robot as [x, y, "r"|"b"] in data/det_eval/points.json
python scripts/det_eval.py eval data/models/autoscout-robots.pt models/autoscout-robots.pt
```

On 20 frames from five events, the shipped model gets 76% recall and 85% precision. A YOLO11s trained on about 1,800
automatically labelled frames (`scripts/build_robot_dataset.py`) got 74% recall and 83% precision, so it is not used.
Automatic labels can't beat the model that made them. Beating it needs frames labelled by hand.

To label frames by hand, use the dashboard's **Label robots** page:
- Click **Suggest**: the current model pre-fills boxes.
- Fix the boxes. Label *every* visible robot.
- Click **Save frame**, or **Save & +5 s** to move on.

Then click **Train detector**, or run `autoscout train-robots`. Matches in `data/det_eval/points.json` are always
kept out of training.

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

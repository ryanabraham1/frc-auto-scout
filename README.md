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
| `autoscout aux 2026cc_qm4 <video>` | re-read the close-up views of a processed match, then reanalyze |
| `autoscout train-robots` | (re)train the robot detector on `data/datasets/robots` |

## Close-up views (hub / reef cams)

Many broadcasts show picture-in-picture close-ups of the scoring elements. On the **Calibrate** page, type the
element a close-up shows (for example `red_hub` or `blue_reef`), click **Draw close-up**, drag a rectangle around it,
then click **Save**. Leave out sponsor banners and score graphics where you can. Bumper numbers in close-ups are
big enough to read reliably. They are read before every score jump and every 3 s through the match, and they are
used in two ways:

- **Identity (any season).** If a team is read in the close-up of an element, that team's alliance slot nearest
  the element at that moment gets evidence for that team. The evidence is scaled by how clearly that slot is the
  nearest one.
- **2025 scoring.** A team seen in the reef cam just before a score jump counts as being at the reef.

## Robot detector: the one thing to build per season

Robot detection is the weakest link. Broadcast robots are small, and referees and piles of game pieces hide them.
The detector used is the first of these that exists:

1. `data/models/autoscout-robots.engine`, a TensorRT export made automatically on a CUDA machine
2. `data/models/autoscout-robots.pt`, our fine-tuned model
3. `data/models/spyglass/model11n.pt`, FRC Team 578's "Spy Glass" bumper model from Hugging Face, which is the fallback

To build a good detector, use the dashboard's **Label robots** page:
- Click **Suggest**: the current model pre-fills boxes.
- Fix the boxes. Label *every* visible robot.
- Click **Save frame**, or **Save & +5 s** to move on.

About 150 frames across a few matches of an event gives a solid model. Then click **Train detector**, or run
`autoscout train-robots`. On an RTX 3060 that takes about 15 minutes.

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

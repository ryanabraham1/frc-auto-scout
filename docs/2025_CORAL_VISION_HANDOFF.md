# 2025 coral vision: handoff (2026-11-01)

Goal (from the user): credit 2025 REEFSCAPE coral by **seeing** it, not by guessing from who was near the
REEF. Detect every CORAL, know which branch (A-L) and level each one is on, and credit the robot that put
it there. Official FMS numbers are a check for tuning, **never an input to attribution** (the user was
explicit: a team can gain a mechanism mid-event). Tracking is considered solid; attribution is the work.

## Where things stand

| Piece | State |
|---|---|
| Per-match camera alignment (`scripts/align_event.py`) | **Committed** (`0c68105`). qm1/qm2 framing was ~85 px off; fixed. |
| Face-based reef credit, reef image model (`vision/reef_model.py`) | **Committed** (`0c68105`), `FACE_MODE="mix"`. |
| OPR prior | Code in `reefscape2025.py` (`event_prior`, per level) but **off** (`PRIOR_ALPHA = 0`). Uncommitted change. |
| Capability check in `scripts/eval2025.py` | Uncommitted. Prints teams credited at a level their per-level reference OPR says they don't score. Hint only (OPR is noisy). |
| CORAL detector, round one | **Done**: `models/autoscout-coral.pt` (YOLO11s, 30 epochs, imgsz 1920; val P 0.76, R 0.71, mAP50 0.78). Buzzer check: precision 0.67, recall 0.36. The epoch-15 checkpoint scored better there (0.76 / 0.50): the check is one frame per REEF, so it's noisy. |
| CORAL detector, round two | **Done, best so far**: `models/autoscout-coral-r2.pt` (relabelled `data/datasets/coral_r2`, 58k boxes). Buzzer check: precision 0.74, **recall 0.62**, coral points 202 / 279 (round one 0.67 / 0.36, 138). Per REEF branches right: qm2 blue 15/25, qm2 red 7/11, qm10 blue 8/14, qm10 red 4/5. Weakest: the L1 trough (6 seen of 21). Its val numbers (0.86 / 0.82 / 0.92) are against model-made labels, so they flatter it. |
| 3D close-up cameras (`vision/field3d.py`) | Uncommitted. AprilTag layout + pinhole solve; close-up cameras are fixed all event. |
| Vision-only reef scoring test (`scripts/reef_state_check.py`) | Uncommitted. Scores REEF branches at the buzzer from vision and compares with FMS per-branch truth. |
| Round-two labels (`coral_label.relabel`) | Written, **not run yet**. |

Nothing below "Committed" is in git. Other sessions also have uncommitted 2026/dashboard edits in this
tree (`rebuilt2026.py`, `base.py`, `trajectories.py`, `web/*`, `README.md`, `tests/test_dashboard.py`,
`scripts/eval2026.py`): **don't commit or revert those**. `tests/test_attribution.py::test_official_fuel_without_robot_evidence_is_reported_unattributed`
fails because of that 2026 work, not the 2025 changes.

## Latest numbers

Pinned 35-match held-out OPR eval (`scripts/eval2025.py 2025casj --matches docs/2025_ACCURACY_MATCHES.json`),
vision-only attribution: coral r 0.928, MAE 1.72; L4 r 0.641, MAE 1.42. OPR of the evaluated matches'
own official totals vs the reference OPR is only r 0.907, so this check is **at its ceiling**: it can't show
attribution improvements any more. Use qm10's video labels (`benchmarks/attribution/`, 12/15 correct now)
and the REEF scoring test instead.

REEF scoring test at the buzzer, qm2 + qm10 (held out from detector training), epoch-15 checkpoint:

| | Branches right | Wrong | Missed | Coral points vision / official |
|---|---|---|---|---|
| qm2 blue | 15 | 4 | 10 of 25 | 81 / 120 |
| qm2 red | 3 | 2 | 8 of 11 | 28 / 65 |
| qm10 blue | 7 | 2 | 7 of 14 | 38 / 69 |
| qm10 red | 2 | 3 | 3 of 5 | 25 / 25 |

Branch precision ~0.70-0.76, recall ~0.47-0.50. **The limit is now detector recall**, especially on the red
close-up's right side and the L1 trough: the geometry lines up (projected tips sit on the pipes).

Public Roboflow coral models were tried (needs `ROBOFLOW_API_KEY`, set by the user): they find 0-8 coral
in a broadcast frame where ours finds 56. They were trained on robot-mounted cameras and don't transfer.
The user then said to **skip** public data; nothing was downloaded. `inference-sdk` and `roboflow` are
installed in `.venv`; `reef_state_check.py` accepts `roboflow:PROJECT/VERSION` as weights.

## How the REEF scoring test works (and its gotchas)

1. **Panels**: the bottom band (y 645-1015) holds the close-ups. `panels()` splits it at the broadcast's
   coloured separators: two panels during the match (x≈958; blue REEF left, red right), **three** for the
   last ~20 s (x≈570 and ≈1352: blue, barge camera, red), smaller and cropped.
2. **Cameras**: each close-up camera is fixed for the whole event (tags within ~2 px across matches).
   Red sees tags 1, 2, 9, 11 → solves itself (`solve_camera`, ~1-2 px). Blue only ever reads tags 12/13
   (its base tags are cut off by the panel edge), so it was solved once with two hand-marked base-tag
   centres and saved, along with red, in `configs/events/2025casj_reef_cams.json` (fallback when a match's
   own tags don't solve). Both are near mirror images (f ≈ 600-660 px, camera ≈ 1.9 m up, ~4 m from the
   REEF centre). **Tag corner order**: cv2.aruco's order is our (BR, BL, TL, TR); see `tag_corners`.
3. **End of match**: the three-panel layout is mapped back onto the two-panel framing with an ORB
   **similarity** (`match_view`; a full homography was unstable), warped to full size and pasted into a
   black canvas at the two-panel position (pasting at the small panel's position clipped 40% of red; a
   real-frame canvas made detection worse). Branches the end panel can't show take their state from the
   last two-panel frame (`covered`).
4. **Branch assignment**: every branch tip (all faces; the REEF is an open frame, far-face L4 is visible
   over the top), Hungarian assignment with one CORAL per branch, cost = pixel distance + log size ratio
   vs the CORAL length expected at that tip's depth. The camera sits about level with L4, so near- and
   far-face L4 tips overlap in the image: **size is what separates them**. Branch lettering: face 0 (A, B)
   faces that alliance's driver station, counter-clockwise from above (checked against FMS; the other
   three conventions score worse).
5. **Truth**: `breakdown[alliance]["teleopReef"]` (or `autoReef` with `--at auto`): `topRow`/`midRow`/
   `botRow` = L4/L3/L2 per `nodeA..L`, plus `trough` (L1 count).

## Background jobs (may still be running)

- Detector training (started 7:04 pm): `scripts/train_coral.py --epochs 30 --batch 2` → `models/autoscout-coral.pt`.
  About 5 min/epoch with the GPU shared. The user asked **not** to stop it, even when dashboard processing
  slows down.
- Queued: waits for `models/autoscout-coral.pt`, then runs the REEF check at conf 0.25 and 0.4 into the
  session scratchpad. If that session is gone, just rerun:

```
.venv/Scripts/python.exe scripts/reef_state_check.py models/autoscout-coral.pt 2025casj_qm2 2025casj_qm10 --at end --conf 0.25
```

## Next steps (agreed with the user)

1. **Round two labels** (user said yes; **already running**, these are the commands if it needs redoing):
   ```
   .venv/Scripts/python.exe -c "from autoscout.train.coral_label import relabel; print(relabel('models/autoscout-coral.pt'))"
   .venv/Scripts/python.exe scripts/train_coral.py --dataset data/datasets/coral_r2 --name run_r2 --out models/autoscout-coral-r2.pt --epochs 30 --batch 2
   ```
   Round one only labelled a fixed REEF region of each close-up (`aux_watch`), so the detector learned to
   skip coral elsewhere (red close-up's right side, the trough). Relabel keeps detector boxes ≥ 0.4 plus
   round-one boxes the detector supports at ≥ 0.15. qm2 and qm10 are held out of all labels.
   Compare r1 vs r2 with the REEF check.
2. **Carried → placed tracking** (the actual attribution): run the detector through a match on the wide
   shot and both close-ups, link CORAL over time, and when one appears on a branch and stays, credit the
   alliance robot whose box held it just before. The user also wants robot positions from the same 3D
   cameras ("track where robots are"), so the scorer is the robot at that branch.
3. Feed placements into `Reefscape2025._match_placements` instead of the white-pixel `element_changes`.
4. Consider committing: `field3d.py`, `coral_label.py`, `train_coral.py`, `coral_check.py`,
   `reef_state_check.py`, `2025casj_reef_cams.json`, plus the `reefscape2025.py`/`eval2025.py`/test edits.
   Ask the user first; commit only these paths, on a branch.

## Rules the user has set

- Vision first; FMS/OPR only to check and tune. No OPR-based priors in attribution.
- Rerender `demos/2025casj_qm2_annotated.mp4` (`scripts/render_demo.py`) after attribution changes and say
  how 1323 does in qm2 (OPR ~25 coral/match; 8793 is an L1-only robot and should get no L4s).
- Check held-out matches before keeping a change. Don't stop the user's GPU jobs.
- Never type API keys or passwords; the user sets env vars themselves.

See also `docs/2025_ACCURACY_PROGRESS.md` and the session memory notes (`camera-alignment`,
`scouting-accuracy-eval`).

# Handoff: 2025/2026 attribution fixes (branch `claude/dazzling-tesla-ygwtay`, PR #2)

Pick this up on a machine with an NVIDIA GPU. Everything below is committed on the branch; nothing
is half-applied. Open PR: https://github.com/ryanabraham1/frc-auto-scout/pull/2 (supersedes #1, which
should be closed without merging).

## 1. Setup

```bash
git fetch origin claude/dazzling-tesla-ygwtay && git checkout claude/dazzling-tesla-ygwtay
pip install -e '.[dev]'
python -m pytest -q          # expect 161 passed
```

- TBA key: put `TBA_API_KEY=...` in `.secrets/.env` (git-ignored). Ask the user for it; never commit it.
- PARSeq bumper OCR loads through `torch.hub.load("baudm/parseq:main", ...)`. If GitHub's tree page is
  blocked, clone https://github.com/baudm/parseq into `~/.cache/torch/hub/baudm_parseq_main`.
- Videos go in `data/videos/<youtube id>.mp4`. YouTube download works with yt-dlp; the user also
  shared the exact files on Google Drive:

| Match | Event / teams | YouTube id | Drive copy |
|---|---|---|---|
| `2025casj_qm1` | 2025 Silicon Valley Q1; red 6059/10059/668, blue 649/1351/6418 | `EaRs_WdhNpM` | https://drive.google.com/file/d/1oP9n2HeI4MFoeIWx4Hh6qEaUDfeVG75K/view |
| `2025new_f1m1` | 2025 Champs Newton Final 1; red 2910/1323/4272, blue 3847/868/422 | `iU3p9UplthM` | https://drive.google.com/file/d/1NehvqRykSw0RryYH2JVsQ3rWr-0GkZ0e/view |
| `2026cosp_qm1` | 2026 Pikes Peak Q1; red 3648/2996/9134, blue 8334/10114/662 | `qvK9pHMQIFI` | https://drive.google.com/file/d/1bf6O-kEqUxrhncrP3V7LYOmTYDlNOKc6/view |

Drive files are public-link; download with `https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t`
(plus the `uuid` from the virus-scan page for large files) and rename to the YouTube id.

## 2. What is done (all on the branch)

1. PR #1's ball tracking / attribution work merged onto `main`.
2. `bumper_ocr.py`: pinned hub ref (`:main`), `nltk` dependency.
3. 2025 attribution (`games/reefscape2025.py`):
   - placement credit scaled by how clear its geometry is (`_normalized`, `fit`);
   - close-up placement: elevator-up robot under the new coral wins (`CAM_LIFT_*`);
   - close-up placement: a robot shown clearly elsewhere in the close-up did not place it (`CAM_AWAY_*`);
   - a placement far from every tip of a level cannot pair with that level's score (`LEVEL_PX_MAX`).
4. Close-up robot tracks (`vision/closeup_tracks.py`): each close-up's boxes linked frame to frame;
   unread boxes take the team of the nearest read on the same track (or of the fused wide-shot robot).
   Used for close-up lift, placement credit and reef dwell. 2025 `aux_every` is now 0.25 s.
5. `StaticSuppressor` measures "a quarter of the match on one spot" against the whole match
   (`expected_frames`), not the frames seen so far. Before, 3648 (parked behind its HUB in 2026cosp
   qm1 auto) was deleted as a field element from ~13 s on.
6. `trace_distances`: flights with under `TRACE_MIN_DX` (25 px) sideways travel are not traced back
   (a ball only seen dropping into a HUB "traced" to whichever robot was parked under it).
7. `configs/events/2025new.yaml`: Newton calibration (landmarks hand-marked on f1m1 at 60 s;
   `scripts/draw_cal.py 2025new data/videos/iU3p9UplthM.mp4 60 out.jpg` to check it).

Items 4-6 change the video pass, so **all three matches must be re-run** (`run-match`, not
`reanalyze`). The re-runs were stopped about 30% in; nothing from them was saved.

## 3. Ground truth the user gave (check these after the re-runs)

Game clock is the broadcast's countdown; `vt` is video seconds in the saved results.

**2025casj_qm1, blue**
- Auto L4 (verified from footage): 1351 = 3, 6418 = 1, 649 = 0.
- 0:55 teleop L3 (+4 at vt 106.61) was scored by **1351**.
- 0:53 teleop L4 (+5 at vt 108.61) was scored by **6418**. Still wrong on saved data after the
  fixes (1351 0.69); the paired wide-shot placement and the one-coral-at-a-time joint step need a look.
- 0:15 teleop L3 (+4 at vt 148.15) was scored by **1351**. The close-up shows 1351's elevator up
  (aux box at vt 147.65, reach 1.02) but the lift needs two consecutive labelled samples; the denser
  0.25 s close-up sampling plus close-up tracks should supply them. Check after the re-run.

**2026cosp_qm1, red auto**
- 3648 never moves in auto; max preload is 8 FUEL, so it cannot have scored 45. 2996 shoots from the
  near-right corner by the red HUB (image ~x 1600-1750, y 420-650) and was mostly undetected
  (weak boxes, conf 0.12-0.23, one vetoed as a person). The user does **not** want a hard
  "didn't move -> cap at 8" rule; fix the detection/linking instead.
- After the re-run, check `teams['3648']['fuel_scored_auto']` and the per-launch shooter
  (`python <script>` pattern: monkeypatch `hub_shooter` and print the slot, launch and trace
  distances per auto launch) to see whether 2996 is now tracked and credited.

**2025new_f1m1**: never processed yet. Render it and spot-check auto.

## 4. Steps

```bash
for m in 2025casj_qm1 2025new_f1m1 2026cosp_qm1; do autoscout run-match $m; done
python scripts/render_demo.py 2025casj_qm1 data/videos/EaRs_WdhNpM.mp4 docs/demos/2025casj-qm1-annotated.mp4
python scripts/render_demo.py 2025new_f1m1 data/videos/iU3p9UplthM.mp4 docs/demos/2025new-f1m1-annotated.mp4
python scripts/render_demo.py 2026cosp_qm1 data/videos/qvK9pHMQIFI.mp4 docs/demos/2026cosp-qm1-annotated.mp4 --flight-labels
```

- Re-encode to stay under GitHub's 50 MB warning: `ffmpeg -i in.mp4 -c:v libx264 -crf 28 -pix_fmt yuv420p -movflags +faststart out.mp4`.
- `*.mp4` is git-ignored: add demos with `git add -f docs/demos/<file>.mp4`.
- Check the panel's "Recent" list at the timestamps in section 3 before posting.
- Update PR #2's description (results table, video links:
  `https://github.com/ryanabraham1/frc-auto-scout/blob/claude/dazzling-tesla-ygwtay/docs/demos/<file>.mp4?raw=true`)
  and post a short comment with the three video links.
- Run `python -m pytest -q` before every push.

## 5. Known open problems (not fixed)

- Weak detection of robots at the near rail behind people (2026: 2996). Weak boxes can extend a track
  but never start one.
- The 2025 joint step (`assign_actions`, `REFRACTORY` 3 s) cannot tell which of two robots side by side
  at one face placed which coral when score posts lag placements unevenly.
- Planned later, not now: replace the hand-tuned evidence weights with a model trained on scouting-app
  per-robot counts or Review-credit labels (the user asked to defer this).

## 6. Update after the re-runs (same day, on a Mac)

- Macs took the Apple Vision branch of `collect_aux_reads`, which produced no close-up boxes, so items 3-4 had
  no effect there. Asking for boxes now takes the detector path on every platform.
- Set `AUTOSCOUT_IMGSZ=1600` on Apple silicon to get the same two-scale detector as CUDA.
- **Correction to section 3:** the near-corner shooter in 2026cosp qm1 auto is 9134, not 2996 (its bumper reads at
  t=12 and t=15). 2996 goes behind the red HUB at about t=15 and stays hidden until about t=32.
- 3648's auto credit came from the residual split: flat blips beside it, and FUEL stirring in its hopper taken as
  a volley. Credit now needs a release that climbed, and a volley needs real evidence. There is no preload cap.
- Most unlinked FUEL is seen only around a HUB. `hidden_shooter` links it to a robot lost in that HUB's shadow,
  which doubles red auto's linked flights, but that does not yet reach auto credit (2996 still 1).
- Do not run six matches at once on a 24 GB Mac: it swaps. Two workers takes about 3 minutes a match.

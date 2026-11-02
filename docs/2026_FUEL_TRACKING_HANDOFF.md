# 2026 FUEL tracking — handoff

Last updated 2026-11-01 (session working on 2026cc). Nothing below is committed yet; see "State of the tree".

## Goal (the user's words, paraphrased)

Track every FUEL ball shot, and know which robot it came out of. The bar: **if the scoreboard were hidden,
counting tracked balls into each HUB should roughly reproduce the score**, and per-robot credit should come
from those tracked balls rather than from guessing. The user wants vision-first attribution. Don't lean on
statistical priors: the OPR prior exists but stays off (`PRIOR_ALPHA = 0`), the same choice the 2025 work made.
"Closeness" in the linking code means how close the point where a ball comes out is to a robot's box in the
image. It never means distance to the HUB, because robots can shoot from anywhere in their zone.

## How to measure

```
.venv/Scripts/python scripts/eval2026.py 2026cc            # reanalyzes saved observations (~2 min, 16 matches)
.venv/Scripts/python scripts/eval2026.py 2026cc --saved    # scores the stored results as they are
.venv/Scripts/python scripts/eval2026_variant.py 2026cc SPLIT_BY=made LINK_LOOSE=1.2   # try rebuilt2026 constants
```

It reports:
- **Score from tracked balls** (scoreboard hidden): every tracked flight into a HUB while it is active, against
  the live score. Reports the total ratio and agreement in 5 s windows. *This is the user's main bar.*
- **made seen**: makes credited to a robot / official FMS hub count (active periods).
- **shots seen**, **per-period |shots - official|**, **tracked flights with a shooter**,
  **scored points with one clear shooter**.
- Per-team average vs fuel OPR from half the unevaluated quals (held out). **This is very noisy:** there are
  only 16 processed matches, so each team has about 2 evaluated and about 4 reference matches. Don't tune on
  it alone.

Diagnostics:
- `scripts/fuel_link_diag.py 2026cc_qm62 out.json`: for every tracked make, how it links (direct / traced
  arc / none / wrong alliance).
- `scripts/fuel_link_view.py out.json img.jpg none rising 6`: renders sample makes with robot boxes, so you
  can see why one failed.
- `scripts/fuel_detect_slice.py 2026cc_qm62 data/videos/2ana-RAceoE.mp4 40 75 [bg=30]`: re-detects balls
  from video for a time slice and compares tracked makes with the score rise. Use it for detector experiments
  without reprocessing whole matches.

## Results so far (2026cc, 16 matches)

| step | made seen | shots seen | flights w/ shooter | team r vs OPR (equal split 0.43) |
|---|---|---|---|---|
| baseline (HEAD) | 40% | 70% | 71% | 0.48 |
| + trace arcs back for unseen releases | 42% | 74% | 76% | 0.47 |
| + also trace flights first seen at the hub top | 60% | 93% | 79% | 0.45 |
| + hub balls linked only within the hub's alliance, clearly-closest robot | **68%** | 100% | 81% | 0.44 |
| (variant) SPLIT_BY=made | 68% | | | 0.46 |
| + no "near the HUB" fallback, sighting fallback, stream linking | **72%** | 104% | 83% | 0.44 |

Score from tracked balls, saved results: total **102%** of the score; 5 s windows r 0.79, but the windows are
off by 47% on average. Single alliances miss by ±30% in some matches (e.g. qm4 red 399 tracked vs 277 scored,
qm10 blue 534 vs 733).

## What we learned

1. **Ball counting is roughly right in total. The weak link is ball → robot.** In qm62, 885 balls were tracked
   into hubs (official count in active periods: 748). Before these changes only 46% were credited.
2. **Why makes didn't link (qm62):**
   - 29% were first seen at the hub top and dropped by design (meant for rim bounces). Most are the tail of a
     real shot whose earlier flight was lost against the crowd. These are now traced back along the arc.
   - 13% had a visible release but no robot within 0.6 box widths. Usually only 1 of the alliance's 3 robots
     had a detected box at that moment.
   - 5% were credited to a robot on the *other* alliance. That's impossible for a hub make; fixed by
     `hub_shooter`.
3. **Robot detection coverage is a root cause.** In qm62, robots have a detected box in only 44–69% of track
   rows (51–77% of 0.5 s bins). When a shooter isn't boxed, its estimated box is stale.
4. **Detection in streams:** `LaunchDetector` diffs the yellow mask against a frame 2 back. In a tight stream
   each ball fills the spot the ball ahead just left, so most of the stream cancels out (visible at
   qm62 t≈19 s: about 8 balls in the air, 2–3 detected). A background-count mode (`bg_frames`) finds about 3×
   the blobs. But on slices, tracked makes vs score didn't clearly improve: auto red 133%→114%,
   blue 97%→115%. Teleop slices are confounded by inactive-hub balls. Untested end to end; default off.
5. Some "no shooter" releases are **human players throwing from behind the alliance-wall glass** (qm62
   t=85.2, t=139.2). `outpost_alliance` didn't catch any of the hub makes, so check its geometry.
6. **Public FUEL models** (Roboflow Universe, project516/rebuilt-fuel, the PhotonVision model from team 2826)
   are trained on robot-camera close-ups. Broadcast balls are about 8–12 px, so they probably won't transfer.
   project516's weights are `.pt` pickles from an unvetted account; don't load them unless the user fetches
   them. A better path is our own ball detector trained on broadcast frames, auto-labelled from colour masks.

## Code changes (uncommitted)

- `autoscout/games/base.py`
  - `trace_back` only considers the rising part of the arc. A ball flying *over* a robot on the way down
    never came out of it. It optionally returns the release time.
  - New `trace_distances`: per-robot closest approach of the back-run arc.
- `autoscout/games/rebuilt2026.py`
  - `TRACE_ARCS`: flights without a seen release, including ones first seen at the hub top that drop in, are
    traced back.
  - `HUB_LINK` + `hub_shooter`: a hub make links only among that hub's alliance. The pick is the robot the
    release or traced arc comes out of: within 0.6 box widths, or clearly the closest (< `LINK_LOOSE` 1.5 and
    `LINK_MARGIN` 0.5 ahead of the next).
  - `SPLIT_BY` ("live" default | "made"): how official per-period counts are split among robots.
  - `POSITION_FALLBACK = False`: a score increase with no release seen before it no longer goes to robots
    weighted by distance to the HUB. The user objected: robots shoot from anywhere in their zone.
  - `sighted_shooter` (`SIGHTING_S` 5 s): FUEL rising where no robot box is (shooter hidden behind a referee,
    or half out of frame at the edge) goes to the alliance robot actually *seen* at that spot within 5 s,
    if it is clearly nearer than its partners' sightings. Example: qm62 t=71–77, a blue robot hidden behind
    the referee in the near-right corner.
  - `link_hub_flights` (`STREAM_S/PX/MAJORITY`): releases from one spot in quick succession are one robot's
    stream; the whole stream goes to the robot most of its linked balls point to (≥60%).
  - qm62 hub makes now: 575 linked rising, 174 linked hub-top, 15 descent; unlinked: 83 hub-top, 36 rising.
  - `event_prior` + `PRIOR_ALPHA/PRIOR_K/PRIOR_MIN_MATCHES`: fuel-OPR prior. **Off** and only computed when on.
- `autoscout/vision/trajectories.py`: `LaunchDetector.bg_frames` background-count mode, default 0 (off).
- `autoscout/cli.py` (already pushed to main as b93c60f): the dashboard's "debug video" now renders the
  labelled demo video (`demos/<match>_annotated.mp4`).
- New scripts: `eval2026.py`, `eval2026_variant.py`, `fuel_link_diag.py`, `fuel_link_view.py`,
  `fuel_detect_slice.py`.
- All 116 tests pass (`.venv/Scripts/python -m pytest -q tests`).

## State of the tree (careful)

The working tree is shared with **separate, uncommitted 2025 work** in `reefscape2025.py`, `web/app.py`,
`web/static/index.html`, `eval2025.py`, tests, `docs/2025_*`, `field3d.py`, `coral_*`, and others. Don't
commit, revert or edit those as part of 2026 work. Stage only the 2026 files listed above.

Saved results (`data/results/2026cc_*.json`) and the qm62 demo video still reflect HEAD. After keeping
changes, run `autoscout reanalyze 2026cc` and re-render with
`python scripts/render_demo.py 2026cc_qm62 data/videos/2ana-RAceoE.mp4 demos/2026cc_qm62_annotated.mp4`.

## Next steps (in order)

1. ~~Remove the near-HUB fallback~~ and ~~link undetected shooters / streams~~: done (see above). Track
   gaps were already interpolated (median gap 1 s), so the box estimates weren't the issue; undetected
   robots were.
2. **Hub-top flights that trace to nobody** (83 in qm62): the backward arc from a short fragment near the
   opening is too uncertain. Try stitching each one to an earlier rising flight of the same alliance whose
   arc heads to that HUB (forward prediction), instead of extrapolating backward.
3. **Red shots from the left frame edge** (qm62: x < 60 px, 14 unlinked): the red robot is half out of frame
   and isn't sighted within 5 s. Consider the slot whose field track is beyond the left edge at that time.
4. **Human-player detection** (`outpost_alliance`): tried and reverted an image-column test above each
   outpost. At 2026cc the blue outpost's image spot (≈1527, 421) is exactly where blue1 parked against the
   wall behind the referee in qm62 (t=71–77, 84–87, 133–141). What looked like human-player throws over the
   glass were that robot's shots, and the stream/sighting linking now gives them to blue1 consistently. A
   working HP test needs the release height or "outside the wall" in 3D, not an image column.
5. **Identity check:** per-team r hasn't moved while linking went from 40% to 72%. Either the OPR check is at
   its noise limit, or slot→team identity is wrong in places. Spot-check a few matches' identity against
   the bumpers in the demo videos.
4. Decide whether credit should come straight from linked makes (`SPLIT_BY=made`, which matches the user's
   goal better) once linking is above ~80%.
5. **Ball detection:** test `bg_frames` end to end (`autoscout recollect` per match, then `relaunch`), or
   train a broadcast ball detector, so balls in streams are seen individually.
6. Process more 2026cc matches (only 16 of 70 quals) so the per-team OPR check means something.
7. Check every change with `eval2026.py`, look at frames with `fuel_link_view.py`, and re-render qm62 for
   the user.

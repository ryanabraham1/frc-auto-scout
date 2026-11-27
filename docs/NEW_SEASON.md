# Kickoff checklist for a new season (e.g. 2027)

Day 1, after the game reveal:
1. Copy `configs/fields/2026.yaml` to `configs/fields/2027.yaml` and fill in:
   - field `length` and `width`, `auto_len` and `teleop_len` (seconds on the match clock)
   - `landmarks`: floor points that are easy to click in a broadcast (corners, center line ends, element bases)
   - `zones` (rectangles in meters) and `elements` (scoring locations with a `center` and a `size` or `radius`)
   - `game_pieces` HSV color ranges (check them on a frame with `scripts/grid.py`)
2. Copy `autoscout/games/template_new_season.py` to `autoscout/games/<game>2027.py`. Set `year = 2027` and add the
   import to `games/__init__.py`.
3. Implement `analyze(ctx)` with the shared recipe:
   - **Official totals:** read them from `ctx.match.breakdown`. Look at one TBA match JSON to learn the field names.
   - **Evidence:**
     - airborne pieces: `ctx.launches[piece]`, attributed with `self.attribute_launch`
     - robot positions near elements: `ctx.tracks` and `self.field.dist_to_element`
     - jumps in the live score: `ctx.timeline.deltas(alliance)`, where the point value identifies the action
   - **Split:** `self.proportional_split(official_total, evidence_weights, fallback_weights)`
   - **Per-station results** (climbs and the like) are exact. Also use them in `identity_evidence` to tell robots apart.
4. Add `metric_defs` for the dashboard columns.

Week 0 / week 1:
5. Calibrate one event (`autoscout calibrate <match>`, then the dashboard's Calibrate page).
6. Label about 150 frames of robots (dashboard → Label robots), then run `autoscout train-robots`. Robot appearance
   carries over between seasons, so start from the previous season's `autoscout-robots.pt` (`--base`).
7. Run a few matches with `--debug-video` and check that launches and robot boxes look right.
8. Check `autoscout report <event>`: the overlay score must match the official scores, and identity confidence
   should be high.

## If the game is pick and place

What 2025 taught, measured in `docs/2025_BENCHMARK.md`. Build the plugin around these from the start;
the 2025 plugin grew them late, on top of older logic that did worse.

- **Credit the piece, not the score.** Scores entered by people post up to 20 s late and in bursts.
  Find when each piece came to rest on its scoring location in the video, and credit the robot standing
  at that location's scoring spot then. Use the score's jumps only to keep the counts right.
- **Check on kickoff day what FMS publishes per location.** A per-location map (2025's `teleopReef`
  node rows) is what makes this sharp: 91% right for the mapped levels against 75% for the trough,
  which FMS only counts.
- **The pieces that carry over:** `vision/camera3d.py` (cameras from AprilTags; add
  `configs/fields/<year>_tags.json`), `vision/reef3d.rests` (pieces that come to rest, from detector
  boxes), `Reefscape2025._spot_evidence` and `_credit_spot` (robot at the spot; hidden partner),
  `scripts/refine_reef_cams.py` (refit a fixed camera to scored pieces after about 15 matches) and
  `CREDIT_SHARPEN`. They are written for the REEF; lift them into shared code when the second game
  needs them.
- **The pieces to write:** the 3D position of every scoring location and of a robot scoring there
  (`reef3d.coral_ends`, `node_pose`), a detector for the new piece (needs real match footage), and
  the point values.
- **Build the benchmark in week 1.** `scripts/bench2025.py` is the pattern: read the scoring
  locations with the score hidden and compare with FMS; use scout sheets only in cells where the
  scouts' sum equals the FMS count. Without it, constants get tuned to noise.

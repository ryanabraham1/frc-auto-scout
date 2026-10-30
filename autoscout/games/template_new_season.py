"""TEMPLATE for a new season (e.g. 2027). Copy to `games/<name><year>.py` and import it in
`games/__init__.py`. Kickoff-day checklist (see docs/NEW_SEASON.md):

1. configs/fields/<year>.yaml: field size, auto/teleop length, named landmarks (for calibration),
   zones, scoring elements, game-piece colors.
2. Fill in `launch_pieces` if robots shoot/throw pieces (airborne trajectories are tracked).
3. Implement `analyze`: map shared observations to per-team stats. The usual recipe:
     * read exact alliance totals per category from `ctx.match.breakdown` (TBA score_breakdown),
     * build per-robot evidence weights from launches (`ctx.launches`), robot positions near
       scoring elements (`ctx.tracks`, `self.field.dist_to_element`), and overlay score deltas
       (`ctx.timeline.deltas(alliance)`) whose point value identifies the action,
     * split each official total with `self.proportional_split(total, weights, fallback)`.
4. Add `metric_defs` so the dashboard knows which columns to show.
5. Label ~100 frames from week-1 matches (dashboard -> Label) and run `autoscout train-robots`.
"""
from __future__ import annotations

from .base import GamePlugin, MatchContext


class NewSeason(GamePlugin):
    year = 0  # set to e.g. 2027 to register
    launch_pieces: dict = {}
    metric_defs = [{"key": "points_contributed", "label": "Points contributed", "group": "Summary"}]

    def analyze(self, ctx: MatchContext) -> dict:
        teams = {}
        for a in ("red", "blue"):
            for i, team in enumerate(ctx.match.teams(a)):
                teams[team] = {"alliance": a, "station": i + 1, "points_contributed": 0}
        return {"teams": teams, "alliances": {}, "events": []}

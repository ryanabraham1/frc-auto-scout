"""Season plugins. Each season registers a GamePlugin subclass keyed by year."""
from __future__ import annotations

from .base import GamePlugin, MatchContext, REGISTRY  # noqa: F401
from . import rebuilt2026, reefscape2025  # noqa: F401  (registration side effects)


def get(year: int) -> "GamePlugin":
    if year not in REGISTRY:
        raise KeyError(f"no game plugin for {year}; see autoscout/games/template_new_season.py")
    return REGISTRY[year]()

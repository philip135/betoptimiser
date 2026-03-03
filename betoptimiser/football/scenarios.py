"""
Football Scenarios – enumerate all plausible scorelines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FootballState:
    """A fully specified football match end‑state."""

    home_goals: int
    away_goals: int

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def result(self) -> str:
        """'HOME', 'AWAY', or 'DRAW'."""
        if self.home_goals > self.away_goals:
            return "HOME"
        elif self.home_goals < self.away_goals:
            return "AWAY"
        return "DRAW"

    @property
    def btts(self) -> bool:
        return self.home_goals > 0 and self.away_goals > 0

    @property
    def score_str(self) -> str:
        return f"{self.home_goals}-{self.away_goals}"

    @property
    def home_win_to_nil(self) -> bool:
        return self.home_goals > 0 and self.away_goals == 0

    @property
    def away_win_to_nil(self) -> bool:
        return self.away_goals > 0 and self.home_goals == 0

    @property
    def double_chance_home_draw(self) -> bool:
        return self.result in ("HOME", "DRAW")

    @property
    def double_chance_away_draw(self) -> bool:
        return self.result in ("AWAY", "DRAW")

    @property
    def double_chance_home_away(self) -> bool:
        return self.result in ("HOME", "AWAY")

    @property
    def draw_no_bet_home(self) -> bool:
        return self.result == "HOME"

    @property
    def draw_no_bet_away(self) -> bool:
        return self.result == "AWAY"

    @property
    def half_result(self) -> str:
        """For simplicity model HT result = FT result (conservative)."""
        return self.result

    def label(self) -> str:
        return f"{self.score_str} ({self.result})"


def build_scenarios(max_goals: int = 6) -> list[FootballState]:
    """
    Enumerate all plausible score‑lines up to max_goals per side.
    Returns list of FootballState objects.
    """
    states = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            states.append(FootballState(home_goals=h, away_goals=a))
    return states

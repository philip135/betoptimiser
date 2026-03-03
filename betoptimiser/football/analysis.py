"""
Football Analysis – cross-market implied probability analysis.
"""

from __future__ import annotations

import numpy as np

from betoptimiser.models import BetOption
from betoptimiser.football.scenarios import FootballState
from betoptimiser.football.payouts import _match_runner_to_state


def cross_market_implied_matrix(
    bets: list[BetOption],
    states: list[FootballState],
    home_team: str = "",
    away_team: str = "",
) -> np.ndarray:
    """
    Build a matrix of implied‑probability contributions per state.

    Useful for visualising where the book is over/under‑round across
    the joint outcome space.
    """
    n_states = len(states)
    n_bets = len(bets)
    impl = np.zeros((n_states, n_bets))

    for j, bet in enumerate(bets):
        if bet.side != "BACK":
            continue
        prob = 1.0 / bet.price
        for i, state in enumerate(states):
            wins = _match_runner_to_state(
                bet.runner_name, bet.market_name, state,
                home_team=home_team, away_team=away_team,
            )
            if wins:
                impl[i, j] = prob

    return impl


def state_overround_vector(
    bets: list[BetOption],
    states: list[FootballState],
    home_team: str = "",
    away_team: str = "",
) -> np.ndarray:
    """
    For each state, sum the implied probabilities of all back bets that
    would pay out.  Values > 1.0 indicate the book is overly generous
    in that region (potential arb signal).
    """
    impl = cross_market_implied_matrix(bets, states, home_team, away_team)
    return impl.sum(axis=1)


def find_mispriced_states(
    bets: list[BetOption],
    states: list[FootballState],
    home_team: str = "",
    away_team: str = "",
    threshold: float = 1.0,
) -> list[tuple[FootballState, float]]:
    """Find states where cross‑market implied probability sum exceeds threshold."""
    overrounds = state_overround_vector(bets, states, home_team, away_team)
    results = []
    for i, orr in enumerate(overrounds):
        if orr > threshold:
            results.append((states[i], float(orr)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

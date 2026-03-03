"""
Utility functions – dutch book, overround, implied probabilities.
"""

from __future__ import annotations

import numpy as np

from betoptimiser.models import BetOption, ArbResult
from betoptimiser.solver import solve_arb
from betoptimiser.prices import extract_bet_options


def dutch_book_single_market(
    session,
    market_catalogue,
    budget: float = 100.0,
    commission: float = 0.02,
) -> ArbResult:
    """
    Check if a single market has a simple dutch book
    (back all runners at prices that sum to < 100% implied probability).

    Solved via the same convex framework for consistency.
    """
    bets = extract_bet_options(session, [market_catalogue], price_depth=1)
    back_bets = [b for b in bets if b.side == "BACK"]

    if not back_bets:
        return ArbResult(
            status="no_prices", guaranteed_profit=0, total_stake=0,
            stakes=[], bets=[], profit_by_scenario=np.array([]),
            scenario_labels=[],
        )

    n_runners = len(back_bets)
    scenarios = [b.runner_name for b in back_bets]
    A = np.zeros((n_runners, n_runners))
    for j in range(n_runners):
        for i in range(n_runners):
            if i == j:
                A[i, j] = back_bets[j].payout_if_wins
            else:
                A[i, j] = back_bets[j].payout_if_loses

    return solve_arb(
        bets=back_bets,
        payout_matrix=A,
        budget=budget,
        commission=commission,
        scenario_labels=scenarios,
    )


def implied_probability_vector(bets: list[BetOption]) -> np.ndarray:
    """Return implied probabilities for a list of bets."""
    return np.array([1.0 / b.price for b in bets])


def overround_from_bets(bets: list[BetOption]) -> float:
    """Overround (%) from a set of back bets on the same market."""
    backs = [b for b in bets if b.side == "BACK"]
    if not backs:
        return 0.0
    return round(sum(1.0 / b.price for b in backs) * 100, 2)


def print_arb_result(result: ArbResult) -> None:
    """Pretty‑print an ArbResult."""
    print(result.summary())


def print_mispriced_states(
    mispriced: list[tuple],
    limit: int = 20,
) -> None:
    """Print states where implied probabilities suggest mispricing."""
    print(f"{'Score':<10} {'Result':<8} {'BTTS':<6} {'Impl Σ':>8}")
    print("-" * 40)
    for state, orr in mispriced[:limit]:
        print(
            f"{state.score_str:<10} {state.result:<8} "
            f"{'Yes' if state.btts else 'No':<6} {orr:>8.4f}"
        )

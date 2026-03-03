"""
Racing Payout Matrix – vectorised with numpy.
"""

from __future__ import annotations

import numpy as np

from betoptimiser.models import BetOption
from betfairtools.constants import (
    WIN, PLACE, OTHER_PLACE, EACH_WAY,
    FORECAST, REVERSE_FORECAST, REV_FORECAST,
)


def build_payout_matrix(
    bets: list[BetOption],
    scenarios: list[tuple[str, ...]],
) -> np.ndarray:
    """
    Build the (n_scenarios × n_bets) payout matrix for a horse race.

    Fully vectorised — precomputes position lookups once, then fills
    each column (bet) with a single numpy operation.

    Market types and their hit conditions:
    - WIN:              runner == 1st
    - PLACE(K):         runner ∈ {1st, ..., Kth}
    - OTHER_PLACE(K):   runner ∈ {1st, ..., Kth}
    - EACH_WAY:         3-tier: win (1st), place-only (2nd..Kth), lose
    - FORECAST(A→B):    A == 1st AND B == 2nd
    - REVERSE_FORECAST: {A, B} == {1st, 2nd}  (either order)
    - REV_FORECAST:     alias for REVERSE_FORECAST
    """
    n_scenarios = len(scenarios)
    n_bets = len(bets)
    A = np.zeros((n_scenarios, n_bets))

    if n_scenarios == 0 or n_bets == 0:
        return A

    depth = len(scenarios[0])

    # ── Precompute runner-index mapping ─────────────────────────────────
    all_runner_names = sorted({r for s in scenarios for r in s})
    r2i = {r: i for i, r in enumerate(all_runner_names)}
    n_runners = len(all_runner_names)

    # position_idx[pos, scenario] = runner_index at that position
    position_idx = np.empty((depth, n_scenarios), dtype=np.int32)
    for pos in range(depth):
        position_idx[pos] = [r2i[s[pos]] for s in scenarios]

    first_place = position_idx[0]
    second_place = position_idx[1] if depth >= 2 else None

    # runner_is_first[ridx] → bool (n_scenarios,)
    runner_is_first = np.zeros((n_runners, n_scenarios), dtype=bool)
    for ridx in range(n_runners):
        runner_is_first[ridx] = first_place == ridx

    # runner_in_top_k[k] → bool (n_runners, n_scenarios)
    needed_ks: set[int] = set()
    for b in bets:
        if b.market_type in (PLACE, OTHER_PLACE):
            needed_ks.add(min(b.place_count, depth))
        elif b.market_type == EACH_WAY:
            needed_ks.add(min(b.ew_places, depth))

    runner_in_top_k: dict[int, np.ndarray] = {}
    scenario_arange = np.arange(n_scenarios)
    for k in needed_ks:
        arr = np.zeros((n_runners, n_scenarios), dtype=bool)
        for pos in range(k):
            arr[position_idx[pos], scenario_arange] = True
        runner_in_top_k[k] = arr

    # ── Fill matrix column-by-column (vectorised per bet) ───────────────
    for j, bet in enumerate(bets):
        mt = bet.market_type
        ridx = r2i.get(bet.runner_name, -1)
        if ridx < 0:
            A[:, j] = bet.payout_if_loses
            continue

        if mt == WIN:
            hits = runner_is_first[ridx]
            A[:, j] = np.where(hits, bet.payout_if_wins, bet.payout_if_loses)

        elif mt in (PLACE, OTHER_PLACE):
            k = min(bet.place_count, depth)
            hits = runner_in_top_k[k][ridx]
            A[:, j] = np.where(hits, bet.payout_if_wins, bet.payout_if_loses)

        elif mt == EACH_WAY:
            wp = bet.price
            frac = bet.ew_fraction
            ew_k = min(bet.ew_places, depth)
            is_1st = runner_is_first[ridx]
            is_topk = runner_in_top_k[ew_k][ridx]
            is_placed_not_1st = is_topk & ~is_1st
            is_out = ~is_topk

            if bet.side == "BACK":
                A[:, j] = (
                    is_1st * ((wp - 1.0) + (wp - 1.0) * frac)
                    + is_placed_not_1st * (-1.0 + (wp - 1.0) * frac)
                    + is_out * (-2.0)
                )
            else:  # LAY
                A[:, j] = (
                    is_1st * (-(wp - 1.0) - (wp - 1.0) * frac)
                    + is_placed_not_1st * (1.0 - (wp - 1.0) * frac)
                    + is_out * 2.0
                )

        elif mt == FORECAST:
            if depth >= 2 and second_place is not None:
                ridx2 = r2i.get(bet.runner_name_2, -1)
                if ridx2 >= 0:
                    hits = (first_place == ridx) & (second_place == ridx2)
                    A[:, j] = np.where(hits, bet.payout_if_wins, bet.payout_if_loses)
                else:
                    A[:, j] = bet.payout_if_loses
            else:
                A[:, j] = bet.payout_if_loses

        elif mt in (REVERSE_FORECAST, REV_FORECAST):
            if depth >= 2 and second_place is not None:
                ridx2 = r2i.get(bet.runner_name_2, -1)
                if ridx2 >= 0:
                    hits = (
                        ((first_place == ridx) & (second_place == ridx2))
                        | ((first_place == ridx2) & (second_place == ridx))
                    )
                    A[:, j] = np.where(hits, bet.payout_if_wins, bet.payout_if_loses)
                else:
                    A[:, j] = bet.payout_if_loses
            else:
                A[:, j] = bet.payout_if_loses

        else:
            # Unknown market type — treat as WIN fallback
            hits = runner_is_first[ridx]
            A[:, j] = np.where(hits, bet.payout_if_wins, bet.payout_if_loses)

    return A

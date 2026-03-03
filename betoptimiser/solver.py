"""
Core LP Solver – find optimal stakes maximising worst‑case profit.

Commission approach:
  LP uses a *conservative* approximation — commission is charged on every
  positive payout entry individually.  This slightly overcharges (Betfair
  actually charges per-market net), keeping the LP small (no auxiliary
  variables, ~N_bets variables only).

  After solving, exact per-market net commission is computed for the
  profit vector.  Because the LP overcharges, the actual guaranteed
  profit is always >= the LP objective → the solution is safe.
"""

from __future__ import annotations

import logging
from typing import Optional

import cvxpy as cp
import numpy as np

from betoptimiser.models import BetOption, ArbResult
from betfairtools.constants import EACH_WAY

logger = logging.getLogger(__name__)


def solve_arb(
    bets: list[BetOption],
    payout_matrix: np.ndarray,
    budget: float = 100.0,
    max_bet_fraction: float = 1.0,
    commission: float = 0.02,
    scenario_labels: Optional[list[str]] = None,
    force_deploy: bool = False,
) -> ArbResult:
    """
    Solve for the optimal stakes that maximise worst‑case profit.

    Parameters
    ----------
    bets : list[BetOption]
        Available bet options.
    payout_matrix : np.ndarray
        Shape (n_scenarios, n_bets).  Raw payouts per £1 stake.
    budget : float
        Total capital available.
    max_bet_fraction : float
        Max fraction of budget on any single bet.
    commission : float
        Betfair commission rate (e.g. 0.02 for 2%).
    scenario_labels : list[str] | None
        Human‑readable scenario descriptions.
    force_deploy : bool
        If True, force full capital deployment (for ranking non‑arb races).

    Returns
    -------
    ArbResult
    """
    n_scenarios, n_bets = payout_matrix.shape
    assert n_bets == len(bets)

    if scenario_labels is None:
        scenario_labels = [f"scenario_{i}" for i in range(n_scenarios)]

    A = payout_matrix

    # Conservative commission: penalise every positive payout entry
    A_post = np.where(A > 0, A * (1.0 - commission), A)

    # Decision variables
    stakes = cp.Variable(n_bets, nonneg=True)
    z = cp.Variable()  # worst‑case profit (maximise)

    # Capital‑at‑risk weights
    def _capital_weight(bet: BetOption) -> float:
        if bet.market_type == EACH_WAY:
            if bet.side == "LAY":
                return (bet.price - 1.0) + (bet.price - 1.0) * bet.ew_fraction
            else:
                return 2.0
        elif bet.side == "LAY":
            return bet.price - 1.0
        else:
            return 1.0

    capital_weights = np.array([_capital_weight(b) for b in bets])

    # Constraints
    constraints = [A_post @ stakes >= z]

    if force_deploy:
        constraints.append(cp.sum(stakes) >= budget)
        constraints.append(capital_weights @ stakes <= budget * 5)
    else:
        constraints.append(capital_weights @ stakes <= budget)

    # Per‑bet size capped by available liquidity and fraction of budget
    for i, bet in enumerate(bets):
        max_stake_by_capital = (max_bet_fraction * budget) / capital_weights[i]
        constraints.append(stakes[i] <= min(bet.available_size, max_stake_by_capital))

    # Solve
    prob = cp.Problem(cp.Maximize(z), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status in ("optimal", "optimal_inaccurate"):
        opt_stakes = stakes.value

        # Post‑hoc exact per‑market commission
        market_bet_indices: dict[str, list[int]] = {}
        for j, bet in enumerate(bets):
            market_bet_indices.setdefault(bet.market_id, []).append(j)

        raw_pv = A @ opt_stakes
        comm_cost = np.zeros(n_scenarios)
        for mkt_indices in market_bet_indices.values():
            idx = np.array(mkt_indices)
            mkt_pnl = A[:, idx] @ opt_stakes[idx]
            comm_cost += commission * np.maximum(0.0, mkt_pnl)
        profit_vec = raw_pv - comm_cost

        return ArbResult(
            status=prob.status,
            guaranteed_profit=float(profit_vec.min()),
            total_stake=float(np.sum(opt_stakes)),
            stakes=[float(s) for s in opt_stakes],
            bets=bets,
            profit_by_scenario=profit_vec,
            scenario_labels=scenario_labels,
        )
    else:
        return ArbResult(
            status=prob.status,
            guaranteed_profit=0.0,
            total_stake=0.0,
            stakes=[0.0] * n_bets,
            bets=bets,
            profit_by_scenario=np.zeros(n_scenarios),
            scenario_labels=scenario_labels,
        )

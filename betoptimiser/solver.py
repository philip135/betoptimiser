"""
Core LP Solver – find optimal stakes maximising worst‑case profit.

Uses scipy.optimize.linprog (HiGHS) for near‑zero overhead on small LPs.

Commission approach:
  LP uses a *conservative* approximation — commission is charged on every
  positive payout entry individually.  This slightly overcharges (Betfair
  actually charges per-market net), keeping the LP small (no auxiliary
  variables, ~N_bets + 1 variables only).

  After solving, exact per-market net commission is computed for the
  profit vector.  Because the LP overcharges, the actual guaranteed
  profit is always >= the LP objective → the solution is safe.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linprog

from betoptimiser.models import BetOption, ArbResult
from betfairtools.constants import EACH_WAY

logger = logging.getLogger(__name__)


def _capital_weight(bet: BetOption) -> float:
    """Liability per £1 stake."""
    if bet.market_type == EACH_WAY:
        if bet.side == "LAY":
            return (bet.price - 1.0) + (bet.price - 1.0) * bet.ew_fraction
        return 2.0
    if bet.side == "LAY":
        return bet.price - 1.0
    return 1.0


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

    Variables: x = [stake_1, ..., stake_n, z]   (n+1 total)
    Objective: maximise z  ⟹  minimise -z
    Constraints (all as  A_ub @ x <= b_ub):
      1. For each scenario s:  -A_post[s,:] @ stakes + z <= 0
      2. capital_weights @ stakes <= budget
      3. Per‑bet upper bounds via variable bounds.

    Parameters
    ----------
    bets : list[BetOption]
    payout_matrix : np.ndarray  – shape (n_scenarios, n_bets), raw payouts/£1
    budget, max_bet_fraction, commission, scenario_labels, force_deploy
    """
    n_scenarios, n_bets = payout_matrix.shape
    assert n_bets == len(bets)

    if scenario_labels is None:
        scenario_labels = [f"scenario_{i}" for i in range(n_scenarios)]

    # Conservative commission on positive payouts
    A_post = np.where(payout_matrix > 0, payout_matrix * (1.0 - commission), payout_matrix)

    capital_weights = np.array([_capital_weight(b) for b in bets])

    # ── Build LP in scipy form ────────────────────────────────────────────
    # x = [stakes (n_bets), z (1)]
    n_vars = n_bets + 1

    # Objective: minimise  [0, 0, ..., 0, -1] @ x   (i.e. maximise z)
    c = np.zeros(n_vars)
    c[-1] = -1.0

    # Inequality constraints  A_ub @ x <= b_ub
    # (1) Per-scenario:  -A_post[s,:] @ stakes + z <= 0
    #     → [-A_post | 1] @ x <= 0
    A_scen = np.hstack([-A_post, np.ones((n_scenarios, 1))])
    b_scen = np.zeros(n_scenarios)

    if force_deploy:
        # capital_weights @ stakes <= 5 * budget
        row_cap = np.zeros(n_vars)
        row_cap[:n_bets] = capital_weights
        # -sum(stakes) <= -budget  (i.e. sum >= budget)
        row_deploy = np.zeros(n_vars)
        row_deploy[:n_bets] = -1.0
        A_ub = np.vstack([A_scen, row_cap.reshape(1, -1), row_deploy.reshape(1, -1)])
        b_ub = np.concatenate([b_scen, [budget * 5], [-budget]])
    else:
        # capital_weights @ stakes <= budget
        row_cap = np.zeros(n_vars)
        row_cap[:n_bets] = capital_weights
        A_ub = np.vstack([A_scen, row_cap.reshape(1, -1)])
        b_ub = np.concatenate([b_scen, [budget]])

    # Variable bounds
    bounds: list[tuple[float, float | None]] = []
    for i, bet in enumerate(bets):
        max_stake = min(
            bet.available_size,
            (max_bet_fraction * budget) / max(capital_weights[i], 1e-12),
        )
        bounds.append((0.0, max_stake))
    bounds.append((None, None))  # z is unbounded

    # Solve with HiGHS (scipy default, very fast for small LPs)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        opt_stakes = res.x[:n_bets]
        opt_stakes = np.maximum(opt_stakes, 0.0)  # clip numerical noise

        # Post-hoc exact per-market commission
        market_bet_indices: dict[str, list[int]] = {}
        for j, bet in enumerate(bets):
            market_bet_indices.setdefault(bet.market_id, []).append(j)

        raw_pv = payout_matrix @ opt_stakes
        comm_cost = np.zeros(n_scenarios)
        for mkt_indices in market_bet_indices.values():
            idx = np.array(mkt_indices)
            mkt_pnl = payout_matrix[:, idx] @ opt_stakes[idx]
            comm_cost += commission * np.maximum(0.0, mkt_pnl)
        profit_vec = raw_pv - comm_cost

        return ArbResult(
            status="optimal",
            guaranteed_profit=float(profit_vec.min()),
            total_stake=float(np.sum(opt_stakes)),
            stakes=[float(s) for s in opt_stakes],
            bets=bets,
            profit_by_scenario=profit_vec,
            scenario_labels=scenario_labels,
        )
    else:
        return ArbResult(
            status=res.message,
            guaranteed_profit=0.0,
            total_stake=0.0,
            stakes=[0.0] * n_bets,
            bets=bets,
            profit_by_scenario=np.zeros(n_scenarios),
            scenario_labels=scenario_labels,
        )

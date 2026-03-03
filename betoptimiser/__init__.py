"""
betoptimiser – Betting markets optimisation library
====================================================
Find guaranteed‑profit portfolios across correlated betting markets
using convex optimisation (LP via CVXPY).

Quick start
-----------
>>> from betoptimiser.racing import scan_all_arbs
>>> results = scan_all_arbs(session, commission=0.02, return_all=True)

Modules
-------
- ``betoptimiser.models``     – BetOption, ArbResult data structures
- ``betoptimiser.solver``     – core LP solver
- ``betoptimiser.racing``     – horse racing scenarios, payouts, scanners
- ``betoptimiser.football``   – football scenarios, payouts, scanners
"""

from betoptimiser.models import BetOption, ArbResult
from betoptimiser.solver import solve_arb

__all__ = [
    "BetOption",
    "ArbResult",
    "solve_arb",
]

"""
betoptimiser.football – Football optimisation.

>>> from betoptimiser.football import scan_arb, scan_all_arbs
"""

from betoptimiser.football.scenarios import FootballState, build_scenarios
from betoptimiser.football.payouts import build_payout_matrix
from betoptimiser.football.scanner import scan_arb, scan_event_arbs, scan_all_arbs

__all__ = [
    "FootballState",
    "build_scenarios",
    "build_payout_matrix",
    "scan_arb",
    "scan_event_arbs",
    "scan_all_arbs",
]

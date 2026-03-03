"""
betoptimiser.racing – Horse racing optimisation.

>>> from betoptimiser.racing import scan_all_arbs
>>> results = scan_all_arbs(session, commission=0.02, return_all=True)
"""

from betoptimiser.racing.scanner import scan_arb, scan_all_arbs
from betoptimiser.racing.scenarios import build_scenarios
from betoptimiser.racing.payouts import build_payout_matrix
from betoptimiser.racing.detection import (
    detect_place_count,
    detect_ew_terms,
    classify_market_type,
)

__all__ = [
    "scan_arb",
    "scan_all_arbs",
    "build_scenarios",
    "build_payout_matrix",
    "detect_place_count",
    "detect_ew_terms",
    "classify_market_type",
]

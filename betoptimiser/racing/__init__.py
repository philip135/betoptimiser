"""
betoptimiser.racing – Horse racing optimisation.

>>> from betoptimiser.racing import scan_all_arbs
>>> arbs = scan_all_arbs(session, commission=0.02)
"""

from betoptimiser.racing.scanner import scan_runner_arb, scan_all_arbs
from betoptimiser.racing.detection import (
    detect_place_count,
    detect_ew_terms,
    classify_market_type,
)

__all__ = [
    "scan_runner_arb",
    "scan_all_arbs",
    "detect_place_count",
    "detect_ew_terms",
    "classify_market_type",
]

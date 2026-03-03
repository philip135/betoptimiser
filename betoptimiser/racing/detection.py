"""
Racing Detection – place count, EW terms, market type classification.
"""

from __future__ import annotations

import re
import logging

from betfairtools.constants import (
    WIN, PLACE, EACH_WAY, FORECAST, REVERSE_FORECAST,
    REV_FORECAST, OTHER_PLACE, MATCH_BET, WITHOUT_FAV,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLACE COUNT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_places_from_rules(rules: str) -> int:
    """
    Parse number of places from rules text like
    "Who will finish 1st, 2nd or 3rd in this race?"
    """
    m = re.search(r'finish\s+(.+?)\s+in\s+this', rules, re.I)
    if m:
        text = m.group(1)
        ordinals = re.findall(r'\d+(?:st|nd|rd|th)', text, re.I)
        if ordinals:
            return len(ordinals)
    return 0


def _parse_place_count_from_name(market_name: str) -> int:
    """Parse place count from OTHER_PLACE names like '2 TBP', '4 TBP'."""
    m = re.match(r'(\d+)\s*TBP', market_name, re.I)
    if m:
        return int(m.group(1))
    return 0


def _heuristic_place_count(n_runners: int, place_market_exists: bool = True) -> int:
    """Heuristic place count from number of runners."""
    min_places = 2 if place_market_exists else 1
    if n_runners <= 7:
        return max(min_places, 2)
    elif n_runners <= 15:
        return 3
    else:
        return 4


def detect_place_count(
    desc=None,
    market_name: str = "",
    n_runners: int = 0,
) -> int:
    """
    Detect number of places paid from the market description object.

    Strategy:
    1. Parse rules text for ordinal pattern ("1st, 2nd or 3rd").
    2. For OTHER_PLACE, parse from market name ("2 TBP", "4 TBP").
    3. Fall back to heuristic by runner count.
    """
    if desc:
        rules = getattr(desc, "rules", "") or ""
        places = _parse_places_from_rules(rules)
        if places > 0:
            return places

    if market_name:
        from_name = _parse_place_count_from_name(market_name)
        if from_name > 0:
            return from_name

    if n_runners > 0:
        return _heuristic_place_count(n_runners, place_market_exists=True)
    return 3  # conservative default


# ═══════════════════════════════════════════════════════════════════════════════
#  EACH-WAY TERMS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _heuristic_ew_places(n_runners: int) -> int:
    """Heuristic EW place count from runner count."""
    if n_runners <= 4:
        return 0
    elif n_runners <= 7:
        return 2
    elif n_runners <= 15:
        return 3
    else:
        return 4


def detect_ew_terms(
    desc=None,
    n_runners: int = 0,
) -> tuple[float, int]:
    """
    Detect each-way fraction and place count from market description.

    Uses desc.each_way_divisor (Betfair native field) for fraction,
    and parses rules text ordinals for number of places.

    Returns (fraction, places) e.g. (0.2, 3) for 1/5 odds, 3 places.
    Falls back to standard UK/IE EW terms if fields not found.
    """
    fraction = 0.0
    places = 0

    if desc:
        # Fraction: use Betfair's each_way_divisor field directly
        ew_divisor = getattr(desc, "each_way_divisor", None)
        if ew_divisor and float(ew_divisor) > 0:
            fraction = 1.0 / float(ew_divisor)
            logger.debug(f"EW fraction from each_way_divisor={ew_divisor}: {fraction}")

        # Places: parse from rules text ordinals
        rules = getattr(desc, "rules", "") or ""
        places = _parse_places_from_rules(rules)

    if fraction > 0 and places > 0:
        return fraction, places

    # Partial detection — fill in missing piece
    if fraction > 0 and places == 0:
        places = _heuristic_ew_places(n_runners)
        if places > 0:
            return fraction, places

    # Full fallback to standard UK/IE EW terms
    if n_runners <= 4:
        return 0.0, 0
    elif n_runners <= 7:
        return 0.25, 2
    elif n_runners <= 11:
        return 0.25, 3
    elif n_runners <= 15:
        return 0.2, 3
    else:
        return 0.25, 4


# ═══════════════════════════════════════════════════════════════════════════════
#  MARKET TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_market_type(market_name: str, market_type_code: str) -> str:
    """Classify a Betfair market type code + name into internal types."""
    code_map = {
        WIN: WIN,
        PLACE: PLACE,
        EACH_WAY: EACH_WAY,
        FORECAST: FORECAST,
        REVERSE_FORECAST: REVERSE_FORECAST,
        REV_FORECAST: REV_FORECAST,
        OTHER_PLACE: OTHER_PLACE,
        MATCH_BET: MATCH_BET,
        WITHOUT_FAV: WITHOUT_FAV,
    }
    if market_type_code in code_map:
        return code_map[market_type_code]

    # Fallback: name-based heuristic
    name_lower = market_name.lower()
    if "each way" in name_lower or "each-way" in name_lower:
        return EACH_WAY
    elif "reverse forecast" in name_lower:
        return REV_FORECAST
    elif "forecast" in name_lower:
        return FORECAST
    elif "tbp" in name_lower:
        return OTHER_PLACE
    elif "place" in name_lower:
        return PLACE
    return WIN

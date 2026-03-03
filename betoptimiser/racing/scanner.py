"""
Racing Scanner – per-runner cross-market arb detection via convex optimisation.

Instead of enumerating all ordered finish permutations (O(N!)), analyses each
runner independently across WIN, PLACE, EACH_WAY, and OTHER_PLACE markets.

For each runner, the possible finishing outcomes are a small set of states
(typically 3–5: wins, places-only, doesn't place).  This is the convex hull
of outcomes — we solve a tiny LP over these states in microseconds.

Example: WIN + PLACE(3) + EACH_WAY(1/4, 3pl) gives 3 states per runner:
  1. Wins (1st)
  2. Places (2nd–3rd)
  3. Doesn't place (>3rd)

The LP finds stakes across all available bets (back/lay on each market) that
guarantee positive profit in every state.  If it succeeds → cross-market arb.

Total work: N_runners × 3-state LP  (~200 runners → < 100ms total)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from betoptimiser.models import ArbResult, BetOption
from betoptimiser.solver import solve_arb
from betoptimiser.prices import extract_bet_options
from betoptimiser.racing.detection import (
    detect_place_count,
    detect_ew_terms,
    classify_market_type,
)
from betfairtools.constants import (
    WIN, PLACE, EACH_WAY,
    OTHER_PLACE,
)

logger = logging.getLogger(__name__)

# Market types where outcome depends only on THIS runner's finishing position.
SINGLE_RUNNER_TYPES = {WIN, PLACE, EACH_WAY, OTHER_PLACE}

# Only fetch these market types (skip FORECAST, MATCH_BET, etc.)
SCAN_MARKET_TYPES = [WIN, PLACE, EACH_WAY, OTHER_PLACE]


# ── Per-runner payout logic ───────────────────────────────────────────────────

def _bet_payout_at_position(bet: BetOption, position: int) -> float:
    """
    Net profit per £1 unit stake if the runner finishes at ``position``.

    position = 1 → wins.  position > K → doesn't place for a K-place market.

    Market types:
      WIN:            hit if position == 1
      PLACE(K):       hit if position ≤ K
      OTHER_PLACE(K): hit if position ≤ K
      EACH_WAY(f,K):  3-tier: win (pos=1), place-only (1<pos≤K), lose (pos>K)
                       Note: £1 unit EW = £1 win-part + £1 place-part = £2 total
    """
    mt = bet.market_type
    side = bet.side
    p = bet.price

    if mt == WIN:
        hit = position == 1
        return (p - 1.0 if hit else -1.0) if side == "BACK" else (-(p - 1.0) if hit else 1.0)

    if mt in (PLACE, OTHER_PLACE):
        hit = position <= bet.place_count
        return (p - 1.0 if hit else -1.0) if side == "BACK" else (-(p - 1.0) if hit else 1.0)

    if mt == EACH_WAY:
        f = bet.ew_fraction
        wins = position == 1
        places = position <= bet.ew_places

        if side == "BACK":
            if wins:
                return (p - 1.0) * (1.0 + f)      # both parts win
            elif places:
                return (p - 1.0) * f - 1.0          # place part wins, win part loses
            else:
                return -2.0                          # both parts lose
        else:  # LAY
            if wins:
                return -(p - 1.0) * (1.0 + f)
            elif places:
                return 1.0 - (p - 1.0) * f
            else:
                return 2.0

    # Unknown type — treat as WIN
    hit = position == 1
    return (p - 1.0 if hit else -1.0) if side == "BACK" else (-(p - 1.0) if hit else 1.0)


def _runner_outcome_states(
    bets: list[BetOption],
) -> list[tuple[int, str]]:
    """
    Determine the distinct finishing-position states for a runner.

    Each market defines a "cut point" — a position threshold that changes
    the payout.  All positions within a band (between consecutive cut points)
    produce identical payouts for every bet, so we only need one representative
    per band.

    Returns list of (representative_position, human_label) pairs.

    Example – WIN + PLACE(3) + EW(3pl):
      Cut points: {1, 3}
      States: [(1, "Wins"), (3, "Places 2nd-3rd"), (4, "Doesn't place")]

    Example – WIN + PLACE(3) + OTHER_PLACE(2):
      Cut points: {1, 2, 3}
      States: [(1, "Wins"), (2, "Finishes 2nd"), (3, "Finishes 3rd"), (4, "Doesn't place")]
    """
    cut_points = {1}  # WIN always distinguishes 1st vs rest
    for b in bets:
        if b.market_type in (PLACE, OTHER_PLACE) and b.place_count > 0:
            cut_points.add(b.place_count)
        if b.market_type == EACH_WAY and b.ew_places > 0:
            cut_points.add(b.ew_places)

    sorted_cuts = sorted(cut_points)
    ordinal = lambda n: {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")

    states: list[tuple[int, str]] = []
    prev = 0
    for c in sorted_cuts:
        lo, hi = prev + 1, c
        if lo == 1 and hi == 1:
            label = "Wins"
        elif lo == hi:
            label = f"Finishes {ordinal(lo)}"
        else:
            label = f"Places {ordinal(lo)}-{ordinal(hi)}"
        states.append((c, label))
        prev = c

    max_place = sorted_cuts[-1]
    states.append((max_place + 1, "Doesn't place"))
    return states


def _build_runner_payout_matrix(
    bets: list[BetOption],
    positions: list[int],
) -> np.ndarray:
    """
    Build the (n_states × n_bets) payout matrix for one runner.
    Typically 3 rows × 4-8 columns — trivially small.
    """
    A = np.zeros((len(positions), len(bets)))
    for i, pos in enumerate(positions):
        for j, bet in enumerate(bets):
            A[i, j] = _bet_payout_at_position(bet, pos)
    return A


# ── Per-runner arb scan ───────────────────────────────────────────────────────

def scan_runner_arb(
    runner_name: str,
    runner_bets: list[BetOption],
    budget: float = 100.0,
    commission: float = 0.02,
    race_name: str = "",
) -> ArbResult:
    """
    Check a single runner for a cross-market arb.

    Builds a tiny payout matrix over the runner's possible finishing
    positions (3–5 states) and solves the LP in microseconds.

    Parameters
    ----------
    runner_name : str
    runner_bets : list[BetOption]
        All bets for this runner across all markets.
    budget, commission : float
    race_name : str
        Race identifier for display.
    """
    # Only consider single-runner market types (not FORECAST etc.)
    bets = [b for b in runner_bets if b.market_type in SINGLE_RUNNER_TYPES]

    if len(bets) < 2:
        return ArbResult(
            status="insufficient_bets",
            guaranteed_profit=0.0,
            total_stake=0.0,
            stakes=[0.0] * len(bets),
            bets=bets,
            profit_by_scenario=np.array([]),
            scenario_labels=[],
            race_name=f"{race_name} – {runner_name}",
        )

    # Only worth checking if bets span ≥ 2 distinct market types
    market_types = {b.market_type for b in bets}
    if len(market_types) < 2:
        return ArbResult(
            status="single_market",
            guaranteed_profit=0.0,
            total_stake=0.0,
            stakes=[0.0] * len(bets),
            bets=bets,
            profit_by_scenario=np.array([]),
            scenario_labels=[],
            race_name=f"{race_name} – {runner_name}",
        )

    # Determine outcome states (3–5 typically)
    states = _runner_outcome_states(bets)
    positions = [s[0] for s in states]
    labels = [s[1] for s in states]

    # Build tiny payout matrix and solve
    A = _build_runner_payout_matrix(bets, positions)
    result = solve_arb(
        bets=bets,
        payout_matrix=A,
        budget=budget,
        commission=commission,
        scenario_labels=labels,
    )

    result.race_name = f"{race_name} – {runner_name}"
    result.n_scenarios = len(states)
    if result.is_arb:
        result.arb_margin = result.guaranteed_profit / budget
    else:
        result.arb_margin = -1.0

    return result


# ── Race-level scan ───────────────────────────────────────────────────────────

def scan_race(
    session,
    race_catalogues: list,
    budget: float = 100.0,
    commission: float = 0.02,
    market_meta: Optional[dict[str, dict]] = None,
    race_name: str = "",
    n_runners: int = 0,
    pre_fetched_books: Optional[dict] = None,
) -> list[ArbResult]:
    """
    Scan every runner in a race for cross-market arbs.

    Returns a list of ArbResult — one per runner with an arb found.
    Empty list if no arbs in this race.
    """
    if market_meta is None:
        market_meta = {}

    bets = extract_bet_options(
        session, race_catalogues, price_depth=2,
        market_meta=market_meta,
        pre_fetched_books=pre_fetched_books,
    )

    if not bets:
        return []

    # Group bets by runner
    runner_bets: dict[str, list[BetOption]] = {}
    for b in bets:
        if b.market_type in SINGLE_RUNNER_TYPES:
            runner_bets.setdefault(b.runner_name, []).append(b)

    results: list[ArbResult] = []
    for rname, rbets in runner_bets.items():
        # Need bets across ≥ 2 market types for cross-market arb
        if len({b.market_type for b in rbets}) < 2:
            continue

        result = scan_runner_arb(
            runner_name=rname,
            runner_bets=rbets,
            budget=budget,
            commission=commission,
            race_name=race_name,
        )

        if result.is_arb:
            results.append(result)

    return results


# ── Batch scan ────────────────────────────────────────────────────────────────

def scan_all_arbs(
    session,
    days: int = 0,
    hours: float = 0,
    countries: Optional[list[str]] = None,
    budget: float = 100.0,
    commission: float = 0.02,
    min_profit: float = 0.01,
    market_types: Optional[list[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> list[ArbResult]:
    """
    Scan all races for cross-market per-runner arbs.

    Parameters
    ----------
    verbose : bool
        If True, log detailed progress (DEBUG-level info).
        If False (default), only log arb finds, warnings, and errors.

    Returns only ArbResult objects with guaranteed_profit ≥ min_profit.
    """
    # Set log level for this scan
    prev_level = logger.level
    scan_level = logging.DEBUG if verbose else logging.WARNING
    logger.setLevel(scan_level)

    # Also control betfairtools logging noise
    _bt_logger = logging.getLogger("betfairtools")
    _bt_prev = _bt_logger.level
    _bt_logger.setLevel(scan_level)

    countries = countries or ["GB", "IE"]
    market_types = market_types or SCAN_MARKET_TYPES

    if not days and not hours:
        days = 1  # default to 1 day

    # ── Fetch catalogues ──────────────────────────────────────────────────
    all_markets: list = []
    try:
        all_markets = session.racing.markets(
            days=days, hours=hours, countries=countries,
            market_type_codes=market_types,
        )
    except Exception:
        logger.info("Bulk catalogue fetch failed, falling back to per-type")
        for mt_code in market_types:
            try:
                batch = session.racing.markets(
                    days=days, hours=hours, countries=countries,
                    market_type_codes=[mt_code],
                )
                all_markets.extend(batch)
            except Exception as e:
                logger.warning(f"Could not fetch {mt_code} markets: {e}")

    logger.info(f"Fetched {len(all_markets)} total racing markets")

    # Group by race (event_id + start_time)
    race_groups: dict[tuple[str, str], list] = {}
    for cat in all_markets:
        eid = cat.event.id if cat.event else None
        start = (
            cat.market_start_time.isoformat()
            if cat.market_start_time else None
        )
        if eid and start:
            race_groups.setdefault((eid, start), []).append(cat)

    logger.info(f"Grouped into {len(race_groups)} distinct races")

    # Count market types
    type_counts: dict[str, int] = {}
    for cats in race_groups.values():
        for cat in cats:
            desc = getattr(cat, "description", None)
            mt_code = getattr(desc, "market_type", None) if desc else None
            mt_code = mt_code or ""
            mname = cat.market_name or ""
            _type = classify_market_type(mname, mt_code)
            type_counts[_type] = type_counts.get(_type, 0) + 1
    logger.info(f"Market type breakdown: {type_counts}")

    # ── Batch-fetch ALL price books upfront ───────────────────────────────
    all_market_ids = [cat.market_id for cats in race_groups.values() for cat in cats]
    logger.info(f"Fetching price books for {len(all_market_ids)} markets")
    all_book_list: list = []
    CHUNK = 10
    for i in range(0, len(all_market_ids), CHUNK):
        chunk = all_market_ids[i : i + CHUNK]
        try:
            all_book_list.extend(session.book(chunk, price_depth=2))
        except Exception as e:
            logger.warning(f"Book fetch error for chunk {i}: {e}")
    pre_fetched_books: dict = {b.market_id: b for b in all_book_list}
    logger.info(f"Fetched {len(pre_fetched_books)} books")

    # ── Scan each race ────────────────────────────────────────────────────
    all_results: list[ArbResult] = []
    total_runners = 0

    for (eid, start_iso), cats in race_groups.items():
        # Find WIN catalogue for race name
        win_cat = None
        for c in cats:
            desc = getattr(c, "description", None)
            mt_code = getattr(desc, "market_type", None) if desc else None
            mt_code = mt_code or ""
            classified = classify_market_type(c.market_name or "", mt_code)
            if classified == WIN:
                win_cat = c
                break
        if not win_cat:
            win_cat = cats[0]

        course = win_cat.event.name if win_cat.event else "Unknown"
        race_desc = win_cat.market_name or ""
        start_time_str = (
            win_cat.market_start_time.strftime("%H:%M")
            if win_cat.market_start_time else "??"
        )
        race_label = f"{course} {start_time_str} – {race_desc}"
        n_runners = len(win_cat.runners) if win_cat.runners else 0
        total_runners += n_runners

        # Build per-market metadata
        meta: dict[str, dict] = {}
        for cat in cats:
            mname = cat.market_name or ""
            desc = getattr(cat, "description", None)
            mt_code = getattr(desc, "market_type", None) if desc else None
            mt_code = mt_code or ""
            mt = classify_market_type(mname, mt_code)

            m: dict = {"market_type": mt}
            if mt in (PLACE, OTHER_PLACE):
                pc = detect_place_count(
                    desc, market_name=mname, n_runners=n_runners
                )
                m["place_count"] = pc
            elif mt == EACH_WAY:
                frac, places = detect_ew_terms(desc, n_runners=n_runners)
                m["ew_fraction"] = frac
                m["ew_places"] = places

            meta[cat.market_id] = m

        try:
            race_arbs = scan_race(
                session, cats,
                budget=budget, commission=commission,
                market_meta=meta,
                race_name=race_label,
                n_runners=n_runners,
                pre_fetched_books=pre_fetched_books,
            )

            for r in race_arbs:
                if r.guaranteed_profit >= min_profit:
                    logger.info(
                        f"ARB FOUND: {r.race_name} | "
                        f"profit=£{r.guaranteed_profit:.4f}"
                    )
                    all_results.append(r)

            if not race_arbs:
                logger.info(
                    f"No arbs: {race_label} "
                    f"({n_runners} runners, {len(cats)} markets)"
                )
        except Exception as e:
            logger.warning(f"Error scanning {race_label}: {e}")

    logger.info(
        f"Scan complete: {len(all_results)} arbs "
        f"across {len(race_groups)} races, {total_runners} runners"
    )

    # Restore previous log levels
    logger.setLevel(prev_level)
    _bt_logger.setLevel(_bt_prev)

    return all_results

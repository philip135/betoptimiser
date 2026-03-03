"""
Racing Scanner – per‑race and batch arb scanning.
"""

from __future__ import annotations

import logging
from math import perm
from typing import Optional

import numpy as np

from betoptimiser.models import ArbResult, BetOption
from betoptimiser.solver import solve_arb
from betoptimiser.prices import extract_bet_options
from betoptimiser.racing.scenarios import runner_names, build_scenarios
from betoptimiser.racing.payouts import build_payout_matrix
from betoptimiser.racing.detection import (
    detect_place_count,
    detect_ew_terms,
    classify_market_type,
)
from betfairtools.constants import (
    WIN, PLACE, EACH_WAY, FORECAST, REVERSE_FORECAST,
    REV_FORECAST, OTHER_PLACE, MATCH_BET, WITHOUT_FAV,
    ALL_RACING_TYPES,
)

logger = logging.getLogger(__name__)


def scan_arb(
    session,
    race_catalogues: list,
    budget: float = 100.0,
    commission: float = 0.02,
    market_meta: Optional[dict[str, dict]] = None,
    race_name: str = "",
    n_runners: int = 0,
) -> ArbResult:
    """
    End‑to‑end arb scan for a single horse race across ALL available
    market types.

    Parameters
    ----------
    session : betfairtools.Session
        Authenticated session.
    race_catalogues : list[MarketCatalogue]
        All market catalogues for this race (matched by event_id + start_time).
    budget : float
    commission : float
    market_meta : dict[market_id, dict] | None
        Per-market metadata (market_type, place_count, ew_fraction, ew_places).
    race_name : str
    n_runners : int
    """
    if market_meta is None:
        market_meta = {}

    # Determine scenario depth from the markets present
    max_depth = 1
    market_types_present: set[str] = set()

    for cat in race_catalogues:
        meta = market_meta.get(cat.market_id, {})
        mt = meta.get("market_type", WIN)
        market_types_present.add(mt)

        if mt in (PLACE, OTHER_PLACE):
            pc = meta.get("place_count", 0)
            if pc > max_depth:
                max_depth = pc
        elif mt == EACH_WAY:
            ewp = meta.get("ew_places", 0)
            if ewp > max_depth:
                max_depth = ewp
        elif mt in (FORECAST, REVERSE_FORECAST, REV_FORECAST):
            max_depth = max(max_depth, 2)

    # Build runner list and extract bets
    runners = runner_names(race_catalogues)
    bets = extract_bet_options(
        session, race_catalogues, price_depth=2, market_meta=market_meta
    )

    if not bets:
        return ArbResult(
            status="no_prices", guaranteed_profit=0, total_stake=0,
            stakes=[], bets=[], profit_by_scenario=np.array([]),
            scenario_labels=[], race_name=race_name,
        )

    # Check scenario count isn't too large
    n = len(runners)
    n_scenarios_est = perm(n, min(max_depth, n))
    if n_scenarios_est > 50000:
        logger.warning(
            f"Skipping {race_name}: {n_scenarios_est} scenarios too large "
            f"({n} runners, depth={max_depth})"
        )
        return ArbResult(
            status="too_many_scenarios", guaranteed_profit=0, total_stake=0,
            stakes=[], bets=[], profit_by_scenario=np.array([]),
            scenario_labels=[], race_name=race_name,
        )

    labels, scenarios = build_scenarios(runners, max_places=max_depth)
    A = build_payout_matrix(bets, scenarios)

    mkt_summary = ", ".join(sorted(market_types_present))
    logger.info(
        f"Scanning {race_name} | "
        f"markets=[{mkt_summary}], "
        f"depth={max_depth}, runners={n}, "
        f"scenarios={len(scenarios)}, bets={len(bets)}"
    )

    result = solve_arb(
        bets=bets,
        payout_matrix=A,
        budget=budget,
        commission=commission,
        scenario_labels=labels,
    )

    if result.is_arb:
        result.arb_margin = result.guaranteed_profit / budget
    else:
        # Re-solve with forced stake deployment for arb margin ranking
        forced = solve_arb(
            bets=bets,
            payout_matrix=A,
            budget=budget,
            commission=commission,
            scenario_labels=labels,
            force_deploy=True,
        )
        if forced.status in ("optimal", "optimal_inaccurate"):
            result = forced
            total_staked = max(forced.total_stake, 1e-9)
            result.arb_margin = forced.guaranteed_profit / total_staked
        else:
            result.arb_margin = -1.0

    result.race_name = race_name
    result.n_scenarios = len(scenarios)
    return result


def scan_all_arbs(
    session,
    days: int = 1,
    countries: Optional[list[str]] = None,
    budget: float = 100.0,
    commission: float = 0.02,
    min_profit: float = 0.01,
    return_all: bool = False,
    market_types: Optional[list[str]] = None,
) -> list[ArbResult]:
    """
    Scan all UK/IE races across ALL available market types.

    Fetches markets, groups by race (event_id + start_time), solves
    for arbs across all correlated markets simultaneously.

    Returns only results with guaranteed_profit > min_profit,
    unless return_all=True.
    """
    countries = countries or ["GB", "IE"]
    market_types = market_types or ALL_RACING_TYPES

    # Fetch market types in batches to avoid TOO_MUCH_DATA API errors
    all_markets: list = []
    for mt_code in market_types:
        try:
            batch = session.racing.markets(
                days=days, countries=countries,
                market_type_codes=[mt_code],
            )
            all_markets.extend(batch)
        except Exception as e:
            logger.warning(f"Could not fetch {mt_code} markets: {e}")

    logger.info(f"Fetched {len(all_markets)} total racing markets")

    # Group all markets by race: (event_id, market_start_time)
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

    # Count market types across all races
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

    all_results: list[ArbResult] = []
    arb_results: list[ArbResult] = []

    for (eid, start_iso), cats in race_groups.items():
        # Find the WIN catalogue for race name and runner count
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
                logger.info(
                    f"EW terms for {mname}: fraction=1/{int(1/frac) if frac else '?'} "
                    f"({frac:.4f}), places={places}"
                )

            meta[cat.market_id] = m

        try:
            res = scan_arb(
                session, cats,
                budget=budget, commission=commission,
                market_meta=meta,
                race_name=race_label,
                n_runners=n_runners,
            )
            all_results.append(res)
            if res.is_arb and res.guaranteed_profit >= min_profit:
                mkt_types = ", ".join(sorted(set(
                    m.get("market_type", "?") for m in meta.values()
                )))
                logger.info(
                    f"ARB FOUND: {race_label} | "
                    f"profit=£{res.guaranteed_profit:.4f} | "
                    f"markets=[{mkt_types}]"
                )
                arb_results.append(res)
            else:
                logger.info(
                    f"No arb: {race_label} | "
                    f"margin={res.arb_margin:+.4%} | "
                    f"markets={len(cats)} | "
                    f"scenarios={res.n_scenarios}"
                )
        except Exception as e:
            logger.warning(f"Error scanning {race_label}: {e}")

    logger.info(
        f"Racing arb scan complete: "
        f"{len(arb_results)}/{len(all_results)} races have arbs"
    )
    if return_all:
        return all_results
    return arb_results

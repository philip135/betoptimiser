"""
Football Scanner – per‑match and batch arb scanning.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from betfairlightweight.filters import market_filter as mf_filter

from betoptimiser.models import ArbResult
from betoptimiser.solver import solve_arb
from betoptimiser.prices import extract_bet_options
from betoptimiser.football.scenarios import build_scenarios
from betoptimiser.football.payouts import build_payout_matrix
from betfairtools.constants import (
    MATCH_ODDS, OVER_UNDER_15, OVER_UNDER_25, OVER_UNDER_35,
    BOTH_TEAMS_TO_SCORE, CORRECT_SCORE,
)

logger = logging.getLogger(__name__)


def scan_arb(
    session,
    market_catalogues: list,
    home_team: str = "",
    away_team: str = "",
    budget: float = 100.0,
    commission: float = 0.02,
    max_goals: int = 6,
) -> ArbResult:
    """
    Scan across multiple correlated football markets for a single match.

    Pass all market catalogues for one event and this will look for a
    guaranteed‑profit portfolio.
    """
    bets = extract_bet_options(session, market_catalogues, price_depth=1)
    if not bets:
        return ArbResult(
            status="no_prices", guaranteed_profit=0, total_stake=0,
            stakes=[], bets=[], profit_by_scenario=np.array([]),
            scenario_labels=[],
        )

    states = build_scenarios(max_goals)
    A = build_payout_matrix(bets, states, home_team, away_team)
    labels = [s.label() for s in states]

    return solve_arb(
        bets=bets,
        payout_matrix=A,
        budget=budget,
        commission=commission,
        scenario_labels=labels,
    )


def scan_event_arbs(
    session,
    event_id: str,
    home_team: str = "",
    away_team: str = "",
    budget: float = 100.0,
    commission: float = 0.02,
    max_goals: int = 6,
    market_type_codes: Optional[list[str]] = None,
) -> ArbResult:
    """
    High‑level: fetch all markets for a football event and scan for arbs.
    """
    type_codes = market_type_codes or [
        MATCH_ODDS,
        OVER_UNDER_15,
        OVER_UNDER_25,
        OVER_UNDER_35,
        BOTH_TEAMS_TO_SCORE,
        CORRECT_SCORE,
        "DOUBLE_CHANCE",
        "DRAW_NO_BET",
        "WIN_TO_NIL",
    ]

    catalogues = session.client.betting.list_market_catalogue(
        filter=mf_filter(event_ids=[event_id], market_type_codes=type_codes),
        market_projection=[
            "EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION",
        ],
        max_results="100",
        sort="FIRST_TO_START",
    )

    logger.info(
        f"Football arb scan: event={event_id}, "
        f"{len(catalogues)} markets, types={type_codes}"
    )

    if not catalogues:
        return ArbResult(
            status="no_markets", guaranteed_profit=0, total_stake=0,
            stakes=[], bets=[], profit_by_scenario=np.array([]),
            scenario_labels=[],
        )

    # Auto‑detect team names from Match Odds runners
    if not home_team or not away_team:
        for cat in catalogues:
            if "match odds" in cat.market_name.lower():
                runners = [r.runner_name for r in cat.runners]
                non_draw = [r for r in runners if r.lower() not in ("the draw", "draw")]
                if len(non_draw) >= 2:
                    home_team = home_team or non_draw[0]
                    away_team = away_team or non_draw[1]
                break

    return scan_arb(
        session, catalogues,
        home_team=home_team, away_team=away_team,
        budget=budget, commission=commission, max_goals=max_goals,
    )


def scan_all_arbs(
    session,
    days: int = 1,
    budget: float = 100.0,
    commission: float = 0.02,
    min_profit: float = 0.01,
    competition_ids: Optional[list[str]] = None,
    max_events: int = 50,
) -> list[tuple[str, ArbResult]]:
    """
    Scan multiple football events for cross‑market arb opportunities.

    Returns list of (event_name, ArbResult) for profitable opportunities.
    """
    events = session.football.events(days=days, competition_ids=competition_ids)
    results = []

    for ev_result in events[:max_events]:
        ev = ev_result.event
        try:
            res = scan_event_arbs(
                session, event_id=ev.id,
                budget=budget, commission=commission,
            )
            if res.is_arb and res.guaranteed_profit >= min_profit:
                logger.info(
                    f"FOOTBALL ARB: {ev.name} | "
                    f"profit=£{res.guaranteed_profit:.4f}"
                )
                results.append((ev.name, res))
        except Exception as e:
            logger.warning(f"Error scanning football event {ev.name}: {e}")

    logger.info(f"Football arb scan complete: {len(results)} opportunities found")
    return results

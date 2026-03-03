"""
Football Payout Matrix – map scorelines × bets.
"""

from __future__ import annotations

import re

import numpy as np

from betoptimiser.models import BetOption
from betoptimiser.football.scenarios import FootballState


def _match_runner_to_state(
    runner_name: str,
    market_name: str,
    state: FootballState,
    home_team: str = "",
    away_team: str = "",
) -> bool:
    """
    Determine if a runner 'wins' in a given FootballState.

    Handles: Match Odds, Over/Under, BTTS, Correct Score,
    Double Chance, Draw No Bet, Win to Nil.
    """
    rn = runner_name.strip().lower()
    mn = market_name.strip().lower()
    ht = home_team.strip().lower()
    at = away_team.strip().lower()

    # ── Match Odds ────────────────────────────────────────────────────────
    if "match odds" in mn:
        if rn == ht or rn == "home":
            return state.result == "HOME"
        if rn == at or rn == "away":
            return state.result == "AWAY"
        if rn in ("the draw", "draw"):
            return state.result == "DRAW"

    # ── Over / Under goals ────────────────────────────────────────────────
    if "over/under" in mn or "over_under" in mn:
        line_match = re.search(
            r"(\d+\.?\d*)",
            mn.replace("over/under", "").replace("over_under", ""),
        )
        line = float(line_match.group(1)) if line_match else 2.5
        if rn.startswith("over"):
            return state.total_goals > line
        if rn.startswith("under"):
            return state.total_goals < line

    # ── Both Teams To Score ───────────────────────────────────────────────
    if "both teams to score" in mn or "btts" in mn:
        if rn == "yes":
            return state.btts
        if rn == "no":
            return not state.btts

    # ── Correct Score ─────────────────────────────────────────────────────
    if "correct score" in mn:
        if rn == "any other home win":
            return state.result == "HOME" and state.home_goals > 3
        if rn == "any other away win":
            return state.result == "AWAY" and state.away_goals > 3
        if rn == "any other draw":
            return state.result == "DRAW" and state.home_goals > 3
        score_match = re.match(r"(\d+)\s*-\s*(\d+)", rn)
        if score_match:
            return (
                state.home_goals == int(score_match.group(1))
                and state.away_goals == int(score_match.group(2))
            )

    # ── Win To Nil ────────────────────────────────────────────────────────
    if "win to nil" in mn:
        if rn == "yes":
            if ht and ht in mn:
                return state.home_win_to_nil
            if at and at in mn:
                return state.away_win_to_nil
            return state.home_win_to_nil or state.away_win_to_nil
        if rn == "no":
            if ht and ht in mn:
                return not state.home_win_to_nil
            if at and at in mn:
                return not state.away_win_to_nil
            return not (state.home_win_to_nil or state.away_win_to_nil)

    # ── Double Chance ─────────────────────────────────────────────────────
    if "double chance" in mn:
        if ht and at:
            if rn == f"{ht} or draw" or rn == "home or draw":
                return state.double_chance_home_draw
            if rn == f"{at} or draw" or rn == "away or draw":
                return state.double_chance_away_draw
            if rn in (f"{ht} or {at}", "home or away"):
                return state.double_chance_home_away
        else:
            if "home" in rn and "draw" in rn:
                return state.double_chance_home_draw
            if "away" in rn and "draw" in rn:
                return state.double_chance_away_draw

    # ── Draw No Bet ───────────────────────────────────────────────────────
    if "draw no bet" in mn:
        if rn == ht or rn == "home":
            return state.draw_no_bet_home
        if rn == at or rn == "away":
            return state.draw_no_bet_away

    # ── Half Time (conservative: skip) ────────────────────────────────────
    if "half time" in mn and "full time" not in mn:
        pass

    return False


def build_payout_matrix(
    bets: list[BetOption],
    states: list[FootballState],
    home_team: str = "",
    away_team: str = "",
) -> np.ndarray:
    """
    Build (n_states × n_bets) payout matrix for a football match.

    Each state is a possible final scoreline.
    """
    n_states = len(states)
    n_bets = len(bets)
    A = np.zeros((n_states, n_bets))

    for j, bet in enumerate(bets):
        for i, state in enumerate(states):
            runner_wins = _match_runner_to_state(
                bet.runner_name, bet.market_name, state,
                home_team=home_team, away_team=away_team,
            )
            A[i, j] = bet.payout_if_wins if runner_wins else bet.payout_if_loses

    return A

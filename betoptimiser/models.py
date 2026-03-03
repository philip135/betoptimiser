"""
Data Models – BetOption and ArbResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from betfairtools.constants import (
    EACH_WAY, PLACE, OTHER_PLACE, FORECAST, REVERSE_FORECAST, REV_FORECAST,
)


@dataclass
class BetOption:
    """A single actionable bet (one side of one runner in one market)."""

    market_id: str
    market_name: str
    selection_id: int
    runner_name: str
    side: str               # "BACK" or "LAY"
    price: float            # decimal odds
    available_size: float   # max stake at this price
    market_type: str = ""   # WIN, PLACE, EACH_WAY, FORECAST, etc.
    place_count: int = 0    # for PLACE bets: how many places this market pays
    ew_fraction: float = 0.0  # for EACH_WAY: place fraction e.g. 0.25 (1/4)
    ew_places: int = 0      # for EACH_WAY: number of EW places
    runner_name_2: str = "" # for FORECAST: 2nd runner in the forecast

    @property
    def payout_if_wins(self) -> float:
        """Net P&L per unit stake if this runner's outcome OCCURS."""
        if self.side == "BACK":
            return self.price - 1.0
        else:  # LAY
            return -(self.price - 1.0)

    @property
    def payout_if_loses(self) -> float:
        """Net P&L per unit stake if this runner's outcome does NOT occur."""
        if self.side == "BACK":
            return -1.0
        else:  # LAY
            return 1.0


@dataclass
class ArbResult:
    """Result of an arbitrage optimisation."""

    status: str                     # "optimal", "infeasible", …
    guaranteed_profit: float        # worst‑case profit (per unit budget)
    total_stake: float
    stakes: list[float]             # stake per BetOption
    bets: list[BetOption]
    profit_by_scenario: np.ndarray  # profit vector across all scenarios
    scenario_labels: list[str]      # human‑readable labels
    race_name: str = ""             # event / race name for display
    n_scenarios: int = 0            # total scenarios enumerated
    place_count: int = 0            # number of places (racing)
    arb_margin: float = 0.0        # worst‑case profit per £1 capital deployed

    @property
    def is_arb(self) -> bool:
        return self.status == "optimal" and self.guaranteed_profit > 1e-6

    def summary(self, top_n: int = 20) -> str:
        """Human‑readable summary of this arb result."""
        lines = []
        if self.race_name:
            lines.append(f"Race: {self.race_name}")
        lines += [
            f"Status: {self.status}",
            f"Guaranteed profit: £{self.guaranteed_profit:.4f}",
            f"Total stake: £{self.total_stake:.2f}",
            f"Arb margin: {self.arb_margin:+.4%} per £1 capital",
        ]
        if self.n_scenarios:
            lines.append(f"Scenarios enumerated: {self.n_scenarios}")

        # Market types present
        if self.bets:
            types = sorted(set(b.market_type for b in self.bets if b.market_type))
            if types:
                lines.append(f"Market types: {', '.join(types)}")

            # EW terms and place counts per market
            seen_markets: set[str] = set()
            for b in self.bets:
                if b.market_id in seen_markets:
                    continue
                seen_markets.add(b.market_id)
                if b.market_type == EACH_WAY and b.ew_fraction > 0:
                    divisor = int(round(1.0 / b.ew_fraction)) if b.ew_fraction > 0 else "?"
                    lines.append(
                        f"  EW terms: 1/{divisor} odds, {b.ew_places} places "
                        f"({b.market_name})"
                    )
                elif b.market_type in (PLACE, OTHER_PLACE) and b.place_count > 0:
                    lines.append(
                        f"  Place count: {b.place_count} ({b.market_name})"
                    )

        # ── Dutch book vector ────────────────────────────────────────────
        lines.append("")
        lines.append("Dutch Book Vector (optimal stakes):")
        lines.append(
            f"  {'Side':<5} {'Runner':<25} {'Price':>7}"
            f" {'Stake':>10} {'Liability':>10} {'Market'}"
        )
        lines.append(f"  {'-'*90}")

        active_bets = [
            (bet, stake) for bet, stake in zip(self.bets, self.stakes)
            if abs(stake) > 1e-4
        ]
        total_capital = 0.0
        for bet, stake in sorted(active_bets, key=lambda x: -abs(x[1])):
            if bet.market_type == EACH_WAY:
                if bet.side == "LAY":
                    liability = ((bet.price - 1) + (bet.price - 1) * bet.ew_fraction) * stake
                else:
                    liability = 2.0 * stake
            elif bet.side == "LAY":
                liability = (bet.price - 1) * stake
            else:
                liability = stake
            total_capital += liability

            mtype_tag = f"[{bet.market_type}]" if bet.market_type else ""
            if bet.market_type == EACH_WAY and bet.ew_fraction > 0:
                divisor = int(round(1.0 / bet.ew_fraction))
                mtype_tag = f"[EW 1/{divisor} {bet.ew_places}pl]"
            elif bet.market_type in (PLACE, OTHER_PLACE) and bet.place_count > 0:
                mtype_tag = f"[{bet.market_type} {bet.place_count}pl]"

            lines.append(
                f"  {bet.side:<5} {bet.runner_name:<25} {bet.price:>7.2f}"
                f" £{stake:>9.2f} £{liability:>9.2f}  {bet.market_name} {mtype_tag}"
            )
        lines.append(f"  {'':>38} {'':>10} £{total_capital:>9.2f}  TOTAL CAPITAL AT RISK")

        # ── Scenario profit distribution ─────────────────────────────────
        lines.append("")
        if len(self.profit_by_scenario) > 0:
            lines.append(
                f"Profit range: £{self.profit_by_scenario.min():.4f}"
                f" → £{self.profit_by_scenario.max():.4f}"
            )
            lines.append("")
            lines.append("Worst scenarios:")
            idx = np.argsort(self.profit_by_scenario)
            for i in idx[:top_n]:
                lines.append(
                    f"  £{self.profit_by_scenario[i]:>8.4f}  {self.scenario_labels[i]}"
                )
        return "\n".join(lines)

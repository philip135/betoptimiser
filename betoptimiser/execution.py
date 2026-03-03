"""
Execution – place arb bets on Betfair via the session's orders API.

Safety features:
  - Requires explicit execute=True flag
  - All-or-nothing: skips arb if any stake is below minimum
  - Match verification: checks every bet is fully matched
  - Automatic rollback: cancels all placed bets if any leg fails or is unmatched
  - Dry-run mode prints what would be placed without touching the API
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from betfairlightweight.filters import cancel_instruction

from betoptimiser.models import ArbResult, BetOption

logger = logging.getLogger(__name__)

BETFAIR_MIN_STAKE = 1.00  # Betfair minimum bet size (EUR/GBP)


@dataclass
class ExecutionReport:
    """Summary of an arb execution attempt."""

    arb: ArbResult
    orders_placed: list[dict] = field(default_factory=list)
    orders_failed: list[dict] = field(default_factory=list)
    orders_cancelled: list[dict] = field(default_factory=list)
    dry_run: bool = True

    @property
    def all_success(self) -> bool:
        return (
            len(self.orders_failed) == 0
            and len(self.orders_cancelled) == 0
            and len(self.orders_placed) > 0
        )

    def summary(self) -> str:
        lines = [
            f"Execution {'(DRY RUN)' if self.dry_run else '(LIVE)'}",
            f"  Race: {self.arb.race_name}",
            f"  Expected profit: £{self.arb.guaranteed_profit:.4f}",
            f"  Orders placed:   {len(self.orders_placed)}",
            f"  Orders failed:   {len(self.orders_failed)}",
        ]
        if self.orders_cancelled:
            lines.append(f"  Orders cancelled (rollback): {len(self.orders_cancelled)}")
        if self.orders_placed:
            lines.append("")
            lines.append(f"  {'Side':<5} {'Runner':<20} {'Price':>7} {'Stake':>8} {'Matched':>8}  Market")
            lines.append(f"  {'-'*78}")
            for o in self.orders_placed:
                matched = o.get("size_matched", "?")
                matched_str = f"£{matched:>7.2f}" if isinstance(matched, (int, float)) else f"  {matched:>5}"
                lines.append(
                    f"  {o['side']:<5} {o['runner']:<20} {o['price']:>7.2f} "
                    f"£{o['stake']:>7.2f} {matched_str}  {o['market_name']} [{o['market_type']}]"
                )
        if self.orders_failed:
            lines.append("")
            lines.append("  FAILED ORDERS:")
            for o in self.orders_failed:
                lines.append(
                    f"  {o['side']} {o['runner']} @ {o['price']} "
                    f"£{o['stake']:.2f} – {o.get('error', 'unknown')}"
                )
        if self.orders_cancelled:
            lines.append("")
            lines.append("  CANCELLED (rollback):")
            for o in self.orders_cancelled:
                lines.append(
                    f"  {o['side']} {o['runner']} @ {o['price']} "
                    f"£{o['stake']:.2f} bet_id={o.get('bet_id', '?')}"
                )
        return "\n".join(lines)


def _cancel_bets(session, placed_orders: list[dict]) -> list[dict]:
    """Cancel all previously placed bets. Returns list of cancelled order dicts."""
    cancelled = []
    # Group by market_id for batch cancel
    by_market: dict[str, list[dict]] = {}
    for o in placed_orders:
        if o.get("bet_id"):
            by_market.setdefault(o["market_id"], []).append(o)

    for market_id, orders in by_market.items():
        instructions = [cancel_instruction(bet_id=str(o["bet_id"])) for o in orders]
        try:
            result = session._client.betting.cancel_orders(
                market_id=market_id,
                instructions=instructions,
            )
            status = getattr(result, "status", "UNKNOWN")
            if status == "SUCCESS":
                for o in orders:
                    logger.info(f"CANCELLED: bet_id={o['bet_id']} {o['side']} {o['runner']}")
                    cancelled.append(o)
            else:
                logger.error(f"Cancel failed for market {market_id}: {status}")
        except Exception as e:
            logger.error(f"Cancel exception for market {market_id}: {e}")

    return cancelled


def execute_arb(
    session,
    arb: ArbResult,
    execute: bool = False,
    min_stake: float = BETFAIR_MIN_STAKE,
    strategy_ref: str = "betoptimiser",
) -> ExecutionReport:
    """
    Place all bets in an ArbResult on Betfair.

    All-or-nothing execution:
    - All bets must meet min_stake or the arb is skipped
    - All bets must be placed and fully matched
    - If any bet fails or is unmatched, ALL placed bets are cancelled

    Parameters
    ----------
    session : betfairtools.Session
        Authenticated session with orders API.
    arb : ArbResult
        A result where arb.is_arb is True.
    execute : bool
        **Must be True to actually place bets.**  Default False = dry run.
    min_stake : float
        Minimum stake per bet (default £1). Entire arb skipped if any
        bet is below this.
    strategy_ref : str
        Customer strategy reference for order tracking (max 15 chars).

    Returns
    -------
    ExecutionReport
    """
    report = ExecutionReport(arb=arb, dry_run=not execute)

    if not arb.is_arb:
        logger.warning("execute_arb called on non-arb result – nothing to do")
        return report

    # ── Collect bets with non-zero stakes ─────────────────────────────────
    all_bets = [
        (bet, stake)
        for bet, stake in zip(arb.bets, arb.stakes)
        if abs(stake) > 1e-9
    ]

    # All-or-nothing: every stake must meet minimum
    too_small = [(b, s) for b, s in all_bets if abs(s) < min_stake]
    if too_small:
        names = ", ".join(f"{b.side} {b.runner_name} £{s:.2f}" for b, s in too_small)
        logger.warning(
            f"Skipping arb – {len(too_small)} bet(s) below £{min_stake:.2f} minimum: {names}"
        )
        return report

    # ── Dry run ───────────────────────────────────────────────────────────
    if not execute:
        for bet, stake in all_bets:
            report.orders_placed.append({
                "market_id": bet.market_id,
                "market_name": bet.market_name,
                "market_type": bet.market_type,
                "selection_id": bet.selection_id,
                "runner": bet.runner_name,
                "side": bet.side,
                "price": bet.price,
                "stake": round(stake, 2),
                "size_matched": round(stake, 2),
            })
        return report

    # ── Live execution ────────────────────────────────────────────────────
    placed: list[dict] = []
    failed = False

    for bet, stake in all_bets:
        order_info = {
            "market_id": bet.market_id,
            "market_name": bet.market_name,
            "market_type": bet.market_type,
            "selection_id": bet.selection_id,
            "runner": bet.runner_name,
            "side": bet.side,
            "price": bet.price,
            "stake": round(stake, 2),
        }

        try:
            result = session.orders.place_limit(
                market_id=bet.market_id,
                selection_id=bet.selection_id,
                side=bet.side,
                price=bet.price,
                size=round(stake, 2),
                persistence_type="LAPSE",
                customer_strategy_ref=strategy_ref,
            )

            status = getattr(result, "status", "UNKNOWN")
            if status != "SUCCESS":
                error_msg = str(getattr(result, "error_code", status))
                logger.error(
                    f"FAILED: {bet.side} {bet.runner_name} "
                    f"@ {bet.price} £{stake:.2f} – {error_msg}"
                )
                order_info["error"] = error_msg
                report.orders_failed.append(order_info)
                failed = True
                break

            # Extract match info from instruction report
            instr_reports = getattr(result, "place_instruction_reports", [])
            instr = instr_reports[0] if instr_reports else None

            bet_id = getattr(instr, "bet_id", None) if instr else None
            size_matched = getattr(instr, "size_matched", 0) if instr else 0
            order_status = getattr(instr, "order_status", "UNKNOWN") if instr else "UNKNOWN"

            order_info["bet_id"] = bet_id
            order_info["size_matched"] = size_matched or 0
            order_info["order_status"] = order_status

            logger.info(
                f"PLACED: {bet.side} {bet.runner_name} "
                f"@ {bet.price} £{stake:.2f} | matched=£{size_matched or 0:.2f} "
                f"bet_id={bet_id} status={order_status}"
            )

            # Check if fully matched
            if (size_matched or 0) < round(stake, 2) - 0.01:
                logger.warning(
                    f"UNMATCHED: {bet.side} {bet.runner_name} – "
                    f"wanted £{stake:.2f}, matched £{size_matched or 0:.2f}"
                )
                placed.append(order_info)
                failed = True
                break

            placed.append(order_info)

        except Exception as e:
            logger.error(f"EXCEPTION placing {bet.side} {bet.runner_name}: {e}")
            order_info["error"] = str(e)
            report.orders_failed.append(order_info)
            failed = True
            break

    # ── Rollback if anything went wrong ───────────────────────────────────
    if failed and placed:
        logger.warning(
            f"Rolling back {len(placed)} placed bet(s) due to failed/unmatched leg"
        )
        cancelled = _cancel_bets(session, placed)
        report.orders_cancelled = cancelled
        # Move placed to failed since arb is broken
        for o in placed:
            if o not in cancelled:
                report.orders_failed.append(o)
    else:
        report.orders_placed = placed

    mode = "LIVE"
    logger.info(
        f"Execution complete ({mode}): "
        f"{len(report.orders_placed)} placed, "
        f"{len(report.orders_failed)} failed, "
        f"{len(report.orders_cancelled)} cancelled"
    )
    return report

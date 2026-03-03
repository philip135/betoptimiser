"""
Price Extraction – pull live prices from Betfair and build BetOption lists.
"""

from __future__ import annotations

import logging
from typing import Optional

from betoptimiser.models import BetOption
from betfairtools.constants import (
    WIN, PLACE, EACH_WAY, FORECAST, REVERSE_FORECAST,
    REV_FORECAST, OTHER_PLACE,
)

logger = logging.getLogger(__name__)


def extract_bet_options(
    session,
    market_catalogues: list,
    price_depth: int = 1,
    market_meta: Optional[dict[str, dict]] = None,
    pre_fetched_books: Optional[dict] = None,
) -> list[BetOption]:
    """
    Build BetOption objects from live prices.

    Parameters
    ----------
    session : betfairtools.Session
        Authenticated session (or anything with a .book() method).
        Ignored if *pre_fetched_books* is provided.
    market_catalogues : list
        MarketCatalogue objects.
    price_depth : int
        Depth of price ladder to pull.
    market_meta : dict[market_id, dict] | None
        Per-market metadata with keys:
          market_type, place_count, ew_fraction, ew_places
        If None, market_type is inferred from market_name heuristics.
    pre_fetched_books : dict[market_id, MarketBook] | None
        If supplied, uses these instead of calling the API.
    """
    market_ids = [m.market_id for m in market_catalogues]

    if pre_fetched_books is not None:
        all_books = [pre_fetched_books[mid] for mid in market_ids if mid in pre_fetched_books]
    else:
        # Betfair allows max 40 market IDs per request
        all_books = []
        for i in range(0, len(market_ids), 40):
            chunk = market_ids[i : i + 40]
            all_books.extend(session.book(chunk, price_depth=price_depth))

    # Build name lookups
    name_map: dict[str, dict[int, str]] = {}
    mkt_name_map: dict[str, str] = {}
    for cat in market_catalogues:
        mkt_name_map[cat.market_id] = cat.market_name
        name_map[cat.market_id] = {
            r.selection_id: r.runner_name for r in cat.runners
        }

    options: list[BetOption] = []
    for book in all_books:
        mid = book.market_id
        meta = (market_meta or {}).get(mid, {})
        mtype = meta.get("market_type", "")
        pc = meta.get("place_count", 0)
        ewf = meta.get("ew_fraction", 0.0)
        ewp = meta.get("ew_places", 0)

        # Infer market type from name if not explicitly provided
        if not mtype:
            mname_lower = mkt_name_map.get(mid, "").lower()
            if "each way" in mname_lower or "each-way" in mname_lower:
                mtype = EACH_WAY
            elif "forecast" in mname_lower and "reverse" in mname_lower:
                mtype = REV_FORECAST
            elif "forecast" in mname_lower:
                mtype = FORECAST
            elif "tbp" in mname_lower:
                mtype = OTHER_PLACE
            elif "place" in mname_lower:
                mtype = PLACE
            else:
                mtype = WIN

        for runner in book.runners:
            rname = name_map.get(mid, {}).get(
                runner.selection_id, str(runner.selection_id)
            )
            mname = mkt_name_map.get(mid, mid)

            # For FORECAST markets, runner_name may be "A/B" composite
            runner_name_2 = ""
            if mtype in (FORECAST, REVERSE_FORECAST, REV_FORECAST) and "/" in rname:
                parts = rname.split("/", 1)
                rname = parts[0].strip()
                runner_name_2 = parts[1].strip()

            for back in runner.ex.available_to_back[:price_depth]:
                options.append(
                    BetOption(
                        market_id=mid,
                        market_name=mname,
                        selection_id=runner.selection_id,
                        runner_name=rname,
                        side="BACK",
                        price=back.price,
                        available_size=back.size,
                        market_type=mtype,
                        place_count=pc,
                        ew_fraction=ewf,
                        ew_places=ewp,
                        runner_name_2=runner_name_2,
                    )
                )
            for lay in runner.ex.available_to_lay[:price_depth]:
                options.append(
                    BetOption(
                        market_id=mid,
                        market_name=mname,
                        selection_id=runner.selection_id,
                        runner_name=rname,
                        side="LAY",
                        price=lay.price,
                        available_size=lay.size,
                        market_type=mtype,
                        place_count=pc,
                        ew_fraction=ewf,
                        ew_places=ewp,
                        runner_name_2=runner_name_2,
                    )
                )
    return options

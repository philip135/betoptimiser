# betoptimiser

Betting markets optimisation library — find guaranteed-profit portfolios across correlated betting markets using convex optimisation (LP via CVXPY).

## Installation

```bash
pip install betoptimiser
```

Or install from source:

```bash
pip install git+https://github.com/philip135/betoptimiser.git
```

**Requires** [betfairtools](https://github.com/philip135/betfairtools) (installed automatically as a dependency).

## Quick Start

### Horse Racing – Arb Scanner

```python
from betfairtools import Session
from betoptimiser.racing import scan_all_arbs
from betoptimiser.utils import print_arb_result

session = Session(username="...", password="...", app_key="...")

# Scan all UK/IE races across WIN, PLACE, EACH_WAY, FORECAST, etc.
results = scan_all_arbs(
    session,
    commission=0.02,
    return_all=True,
)

# Show arbs
for r in results:
    if r.is_arb:
        print_arb_result(r)

# Rank all races by arb margin
ranked = sorted(results, key=lambda r: r.arb_margin, reverse=True)
for r in ranked[:10]:
    print(f"{r.arb_margin:+.4%}  {r.race_name}")
```

### Football – Cross-Market Arbs

```python
from betoptimiser.football import scan_event_arbs

result = scan_event_arbs(session, event_id="12345678")
if result.is_arb:
    print_arb_result(result)
```

### Single Market Dutch Book

```python
from betoptimiser.utils import dutch_book_single_market

markets = session.racing.win_markets()
result = dutch_book_single_market(session, markets[0])
```

## Architecture

| Module | Description |
|--------|-------------|
| `betoptimiser.models` | `BetOption` and `ArbResult` data structures |
| `betoptimiser.solver` | Core LP solver (CVXPY + CLARABEL) |
| `betoptimiser.prices` | Extract live bet options from Betfair |
| `betoptimiser.racing` | Horse racing scenarios, payouts, scanners |
| `betoptimiser.football` | Football scenarios, payouts, scanners |
| `betoptimiser.utils` | Dutch book, display helpers |

## How It Works

1. **Enumerate scenarios** — all possible race finishes (ordered permutations) or football scorelines
2. **Build payout matrix** — map each bet × scenario to net P&L per £1 staked
3. **Solve LP** — maximise worst-case profit subject to capital constraints
4. **Post-hoc commission** — exact per-market-net Betfair commission applied after solving

The LP uses conservative commission in the objective (per-positive-entry) to keep the problem small (~N_bets variables, no auxiliaries), then applies exact commission post-hoc. Since the LP overcharges, actual profit ≥ LP objective — the solution is always safe.

## Market Types Supported

**Horse Racing:** WIN, PLACE, EACH_WAY, FORECAST, REV_FORECAST, OTHER_PLACE, MATCH_BET, WITHOUT_FAV

**Football:** Match Odds, Over/Under, BTTS, Correct Score, Double Chance, Draw No Bet, Win to Nil

## License

MIT

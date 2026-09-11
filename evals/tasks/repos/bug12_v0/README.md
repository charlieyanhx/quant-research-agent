# Spread mean-reversion strategy — backtest

Sells a two-leg spread when its net mid is more than one trailing-20-day standard
deviation rich, holds 10 days, marks to market daily from per-leg quotes, and fills at
full cross-spread on entry and exit (Line 3). Reported Sharpe is on the full calendar.

Run: `python backtest.py`

Files: `trades.csv` (candidate signal days), `marks.csv` (per-leg bid/ask by day; quote
outages leave some days missing), `spot.csv` (underlying close).

The strategy is delta-hedged and market-neutral by construction; the Sharpe above is alpha.

"""Two-leg spread strategy: sell when the net mid is rich, hold HOLD days.

Data: trades.csv (candidate signal days), marks.csv (per-leg bid/ask per day; some days are
missing for some legs — quote outages), spot.csv (underlying close).
"""
import numpy as np
import pandas as pd

HOLD = 10
Z_ENTRY = 1.0
LOOKBACK = 20
TRADING_DAYS = 252


def load():
    trades = pd.read_csv("trades.csv")
    marks = pd.read_csv("marks.csv")
    spot = pd.read_csv("spot.csv").set_index("day")["close"]
    calendar = np.arange(int(spot.index.min()), int(spot.index.max()) + 1)
    return trades, marks, spot, calendar


def net_quotes(marks):
    """Per (trade_id, day): net mid and net half-spread of short − long legs.
    A leg with no quote that day carries its last mark forward (a missing quote is not a
    price of zero)."""
    m = marks.assign(mid=(marks["bid"] + marks["ask"]) / 2, half=(marks["ask"] - marks["bid"]) / 2)
    mid = m.pivot_table(index=["trade_id", "day"], columns="leg", values="mid")
    half = m.pivot_table(index=["trade_id", "day"], columns="leg", values="half")
    mid = mid.groupby(level=0).ffill()
    half = half.groupby(level=0).ffill()
    net_mid = mid["short"] - mid["long"]
    net_half = half["short"] + half["long"]
    return net_mid, net_half


def signal_z(path, day):
    """z-score of the net mid on `day` against the trailing LOOKBACK days (ending at day)."""
    window = path.loc[:day].tail(LOOKBACK)
    if len(window) < LOOKBACK or window.std() == 0:
        return np.nan
    return (path.loc[day] - window.mean()) / window.std()


def trade_pnl(trade, net_mid, net_half, calendar):
    """Daily P&L series (indexed by calendar day) for one trade, or None if no signal.

    Entry fills at the first quote AFTER the signal day. Exit is by calendar day, HOLD days
    after entry, force-closed at the last calendar day if the window ends first. Marks are
    reindexed to the calendar so every daily difference spans exactly one day. Fills cross
    the spread on both entry and exit (Line 3)."""
    tid = trade["trade_id"]
    path = net_mid.loc[tid]
    half = net_half.loc[tid]
    z = signal_z(path, trade["signal_day"])
    if not (z > Z_ENTRY):
        return None
    side = -1.0  # rich → sell the spread
    entry_day = int(path.index[path.index > trade["signal_day"]][0])  # first quote after the signal
    exit_day = min(entry_day + HOLD, calendar[-1])
    path = path.reindex(range(entry_day, exit_day + 1))
    half = half.reindex(range(entry_day, exit_day + 1)).ffill()
    daily = (side * path.dropna().diff().fillna(0.0)).reindex(path.index, fill_value=0.0)
    cost = half.loc[entry_day] + half.loc[exit_day]
    daily.loc[entry_day] -= half.loc[entry_day]
    daily.loc[exit_day] -= half.loc[exit_day]
    return daily, cost


def run():
    trades, marks, spot, calendar = load()
    net_mid, net_half = net_quotes(marks)
    ledger = []
    per_trade = []
    for _, trade in trades.iterrows():
        out = trade_pnl(trade, net_mid, net_half, calendar)
        if out is None:
            continue
        daily, cost = out
        ledger.append(daily)
        per_trade.append({"trade_id": trade["trade_id"], "net": daily.sum(), "gross": daily.sum() + cost})
    tape = pd.DataFrame(per_trade)
    book = pd.concat(ledger, axis=1).sum(axis=1, min_count=1)
    book = book.reindex(calendar, fill_value=0.0)
    sharpe = book.mean() / book.std() * np.sqrt(TRADING_DAYS)

    spot_ret = spot.reindex(calendar).pct_change().fillna(0.0)
    beta = np.cov(book, spot_ret)[0, 1] / spot_ret.var()
    beta_pnl = beta * spot_ret
    beta_share = float(np.cov(book, beta_pnl)[0, 1] / book.var()) if book.var() > 0 else float("nan")

    print(f"trades entered: {len(tape)} of {len(trades)} candidates")
    print(f"gross P&L (Line 1, mid fills): {tape['gross'].sum():.2f}")
    print(f"net P&L   (Line 3, cross-spread): {tape['net'].sum():.2f}")
    print(f"Sharpe (Line 3, MTM daily, full calendar {len(calendar)} days): {sharpe:.2f}")
    print(f"beta to spot: {beta:.3f}; share of P&L variance explained by beta: {beta_share:.1%}")
    return tape, book, sharpe


if __name__ == "__main__":
    run()

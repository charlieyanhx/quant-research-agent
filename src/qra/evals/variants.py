"""Paraphrased expressions of each bug class.

Variant 0 is `template.MUTATIONS` (the expression the regex baseline was written against).
Variants 1 and 2 plant the *same* class through a different edit — different anchor line,
different idiom, sometimes a different function — so a reviewer that matched the shape of
v0 rather than the mechanism should miss them. The baseline is frozen at v0 on purpose.

Each variant is a list of (old, new) patches applied to CLEAN plus an optional README
override (class 12 lives partly in the claim, not the code).
"""

from __future__ import annotations

from dataclasses import dataclass

from .template import MUTATIONS, MUTATIONS_EXTRA

README_BASE = """# Spread mean-reversion strategy — backtest

Sells a two-leg spread when its net mid is more than one trailing-20-day standard
deviation rich, holds {hold} days, marks to market daily from per-leg quotes, and fills at
full cross-spread on entry and exit (Line 3). Reported Sharpe is on the full calendar.

Run: `python backtest.py`

Files: `trades.csv` (candidate signal days), `marks.csv` (per-leg bid/ask by day; quote
outages leave some days missing), `spot.csv` (underlying close).
"""

README_NEUTRAL_V0 = README_BASE + """
The strategy is delta-hedged and market-neutral by construction; the Sharpe above is alpha.
"""
README_NEUTRAL_V1 = README_BASE + """
Exposure: the beta line printed by the backtest confirms the book carries no directional
risk, so the Sharpe is quoted as pure alpha.
"""
README_NEUTRAL_V2 = README_BASE + """
Because both legs are on the same underlying the spread is dollar-neutral and its P&L is
uncorrelated with the index; no hedge is needed and the result is reported as alpha.
"""


@dataclass(frozen=True)
class Variant:
    patches: list[tuple[str, str]]
    readme: str | None = None
    note: str = ""


def _v0(k: int) -> Variant:
    patches = [MUTATIONS[k]] + ([MUTATIONS_EXTRA[k]] if k in MUTATIONS_EXTRA else [])
    return Variant(patches, README_NEUTRAL_V0 if k == 12 else None, "baseline expression")


_ENTRY = "    entry_day = int(path.index[path.index > trade[\"signal_day\"]][0])  # first quote after the signal\n"
_EXIT = "    exit_day = min(entry_day + HOLD, calendar[-1])\n"
_REINDEX = ("    path = path.reindex(range(entry_day, exit_day + 1)).ffill()\n"
            "    half = half.reindex(range(entry_day, exit_day + 1)).ffill()\n")
_DAILY = "    daily = side * path.diff().fillna(0.0)\n"
_COST = "    cost = half.loc[entry_day] + half.loc[exit_day]\n"
_FILLS = ("    daily.loc[entry_day] -= half.loc[entry_day]\n"
          "    daily.loc[exit_day] -= half.loc[exit_day]\n")
_FFILL = ("    mid = mid.groupby(level=0).ffill()\n"
          "    half = half.groupby(level=0).ffill()\n")
_PIVOT_MID = "    mid = m.pivot_table(index=[\"trade_id\", \"day\"], columns=\"leg\", values=\"mid\")\n"
_PATH = ("    path = net_mid.loc[tid]\n"
         "    half = net_half.loc[tid]\n")
_APPEND = "        daily, cost = out\n"
_BOOK = "    book = book.reindex(calendar, fill_value=0.0)\n"
_SHARPE = "    sharpe = book.mean() / book.std() * np.sqrt(TRADING_DAYS)\n"
_RETURN_LOAD = "    return trades, marks, spot, calendar\n"
_SPOT = "    spot_ret = spot.reindex(calendar).pct_change().fillna(0.0)\n"
_BETA_BLOCK = (_SPOT
               + "    beta = np.cov(book, spot_ret)[0, 1] / spot_ret.var()\n"
               + "    beta_pnl = beta * spot_ret\n"
               + "    beta_share = float(np.cov(book, beta_pnl)[0, 1] / book.var()) if book.var() > 0 else float(\"nan\")\n")
_BETA_PRINT = "    print(f\"beta to spot: {beta:.3f}; share of P&L variance explained by beta: {beta_share:.1%}\")\n"
_NET_GROSS = "        per_trade.append({\"trade_id\": trade[\"trade_id\"], \"net\": daily.sum(), \"gross\": daily.sum() + cost})\n"

VARIANTS: dict[int, list[Variant]] = {
    1: [_v0(1),
        Variant([(_DAILY,
                  "    daily = pd.Series({exit_day: side * (path.iloc[-1] - path.iloc[0])}).reindex(path.index, fill_value=0.0)\n")],
                note="realized P&L placed on the close day via a one-element series"),
        Variant([("        ledger.append(daily)\n",
                  "        ledger.append(pd.Series({daily.index[-1]: daily.sum()}))  # book realized P&L when the trade closes\n")],
                note="lumping moved from trade_pnl into the aggregation step")],
    2: [_v0(2),
        Variant([(_EXIT + _REINDEX,
                  "    traj = path[path.index >= entry_day].iloc[:HOLD + 1]\n"
                  "    exit_day = int(traj.index[-1])\n" + _REINDEX)],
                note="positional slice of the trade's own series"),
        Variant([(_EXIT + _REINDEX,
                  "    window = path.loc[entry_day:].head(HOLD + 1)\n"
                  "    exit_day = int(window.index.max())\n" + _REINDEX)],
                note="head(HOLD+1) on a gapped series")],
    3: [_v0(3),
        Variant([(_RETURN_LOAD,
                  "    quarter = 63\n"
                  "    trades = trades[(trades[\"signal_day\"] % quarter) + HOLD + 1 < quarter]  # trades that complete within their quarter\n"
                  + _RETURN_LOAD)],
                note="quarterly boundary filter at load time"),
        Variant([(_EXIT,
                  _EXIT + "    if entry_day // 378 != exit_day // 378:\n"
                          "        return None  # straddles the in-sample / out-of-sample split; scored in neither half\n")],
                note="sample-split boundary inside trade_pnl")],
    4: [_v0(4),
        Variant([(_FFILL, "    mid = mid.fillna({\"short\": 0.0, \"long\": 0.0})\n"
                          "    half = half.fillna({\"short\": 0.0, \"long\": 0.0})\n")],
                note="dict fillna"),
        Variant([(_PIVOT_MID, "    mid = m.pivot_table(index=[\"trade_id\", \"day\"], columns=\"leg\", values=\"mid\", fill_value=0.0)\n"),
                 (_FFILL, "    half = half.groupby(level=0).ffill()\n")],
                note="fill_value=0 inside pivot_table; only half-spread is carried")],
    5: [_v0(5),
        Variant([(_REINDEX, "    path = path.reindex(range(entry_day, exit_day + 1)).ffill()\n"
                            "    half = pd.Series(0.0, index=range(entry_day, exit_day + 1))  # fair-value fills; the spread is noise around mid\n")],
                note="half-spread zeroed upstream, cost lines untouched"),
        Variant([(_NET_GROSS,
                  "        per_trade.append({\"trade_id\": trade[\"trade_id\"], \"net\": daily.sum() + cost, \"gross\": daily.sum() + cost})  # spread already reflected in marks\n")],
                note="cost added back at aggregation; net == gross")],
    6: [_v0(6),
        Variant([(_BOOK, "    book = book[book != 0]  # flat days carry no information about the strategy\n"),
                 (_SPOT, "    spot_ret = spot.reindex(book.index).pct_change().fillna(0.0)\n")],
                note="zero days filtered instead of unpadded"),
        Variant([(_SHARPE, "    active = book[book.abs() > 0]\n"
                           "    sharpe = active.mean() / active.std() * np.sqrt(TRADING_DAYS)\n")],
                note="calendar padded, Sharpe taken on active days only")],
    7: [_v0(7),
        Variant([(_COST, "    cost = -(half.loc[entry_day] + half.loc[exit_day])\n")],
                note="cost sign flipped in the gross reconstruction only"),
        Variant([(_FILLS, "    daily.loc[entry_day] -= side * half.loc[entry_day]\n"
                          "    daily.loc[exit_day] -= side * half.loc[exit_day]\n")],
                note="side applied to the spread so a short pays a negative cost")],
    8: [_v0(8),
        Variant([(_APPEND, _APPEND + "        if daily.index[-1] >= calendar[-1]:\n"
                                     "            continue  # still open at sample end; excluded until it resolves\n")],
                note="open-at-end trades excluded during aggregation"),
        Variant([(_RETURN_LOAD, "    trades = trades[trades[\"signal_day\"] + HOLD + 1 <= calendar[-1]]  # only candidates that can complete\n"
                                + _RETURN_LOAD)],
                note="candidates filtered at load")],
    9: [_v0(9),
        Variant([(_PATH, "    path = net_mid.groupby(level=\"day\").mean()  # spread level for the day\n"
                         "    half = net_half.groupby(level=\"day\").mean()\n")],
                note="cross-sectional mean across instruments"),
        Variant([(_PATH, "    path = net_mid.groupby(level=\"day\").last()\n"
                         "    half = net_half.groupby(level=\"day\").last()\n")],
                note="last instrument quoted that day")],
    10: [_v0(10),
         Variant([(_ENTRY, "    entry_day = int(path.index[path.index >= trade[\"signal_day\"]][0])  # first quote from the signal on\n")],
                 note=">= instead of >"),
         Variant([(_ENTRY, "    entry_day = int(path.loc[:trade[\"signal_day\"]].index[-1])  # last observed quote at decision time\n")],
                 note="last quote at or before the signal")],
    11: [_v0(11),
         Variant([(_DAILY, "    raw = net_mid.loc[tid].loc[entry_day:exit_day]\n"
                           "    daily = (side * raw.diff().fillna(0.0)).reindex(path.index, fill_value=0.0)\n")],
                 note="diff on the raw gapped rows, then placed on the calendar"),
         Variant([(_REINDEX + _DAILY,
                   "    path = path.reindex(range(entry_day, exit_day + 1))\n"
                   "    half = half.reindex(range(entry_day, exit_day + 1)).ffill()\n"
                   "    daily = (side * path.dropna().diff().fillna(0.0)).reindex(path.index, fill_value=0.0)\n")],
                 note="reindex without ffill, dropna before diff")],
    12: [_v0(12),
         Variant([(_SPOT, "    spot_ret = spot.reindex(calendar).pct_change().shift(1).fillna(0.0)\n")],
                 README_NEUTRAL_V1, note="beta measured against lagged spot; prints ~0"),
         Variant([(_BETA_BLOCK, ""), (_BETA_PRINT, "    print(\"exposure: dollar-neutral spread, P&L uncorrelated with the index\")\n")],
                 README_NEUTRAL_V2, note="attribution removed; claim phrased without the baseline's keywords")],
}


def apply(src: str, variant: Variant) -> str:
    for old, new in variant.patches:
        assert src.count(old) == 1, f"patch anchor not unique: {old[:60]!r}"
        src = src.replace(old, new)
    return src


def n_variants() -> int:
    return max(len(v) for v in VARIANTS.values())



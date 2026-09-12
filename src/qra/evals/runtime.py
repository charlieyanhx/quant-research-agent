"""Runtime-invariant baseline: run the repository's backtest and check what it returns.

The regex baseline matches the *shape* of a bug; this one ignores the code and tests the
*numbers* the way an auditor would, using only the repo's data files and the clean
semantics of the task (the twelve invariants of the taxonomy). It has no access to labels.

For each repo: import `backtest.py`, call `run()` → `(tape, book, sharpe)`; independently
rebuild the clean tape from `trades.csv` / `marks.csv` / `spot.csv` with the reference
semantics; then apply invariants, each mapped to the bug class it detects:

  6  calendar coverage       book index must be the full calendar
  7  sign                    net ≤ gross on every trade
  5  spread paid             net < gross on every trade that traded (net == gross ⇒ mid fills)
  3, 8  trade count          trades entered must equal the independent count of signals
  1  lumping                 a trade's P&L must not sit on one day when the mark path moved
  4  missing leg             no daily P&L equal to a whole leg's value (leg marked at zero)
  12 attribution             the README claims neutrality but the book's beta to spot is not ≈ 0
  2, 9, 10, 11  per-trade P&L differs from the reference tape by more than a tolerance —
                 detected, but the class among these four is not identified by a runtime
                 invariant (reported as class 0 = "unclassified deviation")

Everything above is computed in a subprocess with the repo as cwd; a repo that crashes is
reported as class 0 too. The baseline is deliberately conservative: it flags only what an
invariant proves, so its false-alarm rate on clean repos should be zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..agent.tools import Finding

UNCLASSIFIED = 0

_PROBE = r'''
import json, sys, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("backtest", "backtest.py"); m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m); tape, book, sharpe = m.run()
except Exception as e:
    print(json.dumps({"crash": repr(e)[:300]})); sys.exit(0)
trades = pd.read_csv("trades.csv"); marks = pd.read_csv("marks.csv"); spot = pd.read_csv("spot.csv").set_index("day")["close"]
calendar = np.arange(int(spot.index.min()), int(spot.index.max()) + 1)
HOLD, Z, LB = 10, 1.0, 20
mk = marks.assign(mid=(marks.bid + marks.ask) / 2, half=(marks.ask - marks.bid) / 2)
mid = mk.pivot_table(index=["trade_id", "day"], columns="leg", values="mid").groupby(level=0).ffill()
half = mk.pivot_table(index=["trade_id", "day"], columns="leg", values="half").groupby(level=0).ffill()
net = mid["short"] - mid["long"]; nh = half["short"] + half["long"]
legs_per_day = marks.groupby(["trade_id", "day"])["leg"].nunique()
def tape_for(t, p, h, entry_at_signal=False, exit_by_position=False):
    e = int(t.signal_day) if entry_at_signal else int(p.index[p.index > t.signal_day][0])
    if exit_by_position:
        traj = p.loc[e:]; x = int(traj.index[min(HOLD, len(traj) - 1)])
    else:
        x = int(min(e + HOLD, calendar[-1]))
    pp = p.reindex(range(e, x + 1)).ffill(); hh = h.reindex(range(e, x + 1)).ffill()
    d = -1.0 * pp.diff().fillna(0.0); d.loc[e] -= hh.loc[e]; d.loc[x] -= hh.loc[x]
    return e, x, d, hh
ref = {}; ref_daily = {}; ref10 = {}; ref2 = {}; ledger = []
one_leg_any = {}
for _, t in trades.iterrows():
    p0 = net.loc[t.trade_id]; after = p0.index[p0.index > t.signal_day]
    if len(after) == 0: one_leg_any[int(t.trade_id)] = 0; continue
    e0 = int(after[0]); x0 = int(min(e0 + HOLD, calendar[-1]))
    # a leg marked at zero moves the z-score (lookback) as well as the P&L (hold): look at both windows
    one_leg_any[int(t.trade_id)] = int((legs_per_day.loc[t.trade_id].reindex(range(int(t.signal_day) - LB, x0 + 1)).fillna(2) < 2).sum())
for _, t in trades.iterrows():
    p = net.loc[t.trade_id]; h = nh.loc[t.trade_id]
    w = p.loc[:t.signal_day].tail(LB)
    if len(w) < LB or w.std() == 0 or not ((p.loc[t.signal_day] - w.mean()) / w.std() > Z): continue
    e, x, d, hh = tape_for(t, p, h)
    one_leg = int((legs_per_day.loc[t.trade_id].reindex(range(e, x + 1)).fillna(2) < 2).sum())
    ref[int(t.trade_id)] = {"net": float(d.sum()), "gross": float(d.sum() + hh.loc[e] + hh.loc[x]), "entry": e, "exit": x, "one_leg_days": one_leg}
    ref_daily[int(t.trade_id)] = int((d != 0).sum()); ledger.append(d)
    e10, x10, d10, _ = tape_for(t, p, h, entry_at_signal=True); ref10[int(t.trade_id)] = float(d10.sum())
    e2, x2, d2, _ = tape_for(t, p, h, exit_by_position=True); ref2[int(t.trade_id)] = float(d2.sum())
ref_book = pd.concat(ledger, axis=1).sum(axis=1, min_count=1).reindex(calendar, fill_value=0.0) if ledger else pd.Series(0.0, index=calendar)
out = {"n_tape": int(len(tape)), "n_ref": len(ref),
       "tape": {int(r.trade_id): {"net": float(r.net), "gross": float(r.gross)} for r in tape.itertuples()},
       "ref": ref, "ref_active_days": ref_daily, "ref10": ref10, "ref2": ref2, "one_leg_any": one_leg_any,
       "book_len": int(len(book)), "cal_len": int(len(calendar)), "book_nonzero": int((book != 0).sum()),
       "sharpe": float(sharpe), "ref_sharpe": float(ref_book.mean() / ref_book.std() * np.sqrt(252)) if ref_book.std() > 0 else None,
       "book_matches_ref": bool(len(book) == len(calendar) and np.allclose(book.reindex(calendar).fillna(0.0).to_numpy(), ref_book.to_numpy(), atol=1e-6))}
sr = spot.reindex(book.index).pct_change().fillna(0.0)
out["beta"] = float(np.cov(book, sr)[0, 1] / sr.var()) if len(book) > 2 and sr.var() > 0 else None
out["beta_share"] = float(np.cov(book, out["beta"] * sr)[0, 1] / book.var()) if out["beta"] is not None and book.var() > 0 else None
print(json.dumps(out))
'''


def probe(repo: str | Path, timeout: int = 180) -> dict:
    r = subprocess.run([sys.executable, "-c", _PROBE], cwd=repo, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"crash": (r.stderr or r.stdout)[-300:]}


def runtime_findings(repo: str | Path, tol: float = 1e-6) -> list[Finding]:
    repo = Path(repo)
    p = probe(repo)
    if "crash" in p:
        return [Finding(UNCLASSIFIED, "backtest.py", None, f"run() failed: {p['crash']}")]
    found: list[Finding] = []
    readme = (repo / "README.md").read_text().lower() if (repo / "README.md").exists() else ""

    tape = {int(k): v for k, v in p["tape"].items()}
    ref = {int(k): v for k, v in p["ref"].items()}
    ref10 = {int(k): v for k, v in p["ref10"].items()}
    ref2 = {int(k): v for k, v in p["ref2"].items()}
    one_leg_any = {int(k): v for k, v in p["one_leg_any"].items()}
    if p["book_len"] != p["cal_len"]:
        found.append(Finding(6, "backtest.py", None, f"book has {p['book_len']} days, calendar has {p['cal_len']}"))
    elif p["book_matches_ref"] and p["ref_sharpe"] is not None and abs(p["sharpe"] - p["ref_sharpe"]) > 1e-6:
        found.append(Finding(6, "backtest.py", None, f"daily book matches the reference but Sharpe {p['sharpe']:.4f} != {p['ref_sharpe']:.4f} from the same days"))
    if any(v["net"] > v["gross"] + tol for v in tape.values()):
        found.append(Finding(7, "backtest.py", None, "net > gross on at least one trade"))
    elif tape and all(abs(v["net"] - v["gross"]) < tol for v in tape.values()):
        found.append(Finding(5, "backtest.py", None, "net == gross on every trade: no spread paid"))
    missing = sorted(set(ref) - set(tape))
    extra = sorted(set(tape) - set(ref))
    if extra and all(one_leg_any.get(t, 0) > 0 for t in extra):
        found.append(Finding(4, "backtest.py", None,
                             f"{len(extra)} trades on the tape have no signal in their own quotes and every one has a one-leg day: the absent leg was marked at a price and moved the signal"))
    elif extra:
        found.append(Finding(9, "backtest.py", None,
                             f"{len(extra)} trades on the tape have no signal in their own mark series: the signal was computed on another instrument's quotes"))
    elif missing:
        cal_end = p["cal_len"] - 1
        near_end = all(ref[t]["entry"] + 10 >= cal_end for t in missing)
        one_leg = all(ref[t]["one_leg_days"] > 0 for t in missing)
        if near_end:
            cls, why = 8, "every missing trade would have resolved after the calendar end: unresolved candidates dropped"
        elif one_leg:
            cls, why = 4, "every missing trade has a day with one leg unquoted: the absent leg was marked at a price (zero) and the signal moved"
        else:
            cls, why = 3, "missing trades sit inside the sample, not at its end: dropped at a period boundary"
        found.append(Finding(cls, "backtest.py", None, f"{p['n_tape']} trades vs {p['n_ref']} signals; {why}"))
    # lumping: book has far fewer active days than the reference trades' active days imply
    ref_days = sum(p["ref_active_days"].values())
    if ref_days and p["book_nonzero"] < 0.35 * ref_days:
        found.append(Finding(1, "backtest.py", None, f"only {p['book_nonzero']} non-zero P&L days for {ref_days} trade-days of marks"))
    # neutrality claim vs measured beta
    claims_neutral = any(k in readme for k in ("market-neutral", "delta-hedged", "dollar-neutral", "no directional", "uncorrelated with the index"))
    if claims_neutral and p.get("beta_share") is not None and p["beta_share"] > 0.02:
        found.append(Finding(12, "README.md", None, f"README claims neutrality; beta share of P&L variance {p['beta_share']:.1%}"))
    # per-trade deviation from the reference tape: test the semantic hypotheses we can, else report unclassified
    common = sorted(set(tape) & set(ref))
    dev = [t for t in common if abs(tape[t]["net"] - ref[t]["net"]) > 1e-6 * max(1.0, abs(ref[t]["net"]))]
    already = {f.bug_class for f in found}
    if not dev and common and not p["book_matches_ref"] and p["book_len"] == p["cal_len"] and not already:
        found.append(Finding(11, "backtest.py", None,
                             "every trade's total matches the reference but the daily book does not: returns were differenced across gaps"))
    if dev and not already & {3, 4, 5, 7, 8, 9}:
        if all(abs(tape[t]["net"] - ref10[t]) < 1e-6 * max(1.0, abs(ref10[t])) for t in dev):
            found.append(Finding(10, "backtest.py", None,
                                 f"{len(dev)} trades match a tape that fills at the signal day's own quote, not the next one"))
        elif all(ref[t]["one_leg_days"] > 0 for t in dev) and any(ref[t]["one_leg_days"] == 0 for t in common):
            found.append(Finding(4, "backtest.py", None,
                                 f"the {len(dev)} deviating trades are exactly those with a one-leg day: the absent leg is marked at a price"))
        elif all(tape[t]["net"] - ref[t]["net"] > 0 for t in dev) and all(abs(tape[t]["gross"] - ref[t]["gross"]) < 1e-6 * max(1.0, abs(ref[t]["gross"])) for t in dev):
            found.append(Finding(7, "backtest.py", None,
                                 f"gross matches the reference but net is higher on all {len(dev)} deviating trades: a cost entered with the wrong sign"))
        elif all(abs(tape[t]["net"] - ref[t]["gross"]) < 1e-6 * max(1.0, abs(ref[t]["gross"])) for t in dev):
            found.append(Finding(5, "backtest.py", None,
                                 f"net on all {len(dev)} deviating trades equals the reference GROSS: fills at mid, no spread paid"))
        elif all(abs((tape[t]["net"] - ref[t]["net"]) - 2 * (ref[t]["gross"] - ref[t]["net"])) < 1e-6 * max(1.0, abs(ref[t]["gross"])) for t in dev):
            found.append(Finding(7, "backtest.py", None,
                                 f"net exceeds the reference by exactly twice the spread on all {len(dev)} deviating trades: the cost was credited, not charged"))
        elif all(abs(tape[t]["net"] - ref2[t]) < 1e-6 * max(1.0, abs(ref2[t])) for t in dev):
            found.append(Finding(2, "backtest.py", None,
                                 f"{len(dev)} trades match a tape whose exit is the HOLD-th quoted row, not the calendar day HOLD days later"))
        else:
            found.append(Finding(UNCLASSIFIED, "backtest.py", None,
                                 f"{len(dev)} of {len(common)} trades differ from the reference tape (class not identified at runtime)"))
    return found

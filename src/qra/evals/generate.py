"""Generate the seeded-bug task set: N synthetic repos per bug class plus clean controls.

Each task is a directory the agent may read (`backtest.py`, `trades.csv`, `marks.csv`,
`spot.csv`, `README.md`) and a label the agent may not (`labels.yaml`, kept outside the
repo directories). Data differs per variant (seed); the bug is the same textual mutation
of the clean template, so v0.1 measures "can the reviewer find this class when it is
expressed this way" — not robustness to paraphrase, which is v0.2's job.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .template import CLASS_NAMES, CLEAN, MUTATIONS, MUTATIONS_EXTRA

CALENDAR_DAYS = 3 * 252
N_TRADES = 90
SERIES_LEN = 60
LOOKBACK = 20
HOLD = 10

README_CLEAN = """# Spread mean-reversion strategy — backtest

Sells a two-leg spread when its net mid is more than one trailing-20-day standard
deviation rich, holds {hold} days, marks to market daily from per-leg quotes, and fills at
full cross-spread on entry and exit (Line 3). Reported Sharpe is on the full calendar.

Run: `python backtest.py`

Files: `trades.csv` (candidate signal days), `marks.csv` (per-leg bid/ask by day; quote
outages leave some days missing), `spot.csv` (underlying close).
"""

README_NEUTRAL = README_CLEAN + """
The strategy is delta-hedged and market-neutral by construction; the Sharpe above is alpha.
"""


def _ou(rng, n, level, kappa, sigma, x0):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (level - x[t - 1]) + sigma * rng.standard_normal()
    return x


def make_data(seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    days = np.arange(CALENDAR_DAYS)
    spot_ret = rng.normal(0.0003, 0.011, CALENDAR_DAYS)
    spot = 400.0 * np.exp(np.cumsum(spot_ret))
    pd.DataFrame({"day": days, "close": np.round(spot, 2)}).to_csv(out / "spot.csv", index=False)

    starts = np.sort(rng.integers(0, CALENDAR_DAYS - LOOKBACK - 2, N_TRADES))
    trades, rows = [], []
    for tid, start in enumerate(starts):
        n = min(SERIES_LEN, CALENDAR_DAYS - start)
        seg = spot_ret[start:start + n]
        # net spread mid: mean-reverting, and short-spread P&L loads negatively on spot
        # (a put-spread-like exposure) so beta is real and measurable
        level = rng.uniform(1.5, 3.0)
        noise = _ou(rng, n, 0.0, 0.25, 0.10, 0.0)
        net = level + noise - 6.0 * np.cumsum(seg - seg.mean())
        short_mid = net + rng.uniform(4.0, 7.0)
        long_mid = short_mid - net
        half_s = rng.uniform(0.03, 0.06)
        half_l = rng.uniform(0.02, 0.05)
        for i in range(n):
            day = start + i
            if rng.random() < 0.05:  # quote outage: whole day missing
                continue
            drop_leg = rng.random() < 0.03  # one leg missing that day
            for leg, mid, half in (("short", short_mid[i], half_s), ("long", long_mid[i], half_l)):
                if drop_leg and leg == "long":
                    continue
                rows.append({"trade_id": tid, "day": day, "leg": leg,
                             "bid": round(mid - half, 3), "ask": round(mid + half, 3)})
        quoted = sorted({r["day"] for r in rows if r["trade_id"] == tid and r["day"] >= start + LOOKBACK
                         and r["day"] < start + n - 1})
        signal_day = int(rng.choice(quoted)) if quoted else start + LOOKBACK
        trades.append({"trade_id": tid, "signal_day": signal_day})
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "marks.csv", index=False)


def mutate(bug_class: int | None) -> str:
    src = CLEAN
    if bug_class is None:
        return src
    for table in (MUTATIONS, MUTATIONS_EXTRA):
        if bug_class in table:
            old, new = table[bug_class]
            assert src.count(old) == 1, f"class {bug_class}: patch anchor not unique"
            src = src.replace(old, new)
    return src


def build(root: Path, variants: int, seed0: int = 1000) -> dict:
    repos = root / "repos"
    repos.mkdir(parents=True, exist_ok=True)
    labels = {}
    for v in range(variants):
        for bug_class in [None] + list(CLASS_NAMES):
            tag = "clean" if bug_class is None else f"bug{bug_class:02d}"
            task_id = f"{tag}_v{v}"
            d = repos / task_id
            d.mkdir(exist_ok=True)
            seed = seed0 + 100 * v + (bug_class or 0)
            make_data(seed, d)
            (d / "backtest.py").write_text(mutate(bug_class))
            readme = README_NEUTRAL if bug_class == 12 else README_CLEAN
            (d / "README.md").write_text(readme.format(hold=HOLD))
            digest = hashlib.sha256(b"".join(sorted(p.read_bytes() for p in d.iterdir()))).hexdigest()[:12]
            labels[task_id] = {
                "repo": str(d.relative_to(root)),
                "seeded": [] if bug_class is None else [bug_class],
                "seed": seed,
                "sha256": digest,
            }
    (root / "labels.yaml").write_text(yaml.safe_dump(labels, sort_keys=True))
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evals/tasks")
    ap.add_argument("--variants", type=int, default=1)
    a = ap.parse_args()
    labels = build(Path(a.root), a.variants)
    print(f"wrote {len(labels)} tasks under {a.root}")


if __name__ == "__main__":
    main()

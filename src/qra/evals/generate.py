"""Generate the seeded-bug task set: N synthetic repos per bug class plus clean controls.

Each task is a directory the agent may read (`backtest.py`, `trades.csv`, `marks.csv`,
`spot.csv`, `README.md`) and a label the agent may not (`labels.yaml`, kept outside the
repo directories). Data differs per variant (seed); the bug is the same textual mutation
of the clean template, so variant 0 is the
expression the regex baseline was written against; variants 1-2 (`variants.py`) plant the
same class through a different edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .template import CLASS_NAMES, CLEAN
from .variants import README_BASE, VARIANTS, apply

CALENDAR_DAYS = 3 * 252
N_TRADES = 90
SERIES_LEN = 60
LOOKBACK = 20
HOLD = 10

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
        quoted = sorted({r["day"] for r in rows if r["trade_id"] == tid and r["day"] >= start + LOOKBACK})
        quoted = quoted[:-1]  # a signal needs at least one later quote to fill against
        signal_day = int(rng.choice(quoted)) if quoted else start + LOOKBACK
        trades.append({"trade_id": tid, "signal_day": signal_day})
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "marks.csv", index=False)


def mutate(bug_class: int | None, variant: int = 0) -> tuple[str, str]:
    """(backtest.py source, README) for a bug class and variant; clean when bug_class is None."""
    if bug_class is None:
        return CLEAN, README_BASE.format(hold=HOLD)
    v = VARIANTS[bug_class][variant]
    return apply(CLEAN, v), (v.readme or README_BASE).format(hold=HOLD)


_RUN = """
import json, importlib.util, sys
spec = importlib.util.spec_from_file_location("bt", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tape, book, sharpe = m.run()
print(json.dumps({"nets": {int(r.trade_id): [round(float(r.net), 9), round(float(r.gross), 9)] for r in tape.itertuples()},
                  "book": [round(float(v), 9) for v in book.reindex(range(int(book.index.min()), int(book.index.max()) + 1), fill_value=0.0)],
                  "sharpe": round(float(sharpe), 9)}))
"""


def runtime_effect(repo: Path, readme: str) -> str:
    """What the mutation changes on this seed's data: 'numbers' (tape nets/gross, daily book or
    Sharpe differ from the clean template on the same data), 'claim' (only the README differs —
    the attribution class), or 'none' (a bug in form only: findable by reading, invisible to any
    runtime check; the scorer reports these separately)."""
    clean = repo / "_clean_reference.py"
    clean.write_text(CLEAN)
    try:
        outs = []
        for script in ("backtest.py", clean.name):
            r = subprocess.run([sys.executable, "-c", _RUN, script], cwd=repo, capture_output=True, text=True, timeout=180,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            outs.append(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"crash": True})
    finally:
        clean.unlink(missing_ok=True)
    if outs[0] != outs[1]:
        return "numbers"
    return "claim" if readme != README_BASE.format(hold=HOLD) else "none"


def build(root: Path, variants: int, seed0: int = 1000) -> dict:
    repos = root / "repos"
    repos.mkdir(parents=True, exist_ok=True)
    labels = {}
    for v in range(variants):
        for bug_class in [None] + list(CLASS_NAMES):
            if bug_class is not None and v >= len(VARIANTS[bug_class]):
                continue
            tag = "clean" if bug_class is None else f"bug{bug_class:02d}"
            task_id = f"{tag}_v{v}"
            d = repos / task_id
            d.mkdir(exist_ok=True)
            seed = seed0 + 100 * v + (bug_class or 0)
            make_data(seed, d)
            source, readme = mutate(bug_class, v)
            (d / "backtest.py").write_text(source)
            (d / "README.md").write_text(readme)
            digest = hashlib.sha256(b"".join(sorted(p.read_bytes() for p in d.iterdir() if p.is_file()))).hexdigest()[:12]
            labels[task_id] = {
                "repo": str(d.relative_to(root)),
                "seeded": [] if bug_class is None else [bug_class],
                "effect": "none" if bug_class is None else runtime_effect(d, readme),
                "variant": v,
                "note": "" if bug_class is None else VARIANTS[bug_class][v].note,
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

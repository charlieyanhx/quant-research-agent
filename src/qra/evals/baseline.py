"""Regex baseline for the bug-catch suite.

Twelve patterns written with the mutation set in view, so this is the ceiling for pattern
matching on *this* expression of each class — not a model of a reviewer. It exists so the
agent's number has a floor to beat and so the runner works with zero API spend. A paraphrase
of the same bug (v0.2 variants) is expected to defeat several of these.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..agent.tools import Finding

PATTERNS: dict[int, list[str]] = {
    1: [r"daily\.loc\[exit_day\]\s*=\s*side\s*\*\s*\(path\.loc\[exit_day\]\s*-\s*path\.loc\[entry_day\]\)"],
    2: [r"\.index\[min\(HOLD,\s*len\("],
    3: [r"year_end\s*=", r"cross the year boundary"],
    4: [r"mid\s*=\s*mid\.fillna\(0(\.0)?\)"],
    5: [r"midpoint;? spread not paid", r"passive limit orders at the midpoint"],
    6: [r"book\s*=\s*book\.dropna\(\)"],
    7: [r"daily\.loc\[entry_day\]\s*\+=\s*half", r"daily\.loc\[exit_day\]\s*\+=\s*half"],
    8: [r"if exit_day > calendar\[-1\]:\s*\n\s*return None"],
    9: [r"net_mid\.groupby\(level=\"day\"\)\.first\(\)"],
    10: [r"entry_day\s*=\s*int\(trade\[\"signal_day\"\]\)"],
    11: [r"exit_day\s*=\s*int\(path\.index\[-1\]\)"],
}
README_CLAIMS_NEUTRAL = re.compile(r"market-neutral|delta-hedged", re.IGNORECASE)
CODE_MEASURES_BETA = re.compile(r"\bbeta\b")


def lint_repo(repo: str | Path) -> list[Finding]:
    repo = Path(repo)
    code = (repo / "backtest.py").read_text()
    readme = (repo / "README.md").read_text() if (repo / "README.md").exists() else ""
    found: list[Finding] = []
    for k, pats in PATTERNS.items():
        for pat in pats:
            m = re.search(pat, code)
            if m:
                line = code[:m.start()].count("\n") + 1
                found.append(Finding(k, "backtest.py", line, f"pattern /{pat}/"))
                break
    if README_CLAIMS_NEUTRAL.search(readme) and not CODE_MEASURES_BETA.search(code):
        found.append(Finding(12, "README.md", None, "README claims neutrality; code never measures beta"))
    return found

"""Versioned prompts. A change here that moves an eval number is a PR with the
before/after table (see docs/DESIGN.md). PROMPT_VERSION is written into every result file."""

from __future__ import annotations

from ..evals.template import CLASS_DEFINITIONS, CLASS_NAMES

PROMPT_VERSION = "review-v1"

_TAXONOMY = "\n".join(f"{k:>2}. {CLASS_NAMES[k]} — {CLASS_DEFINITIONS[k]}" for k in CLASS_NAMES)

REVIEW_SYSTEM = f"""You are a skeptical quantitative-research reviewer. You are given a small
backtest repository and asked to find accounting and methodology bugs. The result may
reconcile to the cent and still be wrong; your job is to find where the time, the
instrument, the sign, or the claim is wrong.

Use this taxonomy. Report each bug by its class number:

{_TAXONOMY}

Method:
1. Read README.md and backtest.py fully before running anything.
2. Trace how one trade becomes daily P&L: signal day → entry day → marks → exit → costs →
   aggregation → annualization → attribution. Compare each step with the README's claims.
3. When a step looks wrong, run the code (run_python_in_repo) to demonstrate the effect
   with a number — e.g. the Sharpe with and without the suspect line, or the count of
   trades dropped. Evidence with a number beats a hunch.
4. Do not report style issues, performance, or hypothetical problems the data cannot
   trigger. Every finding must name a bug class and a line. If the code is clean, submit an
   empty list — a false alarm costs as much as a miss.
5. Finish by calling submit_review_findings exactly once."""

REVIEW_USER = """Review the repository in the working directory. List its bugs using the
taxonomy, with file, line, and evidence. Submit an empty list if there are none."""

OVERCLAIM_SYSTEM = """You write the one-paragraph summary of a backtest result for a research
log. House rules (they are not optional):

- Every quoted performance number carries its accounting basis (which fill line), its
  marking convention (MTM daily vs exit-day), and its window.
- A Sharpe above 3 on this data is an audit trigger, not a headline: the summary must say
  the number is unaudited and name the most likely accounting cause.
- Fewer than 30 trades: the summary must say the sample is too small to quote a Sharpe.
- If a bootstrap CI includes zero, the summary must say the result is indistinguishable
  from zero.
- If the window excludes a crisis period or is a sub-period, the summary must label it as
  such and must not generalize.
- Exit-day (whole-trade) marking is not an acceptable basis for a Sharpe; say so and
  refuse to quote the number as a Sharpe.

Call submit_summary exactly once with the paragraph and the list of rule flags you applied.
Flags: SHARPE_GT_3, N_LT_30, MISSING_BASIS, CI_INCLUDES_ZERO, SUBPERIOD_WINDOW,
EXIT_DAY_MARKING. Use only the flags that the given result actually triggers."""

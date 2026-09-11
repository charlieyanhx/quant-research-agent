"""Eval runner. Writes evals/results/<suite>_<agent>_<model>_<sha>.json — per-task rows,
aggregates, model ID, prompt version, git SHA, cost. The README table is regenerated from
these files (`qra report`), never edited by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from .baseline import lint_repo
from .overclaim import expected_flags
from .scorers import aggregate_bugcatch, aggregate_overclaim, score_bugcatch, score_overclaim


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def run_bugcatch(tasks_root: Path, agent: str, model: str, limit: int | None, cost_cap: float,
                 transcripts: Path | None) -> dict:
    labels = yaml.safe_load((tasks_root / "labels.yaml").read_text())
    rows, total_cost = [], 0.0
    for i, (task_id, lab) in enumerate(sorted(labels.items())):
        if limit is not None and i >= limit:
            break
        repo = tasks_root / lab["repo"]
        t0 = time.time()
        if agent == "lint":
            findings = lint_repo(repo)
            meta = {"cost_usd": 0.0, "turns": 0, "submitted": True, "stop_reason": "rule"}
        elif agent == "claude":
            from ..agent.orchestrator import review_repo
            res = review_repo(repo, model=model, cost_cap_usd=cost_cap)
            findings = res.findings
            meta = {"cost_usd": res.cost_usd, "turns": res.turns, "submitted": res.submitted,
                    "stop_reason": res.stop_reason, "usage": asdict(res.usage)}
            if transcripts:
                transcripts.mkdir(parents=True, exist_ok=True)
                (transcripts / f"{task_id}.json").write_text(json.dumps(res.to_dict(), indent=1))
        else:
            raise SystemExit(f"unknown agent {agent}")
        found = sorted({f.bug_class for f in findings})
        sc = score_bugcatch(lab["seeded"], found)
        total_cost += meta["cost_usd"]
        rows.append({"task_id": task_id, "seeded": lab["seeded"], "variant": lab.get("variant", 0), "found": found,
                     "findings": [asdict(f) for f in findings],
                     "tp": sc.tp, "fp": sc.fp, "fn": sc.fn, "seconds": round(time.time() - t0, 1), **meta})
        print(f"{task_id:<12} seeded={lab['seeded']} found={found} tp={sc.tp} fp={sc.fp} fn={sc.fn} "
              f"${meta['cost_usd']:.2f}")
    return {"rows": rows, "aggregate": aggregate_bugcatch(rows), "total_cost_usd": total_cost}


def run_overclaim(tasks_root: Path, agent: str, model: str, limit: int | None) -> dict:
    tasks = yaml.safe_load((tasks_root / "overclaim.yaml").read_text())
    rows, total_cost = [], 0.0
    for i, t in enumerate(tasks):
        if limit is not None and i >= limit:
            break
        if agent == "lint":
            got, summary, cost = expected_flags(t["result"]), "(rule)", 0.0
        elif agent == "claude":
            from ..agent.orchestrator import summarize_result
            out = summarize_result(t["result"], model=model)
            got, summary, cost = out["flags"], out["summary"], out["cost_usd"]
        else:
            raise SystemExit(f"unknown agent {agent}")
        sc = score_overclaim(t["expected"], got)
        total_cost += cost
        rows.append({"task_id": t["id"], "expected": t["expected"], "got": got, "score": sc,
                     "summary": summary, "cost_usd": cost})
        print(f"{t['id']:<18} expected={t['expected']} got={got} exact={sc['exact']}")
    return {"rows": rows, "aggregate": aggregate_overclaim(rows), "total_cost_usd": total_cost}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["bugcatch", "overclaim"], default="bugcatch")
    ap.add_argument("--agent", choices=["lint", "claude"], default="lint")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--tasks", default="evals/tasks")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cost-cap", type=float, default=2.0, help="per-task $ cap for the claude agent")
    ap.add_argument("--run-cap", type=float, default=20.0, help="abort the run above this total $")
    ap.add_argument("--transcripts", action="store_true")
    a = ap.parse_args(argv)

    from ..agent.prompts import PROMPT_VERSION
    tasks_root, out = Path(a.tasks), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    tr = out / "transcripts" / f"{a.suite}_{a.model}_{sha}" if a.transcripts else None
    if a.suite == "bugcatch":
        res = run_bugcatch(tasks_root, a.agent, a.model, a.limit, a.cost_cap, tr)
    else:
        res = run_overclaim(tasks_root, a.agent, a.model, a.limit)
    if res["total_cost_usd"] > a.run_cap:
        print(f"WARNING: run cost ${res['total_cost_usd']:.2f} exceeded cap ${a.run_cap:.2f}")
    res.update(suite=a.suite, agent=a.agent, model=a.model if a.agent == "claude" else "n/a",
               prompt_version=PROMPT_VERSION, git_sha=sha, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    name = f"{a.suite}_{a.agent}_{res['model'].replace('/', '-')}_{sha}.json"
    (out / name).write_text(json.dumps(res, indent=1, default=float))
    agg = res["aggregate"]
    print(f"\nwrote {out / name}")
    print(json.dumps({k: v for k, v in agg.items() if k not in ('per_class', 'per_flag')}, indent=1))


if __name__ == "__main__":
    main()

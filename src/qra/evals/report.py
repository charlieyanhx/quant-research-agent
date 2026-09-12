"""Regenerate the results tables in README.md from evals/results/*.json.

Between `<!-- results:start -->` and `<!-- results:end -->`. The latest file per
(suite, agent, model) is used; each row carries its git SHA and prompt version so a number
can always be traced to the run that produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .template import CLASS_NAMES

START, END = "<!-- results:start -->", "<!-- results:end -->"


def _fmt(x) -> str:
    return "—" if x is None or x != x else f"{x:.0%}"


def latest_runs(results: Path) -> dict[tuple, dict]:
    runs = {}
    for p in sorted(results.glob("*.json")):
        r = json.loads(p.read_text())
        runs[(r["suite"], r["agent"], r["model"])] = r
    return runs


def render(results: Path) -> str:
    runs = latest_runs(results)
    bug = {k: v for k, v in runs.items() if k[0] == "bugcatch"}
    over = {k: v for k, v in runs.items() if k[0] == "overclaim"}
    out = [START, "", "### Bug-catch suite — recall by class", ""]
    if bug:
        cols = [f"{k[1]}{'' if k[2] == 'n/a' else ' · ' + k[2]}" for k in bug]
        out.append("| # | class | " + " | ".join(cols) + " |")
        out.append("|---|---|" + "|".join("---" for _ in cols) + "|")
        for c in CLASS_NAMES:
            cells = [_fmt(r["aggregate"]["per_class"][str(c)]["recall"]) for r in bug.values()]
            out.append(f"| {c} | {CLASS_NAMES[c]} | " + " | ".join(cells) + " |")
        out.append("| | **recall (all seeded)** | " + " | ".join(_fmt(r["aggregate"]["recall"]) for r in bug.values()) + " |")
        out.append("| | recall by expression v0 / v1 / v2 | " + " | ".join(
            " / ".join(_fmt(x) for x in r["aggregate"].get("recall_by_variant", {}).values()) for r in bug.values()) + " |")
        out.append("| | **recall on active tasks** (bug changes a number or a claim) | " + " | ".join(_fmt(r["aggregate"].get("recall_active")) for r in bug.values()) + " |")
        out.append("| | detected at all on active tasks (any finding, incl. unclassified) | " + " | ".join(_fmt(r["aggregate"].get("detection_active")) for r in bug.values()) + " |")
        out.append("| | **precision** (labelled findings) | " + " | ".join(_fmt(r["aggregate"]["precision"]) for r in bug.values()) + " |")
        out.append("| | **false alarms on clean controls** | " + " | ".join(_fmt(r["aggregate"]["control_false_alarm_rate"]) for r in bug.values()) + " |")
        out.append("| | tasks / cost / sha / prompt | " + " | ".join(
            f"{r['aggregate']['n_tasks']} / ${r['total_cost_usd']:.2f} / {r['git_sha']} / {r['prompt_version']}" for r in bug.values()) + " |")
        dormant = next((r["aggregate"].get("dormant_tasks") for r in bug.values() if r["aggregate"].get("dormant_tasks")), None)
        if dormant:
            out.append("")
            out.append(f"Dormant seeded tasks (the mutation changes no number and no claim on that seed's data, so only reading can find it): {', '.join(dormant)}.")
    else:
        out.append("_no bug-catch runs yet_")
    out += ["", "### Overclaim-refusal suite", ""]
    if over:
        out.append("| agent | exact flag match | over-flags per task | clean cases wrongly flagged | tasks / cost / sha |")
        out.append("|---|---|---|---|---|")
        for k, r in over.items():
            a = r["aggregate"]
            out.append(f"| {k[1]}{'' if k[2] == 'n/a' else ' · ' + k[2]} | {_fmt(a['exact_rate'])} | "
                       f"{a['over_flag_per_task']:.2f} | {_fmt(a['clean_over_flag_rate'])} | "
                       f"{a['n_tasks']} / ${r['total_cost_usd']:.2f} / {r['git_sha']} |")
    else:
        out.append("_no overclaim runs yet_")
    out += ["", END]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="evals/results")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args(argv)
    readme = Path(a.readme)
    text = readme.read_text()
    block = render(Path(a.results))
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        _, post = rest.split(END, 1)
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    readme.write_text(text)
    print(block)


if __name__ == "__main__":
    main()

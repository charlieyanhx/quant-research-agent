"""`qra` command line: generate tasks, run evals, regenerate the README table, review a repo."""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: qra {generate|run|report|review} ...")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "generate":
        from pathlib import Path

        from .evals.generate import main as gen
        from .evals.overclaim import build as build_overclaim
        sys.argv = ["generate", *rest]
        gen()
        build_overclaim(Path("evals/tasks"))
        print("wrote evals/tasks/overclaim.yaml")
    elif cmd == "run":
        from .evals.run import main as run
        run(rest)
    elif cmd == "report":
        from .evals.report import main as report
        report(rest)
    elif cmd == "review":
        from .agent.orchestrator import review_repo
        if not rest:
            raise SystemExit("usage: qra review <repo_dir> [--model M]")
        model = rest[rest.index("--model") + 1] if "--model" in rest else "claude-opus-5"
        res = review_repo(rest[0], model=model)
        print(json.dumps(res.to_dict(), indent=1))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()

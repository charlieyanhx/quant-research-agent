"""FastMCP server exposing the same tools to Claude Code / Codex / any MCP client.

    pip install -e ".[mcp]"
    qra-mcp --repo path/to/backtest_repo      (or: python -m qra.mcp.server --repo ...)

The repository is fixed at startup so the server, like the agent, is jailed to one tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agent.tools import ToolState, list_files, read_file, run_python
from ..evals.baseline import lint_repo


def build(repo: Path):
    from fastmcp import FastMCP

    state = ToolState(repo=repo.resolve())
    mcp = FastMCP("quant-research-agent")

    @mcp.tool
    def list_repo_files() -> str:
        """List readable files in the backtest repository."""
        return list_files(state)

    @mcp.tool
    def read_repo_file(path: str, start: int = 1, end: int | None = None) -> str:
        """Read a repository file with line numbers."""
        return read_file(state, path, start, end)

    @mcp.tool
    def run_python_in_repo(code: str) -> str:
        """Run Python in the repository directory (60 s, no network)."""
        return run_python(state, code)

    @mcp.tool
    def regex_baseline() -> list[dict]:
        """The twelve-pattern regex baseline's findings for this repository."""
        return [f.__dict__ for f in lint_repo(state.repo)]

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    build(Path(a.repo)).run()


if __name__ == "__main__":
    main()

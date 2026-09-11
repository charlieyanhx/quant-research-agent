"""Typed tools the agent can call, jailed to one repository directory.

Each tool is a plain function so the same implementation is exposed three ways: to the
Anthropic tool runner (`make_tools`), over MCP (`qra.mcp.server`), and to tests. Nothing
here touches the network; `run_python` executes in a subprocess with a wall-clock limit and
an address-space cap, in the repo directory, with a scrubbed environment.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAX_READ_CHARS = 40_000
RUN_TIMEOUT_S = 60
RUN_MEMORY_BYTES = 2 * 1024**3
ALLOWED_SUFFIXES = {".py", ".md", ".csv", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg"}


@dataclass
class Finding:
    bug_class: int
    file: str
    line: int | None
    evidence: str


@dataclass
class ToolState:
    repo: Path
    findings: list[Finding] = field(default_factory=list)
    submitted: bool = False
    calls: list[dict] = field(default_factory=list)


def _jail(state: ToolState, path: str) -> Path:
    p = (state.repo / path).resolve()
    if state.repo.resolve() not in p.parents and p != state.repo.resolve():
        raise ValueError(f"path escapes the repository: {path}")
    return p


def list_files(state: ToolState) -> str:
    rows = []
    for p in sorted(state.repo.rglob("*")):
        if p.is_file() and p.suffix in ALLOWED_SUFFIXES and ".venv" not in p.parts:
            rows.append(f"{p.relative_to(state.repo)}\t{p.stat().st_size} bytes")
    return "\n".join(rows) or "(no files)"


def read_file(state: ToolState, path: str, start: int = 1, end: int | None = None) -> str:
    p = _jail(state, path)
    if not p.is_file():
        return f"error: {path} is not a file"
    if p.suffix not in ALLOWED_SUFFIXES:
        return f"error: {path} has a non-text suffix"
    lines = p.read_text(errors="replace").splitlines()
    end = len(lines) if end is None else min(end, len(lines))
    chunk = lines[max(start, 1) - 1:end]
    text = "\n".join(f"{i}\t{ln}" for i, ln in enumerate(chunk, start=max(start, 1)))
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n… truncated; {len(lines)} lines total, use start/end"
    return text


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (RUN_MEMORY_BYTES, RUN_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (RUN_TIMEOUT_S, RUN_TIMEOUT_S))


def run_python(state: ToolState, code: str) -> str:
    """Run a Python snippet in the repository directory. stdout+stderr, truncated."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": "0", "HOME": str(state.repo)}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=state.repo, env=env, capture_output=True,
            text=True, timeout=RUN_TIMEOUT_S, preexec_fn=_limits if sys.platform != "darwin" else None,
        )
        out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
        out += f"\n[exit {proc.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timed out after {RUN_TIMEOUT_S}s]"
    return out[-MAX_READ_CHARS:]


def submit_findings(state: ToolState, findings_json: str) -> str:
    """Record the final list of findings. Replaces any earlier submission."""
    try:
        raw = json.loads(findings_json)
    except json.JSONDecodeError as exc:
        return f"error: findings_json is not valid JSON ({exc}); submit again"
    if not isinstance(raw, list):
        return "error: findings_json must be a JSON list (possibly empty)"
    parsed: list[Finding] = []
    for i, f in enumerate(raw):
        try:
            parsed.append(Finding(int(f["bug_class"]), str(f.get("file", "")),
                                  int(f["line"]) if f.get("line") is not None else None,
                                  str(f.get("evidence", ""))))
        except (KeyError, TypeError, ValueError) as exc:
            return f"error: finding {i} malformed ({exc}); each needs bug_class, file, line, evidence"
    state.findings = parsed
    state.submitted = True
    return f"recorded {len(parsed)} finding(s)"


def make_tools(state: ToolState) -> list:
    """Bind the tools to one repo and wrap them for the Anthropic tool runner."""
    from anthropic import beta_tool

    def _log(name, **kw):
        state.calls.append({"tool": name, **{k: (v if len(str(v)) < 200 else str(v)[:200] + "…")
                                            for k, v in kw.items()}})

    @beta_tool
    def list_repo_files() -> str:
        """List every readable file in the repository under review with its size."""
        _log("list_repo_files")
        return list_files(state)

    @beta_tool
    def read_repo_file(path: str, start: int = 1, end: int | None = None) -> str:
        """Read a file from the repository, with line numbers.

        Args:
            path: Path relative to the repository root, e.g. backtest.py.
            start: First line to return (1-based).
            end: Last line to return (inclusive); omit for end of file.
        """
        _log("read_repo_file", path=path, start=start, end=end)
        return read_file(state, path, start, end)

    @beta_tool
    def run_python_in_repo(code: str) -> str:
        """Execute Python code with the repository as the working directory (60 s limit,
        no network). Use it to run the backtest, print intermediate frames, or test a
        hypothesis numerically. Returns stdout and stderr.

        Args:
            code: Python source to execute.
        """
        _log("run_python_in_repo", code=code)
        return run_python(state, code)

    @beta_tool
    def submit_review_findings(findings_json: str) -> str:
        """Submit the final review as a JSON list. Call exactly once, last. An empty list
        means the code is clean. Each item: {"bug_class": <1-12>, "file": "<path>",
        "line": <int or null>, "evidence": "<one sentence citing the code and, if you ran
        it, the number that demonstrates the effect>"}.

        Args:
            findings_json: JSON-encoded list of findings.
        """
        _log("submit_review_findings", findings_json=findings_json)
        return submit_findings(state, findings_json)

    return [list_repo_files, read_repo_file, run_python_in_repo, submit_review_findings]

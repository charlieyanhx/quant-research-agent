"""Every generated repo runs, prints a Sharpe, and the mutation actually changed the code."""

import subprocess
import sys

import pytest
import yaml

from qra.evals.generate import build, mutate
from qra.evals.template import CLASS_NAMES, CLEAN


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    root = tmp_path_factory.mktemp("tasks")
    labels = build(root, variants=1)
    return root, labels


def test_thirteen_tasks(tasks):
    _, labels = tasks
    assert len(labels) == 13
    assert sum(1 for v in labels.values() if not v["seeded"]) == 1


@pytest.mark.parametrize("bug_class", list(CLASS_NAMES))
def test_each_mutation_changes_code(bug_class):
    assert mutate(bug_class) != CLEAN


@pytest.mark.parametrize("bug_class", [None] + list(CLASS_NAMES))
def test_each_repo_runs_and_reports(tasks, bug_class):
    root, labels = tasks
    tag = "clean" if bug_class is None else f"bug{bug_class:02d}"
    repo = root / labels[f"{tag}_v0"]["repo"]
    proc = subprocess.run([sys.executable, "backtest.py"], cwd=repo, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "Sharpe" in proc.stdout


def test_labels_do_not_leak_into_repos(tasks):
    root, labels = tasks
    for lab in labels.values():
        for p in (root / lab["repo"]).iterdir():
            assert "seeded" not in p.read_text(errors="ignore")
            assert p.name != "labels.yaml"
    assert yaml.safe_load((root / "labels.yaml").read_text()) == labels

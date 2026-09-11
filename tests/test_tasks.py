"""Every generated repo runs, prints a Sharpe, and the mutation actually changed the code."""

import subprocess
import sys

import pytest
import yaml

from qra.evals.generate import build, mutate
from qra.evals.template import CLASS_NAMES, CLEAN
from qra.evals.variants import VARIANTS

N_VARIANTS = 3
ALL = [(None, v) for v in range(N_VARIANTS)] + [(k, v) for k in CLASS_NAMES for v in range(len(VARIANTS[k]))]


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    root = tmp_path_factory.mktemp("tasks")
    labels = build(root, variants=N_VARIANTS)
    return root, labels


def test_task_count(tasks):
    _, labels = tasks
    assert len(labels) == 39
    assert sum(1 for v in labels.values() if not v["seeded"]) == 3


@pytest.mark.parametrize("bug_class,variant", [t for t in ALL if t[0] is not None])
def test_each_mutation_changes_code(bug_class, variant):
    src, _ = mutate(bug_class, variant)
    assert src != CLEAN
    others = {mutate(bug_class, w)[0] for w in range(len(VARIANTS[bug_class])) if w != variant}
    assert src not in others, "variants of one class must differ from each other"


@pytest.mark.parametrize("bug_class,variant", ALL)
def test_each_repo_runs_and_reports(tasks, bug_class, variant):
    root, labels = tasks
    tag = "clean" if bug_class is None else f"bug{bug_class:02d}"
    repo = root / labels[f"{tag}_v{variant}"]["repo"]
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

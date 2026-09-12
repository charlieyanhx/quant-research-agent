"""Runtime-invariant baseline: zero false alarms on clean repos, and the invariants identify the
classes they are designed for on the committed task set. Dormant tasks are excluded by label."""

from pathlib import Path

import pytest
import yaml

from qra.evals.runtime import UNCLASSIFIED, runtime_findings

ROOT = Path(__file__).parent.parent / "evals" / "tasks"
LABELS = yaml.safe_load((ROOT / "labels.yaml").read_text())


@pytest.mark.parametrize("task_id", [k for k, v in LABELS.items() if not v["seeded"]])
def test_clean_repos_raise_no_finding(task_id):
    assert runtime_findings(ROOT / LABELS[task_id]["repo"]) == []


@pytest.mark.parametrize("task_id", ["bug01_v0", "bug04_v0", "bug05_v1", "bug06_v0", "bug06_v2", "bug07_v1", "bug09_v0",
                                     "bug10_v0", "bug11_v0", "bug12_v0", "bug02_v0"])
def test_invariants_identify_their_class(task_id):
    found = {f.bug_class for f in runtime_findings(ROOT / LABELS[task_id]["repo"])}
    assert set(LABELS[task_id]["seeded"]) <= found, found


def test_every_active_task_is_at_least_detected():
    for task_id, lab in LABELS.items():
        if lab["seeded"] and lab["effect"] != "none":
            assert runtime_findings(ROOT / lab["repo"]), task_id


def test_unclassified_is_never_reported_alongside_a_class():
    for task_id, lab in LABELS.items():
        found = {f.bug_class for f in runtime_findings(ROOT / lab["repo"])}
        assert not (UNCLASSIFIED in found and len(found) > 1), (task_id, found)


def test_dormant_labels_are_the_documented_four():
    assert sorted(k for k, v in LABELS.items() if v["seeded"] and v["effect"] == "none") == \
        ["bug03_v0", "bug08_v2", "bug11_v1", "bug11_v2"]

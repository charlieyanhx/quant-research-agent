import math

from qra.evals.baseline import lint_repo
from qra.evals.generate import build
from qra.evals.overclaim import CASES, expected_flags
from qra.evals.scorers import aggregate_bugcatch, score_bugcatch, score_overclaim


def test_bugcatch_scoring_identities():
    s = score_bugcatch([3], [3, 7])
    assert (s.tp, s.fp, s.fn) == (1, 1, 0) and s.recall == 1.0 and s.precision == 0.5
    c = score_bugcatch([], [])
    assert c.is_control and not c.false_alarm and math.isnan(c.recall)
    assert score_bugcatch([], [2]).false_alarm


def test_aggregate_reports_control_false_alarms():
    rows = [{"seeded": [1], "found": [1]}, {"seeded": [], "found": [5]}, {"seeded": [], "found": []}]
    a = aggregate_bugcatch(rows)
    assert a["recall"] == 1.0 and a["precision"] == 0.5
    assert a["control_false_alarm_rate"] == 0.5 and a["n_controls"] == 2


def test_baseline_finds_each_seeded_class_and_nothing_on_clean(tmp_path):
    labels = build(tmp_path, variants=1)
    for task_id, lab in labels.items():
        found = sorted({f.bug_class for f in lint_repo(tmp_path / lab["repo"])})
        assert found == lab["seeded"], task_id


def test_overclaim_rules_and_boundaries():
    assert expected_flags({"sharpe": 3.0, "n_trades": 30, "basis": "L3", "marking": "MTM daily"}) == []
    assert expected_flags({"sharpe": 3.01, "n_trades": 29, "basis": "", "marking": "exit-day",
                           "sharpe_ci": [-0.1, 6], "window_note": "x"}) == [
        "SHARPE_GT_3", "N_LT_30", "MISSING_BASIS", "CI_INCLUDES_ZERO", "SUBPERIOD_WINDOW", "EXIT_DAY_MARKING"]
    assert any(not expected_flags({k: v for k, v in c.items() if k != "id"}) for c in CASES)
    sc = score_overclaim(["N_LT_30"], ["N_LT_30", "SHARPE_GT_3"])
    assert not sc["exact"] and sc["extra"] == ["SHARPE_GT_3"] and sc["recall"] == 1.0

import json

import pytest

from qra.agent.tools import ToolState, list_files, read_file, run_python, submit_findings


@pytest.fixture
def state(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nprint('hi')\n")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")
    return ToolState(repo=tmp_path)


def test_list_and_read(state):
    assert "a.py" in list_files(state) and "blob.bin" not in list_files(state)
    assert read_file(state, "a.py", 2, 2) == "2\tprint('hi')"


def test_jail(state):
    with pytest.raises(ValueError):
        read_file(state, "../../etc/passwd")


def test_run_python_in_repo(state):
    out = run_python(state, "import os; print(sorted(os.listdir('.')))")
    assert "a.py" in out and "[exit 0]" in out


def test_submit_findings_contract(state):
    assert "error" in submit_findings(state, "not json")
    assert "error" in submit_findings(state, json.dumps([{"file": "x"}]))
    assert submit_findings(state, "[]") == "recorded 0 finding(s)" and state.submitted
    msg = submit_findings(state, json.dumps([{"bug_class": 7, "file": "a.py", "line": 3, "evidence": "sign"}]))
    assert msg == "recorded 1 finding(s)" and state.findings[0].bug_class == 7

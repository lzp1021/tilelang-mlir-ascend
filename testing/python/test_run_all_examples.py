import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import run_all


def test_assign_devices_uses_only_npu_zero_when_sequential():
    assert run_all.assign_devices(["a.py", "b.py", "c.py"], 16, True) == [
        ("a.py", 0),
        ("b.py", 0),
        ("c.py", 0),
    ]


def test_assign_devices_round_robins_when_parallel():
    assert run_all.assign_devices(["a.py", "b.py", "c.py"], 2, False) == [
        ("a.py", 0),
        ("b.py", 1),
        ("c.py", 0),
    ]


def test_run_example_collects_parallel_coverage_in_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    examples_dir = tmp_path / "examples"
    example_path = examples_dir / "vector_add.py"
    recorded = {}

    def record_subprocess(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_all.subprocess, "run", record_subprocess)

    result = run_all.run_example(
        "vector_add.py", examples_dir, device_id=0, coverage=True
    )

    assert result.status == "PASSED"
    assert recorded["command"] == [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        f"--rcfile={tmp_path / '.coveragerc'}",
        str(example_path),
    ]
    assert recorded["kwargs"]["env"]["COVERAGE_FILE"] == str(tmp_path / ".coverage")

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_autotune_coverage as autotune_runner
from scripts.run_autotune_case import run_target
from scripts.run_autotune_coverage import (
    discover_autotune_paths,
    run_autotune_script,
    run_autotune_scripts,
)


def test_discover_autotune_paths_returns_every_nested_python_script(tmp_path: Path):
    autotune_root = tmp_path / "testing" / "autotune"
    (autotune_root / "anneal").mkdir(parents=True)
    (autotune_root / "AscendC").mkdir()
    (autotune_root / "anneal" / "search.py").write_text("pass\n")
    (autotune_root / "AscendC" / "kernel.py").write_text("pass\n")
    (autotune_root / "__init__.py").write_text("")
    (autotune_root / "notes.txt").write_text("")

    assert discover_autotune_paths(tmp_path) == [
        "testing/autotune/AscendC/kernel.py",
        "testing/autotune/anneal/search.py",
    ]


def test_run_target_pins_and_redirects_npu_device_zero(tmp_path: Path, monkeypatch):
    selected_devices = []
    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(set_device=lambda device: selected_devices.append(device))
    )
    target = tmp_path / "autotune_case.py"
    target.write_text("import torch\ntorch.npu.set_device(15)\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    run_target(target)

    assert selected_devices == [0, 0]


def test_run_target_hides_launcher_arguments_from_target_argparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda _device: None))
    target = tmp_path / "argparse_case.py"
    target.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--size', type=int, default=16)\n"
        "assert parser.parse_args().size == 16\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "argv", ["run_autotune_case.py", str(target)])

    run_target(target)


def test_run_autotune_script_collects_parallel_coverage_on_device_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded = {}

    def record_subprocess(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(autotune_runner.subprocess, "run", record_subprocess)

    result = run_autotune_script(
        "testing/autotune/anneal/search.py", tmp_path, timeout=1800
    )

    assert result.status == "PASSED"
    assert recorded["command"] == [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        f"--rcfile={tmp_path / '.coveragerc'}",
        str(tmp_path / "scripts" / "run_autotune_case.py"),
        str(tmp_path / "testing" / "autotune" / "anneal" / "search.py"),
    ]
    assert recorded["kwargs"]["cwd"] == tmp_path
    assert recorded["kwargs"]["timeout"] == 1800
    assert recorded["kwargs"]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    assert recorded["kwargs"]["env"]["COVERAGE_FILE"] == str(tmp_path / ".coverage")


def test_run_autotune_scripts_continues_after_failure_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    return_codes = iter([3, 0])
    executed = []

    def record_subprocess(command, **kwargs):
        executed.append(command[-1])
        return SimpleNamespace(
            returncode=next(return_codes), stdout="", stderr="failed"
        )

    monkeypatch.setattr(autotune_runner.subprocess, "run", record_subprocess)

    result = run_autotune_scripts(
        tmp_path,
        ["testing/autotune/first.py", "testing/autotune/second.py"],
        timeout=1800,
    )

    assert result == 1
    assert executed == [
        str(tmp_path / "testing" / "autotune" / "first.py"),
        str(tmp_path / "testing" / "autotune" / "second.py"),
    ]

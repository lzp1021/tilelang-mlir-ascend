import sys
from pathlib import Path

import pytest

from scripts.run_python_coverage import (
    build_pytest_command,
    discover_test_paths,
    prepare_output_directories,
    run_command,
)


def test_discover_test_paths_returns_sorted_operation_directories(tmp_path: Path):
    (tmp_path / "testing" / "npuir" / "zeta_ops").mkdir(parents=True)
    (tmp_path / "testing" / "npuir" / "alpha_ops").mkdir()
    (tmp_path / "testing" / "npuir" / "broken").mkdir()

    assert discover_test_paths(tmp_path) == [
        "testing/npuir/alpha_ops",
        "testing/npuir/zeta_ops",
    ]


def test_discover_test_paths_rejects_repository_without_operation_tests(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"testing[/\\]npuir[/\\]\*_ops"):
        discover_test_paths(tmp_path)


def test_build_pytest_command_adds_reports_and_preserves_arguments(tmp_path: Path):
    (tmp_path / "testing" / "npuir" / "memory_ops").mkdir(parents=True)

    command = build_pytest_command(
        tmp_path,
        ["--op=copy", "--mode=Developer", "--npu-device=0", "-v"],
    )

    assert command == [
        sys.executable,
        "-m",
        "pytest",
        "testing/npuir/memory_ops",
        "--cov=tilelang",
        "--cov-config=.coveragerc",
        "--cov-report=term-missing",
        "--cov-report=html:coverage/python-html",
        "--cov-report=xml:coverage/python.xml",
        "--html=testing/npuir/output/report.html",
        "--self-contained-html",
        "--junitxml=coverage/junit.xml",
        "--op=copy",
        "--mode=Developer",
        "--npu-device=0",
        "-v",
    ]


def test_run_command_returns_child_exit_code(tmp_path: Path):
    command = [sys.executable, "-c", "raise SystemExit(7)"]

    assert run_command(command, tmp_path) == 7


def test_prepare_output_directories_creates_report_parents(tmp_path: Path):
    prepare_output_directories(tmp_path)

    assert (tmp_path / "coverage").is_dir()
    assert (tmp_path / "testing" / "npuir" / "output").is_dir()

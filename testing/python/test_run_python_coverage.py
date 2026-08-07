import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

from scripts.run_python_coverage import (
    build_pytest_command,
    discover_test_paths,
    prepare_output_directories,
    run_command,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_coverage_policy_measures_tilelang_branches_and_shows_missing_lines():
    config = ConfigParser()
    assert config.read(REPO_ROOT / ".coveragerc")

    omitted_paths = config.get("run", "omit").split()
    assert config.getboolean("run", "branch")
    assert config.getboolean("run", "relative_files")
    assert config.get("run", "source").split() == ["tilelang"]
    assert "*/testing/*" not in omitted_paths
    assert config.getboolean("report", "show_missing")
    assert not config.has_option("report", "fail_under")


def test_coverage_outputs_are_ignored_and_pytest_cov_is_declared():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    requirements = (
        (REPO_ROOT / "requirements-test.txt").read_text(encoding="utf-8").splitlines()
    )

    assert "coverage/" in ignored
    assert "pytest-cov" in requirements


def test_npuir_workflow_runs_shared_coverage_command_and_uploads_reports():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci_npuir.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install pytest pytest-xdist pytest-html pytest-cov numpy" in workflow
    assert "python scripts/run_python_coverage.py -v" in workflow
    assert "path: |\n          testing/npuir/output/\n          coverage/" in workflow


def test_npuir_readme_documents_coverage_runner_and_outputs():
    readme = (REPO_ROOT / "testing" / "npuir" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/run_python_coverage.py" in readme
    assert "testing/npuir/output/report.html" in readme
    assert "coverage/python-html/index.html" in readme
    assert "coverage/python.xml" in readme

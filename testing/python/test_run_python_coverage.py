import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

import scripts.run_python_coverage as coverage_runner
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
        "--cov-append",
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
    stale_profile = tmp_path / "coverage" / "cpp-profraw" / "stale.profraw"
    stale_profile.parent.mkdir(parents=True)
    stale_profile.write_text("old", encoding="utf-8")
    stale_object = tmp_path / "coverage" / "cpp-objects" / "stale.so"
    stale_object.parent.mkdir(parents=True)
    stale_object.write_text("old", encoding="utf-8")
    stale_report = tmp_path / "coverage" / "coverage.xml"
    stale_report.write_text("old", encoding="utf-8")

    prepare_output_directories(tmp_path)

    assert (tmp_path / "coverage").is_dir()
    assert (tmp_path / "coverage" / "cpp-profraw").is_dir()
    assert not stale_profile.exists()
    assert not stale_object.exists()
    assert not stale_report.exists()
    assert (tmp_path / "testing" / "npuir" / "output").is_dir()


def test_coverage_pipeline_combines_examples_before_appending_pytest_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "testing" / "npuir" / "memory_ops").mkdir(parents=True)
    commands = []

    def record_command(command, cwd):
        assert cwd == tmp_path
        commands.append(list(command))
        return 0

    monkeypatch.setattr(coverage_runner, "run_command", record_command)

    assert coverage_runner.run_coverage_pipeline(tmp_path, ["-v"]) == 0
    assert commands == [
        [sys.executable, "-m", "coverage", "erase"],
        [
            sys.executable,
            "examples/run_all.py",
            "--sequential",
            "--coverage",
        ],
        [sys.executable, "scripts/run_autotune_coverage.py"],
        [sys.executable, "-m", "coverage", "combine"],
        [
            sys.executable,
            "-m",
            "pytest",
            "testing/npuir/memory_ops",
            "--cov=tilelang",
            "--cov-config=.coveragerc",
            "--cov-append",
            "--cov-report=term-missing",
            "--cov-report=html:coverage/python-html",
            "--cov-report=xml:coverage/python.xml",
            "--html=testing/npuir/output/report.html",
            "--self-contained-html",
            "--junitxml=coverage/junit.xml",
            "-v",
        ],
        [sys.executable, "scripts/filter_recent_coverage.py", "--months", "8"],
        [sys.executable, "scripts/generate_cpp_coverage.py", "--months", "8"],
        [sys.executable, "scripts/combine_coverage_reports.py", "--months", "8"],
    ]


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
    workflow = (REPO_ROOT / ".github" / "workflows" / "test_npuir_wheel.yml").read_text(
        encoding="utf-8"
    )

    assert "uv pip install pytest pytest-xdist pytest-html pytest-cov numpy" in workflow
    assert "fetch-depth: 0" in workflow
    assert "LLVM_PROFILE_FILE:" in workflow
    assert 'TILELANG_CPP_COVERAGE: "1"' in workflow
    assert "tilelang-npuir-cpp-coverage-py3.11-arm64" in workflow
    assert "python scripts/run_python_coverage.py -v -n 2" in workflow
    assert "python examples/run_all.py" not in workflow
    assert "timeout-minutes: 360" in workflow
    assert "path: |\n          testing/npuir/output/\n          coverage/" in workflow


def test_npuir_readme_documents_coverage_runner_and_outputs():
    readme = (REPO_ROOT / "testing" / "npuir" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/run_python_coverage.py" in readme
    assert "testing/npuir/output/report.html" in readme
    assert "coverage/python-html/index.html" in readme
    assert "coverage/python.xml" in readme
    assert "coverage/python-recent.xml" in readme
    assert "coverage/python-recent.html" in readme
    assert "coverage/index.html" in readme
    assert "coverage/coverage.xml" in readme
    assert "coverage/cpp-recent.xml" in readme
    assert "testing/autotune/**/*.py" in readme


def test_build_workflow_keeps_release_wheel_and_adds_instrumented_arm64_wheel():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build_tilelang_wheel.yml"
    ).read_text(encoding="utf-8")

    assert (
        "name: tilelang-npuir-py${{ env.PYTHON_VERSION }}-${{ matrix.arch }}"
        in workflow
    )
    assert (
        "name: tilelang-npuir-cpp-coverage-py${{ env.PYTHON_VERSION }}-arm64"
        in workflow
    )
    assert "-fprofile-instr-generate -fcoverage-mapping" in workflow
    assert "-DCMAKE_C_COMPILER=clang" in workflow
    assert "-DCMAKE_CXX_COMPILER=clang++" in workflow
    assert (
        "TILELANG_PREBUILT_LIB_DIR=${{ github.workspace }}/build-coverage" in workflow
    )

    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'os.environ.get("TILELANG_PREBUILT_LIB_DIR")' in setup_source


def test_runtime_npu_cpp_extension_supports_llvm_coverage():
    source = (REPO_ROOT / "tilelang" / "utils" / "npu_utils.py").read_text(
        encoding="utf-8"
    )

    assert 'os.environ.get("TILELANG_CPP_COVERAGE", "0")' in source
    assert '"-fprofile-instr-generate", "-fcoverage-mapping"' in source
    assert "TILELANG_CPP_COVERAGE_REPO_ROOT" in source
    assert "TILELANG_CPP_COVERAGE_OBJECT_DIR" in source

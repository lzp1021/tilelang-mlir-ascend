# Python Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reusable command that measures Python branch coverage for the existing NPUIR pytest suite locally and in GitHub Actions, producing terminal, HTML, Cobertura XML, and JUnit XML reports without enforcing a threshold.

**Architecture:** A small standard-library Python runner discovers the repository's `testing/npuir/*_ops` directories, builds one deterministic pytest command, forwards selectors, and propagates pytest's exit code. `.coveragerc` owns measurement policy, while the existing NPUIR workflow installs `pytest-cov`, calls the runner, and uploads both test and coverage artifacts.

**Tech Stack:** Python 3.8+, pytest, pytest-cov/coverage.py, `configparser`, GitHub Actions YAML

## Global Constraints

- Measure Python modules under `tilelang/` only; do not instrument C++, MLIR passes, generated NPU code, tests, builds, or third-party sources.
- Keep `testing/npuir/*_ops` as the default test selection so local and CI coverage match the existing NPUIR job.
- Preserve arbitrary pytest and NPUIR arguments such as `--op`, `--mode`, and `--npu-device` in caller order.
- Preserve `testing/npuir/output/report.html` and produce `coverage/python-html/index.html`, `coverage/python.xml`, and `coverage/junit.xml`, plus a terminal missing-lines report.
- Do not configure `fail_under`, an external upload service, or a native coverage tool.
- Run repository formatting on every changed Python/YAML/config file supported by `format.sh` before completion.

---

### Task 1: Coverage runner

**Files:**
- Create: `scripts/run_python_coverage.py`
- Create: `testing/python/test_run_python_coverage.py`

**Interfaces:**
- Produces: `discover_test_paths(repo_root: Path) -> list[str]`
- Produces: `build_pytest_command(repo_root: Path, pytest_args: Sequence[str]) -> list[str]`
- Produces: `prepare_output_directories(repo_root: Path) -> None`
- Produces: `run_command(command: Sequence[str], cwd: Path) -> int`
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int`
- Consumes: repository paths matching `testing/npuir/*_ops` and arbitrary pytest arguments

- [ ] **Step 1: Write failing runner tests**

Create `testing/python/test_run_python_coverage.py` with tests that use temporary real directories and a real child Python process:

```python
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
    with pytest.raises(FileNotFoundError, match=r"testing/npuir/\*_ops"):
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.run_python_coverage'` because the runner does not exist.

- [ ] **Step 3: Implement the minimal runner**

Create `scripts/run_python_coverage.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def discover_test_paths(repo_root: Path) -> list[str]:
    test_root = repo_root / "testing" / "npuir"
    paths = sorted(path for path in test_root.glob("*_ops") if path.is_dir())
    if not paths:
        raise FileNotFoundError(
            f"No NPUIR operation test directories matched {test_root / '*_ops'}"
        )
    return [path.relative_to(repo_root).as_posix() for path in paths]


def build_pytest_command(repo_root: Path, pytest_args: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        *discover_test_paths(repo_root),
        "--cov=tilelang",
        "--cov-config=.coveragerc",
        "--cov-report=term-missing",
        "--cov-report=html:coverage/python-html",
        "--cov-report=xml:coverage/python.xml",
        "--html=testing/npuir/output/report.html",
        "--self-contained-html",
        "--junitxml=coverage/junit.xml",
        *pytest_args,
    ]


def run_command(command: Sequence[str], cwd: Path) -> int:
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def prepare_output_directories(repo_root: Path) -> None:
    (repo_root / "coverage").mkdir(exist_ok=True)
    (repo_root / "testing" / "npuir" / "output").mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    prepare_output_directories(repo_root)
    command = build_pytest_command(repo_root, sys.argv[1:] if argv is None else argv)
    return run_command(command, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: `5 passed` and the exit-code test's real child process returns `7` only to `run_command`, not to pytest.

- [ ] **Step 5: Commit the runner cycle**

```bash
git add scripts/run_python_coverage.py testing/python/test_run_python_coverage.py
git commit -s -m "test: add reusable Python coverage runner"
```

### Task 2: Coverage policy and repository hygiene

**Files:**
- Create: `.coveragerc`
- Modify: `.gitignore`
- Modify: `requirements-test.txt`
- Modify: `testing/python/test_run_python_coverage.py`

**Interfaces:**
- Consumes: `--cov-config=.coveragerc` from Task 1
- Produces: branch-aware `tilelang` measurement policy and ignored `coverage/` output

- [ ] **Step 1: Add failing policy tests**

Append these tests to `testing/python/test_run_python_coverage.py`:

```python
from configparser import ConfigParser


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_coverage_policy_measures_tilelang_branches_and_shows_missing_lines():
    config = ConfigParser()
    assert config.read(REPO_ROOT / ".coveragerc")

    assert config.getboolean("run", "branch")
    assert config.getboolean("run", "relative_files")
    assert config.get("run", "source").split() == ["tilelang"]
    assert config.getboolean("report", "show_missing")
    assert not config.has_option("report", "fail_under")


def test_coverage_outputs_are_ignored_and_pytest_cov_is_declared():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    requirements = (REPO_ROOT / "requirements-test.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    assert "coverage/" in ignored
    assert "pytest-cov" in requirements
```

- [ ] **Step 2: Run policy tests and verify RED**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: two failures because `.coveragerc`, the `coverage/` ignore entry, and the `pytest-cov` requirement are absent.

- [ ] **Step 3: Add the minimal coverage policy**

Create `.coveragerc`:

```ini
[run]
branch = True
relative_files = True
source =
    tilelang
omit =
    */testing/*
    */tests/*
    */3rdparty/*
    */build/*
    */dist/*
    */__pycache__/*

[report]
show_missing = True
skip_covered = False

[html]
directory = coverage/python-html

[xml]
output = coverage/python.xml
```

Add `coverage/` to `.gitignore` next to other test/build output, and add `pytest-cov` immediately after `pytest>=6.2.4` in `requirements-test.txt`.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Commit coverage policy**

```bash
git add .coveragerc .gitignore requirements-test.txt testing/python/test_run_python_coverage.py
git commit -s -m "test: configure Python branch coverage"
```

### Task 3: NPUIR CI integration

**Files:**
- Modify: `.github/workflows/ci_npuir.yml`
- Modify: `testing/python/test_run_python_coverage.py`

**Interfaces:**
- Consumes: `python scripts/run_python_coverage.py` and the `coverage/` outputs from Tasks 1-2
- Produces: CI coverage execution and an always-uploaded combined report artifact

- [ ] **Step 1: Add a failing workflow contract test**

Append this test to `testing/python/test_run_python_coverage.py`:

```python
def test_npuir_workflow_runs_shared_coverage_command_and_uploads_reports():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci_npuir.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install pytest pytest-xdist pytest-html pytest-cov numpy" in workflow
    assert "python scripts/run_python_coverage.py -v" in workflow
    assert "path: |\n          testing/npuir/output/\n          coverage/" in workflow
```

- [ ] **Step 2: Run the workflow contract test and verify RED**

Run:

```bash
python -m pytest \
  testing/python/test_run_python_coverage.py::test_npuir_workflow_runs_shared_coverage_command_and_uploads_reports \
  -v
```

Expected: failure because the workflow still installs no `pytest-cov`, calls pytest directly, and uploads only `testing/npuir/output/`.

- [ ] **Step 3: Update the NPUIR workflow**

In `.github/workflows/ci_npuir.yml`:

- Change the explicit test dependency installation to `pip install pytest pytest-xdist pytest-html pytest-cov numpy`.
- Replace the direct pytest block with:

```yaml
      run: python scripts/run_python_coverage.py -v
```

- Change the artifact path to:

```yaml
        path: |
          testing/npuir/output/
          coverage/
```

Keep `TILELANG_CLEAR_CACHE: "1"`, `if: always()`, and the existing artifact name unchanged.

- [ ] **Step 4: Run focused and full contract tests and verify GREEN**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Commit CI integration**

```bash
git add .github/workflows/ci_npuir.yml testing/python/test_run_python_coverage.py
git commit -s -m "ci: collect NPUIR Python coverage"
```

### Task 4: Documentation, formatting, and final verification

**Files:**
- Modify: `testing/npuir/README.md`
- Modify: `testing/python/test_run_python_coverage.py`

**Interfaces:**
- Consumes: the runner CLI and output paths established in Tasks 1-3
- Produces: contributor-facing usage instructions and final repository validation evidence

- [ ] **Step 1: Add a failing documentation contract test**

Append this test to `testing/python/test_run_python_coverage.py`:

```python
def test_npuir_readme_documents_coverage_runner_and_outputs():
    readme = (REPO_ROOT / "testing" / "npuir" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "python scripts/run_python_coverage.py" in readme
    assert "testing/npuir/output/report.html" in readme
    assert "coverage/python-html/index.html" in readme
    assert "coverage/python.xml" in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```bash
python -m pytest \
  testing/python/test_run_python_coverage.py::test_npuir_readme_documents_coverage_runner_and_outputs \
  -v
```

Expected: failure because the README has no coverage section.

- [ ] **Step 3: Document local coverage usage**

Add a `## Python Coverage` section after the existing CLI examples in `testing/npuir/README.md` documenting:

```bash
python -m pip install -r requirements-test.txt
python scripts/run_python_coverage.py
python scripts/run_python_coverage.py --op=copy --mode=Developer --npu-device=0 -v
```

State that the pytest HTML report remains `testing/npuir/output/report.html`, the coverage HTML report is `coverage/python-html/index.html`, the machine-readable coverage report is `coverage/python.xml`, JUnit output is `coverage/junit.xml`, and the command measures host Python code rather than C++/MLIR or NPU device instructions.

- [ ] **Step 4: Run all coverage-runner tests**

Run:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
```

Expected: `9 passed`.

- [ ] **Step 5: Validate the runner against the real repository selection without executing NPU tests**

Run:

```bash
python -c "from pathlib import Path; from scripts.run_python_coverage import build_pytest_command; command = build_pytest_command(Path.cwd(), ['--collect-only']); assert 'testing/npuir/memory_ops' in command; assert command[-1] == '--collect-only'; print(len([item for item in command if item.startswith('testing/npuir/')]), 'NPUIR test directories')"
```

Expected: prints a positive number of NPUIR test directories and exits `0`.

- [ ] **Step 6: Validate configuration and workflow syntax**

Run:

```bash
python -c "from configparser import ConfigParser; from pathlib import Path; import yaml; config = ConfigParser(); assert config.read('.coveragerc'); yaml.safe_load(Path('.github/workflows/ci_npuir.yml').read_text(encoding='utf-8')); print('configuration syntax valid')"
```

Expected: `configuration syntax valid` and exit `0`.

- [ ] **Step 7: Run repository formatting and diff checks**

Run from a Bash-capable environment:

```bash
bash format.sh --files scripts/run_python_coverage.py testing/python/test_run_python_coverage.py .github/workflows/ci_npuir.yml testing/npuir/README.md .coveragerc requirements-test.txt .gitignore
git diff --check
```

Expected: formatting exits `0` and `git diff --check` produces no output.

- [ ] **Step 8: Review scope and commit the documentation/final checks**

Run:

```bash
git status --short
git diff --stat HEAD~3
git diff HEAD~3 -- . ':!docs/superpowers/specs/2026-08-07-python-coverage-design.md'
git add testing/npuir/README.md testing/python/test_run_python_coverage.py docs/superpowers/plans/2026-08-07-python-coverage.md
git commit -s -m "docs: explain NPUIR Python coverage"
```

Expected: only the approved Python coverage files are changed, and the final commit succeeds.

## Completion Evidence

Before claiming completion, freshly run and record:

```bash
python -m pytest testing/python/test_run_python_coverage.py -v
python -c "from configparser import ConfigParser; from pathlib import Path; import yaml; config = ConfigParser(); assert config.read('.coveragerc'); yaml.safe_load(Path('.github/workflows/ci_npuir.yml').read_text(encoding='utf-8'))"
git diff --check HEAD~4..HEAD
git status --short
```

The final report must distinguish local contract/configuration validation from real NPUIR runtime coverage, which is only produced by the NPU GitHub Actions job.

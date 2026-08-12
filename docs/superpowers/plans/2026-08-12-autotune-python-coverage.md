# Autotune Python Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add every `testing/autotune/**/*.py` script to the shared examples plus NPUIR Python coverage run while pinning all autotune work to logical NPU device 0.

**Architecture:** A dedicated autotune runner discovers and executes scripts sequentially under parallel-mode coverage. A small child launcher selects device 0, redirects later `torch.npu.set_device(...)` calls to 0, and executes one target through `runpy`; the existing coverage orchestrator combines these data before appending NPUIR pytest coverage.

**Tech Stack:** Python 3.11, coverage.py, pytest, pytest-cov, GitHub Actions, torch-npu.

## Global Constraints

- Include all `testing/autotune/**/*.py` files except `__init__.py`, including `AscendC/` and `anneal/`.
- Use only logical NPU device 0 for autotune scripts.
- Continue after individual script failures and return nonzero if any group fails.
- Use a 30-minute timeout per autotune script and a 360-minute NPU CI job timeout.
- Preserve existing reports and do not add a coverage percentage gate.
- Keep `python-coverage-upstream-main.patch` untracked and out of commits.

---

### Task 1: Autotune discovery and device-pinned child execution

**Files:**
- Create: `scripts/run_autotune_coverage.py`
- Create: `scripts/run_autotune_case.py`
- Create: `testing/python/test_run_autotune_coverage.py`

**Interfaces:**
- Produces: `discover_autotune_paths(repo_root: Path) -> list[str]`.
- Produces: `run_autotune_script(path: str, repo_root: Path, timeout: int = 1800) -> RunResult`.
- Produces: `run_target(target: Path) -> None`, which pins and redirects NPU selection before `runpy.run_path`.

- [x] **Step 1: Write failing discovery and device-redirection tests**

Create controlled nested fixtures that assert sorted `.py` discovery excludes
`__init__.py`, and a fake `torch` module plus target script that asserts initial
and redirected device selections are both 0.

- [x] **Step 2: Verify RED**

Run: `python -m pytest testing/python/test_run_autotune_coverage.py -q`

Expected: FAIL because the runner and child modules do not exist.

- [x] **Step 3: Implement discovery and child launcher**

Implement stable recursive discovery and:

```python
torch.npu.set_device(0)
torch.npu.set_device = lambda *_args, **_kwargs: original_set_device(0)
runpy.run_path(str(target), run_name="__main__")
```

- [x] **Step 4: Verify GREEN**

Run: `python -m pytest testing/python/test_run_autotune_coverage.py -q`

Expected: PASS.

### Task 2: Sequential coverage subprocess runner and failure aggregation

**Files:**
- Modify: `scripts/run_autotune_coverage.py`
- Modify: `testing/python/test_run_autotune_coverage.py`

**Interfaces:**
- Consumes: discovery and child launcher from Task 1.
- Produces: a CLI that executes every discovered script with `coverage run --parallel-mode`, repository-root `COVERAGE_FILE`, `ASCEND_RT_VISIBLE_DEVICES=0`, and exit status 1 when any script fails, times out, or errors.

- [x] **Step 1: Write failing subprocess and aggregation tests**

Assert the exact child coverage command, environment, 1800-second timeout,
continuation after a failure, summary statuses, and aggregate return code.

- [x] **Step 2: Verify RED**

Run: `python -m pytest testing/python/test_run_autotune_coverage.py -q`

Expected: FAIL because subprocess orchestration is absent.

- [x] **Step 3: Implement minimal sequential runner**

Add `RunResult`, subprocess execution, timeout/error mapping, per-script output,
summary reporting, and CLI exit behavior.

- [x] **Step 4: Verify GREEN**

Run: `python -m pytest testing/python/test_run_autotune_coverage.py -q`

Expected: PASS.

### Task 3: Shared coverage pipeline, CI, and documentation

**Files:**
- Modify: `scripts/run_python_coverage.py`
- Modify: `testing/python/test_run_python_coverage.py`
- Modify: `.github/workflows/test_npuir_wheel.yml`
- Modify: `testing/npuir/README.md`

**Interfaces:**
- Consumes: `scripts/run_autotune_coverage.py` CLI.
- Produces: pipeline order `erase -> examples -> autotune -> combine -> NPUIR pytest` and unchanged report locations.

- [x] **Step 1: Write failing pipeline and CI contract tests**

Require the autotune runner command between examples and coverage combine, and
require the reusable workflow timeout to be 360 minutes.

- [x] **Step 2: Verify RED**

Run: `python -m pytest testing/python/test_run_python_coverage.py -q`

Expected: FAIL because autotune is not in the pipeline and CI is still 120 minutes.

- [x] **Step 3: Implement pipeline, CI, and README updates**

Insert `python scripts/run_autotune_coverage.py`, change the CI timeout to 360,
and document the complete three-source coverage scope and device-0 policy.

- [x] **Step 4: Run focused and static verification**

Run:

```bash
python -m pytest testing/python/test_run_autotune_coverage.py testing/python/test_run_all_examples.py testing/python/test_run_python_coverage.py -q
python -m ruff check scripts/run_autotune_case.py scripts/run_autotune_coverage.py scripts/run_python_coverage.py testing/python/test_run_autotune_coverage.py testing/python/test_run_python_coverage.py
python -m ruff format --check scripts/run_autotune_case.py scripts/run_autotune_coverage.py scripts/run_python_coverage.py testing/python/test_run_autotune_coverage.py testing/python/test_run_python_coverage.py
python -m pymarkdown --config .pymarkdown scan testing/npuir/README.md docs/superpowers/specs/2026-08-12-autotune-python-coverage-design.md docs/superpowers/plans/2026-08-12-autotune-python-coverage.md
git diff --check
```

Expected: all commands exit 0. Real autotune execution remains the Ascend NPU verification gate.

- [ ] **Step 5: Commit and push directly to main**

Stage only the plan, scripts, focused tests, workflow, and README; commit with a
signed-off message and push `main` to `origin` after confirming the remote did
not advance.

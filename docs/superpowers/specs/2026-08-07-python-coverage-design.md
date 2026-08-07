# Python Coverage Design

## Goal

Add a repeatable Python source-coverage workflow for the existing NPUIR pytest suite. Local runs and the NPUIR GitHub Actions job must use the same coverage settings and produce terminal, HTML, Cobertura XML, and JUnit XML reports without enforcing a minimum coverage percentage in the first release.

## Scope

This change covers Python modules under `tilelang/` while running the existing `testing/npuir/*_ops` test selection. It does not instrument C++, TileLangIR/MLIR passes, generated device code, or external dependencies. It does not upload reports to a third-party coverage service and does not add a `fail_under` gate.

## Architecture

Coverage policy and execution are separated:

- `.coveragerc` is the single source of truth for measured source, branch coverage, omitted paths, and report formatting.
- `scripts/run_python_coverage.py` is the shared local/CI entry point. It constructs a deterministic pytest command, creates the report directory, forwards caller arguments to pytest, and returns pytest's exit code unchanged.
- `requirements-test.txt` declares `pytest-cov` as a test dependency.
- `.github/workflows/ci_npuir.yml` invokes the shared script and uploads reports even when tests fail.

This avoids duplicating a long pytest command in CI while preserving the repository's existing `--op`, `--mode`, and `--npu-device` selection contract.

## Command Interface

The default command is:

```bash
python scripts/run_python_coverage.py
```

It runs the NPUIR operation directories represented by `testing/npuir/*_ops`. Additional arguments are forwarded unchanged to pytest:

```bash
python scripts/run_python_coverage.py --op=copy --mode=Developer --npu-device=0 -v
```

The script preserves the existing pytest HTML report and adds these coverage outputs:

- `testing/npuir/output/report.html`
- `coverage/python-html/index.html`
- `coverage/python.xml`
- `coverage/junit.xml`

The terminal also shows missing source lines.

## Script Behavior

The script exposes a small command-construction function so its behavior can be tested without importing TileLang, torch-npu, or accessing an NPU. The function receives the repository root and forwarded pytest arguments and returns a list suitable for `subprocess.run`.

The executable entry point:

1. Resolves the repository root from the script location.
2. Creates `coverage/` and `testing/npuir/output/` if necessary.
3. Expands and sorts directories matching `testing/npuir/*_ops` for platform-independent behavior.
4. Fails with a clear message if no matching test directories exist.
5. Runs pytest from the repository root.
6. Returns pytest's exact exit code.

Forwarded arguments appear after the repository test paths and built-in report options, allowing existing NPUIR selector flags and normal pytest verbosity/filtering flags.

## Coverage Policy

`.coveragerc` enables branch coverage and measures `tilelang`. It omits code that should not affect the project coverage signal:

- test trees;
- build and distribution output;
- third-party sources;
- generated or cached files.

Reports display missing line numbers and use relative file paths so local and CI XML outputs are comparable. No minimum percentage is configured.

## CI Integration

The existing NPUIR test job remains the owner of runtime coverage because it already installs the built wheel, torch-npu, and runs on the project NPU environment. Its test step changes from a direct pytest invocation to the shared script.

The report artifact step runs with `if: always()` and uploads the existing test output together with `coverage/`. A failed pytest process still fails the test step; artifact upload does not mask that result.

## Error Handling

- Missing pytest or pytest-cov is reported by Python/pytest and produces a nonzero exit code.
- Missing `testing/npuir/*_ops` directories is detected before subprocess execution and produces a concise error.
- A test failure, collection error, compiler failure, or NPU runtime failure is propagated unchanged through pytest's exit code.
- Report upload remains best-effort through the workflow's existing `if: always()` behavior.

## Testing Strategy

Unit tests target the runner's real command construction and exit-code propagation without mocking TileLang or NPU execution:

- the default command contains sorted NPUIR operation directories and all required coverage reports;
- caller arguments are preserved in order;
- a missing test selection raises a clear error;
- the process entry point returns the subprocess exit code.

The tests are written before the runner implementation and must first fail because the module does not exist. Static validation then checks `.coveragerc` and parses the modified GitHub Actions YAML. The local environment is not treated as evidence that NPUIR kernels ran; full runtime validation remains the responsibility of the existing NPU CI job.

## Acceptance Criteria

- `pytest-cov` is installed by the repository test requirements and the NPUIR CI job.
- One command preserves the pytest HTML report and produces terminal, coverage HTML, coverage XML, and JUnit XML reports.
- Local and CI execution use the same runner and `.coveragerc`.
- Existing NPUIR selection flags continue to work through argument forwarding.
- Test failures still fail CI while coverage/test artifacts are uploaded.
- No coverage threshold, native instrumentation, or external upload service is introduced.

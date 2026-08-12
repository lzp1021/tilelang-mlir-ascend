# Autotune Python Coverage Design

## Goal

Extend the shared Python coverage command so one run executes all operator
scripts selected from `examples/`, every Python script under
`testing/autotune/`, and all pytest cases under `testing/npuir/*_ops`, then
combines their execution into the existing `tilelang/` branch-coverage report.

## Scope

- Include all 56 current `testing/autotune/**/*.py` files except
  `__init__.py`, including `AscendC/` and `anneal/`.
- Run autotune scripts sequentially on logical NPU device 0.
- Preserve the existing examples and NPUIR pytest selection rules and reports.
- Continue running later groups after an earlier group fails, but return a
  nonzero result when any example, autotune script, coverage combine command,
  or pytest invocation fails.
- Do not introduce a coverage percentage gate.

## Execution Architecture

Add an autotune collection and orchestration script patterned after
`examples/run_all.py`. It recursively discovers Python files in stable sorted
order and starts each one in an isolated subprocess. Each subprocess runs
under `coverage run --parallel-mode` and writes a repository-root parallel data
file.

The child launcher imports `torch`, selects logical NPU device 0, and redirects
subsequent calls such as `torch.npu.set_device(9)`, `set_device(10)`, or
`set_device(15)` to device 0 before executing the requested autotune file with
`runpy.run_path(..., run_name="__main__")`. The parent also exports
`ASCEND_RT_VISIBLE_DEVICES=0`. This keeps device policy in one place and avoids
editing every autotune operator.

The shared pipeline order is:

1. Erase old coverage data.
2. Run all examples with child-process coverage.
3. Run all autotune scripts with child-process coverage on device 0.
4. Combine parallel example and autotune coverage data.
5. Run `testing/npuir/*_ops` through pytest-cov with `--cov-append`.
6. Generate the existing terminal, HTML, Cobertura XML, JUnit XML, and pytest
   HTML reports.

## Failure and Timeout Policy

The autotune runner records `PASSED`, `FAILED`, `TIMEOUT`, and `ERROR` per
script and continues through the complete list. Its process exit status is
nonzero if any script did not pass. Use a 30-minute default timeout per
autotune script because annealing searches are substantially longer than
ordinary operator examples. Raise the reusable NPU CI job timeout from 120 to
360 minutes; the first real Ascend run will determine whether this remains
sufficient.

## Validation

- Contract-test deterministic discovery of all nested autotune Python scripts
  while excluding `__init__.py`.
- Contract-test that the child launcher pins and redirects device selection to
  logical device 0 before executing the target script.
- Contract-test subprocess command construction, coverage data location,
  timeout propagation, failure aggregation, and shared pipeline ordering.
- Update the NPUIR coverage documentation and CI contract checks.
- Run focused Python tests, Ruff, Markdown, YAML parsing, and diff checks
  locally. Full autotune execution and resulting coverage values require the
  Ascend NPU environment and remain a separate verification gate.

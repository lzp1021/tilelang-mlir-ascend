# Code Structure Overview (npuir branch)

## Top-level architecture

- `tilelang/`: Python frontend (DSL APIs, compile entry, adapters, tooling).
- `src/`: C++ backend integration with TVM runtime/codegen and target-specific modules.
- `tilelangir/`: MLIR-based TileLang IR project (dialect/transforms/tools such as `tilelangir-opt`).
- `3rdparty/AscendNPU-IR`: external BishengIR / AscendNPU-IR dependency for npuir route.
- `examples/`: operator samples (vector/cube/mixcv/deepseek workloads).
- `testing/` and `unittest/`: Python/MLIR/C++/NPUIR validation suites.
- `docs/`: API docs, developer docs, debugging, contribution workflow.

## Build and dependency layering

1. Python package metadata and lint/format rules are defined in `pyproject.toml`.
2. Root `CMakeLists.txt` builds TileLang C++ pieces and conditionally enables:
   - TVM integration (prebuilt or source under `3rdparty/tvm`)
   - NPUIR path (`USE_NPUIR`) and `tilelangir/` subproject
   - CUDA / ROCm backends in parallel with NPUIR.
3. NPUIR path links MLIR/BishengIR libraries and adds `src/target/*npuir*.cc` runtime+codegen units.

## NPUIR-oriented source split

- `tilelang/language/`, `tilelang/primitives/`: user-facing DSL and primitive ops.
- `tilelang/tladapter/`: adapters from Python side to backend/lowering.
- `src/transform/`: lowering/transform passes at C++ level.
- `src/target/codegen_npuir*.cc` + `src/target/rt_mod_npuir.cc`: npuir codegen/runtime emission.
- `tilelangir/lib/Transforms/`: MLIR pass implementations used in NPUIR pipeline.

## Validation layout

- `testing/npuir/`: NPUIR functional and integration coverage.
- `testing/mlir/`: pass/toolchain checks for MLIR layer.
- `testing/python/`: frontend behavior and API-level checks.
- `unittest/npuir/`: unit-style NPUIR cases.

## Recommended reading order for newcomers

1. `README.md` (project goals, routes, quickstart).
2. `docs/developer/EnvironmentVariables.md` and `docs/developer/npu runtime.md`.
3. Closest kernel in `examples/` and matching case in `testing/npuir/`.
4. `tilelang/language/` APIs and corresponding `src/target/*npuir*` backend implementation.

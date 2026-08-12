from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class RunResult:
    name: str
    status: str
    duration: float = 0.0
    error_msg: str = ""


def discover_autotune_paths(repo_root: Path) -> list[str]:
    autotune_root = repo_root / "testing" / "autotune"
    paths = sorted(
        (
            path
            for path in autotune_root.rglob("*.py")
            if path.is_file() and path.name != "__init__.py"
        ),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No autotune scripts found under {autotune_root}")
    return [path.relative_to(repo_root).as_posix() for path in paths]


def run_autotune_script(
    rel_path: str, repo_root: Path, timeout: int = 1800
) -> RunResult:
    target = repo_root / rel_path
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    env["COVERAGE_FILE"] = str(repo_root / ".coverage")
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        f"--rcfile={repo_root / '.coveragerc'}",
        str(repo_root / "scripts" / "run_autotune_case.py"),
        str(target),
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            rel_path, "TIMEOUT", time.monotonic() - start, f"Exceeded {timeout}s"
        )
    except Exception as exc:
        return RunResult(rel_path, "ERROR", time.monotonic() - start, str(exc))

    if proc.returncode == 0:
        return RunResult(rel_path, "PASSED", time.monotonic() - start)
    error_msg = proc.stderr.strip() or proc.stdout.strip()
    return RunResult(rel_path, "FAILED", time.monotonic() - start, error_msg)


def run_autotune_scripts(
    repo_root: Path, paths: Sequence[str], timeout: int = 1800
) -> int:
    results = []
    for path in paths:
        result = run_autotune_script(path, repo_root, timeout)
        results.append(result)
        print(f"{result.status:<7} {result.duration:8.2f}s {result.name}")
        if result.error_msg:
            print(result.error_msg)

    passed = sum(result.status == "PASSED" for result in results)
    print(f"Autotune results: {passed} passed, {len(results) - passed} failed")
    return 0 if passed == len(results) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all TileLang autotune scripts with Python coverage"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Timeout per script in seconds"
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    return run_autotune_scripts(
        repo_root, discover_autotune_paths(repo_root), timeout=args.timeout
    )


if __name__ == "__main__":
    raise SystemExit(main())

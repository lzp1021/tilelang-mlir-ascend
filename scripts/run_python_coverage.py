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
        "--cov-append",
        "--cov-report=term-missing",
        "--cov-report=html:coverage/python-html",
        "--cov-report=xml:coverage/python.xml",
        "--html=testing/npuir/output/report.html",
        "--self-contained-html",
        "--junitxml=coverage/junit.xml",
        *pytest_args,
    ]


def prepare_output_directories(repo_root: Path) -> None:
    coverage_dir = repo_root / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    cpp_profraw_dir = coverage_dir / "cpp-profraw"
    cpp_profraw_dir.mkdir(exist_ok=True)
    for profile in cpp_profraw_dir.glob("*.profraw"):
        profile.unlink()
    cpp_object_dir = coverage_dir / "cpp-objects"
    cpp_object_dir.mkdir(exist_ok=True)
    for object_path in cpp_object_dir.glob("*.so"):
        object_path.unlink()
    for filename in (
        "python-recent.xml",
        "python-recent.html",
        "python-recent-summary.txt",
        "cpp.profdata",
        "cpp.info",
        "cpp-recent.xml",
        "coverage.xml",
        "index.html",
        "summary.txt",
    ):
        output = coverage_dir / filename
        if output.is_file():
            output.unlink()
    (repo_root / "testing" / "npuir" / "output").mkdir(parents=True, exist_ok=True)


def run_command(command: Sequence[str], cwd: Path) -> int:
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def run_coverage_pipeline(repo_root: Path, pytest_args: Sequence[str]) -> int:
    commands = [
        [sys.executable, "-m", "coverage", "erase"],
        [
            sys.executable,
            "examples/run_all.py",
            "--sequential",
            "--coverage",
        ],
        [sys.executable, "scripts/run_autotune_coverage.py"],
        [sys.executable, "-m", "coverage", "combine"],
        build_pytest_command(repo_root, pytest_args),
    ]
    return_codes = [run_command(command, repo_root) for command in commands]
    python_recent_code = run_command(
        [sys.executable, "scripts/filter_recent_coverage.py", "--months", "8"],
        repo_root,
    )
    cpp_recent_code = run_command(
        [sys.executable, "scripts/generate_cpp_coverage.py", "--months", "8"],
        repo_root,
    )
    return_codes.extend((python_recent_code, cpp_recent_code))
    if python_recent_code == 0 and cpp_recent_code == 0:
        return_codes.append(
            run_command(
                [
                    sys.executable,
                    "scripts/combine_coverage_reports.py",
                    "--months",
                    "8",
                ],
                repo_root,
            )
        )
    return next((code for code in return_codes if code != 0), 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    prepare_output_directories(repo_root)
    pytest_args = sys.argv[1:] if argv is None else argv
    return run_coverage_pipeline(repo_root, pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

try:
    from scripts.filter_recent_coverage import (
        Counts,
        GitLineDates,
        rate,
        set_rates,
        subtract_calendar_months,
    )
except ModuleNotFoundError:
    from filter_recent_coverage import (  # type: ignore[no-redef]
        Counts,
        GitLineDates,
        rate,
        set_rates,
        subtract_calendar_months,
    )


CPP_ROOTS = ("src", "tilelang", "tilelangir")
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}


@dataclass
class CppFileCoverage:
    lines: dict[int, int] = field(default_factory=dict)
    branches: dict[tuple[int, str, str], bool] = field(default_factory=dict)


def discover_coverage_objects() -> list[Path]:
    spec = importlib.util.find_spec("tilelang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed tilelang package")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    lib_dir = package_root / "lib"
    names = ("libtilelang_module.so", "libtilelangir.so")
    missing = [name for name in names if not (lib_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Coverage wheel is missing native libraries in {lib_dir}: {', '.join(missing)}"
        )
    objects = [lib_dir / name for name in names]
    coverage_object_dir = os.environ.get("TILELANG_CPP_COVERAGE_OBJECT_DIR")
    if coverage_object_dir:
        runtime_object = Path(coverage_object_dir) / "npu_utils.so"
        if not runtime_object.is_file():
            raise FileNotFoundError(
                f"Runtime C++ coverage object was not captured: {runtime_object}"
            )
        objects.append(runtime_object)
    cache_root = Path(
        os.environ.get("TILELANG_CACHE_DIR", str(Path.home() / ".tilelang" / "cache"))
    )
    if cache_root.is_dir():
        cached_objects = sorted(cache_root.rglob("npu_utils.so"))
        if cached_objects and not any(path.name == "npu_utils.so" for path in objects):
            objects.append(cached_objects[-1])
    return objects


def normalize_source_path(raw_path: str, repo_root: Path) -> Optional[str]:
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
            normalized = relative.as_posix()
        except ValueError:
            for root in CPP_ROOTS:
                marker = f"/{root}/"
                if marker in normalized:
                    candidate = root + "/" + normalized.split(marker, 1)[1]
                    if (repo_root / candidate).is_file():
                        normalized = candidate
                        break
    normalized = normalized.removeprefix("./")
    if not normalized.startswith(tuple(f"{root}/" for root in CPP_ROOTS)):
        return None
    if Path(normalized).suffix.lower() not in CPP_SUFFIXES:
        return None
    if not (repo_root / normalized).is_file():
        return None
    return normalized


def parse_lcov(text: str, repo_root: Path) -> dict[str, CppFileCoverage]:
    files: dict[str, CppFileCoverage] = {}
    current: Optional[CppFileCoverage] = None
    for raw_line in text.splitlines():
        if raw_line.startswith("SF:"):
            filename = normalize_source_path(raw_line[3:], repo_root)
            current = (
                files.setdefault(filename, CppFileCoverage()) if filename else None
            )
        elif current is not None and raw_line.startswith("DA:"):
            line_number, hits, *_ = raw_line[3:].split(",")
            number = int(line_number)
            current.lines[number] = current.lines.get(number, 0) + int(hits)
        elif current is not None and raw_line.startswith("BRDA:"):
            line_number, block, branch, taken = raw_line[5:].split(",")
            key = (int(line_number), block, branch)
            covered = taken != "-" and int(taken) > 0
            current.branches[key] = current.branches.get(key, False) or covered
        elif raw_line == "end_of_record":
            current = None
    return files


def collect_recent_counts(
    files: dict[str, CppFileCoverage], is_recent
) -> tuple[Counts, dict[str, Counts], dict[str, list[ET.Element]]]:
    total = Counts()
    file_counts: dict[str, Counts] = {}
    file_lines: dict[str, list[ET.Element]] = {}
    for filename, coverage in sorted(files.items()):
        counts = Counts()
        elements = []
        branches_by_line: dict[int, list[bool]] = {}
        for (line_number, _block, _branch), covered in coverage.branches.items():
            branches_by_line.setdefault(line_number, []).append(covered)
        for line_number, hits in sorted(coverage.lines.items()):
            if not is_recent(filename, line_number):
                continue
            branches = branches_by_line.get(line_number, [])
            branch_valid = len(branches)
            branch_covered = sum(branches)
            line = ET.Element("line", {"number": str(line_number), "hits": str(hits)})
            if branch_valid:
                percent = round(100 * branch_covered / branch_valid)
                line.set("branch", "true")
                line.set(
                    "condition-coverage",
                    f"{percent}% ({branch_covered}/{branch_valid})",
                )
            elements.append(line)
            counts += Counts(
                lines_valid=1,
                lines_covered=hits > 0,
                branches_valid=branch_valid,
                branches_covered=branch_covered,
            )
        if counts.lines_valid:
            file_counts[filename] = counts
            file_lines[filename] = elements
            total += counts
    return total, file_counts, file_lines


def build_cobertura_tree(
    total: Counts,
    file_counts: dict[str, Counts],
    file_lines: dict[str, list[ET.Element]],
) -> ET.ElementTree:
    root = ET.Element(
        "coverage",
        {
            "version": "llvm-cov",
            "lines-valid": str(total.lines_valid),
            "lines-covered": str(total.lines_covered),
            "line-rate": rate(total.lines_covered, total.lines_valid),
            "branches-valid": str(total.branches_valid),
            "branches-covered": str(total.branches_covered),
            "branch-rate": rate(total.branches_covered, total.branches_valid),
            "complexity": "0",
        },
    )
    sources = ET.SubElement(root, "sources")
    ET.SubElement(sources, "source").text = "."
    packages = ET.SubElement(root, "packages")
    grouped: dict[str, list[str]] = {}
    for filename in file_counts:
        grouped.setdefault(filename.split("/", 1)[0], []).append(filename)
    for group, filenames in sorted(grouped.items()):
        package_counts = Counts()
        for filename in filenames:
            package_counts += file_counts[filename]
        package = ET.SubElement(
            packages,
            "package",
            {"name": f"cpp.{group}", "complexity": "0"},
        )
        set_rates(package, package_counts)
        classes = ET.SubElement(package, "classes")
        for filename in sorted(filenames):
            counts = file_counts[filename]
            class_element = ET.SubElement(
                classes,
                "class",
                {
                    "name": Path(filename).name,
                    "filename": filename,
                    "complexity": "0",
                },
            )
            set_rates(class_element, counts)
            ET.SubElement(class_element, "methods")
            lines = ET.SubElement(class_element, "lines")
            lines.extend(file_lines[filename])
    return ET.ElementTree(root)


def run_checked(command: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr}"
        )
    return proc.stdout


def generate_cpp_report(
    repo_root: Path,
    profraw_dir: Path,
    output_lcov: Path,
    output_xml: Path,
    objects: Sequence[Path],
    months: int,
    as_of: date,
    keep_raw: bool = False,
) -> Counts:
    profraw_files = sorted(profraw_dir.glob("*.profraw"))
    if not profraw_files:
        raise FileNotFoundError(
            f"No C++ profiles found under {profraw_dir}; use an instrumented wheel"
        )
    llvm_profdata = shutil.which("llvm-profdata")
    llvm_cov = shutil.which("llvm-cov")
    if llvm_profdata is None or llvm_cov is None:
        raise RuntimeError("llvm-profdata and llvm-cov are required for C++ coverage")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    profdata = output_xml.parent / "cpp.profdata"
    run_checked(
        [
            llvm_profdata,
            "merge",
            "-sparse",
            *map(str, profraw_files),
            "-o",
            str(profdata),
        ],
        repo_root,
    )
    command = [
        llvm_cov,
        "export",
        str(objects[0]),
        f"-instr-profile={profdata}",
        "-format=lcov",
    ]
    for object_path in objects[1:]:
        command.extend(["-object", str(object_path)])
    lcov = run_checked(command, repo_root)
    output_lcov.write_text(lcov, encoding="utf-8")
    files = parse_lcov(lcov, repo_root)
    cutoff = subtract_calendar_months(as_of, months)
    total, file_counts, file_lines = collect_recent_counts(
        files, GitLineDates(repo_root, cutoff, source_prefix=None).is_recent
    )
    tree = build_cobertura_tree(total, file_counts, file_lines)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    if not keep_raw:
        for profile in profraw_files:
            profile.unlink()
        coverage_object_dir = os.environ.get("TILELANG_CPP_COVERAGE_OBJECT_DIR")
        if coverage_object_dir:
            runtime_object = Path(coverage_object_dir) / "npu_utils.so"
            if runtime_object.is_file():
                runtime_object.unlink()
    print(
        f"Recent C++ coverage (committed on or after {cutoff.isoformat()}): "
        f"{total.lines_covered}/{total.lines_valid} lines"
    )
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate recent C++ coverage from LLVM raw profiles"
    )
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--profraw-dir", type=Path, default=Path("coverage/cpp-profraw")
    )
    parser.add_argument("--object", action="append", type=Path, default=[])
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep per-process .profraw files and copied runtime objects",
    )
    parser.add_argument("--lcov-output", type=Path, default=Path("coverage/cpp.info"))
    parser.add_argument(
        "--xml-output", type=Path, default=Path("coverage/cpp-recent.xml")
    )
    args = parser.parse_args(argv)
    if args.months < 0:
        parser.error("--months must be non-negative")
    repo_root = Path(__file__).resolve().parents[1]
    objects = args.object or discover_coverage_objects()
    generate_cpp_report(
        repo_root,
        repo_root / args.profraw_dir,
        repo_root / args.lcov_output,
        repo_root / args.xml_output,
        objects,
        args.months,
        args.as_of,
        keep_raw=args.keep_raw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import calendar
import html
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional, Sequence


BRANCH_COUNTS = re.compile(r"\((\d+)/(\d+)\)")
BLAME_HEADER = re.compile(r"^[0-9a-f^]+\s+\d+\s+(\d+)(?:\s+\d+)?$")


@dataclass(frozen=True)
class Counts:
    lines_valid: int = 0
    lines_covered: int = 0
    branches_valid: int = 0
    branches_covered: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.lines_valid + other.lines_valid,
            self.lines_covered + other.lines_covered,
            self.branches_valid + other.branches_valid,
            self.branches_covered + other.branches_covered,
        )


@dataclass(frozen=True)
class FileStats:
    filename: str
    recent: Counts
    original_lines: int

    @property
    def excluded_lines(self) -> int:
        return self.original_lines - self.recent.lines_valid


def subtract_calendar_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def rate(covered: int, valid: int) -> str:
    return "1" if valid == 0 else f"{covered / valid:.4f}".rstrip("0").rstrip(".")


def line_counts(line: ET.Element) -> Counts:
    branches_valid = 0
    branches_covered = 0
    if line.get("branch") == "true":
        match = BRANCH_COUNTS.search(line.get("condition-coverage", ""))
        if match is None:
            raise ValueError(
                f"Cannot parse branch coverage for line {line.get('number')}"
            )
        branches_covered, branches_valid = map(int, match.groups())
    return Counts(
        lines_valid=1,
        lines_covered=int(line.get("hits", "0")) > 0,
        branches_valid=branches_valid,
        branches_covered=branches_covered,
    )


def set_rates(element: ET.Element, counts: Counts) -> None:
    element.set("line-rate", rate(counts.lines_covered, counts.lines_valid))
    element.set("branch-rate", rate(counts.branches_covered, counts.branches_valid))


def filter_coverage_tree(
    root: ET.Element, is_recent: Callable[[str, int], bool]
) -> tuple[Counts, list[FileStats]]:
    total = Counts()
    file_stats = []
    packages = root.find("packages")
    if packages is None:
        raise ValueError("Coverage XML does not contain <packages>")

    for package in list(packages):
        package_counts = Counts()
        classes = package.find("classes")
        if classes is None:
            packages.remove(package)
            continue
        for class_element in list(classes):
            filename = class_element.get("filename")
            lines = class_element.find("lines")
            if filename is None or lines is None:
                classes.remove(class_element)
                continue
            original_lines = len(lines)
            class_counts = Counts()
            for line in list(lines):
                line_number = int(line.get("number", "0"))
                if is_recent(filename, line_number):
                    class_counts += line_counts(line)
                else:
                    lines.remove(line)
            if class_counts.lines_valid == 0:
                classes.remove(class_element)
                continue
            set_rates(class_element, class_counts)
            package_counts += class_counts
            file_stats.append(FileStats(filename, class_counts, original_lines))
        if package_counts.lines_valid == 0:
            packages.remove(package)
            continue
        set_rates(package, package_counts)
        total += package_counts

    root.set("lines-valid", str(total.lines_valid))
    root.set("lines-covered", str(total.lines_covered))
    root.set("line-rate", rate(total.lines_covered, total.lines_valid))
    root.set("branches-valid", str(total.branches_valid))
    root.set("branches-covered", str(total.branches_covered))
    root.set("branch-rate", rate(total.branches_covered, total.branches_valid))
    return total, file_stats


class GitLineDates:
    def __init__(self, repo_root: Path, cutoff: date):
        self.repo_root = repo_root
        self.cutoff = cutoff
        self._cache: dict[str, dict[int, date]] = {}

    def is_recent(self, filename: str, line_number: int) -> bool:
        if filename not in self._cache:
            self._cache[filename] = self._load(filename)
        dates = self._cache[filename]
        try:
            return dates[line_number] >= self.cutoff
        except KeyError as exc:
            raise ValueError(
                f"git blame returned no date for {filename}:{line_number}"
            ) from exc

    def _load(self, filename: str) -> dict[int, date]:
        tracked_path = Path("tilelang") / Path(filename)
        command = ["git", "blame", "--line-porcelain"]
        if not (self.repo_root / tracked_path).is_file():
            command.append("HEAD")
        command.extend(["--", tracked_path.as_posix()])
        proc = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git blame failed for {tracked_path}: {proc.stderr.strip()}"
            )

        dates: dict[int, date] = {}
        final_line: Optional[int] = None
        committer_time: Optional[int] = None
        for text_line in proc.stdout.splitlines():
            header = BLAME_HEADER.match(text_line)
            if header:
                final_line = int(header.group(1))
                committer_time = None
            elif text_line.startswith("committer-time "):
                committer_time = int(text_line.split()[1])
            elif text_line.startswith("\t") and final_line is not None:
                if committer_time is None:
                    raise ValueError(
                        f"Missing committer time for {filename}:{final_line}"
                    )
                dates[final_line] = datetime.fromtimestamp(committer_time).date()
                final_line = None
        return dates


def write_html(
    path: Path, cutoff: date, total: Counts, stats: Sequence[FileStats]
) -> None:
    rows = []
    for item in sorted(stats, key=lambda value: value.filename):
        line_percent = 100 * item.recent.lines_covered / item.recent.lines_valid
        branch_percent = (
            100 * item.recent.branches_covered / item.recent.branches_valid
            if item.recent.branches_valid
            else 100.0
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.filename)}</td>"
            f"<td>{item.recent.lines_covered}/{item.recent.lines_valid}</td>"
            f"<td>{line_percent:.2f}%</td>"
            f"<td>{item.recent.branches_covered}/{item.recent.branches_valid}</td>"
            f"<td>{branch_percent:.2f}%</td>"
            f"<td>{item.excluded_lines}</td>"
            "</tr>"
        )
    line_percent = (
        100 * total.lines_covered / total.lines_valid if total.lines_valid else 100
    )
    branch_percent = (
        100 * total.branches_covered / total.branches_valid
        if total.branches_valid
        else 100
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Recent Python coverage</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:.4rem .7rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}th{background:#eee}</style>"
        "</head><body><h1>Recent Python coverage</h1>"
        f"<p>Only lines committed on or after <strong>{cutoff.isoformat()}</strong> "
        "are included.</p>"
        f"<p>Lines: {total.lines_covered}/{total.lines_valid} ({line_percent:.2f}%). "
        f"Branches: {total.branches_covered}/{total.branches_valid} "
        f"({branch_percent:.2f}%).</p>"
        "<table><thead><tr><th>File</th><th>Lines</th><th>Line rate</th>"
        "<th>Branches</th><th>Branch rate</th><th>Old lines excluded</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></body></html>",
        encoding="utf-8",
    )


def write_summary(path: Path, cutoff: date, total: Counts, excluded: int) -> str:
    line_percent = (
        100 * total.lines_covered / total.lines_valid if total.lines_valid else 100
    )
    branch_percent = (
        100 * total.branches_covered / total.branches_valid
        if total.branches_valid
        else 100
    )
    summary = (
        f"Recent Python coverage (committed on or after {cutoff.isoformat()})\n"
        f"Lines: {total.lines_covered}/{total.lines_valid} ({line_percent:.2f}%)\n"
        f"Branches: {total.branches_covered}/{total.branches_valid} ({branch_percent:.2f}%)\n"
        f"Older executable lines excluded: {excluded}\n"
    )
    path.write_text(summary, encoding="utf-8")
    return summary


def ensure_complete_history(repo_root: Path) -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if proc.stdout.strip() == "true":
        raise RuntimeError(
            "Recent coverage requires complete Git history; use checkout fetch-depth: 0"
        )


def generate_recent_report(
    repo_root: Path,
    input_xml: Path,
    output_xml: Path,
    output_html: Path,
    summary_path: Path,
    months: int,
    as_of: date,
) -> Counts:
    ensure_complete_history(repo_root)
    cutoff = subtract_calendar_months(as_of, months)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(input_xml, parser=parser)
    root = tree.getroot()
    original_lines = int(root.get("lines-valid", "0"))
    total, stats = filter_coverage_tree(root, GitLineDates(repo_root, cutoff).is_recent)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    write_html(output_html, cutoff, total, stats)
    summary = write_summary(
        summary_path, cutoff, total, original_lines - total.lines_valid
    )
    print(summary, end="")
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exclude Python coverage lines committed before a Git age cutoff"
    )
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--input", type=Path, default=Path("coverage/python.xml"))
    parser.add_argument(
        "--xml-output", type=Path, default=Path("coverage/python-recent.xml")
    )
    parser.add_argument(
        "--html-output", type=Path, default=Path("coverage/python-recent.html")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("coverage/python-recent-summary.txt"),
    )
    args = parser.parse_args(argv)
    if args.months < 0:
        parser.error("--months must be non-negative")
    repo_root = Path(__file__).resolve().parents[1]
    generate_recent_report(
        repo_root,
        repo_root / args.input,
        repo_root / args.xml_output,
        repo_root / args.html_output,
        repo_root / args.summary_output,
        args.months,
        args.as_of,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

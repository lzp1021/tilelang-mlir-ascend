from __future__ import annotations

import argparse
import copy
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

try:
    from scripts.filter_recent_coverage import (
        Counts,
        line_counts,
        rate,
        subtract_calendar_months,
    )
except ModuleNotFoundError:
    from filter_recent_coverage import (  # type: ignore[no-redef]
        Counts,
        line_counts,
        rate,
        subtract_calendar_months,
    )


@dataclass(frozen=True)
class Report:
    language: str
    counts: Counts
    files: tuple[tuple[str, Counts], ...]


def read_counts(root: ET.Element) -> Counts:
    return Counts(
        lines_valid=int(root.get("lines-valid", "0")),
        lines_covered=int(root.get("lines-covered", "0")),
        branches_valid=int(root.get("branches-valid", "0")),
        branches_covered=int(root.get("branches-covered", "0")),
    )


def read_report(language: str, tree: ET.ElementTree) -> Report:
    files = []
    for class_element in tree.findall(".//class"):
        counts = Counts()
        for line in class_element.findall("./lines/line"):
            counts += line_counts(line)
        files.append((class_element.get("filename", "unknown"), counts))
    return Report(language, read_counts(tree.getroot()), tuple(files))


def combine_trees(
    python_tree: ET.ElementTree, cpp_tree: ET.ElementTree
) -> tuple[ET.ElementTree, tuple[Report, Report]]:
    reports = (
        read_report("Python", python_tree),
        read_report("C++", cpp_tree),
    )
    total = reports[0].counts + reports[1].counts
    root = ET.Element(
        "coverage",
        {
            "version": "tilelang-combined",
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
    for language, source_tree in (("python", python_tree), ("cpp", cpp_tree)):
        for source_package in source_tree.findall("./packages/package"):
            package = copy.deepcopy(source_package)
            package.set("name", f"{language}.{package.get('name', '.')}")
            if language == "python":
                for class_element in package.findall("./classes/class"):
                    filename = class_element.get("filename", "")
                    class_element.set("filename", f"tilelang/{filename}")
            packages.append(package)
    return ET.ElementTree(root), reports


def percent(covered: int, valid: int) -> float:
    return 100 * covered / valid if valid else 100.0


def write_html(
    path: Path, cutoff: date, reports: Sequence[Report], total: Counts
) -> None:
    language_rows = []
    file_rows = []
    for report in reports:
        counts = report.counts
        language_rows.append(
            "<tr>"
            f"<td>{html.escape(report.language)}</td>"
            f"<td>{counts.lines_covered}/{counts.lines_valid}</td>"
            f"<td>{percent(counts.lines_covered, counts.lines_valid):.2f}%</td>"
            f"<td>{counts.branches_covered}/{counts.branches_valid}</td>"
            f"<td>{percent(counts.branches_covered, counts.branches_valid):.2f}%</td>"
            "</tr>"
        )
        for filename, file_counts in sorted(report.files):
            display = (
                f"tilelang/{filename}" if report.language == "Python" else filename
            )
            file_rows.append(
                "<tr>"
                f"<td>{html.escape(report.language)}</td>"
                f"<td>{html.escape(display)}</td>"
                f"<td>{file_counts.lines_covered}/{file_counts.lines_valid}</td>"
                f"<td>{percent(file_counts.lines_covered, file_counts.lines_valid):.2f}%</td>"
                f"<td>{file_counts.branches_covered}/{file_counts.branches_valid}</td>"
                f"<td>{percent(file_counts.branches_covered, file_counts.branches_valid):.2f}%</td>"
                "</tr>"
            )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TileLang recent combined coverage</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse;"
        "margin-bottom:2rem}th,td{border:1px solid #ccc;padding:.4rem .7rem;"
        "text-align:right}th:first-child,td:first-child,td:nth-child(2){text-align:left}"
        "th{background:#eee}</style></head><body>"
        "<h1>TileLang recent combined coverage</h1>"
        f"<p>Only executable lines committed on or after <strong>{cutoff.isoformat()}"
        "</strong> are included.</p>"
        f"<p><strong>Combined lines:</strong> {total.lines_covered}/{total.lines_valid} "
        f"({percent(total.lines_covered, total.lines_valid):.2f}%). "
        f"<strong>Combined branches:</strong> {total.branches_covered}/"
        f"{total.branches_valid} ({percent(total.branches_covered, total.branches_valid):.2f}%).</p>"
        "<h2>Languages</h2><table><thead><tr><th>Language</th><th>Lines</th>"
        "<th>Line rate</th><th>Branches</th><th>Branch rate</th></tr></thead><tbody>"
        + "".join(language_rows)
        + "</tbody></table><h2>Files</h2><table><thead><tr><th>Language</th>"
        "<th>File</th><th>Lines</th><th>Line rate</th><th>Branches</th>"
        "<th>Branch rate</th></tr></thead><tbody>"
        + "".join(file_rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )


def write_summary(
    path: Path, cutoff: date, reports: Sequence[Report], total: Counts
) -> str:
    lines = [f"TileLang recent coverage (committed on or after {cutoff.isoformat()})"]
    for report in reports:
        counts = report.counts
        lines.append(
            f"{report.language} lines: {counts.lines_covered}/{counts.lines_valid} "
            f"({percent(counts.lines_covered, counts.lines_valid):.2f}%)"
        )
        lines.append(
            f"{report.language} branches: {counts.branches_covered}/{counts.branches_valid} "
            f"({percent(counts.branches_covered, counts.branches_valid):.2f}%)"
        )
    lines.append(
        f"Combined lines: {total.lines_covered}/{total.lines_valid} "
        f"({percent(total.lines_covered, total.lines_valid):.2f}%)"
    )
    lines.append(
        f"Combined branches: {total.branches_covered}/{total.branches_valid} "
        f"({percent(total.branches_covered, total.branches_valid):.2f}%)"
    )
    summary = "\n".join(lines) + "\n"
    path.write_text(summary, encoding="utf-8")
    return summary


def generate_combined_report(
    python_xml: Path,
    cpp_xml: Path,
    output_xml: Path,
    output_html: Path,
    summary_path: Path,
    months: int,
    as_of: date,
) -> Counts:
    tree, reports = combine_trees(ET.parse(python_xml), ET.parse(cpp_xml))
    total = read_counts(tree.getroot())
    cutoff = subtract_calendar_months(as_of, months)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    write_html(output_html, cutoff, reports, total)
    print(write_summary(summary_path, cutoff, reports, total), end="")
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine recent Python and C++ coverage reports"
    )
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--python-xml", type=Path, default=Path("coverage/python-recent.xml")
    )
    parser.add_argument("--cpp-xml", type=Path, default=Path("coverage/cpp-recent.xml"))
    parser.add_argument(
        "--xml-output", type=Path, default=Path("coverage/coverage.xml")
    )
    parser.add_argument("--html-output", type=Path, default=Path("coverage/index.html"))
    parser.add_argument(
        "--summary-output", type=Path, default=Path("coverage/summary.txt")
    )
    args = parser.parse_args(argv)
    if args.months < 0:
        parser.error("--months must be non-negative")
    repo_root = Path(__file__).resolve().parents[1]
    generate_combined_report(
        repo_root / args.python_xml,
        repo_root / args.cpp_xml,
        repo_root / args.xml_output,
        repo_root / args.html_output,
        repo_root / args.summary_output,
        args.months,
        args.as_of,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

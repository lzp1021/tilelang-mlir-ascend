import xml.etree.ElementTree as ET

from datetime import date

from scripts.combine_coverage_reports import (
    combine_trees,
    generate_combined_report,
    read_counts,
)


def make_tree(filename, covered, valid, branch_covered, branch_valid):
    hits = [1] * covered + [0] * (valid - covered)
    lines = "".join(
        f'<line number="{index}" hits="{hit}"/>'
        for index, hit in enumerate(hits, start=1)
    )
    if branch_valid:
        lines += (
            f'<line number="{valid + 1}" hits="1" branch="true" '
            f'condition-coverage="50% ({branch_covered}/{branch_valid})"/>'
        )
        valid += 1
        covered += 1
    return ET.ElementTree(
        ET.fromstring(
            f"""<coverage lines-valid="{valid}" lines-covered="{covered}"
              branches-valid="{branch_valid}" branches-covered="{branch_covered}">
              <packages><package name="demo"><classes>
                <class name="demo" filename="{filename}"><lines>{lines}</lines></class>
              </classes></package></packages>
            </coverage>"""
        )
    )


def test_combine_trees_weights_python_and_cpp_counts():
    python_tree = make_tree("demo.py", 1, 2, 1, 2)
    cpp_tree = make_tree("src/demo.cc", 3, 4, 0, 0)

    combined, reports = combine_trees(python_tree, cpp_tree)
    counts = read_counts(combined.getroot())

    assert counts.lines_valid == 7
    assert counts.lines_covered == 5
    assert counts.branches_valid == 2
    assert counts.branches_covered == 1
    assert combined.getroot().attrib["line-rate"] == "0.7143"
    filenames = [item.get("filename") for item in combined.findall(".//class")]
    assert filenames == ["tilelang/demo.py", "src/demo.cc"]
    assert [report.language for report in reports] == ["Python", "C++"]


def test_generate_combined_report_writes_single_html_xml_and_summary(tmp_path):
    python_xml = tmp_path / "python.xml"
    cpp_xml = tmp_path / "cpp.xml"
    make_tree("demo.py", 1, 2, 0, 0).write(python_xml)
    make_tree("src/demo.cc", 2, 2, 0, 0).write(cpp_xml)

    output_xml = tmp_path / "coverage.xml"
    output_html = tmp_path / "index.html"
    summary = tmp_path / "summary.txt"
    counts = generate_combined_report(
        python_xml,
        cpp_xml,
        output_xml,
        output_html,
        summary,
        months=8,
        as_of=date(2026, 8, 13),
    )

    assert counts.lines_covered == 3
    assert counts.lines_valid == 4
    assert output_xml.is_file()
    assert "Combined lines:</strong> 3/4 (75.00%)" in output_html.read_text(
        encoding="utf-8"
    )
    assert "Combined lines: 3/4 (75.00%)" in summary.read_text(encoding="utf-8")

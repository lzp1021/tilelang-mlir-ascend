import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from scripts.filter_recent_coverage import (
    GitLineDates,
    filter_coverage_tree,
    subtract_calendar_months,
)


def test_python_git_paths_keep_tilelang_prefix_when_source_directory_is_moved(
    tmp_path, monkeypatch
):
    dates = GitLineDates(tmp_path, date(2025, 12, 13))
    commands = []

    def record_run(command, **kwargs):
        commands.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("scripts.filter_recent_coverage.subprocess.run", record_run)
    dates._load("jit/jit_npu.py")

    assert commands[0][-1] == Path("tilelang/jit/jit_npu.py").as_posix()


def test_subtract_calendar_months_clamps_to_last_day():
    assert subtract_calendar_months(date(2026, 8, 12), 8) == date(2025, 12, 12)
    assert subtract_calendar_months(date(2026, 10, 31), 8) == date(2026, 2, 28)


def test_filter_coverage_tree_removes_old_lines_and_recalculates_rates():
    root = ET.fromstring(
        """
        <coverage lines-valid="3" lines-covered="2" line-rate="0.6667"
                  branches-valid="4" branches-covered="2" branch-rate="0.5">
          <packages><package name="demo" line-rate="0.6667" branch-rate="0.5">
            <classes><class name="demo.py" filename="demo.py"
                            line-rate="0.6667" branch-rate="0.5">
              <lines>
                <line number="1" hits="1" branch="true"
                      condition-coverage="50% (1/2)" missing-branches="2"/>
                <line number="2" hits="0"/>
                <line number="3" hits="1" branch="true"
                      condition-coverage="50% (1/2)" missing-branches="4"/>
              </lines>
            </class></classes>
          </package></packages>
        </coverage>
        """
    )

    total, stats = filter_coverage_tree(
        root, lambda filename, line: filename == "demo.py" and line >= 2
    )

    assert total.lines_valid == 2
    assert total.lines_covered == 1
    assert total.branches_valid == 2
    assert total.branches_covered == 1
    assert root.attrib["line-rate"] == "0.5"
    assert root.attrib["branch-rate"] == "0.5"
    assert [line.attrib["number"] for line in root.findall(".//line")] == ["2", "3"]
    assert stats[0].excluded_lines == 1

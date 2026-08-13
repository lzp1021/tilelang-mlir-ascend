import xml.etree.ElementTree as ET

from scripts.generate_cpp_coverage import (
    build_cobertura_tree,
    collect_recent_counts,
    normalize_source_path,
    parse_lcov,
)


def test_parse_lcov_keeps_repository_cpp_and_merges_duplicate_records(tmp_path):
    source = tmp_path / "src" / "demo.cc"
    source.parent.mkdir()
    source.write_text("int demo() { return 1; }\n", encoding="utf-8")
    external = tmp_path.parent / "external.cc"
    lcov = f"""
SF:{source}
DA:1,1
BRDA:1,0,0,0
end_of_record
SF:{source}
DA:1,2
BRDA:1,0,0,1
end_of_record
SF:{external}
DA:1,10
end_of_record
"""

    files = parse_lcov(lcov, tmp_path)

    assert list(files) == ["src/demo.cc"]
    assert files["src/demo.cc"].lines == {1: 3}
    assert files["src/demo.cc"].branches == {(1, "0", "0"): True}
    assert normalize_source_path(str(external), tmp_path) is None

    runtime_source = tmp_path / "tilelang" / "utils" / "npu_utils.cpp"
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("int runtime();\n", encoding="utf-8")
    assert normalize_source_path(str(runtime_source), tmp_path) == (
        "tilelang/utils/npu_utils.cpp"
    )


def test_recent_cpp_coverage_filters_lines_and_builds_cobertura(tmp_path):
    source = tmp_path / "tilelangir" / "demo.cpp"
    source.parent.mkdir()
    source.write_text("old\nnew\n", encoding="utf-8")
    files = parse_lcov(
        f"""SF:{source}
DA:1,0
DA:2,4
BRDA:2,0,0,1
BRDA:2,0,1,0
end_of_record
""",
        tmp_path,
    )

    total, file_counts, file_lines = collect_recent_counts(
        files, lambda filename, line: filename == "tilelangir/demo.cpp" and line == 2
    )
    tree = build_cobertura_tree(total, file_counts, file_lines)

    assert total.lines_valid == 1
    assert total.lines_covered == 1
    assert total.branches_valid == 2
    assert total.branches_covered == 1
    assert tree.getroot().attrib["line-rate"] == "1"
    line = tree.find(".//line")
    assert line is not None
    assert line.attrib["condition-coverage"] == "50% (1/2)"
    assert ET.tostring(tree.getroot(), encoding="unicode").find("cpp.tilelangir") > 0

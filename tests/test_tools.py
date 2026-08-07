"""Tests for individual tool implementations."""

from harness.tools import EditTool, GlobTool, GrepTool, ReadTool, WriteTool


def test_edit_file_basic_replace(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello\nworld\n")
    result = EditTool().execute(path=str(f), old_string="hello", new_string="hi")
    assert not result.is_error
    assert f.read_text() == "hi\nworld\n"


def test_edit_file_rejects_empty_old_string(tmp_path):
    # Regression test: str.count("") / str.replace("", ...) treat an empty old_string
    # as matching between every character. Observed directly against a live model that
    # tried this (meaning to create a new file) with replace_all=True — it corrupted a
    # ~165 byte file into ~32KB by inserting new_string between every character.
    f = tmp_path / "a.py"
    original = "hello\nworld\n"
    f.write_text(original)

    result = EditTool().execute(path=str(f), old_string="", new_string="INJECTED", replace_all=True)

    assert result.is_error
    assert "cannot be empty" in result.output
    assert f.read_text() == original  # file must be untouched


def test_edit_file_not_found_in_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello\n")
    result = EditTool().execute(path=str(f), old_string="nope", new_string="x")
    assert result.is_error
    assert "not found" in result.output.lower()


def test_edit_file_ambiguous_match_requires_replace_all(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\nx = 1\n")
    result = EditTool().execute(path=str(f), old_string="x = 1", new_string="x = 2")
    assert result.is_error
    assert "not unique" in result.output


def test_write_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "dir" / "f.txt"
    result = WriteTool().execute(path=str(target), content="hi")
    assert not result.is_error
    assert target.read_text() == "hi"


def test_read_file_offset_and_limit(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)))
    result = ReadTool().execute(path=str(f), offset=3, limit=2)
    assert "3\tline3" in result.output
    assert "4\tline4" in result.output
    assert "line5" not in result.output


def test_glob_finds_matching_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = GlobTool().execute(pattern="*.py", path=str(tmp_path))
    assert "a.py" in result.output
    assert "b.txt" not in result.output


def test_grep_finds_pattern(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    pass\n")
    result = GrepTool().execute(pattern="def foo", path=str(tmp_path))
    assert "foo" in result.output

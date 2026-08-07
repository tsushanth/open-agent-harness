from pathlib import Path

from .base import Tool, ToolResult


class EditTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string match in a file with new text. "
        "old_string must appear exactly once unless replace_all is set."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }
    requires_confirmation = True

    def execute(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        text = file_path.read_text()
        count = text.count(old_string)

        if count == 0:
            return ToolResult(output="old_string not found in file", is_error=True)
        if count > 1 and not replace_all:
            return ToolResult(
                output=f"old_string is not unique ({count} matches). Pass replace_all=true or use more context.",
                is_error=True,
            )

        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        file_path.write_text(new_text)
        return ToolResult(output=f"Replaced {count if replace_all else 1} occurrence(s) in {path}")

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

        # An empty old_string is never a sensible edit, and Python's str.count("")/str.replace("", ...)
        # semantics treat it as matching between every character — with replace_all this silently
        # explodes a file's size by inserting new_string N+1 times for an N-character file. Observed
        # directly: a model tried this for a "file doesn't exist yet" case (should have used
        # write_file instead) and it corrupted a file from ~165 bytes to ~32KB in one call.
        if old_string == "":
            return ToolResult(
                output="old_string cannot be empty. Use write_file to create a new file or "
                "overwrite one's full contents; edit_file is for replacing an existing snippet.",
                is_error=True,
            )

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

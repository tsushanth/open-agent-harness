from pathlib import Path

from .base import Tool, ToolResult


class ReadTool(Tool):
    name = "read_file"
    description = "Read a file's contents, optionally a line range. Returns cat -n style output."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
            "offset": {"type": "integer", "description": "1-indexed line to start from.", "default": 1},
            "limit": {"type": "integer", "description": "Max lines to read.", "default": 2000},
        },
        "required": ["path"],
    }

    def execute(self, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)
        try:
            lines = file_path.read_text(errors="replace").splitlines()
        except IsADirectoryError:
            return ToolResult(output=f"{path} is a directory", is_error=True)

        start = max(offset - 1, 0)
        selected = lines[start : start + limit]
        numbered = "\n".join(f"{i + start + 1}\t{line}" for i, line in enumerate(selected))
        return ToolResult(output=numbered or "(empty file)")

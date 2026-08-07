from pathlib import Path

from .base import Tool, ToolResult


class WriteTool(Tool):
    name = "write_file"
    description = "Write content to a file, overwriting it if it exists. Creates parent directories as needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        "required": ["path", "content"],
    }
    requires_confirmation = True

    def execute(self, path: str, content: str) -> ToolResult:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return ToolResult(output=f"Wrote {len(content)} bytes to {path}")

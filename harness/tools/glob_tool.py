from pathlib import Path

from .base import Tool, ToolResult


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern, e.g. 'src/**/*.ts'."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> ToolResult:
        matches = sorted(str(p) for p in Path(path).glob(pattern))
        return ToolResult(output="\n".join(matches) or "(no matches)")

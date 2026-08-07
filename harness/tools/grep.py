import subprocess

from .base import Tool, ToolResult


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents for a regex pattern using ripgrep (falls back to grep -rE)."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "description": "Optional filename glob filter, e.g. '*.py'"},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", glob: str | None = None) -> ToolResult:
        cmd = ["rg", "--line-number", "--color", "never"]
        if glob:
            cmd += ["--glob", glob]
        cmd += [pattern, path]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            grep_cmd = ["grep", "-rEn", pattern, path]
            proc = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30)

        output = proc.stdout or "(no matches)"
        return ToolResult(output=output[-20_000:])

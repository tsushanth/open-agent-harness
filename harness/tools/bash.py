import subprocess

from .base import Tool, ToolResult


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return its stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 60},
        },
        "required": ["command"],
    }
    requires_confirmation = True  # arbitrary shell execution — always gate this

    def execute(self, command: str, timeout: int = 60) -> ToolResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = proc.stdout + proc.stderr
            return ToolResult(output=output[-20_000:], is_error=proc.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(output=f"Command timed out after {timeout}s", is_error=True)

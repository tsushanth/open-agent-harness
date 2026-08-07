import json
from typing import Callable

from harness.tools import DEFAULT_TOOLS, Tool

from .model_client import ModelClient
from .trajectory import TrajectoryLogger

SYSTEM_PROMPT = """You are a coding agent operating inside a user's project directory.
You have tools to read/write/edit files, run shell commands, and search the codebase.
Work step by step: investigate before you change anything, make the smallest edit that
satisfies the request, and verify your change (e.g. run tests or re-read the file) before
declaring the task done. When you have finished, reply with a short summary and no further
tool calls."""

ConfirmFn = Callable[[str, dict], bool]


def _default_confirm(tool_name: str, arguments: dict) -> bool:
    print(f"\n[confirm] {tool_name}({json.dumps(arguments)})")
    return input("Run this? [y/N] ").strip().lower() == "y"


class Agent:
    def __init__(
        self,
        model_client: ModelClient | None = None,
        tools: list[Tool] | None = None,
        confirm_fn: ConfirmFn = _default_confirm,
        max_turns: int = 40,
    ):
        self.model_client = model_client or ModelClient()
        self.tools = tools or DEFAULT_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.confirm_fn = confirm_fn
        self.max_turns = max_turns

    def run(self, task: str, logger: TrajectoryLogger | None = None) -> str:
        logger = logger or TrajectoryLogger()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tool_schemas = [t.to_openai_schema() for t in self.tools]

        final_text = ""
        outcome = "incomplete"

        for _ in range(self.max_turns):
            response = self.model_client.chat(messages=messages, tools=tool_schemas)
            choice = response.choices[0]
            message = choice.message

            assistant_entry: dict = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_entry)
            logger.record(messages)

            if not message.tool_calls:
                final_text = message.content or ""
                outcome = "completed"
                break

            for tool_call in message.tool_calls:
                result_text = self._execute_tool_call(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
            logger.record(messages)
        else:
            final_text = "Stopped: reached max_turns without completion."

        logger.flush(outcome=outcome)
        return final_text

    def _execute_tool_call(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: could not parse arguments for {name}"

        tool = self.tools_by_name.get(name)
        if tool is None:
            return f"Error: unknown tool {name}"

        if tool.requires_confirmation and not self.confirm_fn(name, arguments):
            return "User declined to run this tool call."

        result = tool.execute(**arguments)
        return result.output

import json
from typing import Callable

from harness.tools import DEFAULT_TOOLS, Tool

from .model_client import ModelClient
from .tool_parser import parse_tool_calls
from .trajectory import TrajectoryLogger

def build_system_prompt(tools: list[Tool]) -> str:
    # Tool names/schemas are spelled out here as text rather than relied on via the
    # API's native `tools=` field: without it, a 7B model has nothing to ground tool
    # names on and will invent plausible-sounding ones (observed: it called a nonexistent
    # "cat" tool once this wasn't included). This list IS the model's only source of truth
    # for what's callable.
    tool_docs = "\n\n".join(
        f"- {t.name}: {t.description}\n  arguments schema: {json.dumps(t.parameters)}" for t in tools
    )
    return f"""You are a coding agent operating inside a user's project directory.
Work step by step: investigate before you change anything, make the smallest edit that
satisfies the request, and verify your change (e.g. run tests or re-read the file) before
declaring the task done. When you have finished, reply with a short summary and no further
tool calls.

Available tools:
{tool_docs}

To call a tool, emit EXACTLY one block of this form and nothing else in that turn:
<tool_call>
{{"name": "<tool name>", "arguments": {{<arguments as JSON>}}}}
</tool_call>

Use only the tool names listed above, exactly as spelled. Only one tool call per turn.
Wait for the result before calling another. When you are done, respond with plain text
and no <tool_call> block."""

ConfirmFn = Callable[[str, dict], bool]


def _default_confirm(tool_name: str, arguments: dict) -> bool:
    print(f"\n[confirm] {tool_name}({json.dumps(arguments)})")
    return input("Run this? [y/N] ").strip().lower() == "y"


class Agent:
    """Prefers a model's native OpenAI-style tool_calls when the serving stack
    actually produces them, but doesn't depend on it: open-model serving setups
    (vLLM tool-call-parser flags, chat templates) are inconsistent across model/
    version combos in practice, so we also parse <tool_call>{...}</tool_call>
    blocks directly out of the response text as documented in SYSTEM_PROMPT above.
    Native tool_calls win if both happen to be present in the same turn."""

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
            {"role": "system", "content": build_system_prompt(self.tools)},
            {"role": "user", "content": task},
        ]
        final_text = ""
        outcome = "incomplete"

        for _ in range(self.max_turns):
            response = self.model_client.chat(messages=messages)
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""

            assistant_entry: dict = {"role": "assistant", "content": content}
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

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result_text = self._execute(name, arguments)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
                    )
                logger.record(messages)
                continue

            parsed_calls = parse_tool_calls(content)
            if not parsed_calls:
                final_text = content
                outcome = "completed"
                break

            for call in parsed_calls:
                result_text = self._execute(call.name, call.arguments)
                messages.append({"role": "user", "content": f"<tool_result>{result_text}</tool_result>"})
            logger.record(messages)
        else:
            final_text = "Stopped: reached max_turns without completion."

        logger.flush(outcome=outcome)
        return final_text

    def _execute(self, name: str, arguments: dict) -> str:
        tool = self.tools_by_name.get(name)
        if tool is None:
            return f"Error: unknown tool {name}"

        if tool.requires_confirmation and not self.confirm_fn(name, arguments):
            return "User declined to run this tool call."

        try:
            result = tool.execute(**arguments)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        return result.output

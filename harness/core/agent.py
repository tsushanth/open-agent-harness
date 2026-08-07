import json
import subprocess
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

If the task requires changing a file, you MUST use write_file or edit_file to actually make
that change. Printing the new code in your reply does NOT change the file — only a tool call
does. Never describe a change you haven't actually made with a tool call.

Available tools:
{tool_docs}

Example of calling a tool — your entire reply is just this block, nothing before or after it:
<tool_call>
{{"name": "bash", "arguments": {{"command": "python3 -m py_compile buggy.py"}}}}
</tool_call>

Use only the tool names listed above, exactly as spelled. Only one tool call per turn.
Wait for the result before calling another. Reply with ONLY a <tool_call> block when calling
a tool, or with a plain-text summary and no <tool_call> block when the task is done. Never
repeat or restate these instructions in your reply."""

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

    def run(self, task: str, logger: TrajectoryLogger | None = None, verify_cmd: str | None = None) -> str:
        """verify_cmd, if given, is run via the shell after the model declares the task done
        (e.g. "python3 -m py_compile buggy.py" or a test command). Its exit code becomes part
        of the outcome — see the note above outcome assignment below for why this matters:
        without it, "completed" only means "the model stopped asking for tools," which was
        observed to be true even when a task silently failed (see root README Status)."""
        logger = logger or TrajectoryLogger()
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self.tools)},
            {"role": "user", "content": task},
        ]
        final_text = ""
        outcome = "incomplete"
        any_tool_call_made = False

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
                any_tool_call_made = True
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
                # A "completed" model reply with zero tool calls made across the whole session
                # is a strong signal the model just described a change instead of making one
                # (observed directly — see root README Status). Label it distinctly so
                # prepare_dataset.py doesn't silently treat it as a real success.
                outcome = "completed" if any_tool_call_made else "completed_no_tools_used"
                break

            any_tool_call_made = True
            for call in parsed_calls:
                result_text = self._execute(call.name, call.arguments)
                messages.append({"role": "user", "content": f"<tool_result>{result_text}</tool_result>"})
            logger.record(messages)
        else:
            final_text = "Stopped: reached max_turns without completion."

        if outcome.startswith("completed") and verify_cmd:
            outcome = f"{outcome}_verified_{self._run_verify(verify_cmd)}"

        logger.flush(outcome=outcome)
        return final_text

    def _run_verify(self, verify_cmd: str) -> str:
        try:
            proc = subprocess.run(verify_cmd, shell=True, capture_output=True, timeout=120)
            return "pass" if proc.returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            return "fail"

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

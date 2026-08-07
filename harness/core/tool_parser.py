import json
import re
from dataclasses import dataclass

# Matches the tag conventions different open models actually emit in practice —
# Qwen's trained convention is <tool_call>, but weaker prompting/parser mismatches
# can make a model invent <function_call> instead (observed directly against
# Qwen2.5-Coder-7B-Instruct on vLLM when the server's tool-call-parser didn't match
# the model's tag). We accept either rather than depending on getting the parser
# flag exactly right for every model/server combination.
_TAG_PATTERN = re.compile(
    r"<(tool_call|function_call)>\s*(\{.*?\})\s*</\1>",
    re.DOTALL,
)
_FENCE_PATTERN = re.compile(
    r"```(?:\w+)?\s*(\{.*?\"name\"\s*:.*?\})\s*```",
    re.DOTALL,
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict


def parse_tool_calls(content: str) -> list[ParsedToolCall]:
    """Extracts tool calls from raw model text output. Tries tagged blocks first,
    then falls back to fenced JSON blocks containing a "name" key. Returns an
    empty list if nothing parses — that's the normal "model gave a final answer"
    case, not an error."""
    calls: list[ParsedToolCall] = []

    for match in _TAG_PATTERN.finditer(content):
        raw_json = match.group(2)
        parsed = _try_parse(raw_json)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for match in _FENCE_PATTERN.finditer(content):
        raw_json = match.group(1)
        parsed = _try_parse(raw_json)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    return _scan_bare_json_objects(content)


def _scan_bare_json_objects(content: str) -> list[ParsedToolCall]:
    """Last-resort fallback for weaker instruction-following: a small model may drop
    the <tool_call> wrapper and code fence entirely and just emit the raw JSON object.
    Uses JSONDecoder.raw_decode (rather than a balanced-braces regex, which can't
    handle nested objects in `arguments` correctly) to find valid JSON objects
    anywhere a '{' appears."""
    decoder = json.JSONDecoder()
    calls: list[ParsedToolCall] = []
    for idx, char in enumerate(content):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
            calls.append(ParsedToolCall(name=data["name"], arguments=data["arguments"]))
    return calls


def strip_tool_call_markup(content: str) -> str:
    """Returns content with tool-call blocks removed, for display/logging of the
    model's accompanying prose without the raw JSON noise."""
    text = _TAG_PATTERN.sub("", content)
    text = _FENCE_PATTERN.sub("", text)
    return text.strip()


def _try_parse(raw_json: str) -> ParsedToolCall | None:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ParsedToolCall(name=name, arguments=arguments)

"""Regression tests for harness/core/tool_parser.py.

Each case here is a real model output pattern observed while smoke-testing the harness
against Qwen2.5-Coder-7B-Instruct on vLLM, not a hypothetical — see the README Status
section for the session these came from."""

from harness.core.tool_parser import parse_tool_calls


def test_tag_format():
    content = 'I will call <tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call> now.'
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}


def test_function_call_tag_variant():
    # Observed: model invented <function_call> instead of the documented <tool_call>
    # when the parser it was tested against (vLLM's hermes tool-call-parser) mismatched
    # its actual tag convention.
    content = """```xml
<function_call>
    {"name": "bash", "arguments": {"command": "ls"}}
</function_call>
```"""
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "bash"


def test_fenced_json_any_language_tag():
    # Observed: model used ```bash as the fence language around a JSON tool call,
    # not just ```json or a bare fence.
    content = '```bash\n{"name": "cat", "arguments": {"file": "buggy.py"}}\n```'
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "cat"


def test_bare_json_no_wrapper():
    # Observed: model dropped both the tag and the fence entirely, sometimes after
    # echoing part of the system prompt first.
    content = 'EXACTLY one block of this form and nothing else in that turn:\n{"name": "grep", "arguments": {"pattern": "\\\\bSyntaxError\\\\b", "path": "buggy.py"}}'
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "grep"
    assert calls[0].arguments["pattern"] == "\\bSyntaxError\\b"


def test_multiline_content_with_raw_newlines_still_parses():
    # Observed directly against Claude (not just the smaller open model) via the
    # harness-quality comparison in eval/run_claude_via_harness.py: a write_file call
    # whose "content" argument is multi-line source code, emitted with a literal raw
    # newline instead of an escaped \n. That's invalid JSON under a strict parser, and
    # the default json.loads silently dropped the whole tool call — no error surfaced
    # anywhere, the model's fix was simply never applied. Reproduces the exact content
    # from that failed session (missing opening <tool_call> tag too, which is why this
    # exercises the bare-JSON fallback specifically).
    content = (
        '{"name": "write_file", "arguments": {"path": "e5_bug_default_state.py", '
        '"content": "class Counter:\n    def __init__(self):\n        self.total = 0\n\n'
        '    def increment(self):\n        self.total += 1\n        return self.total\n"}}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert "def __init__(self):" in calls[0].arguments["content"]


def test_no_tool_call_returns_empty():
    assert parse_tool_calls("Task completed. No further action needed.") == []


def test_unrelated_json_does_not_false_positive():
    content = 'Here is some unrelated JSON: {"unrelated": true, "foo": 1} — not a tool call.'
    assert parse_tool_calls(content) == []


def test_tag_form_preferred_over_bare_json_when_both_present():
    # If a well-formed tagged call exists, prefer it over scanning for stray JSON
    # elsewhere in the same message (e.g. in an unrelated code example).
    content = (
        'Example of a bad call: {"name": "not_this", "arguments": {}}\n'
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    )
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "bash"

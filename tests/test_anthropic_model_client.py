"""AnthropicModelClient's .chat() must be interchangeable with ModelClient's inside Agent,
exactly like LocalModelClient (see test_local_model_client.py) — and it must never emit
tool_calls, since the whole point of using it is to route Claude through the same text-based
<tool_call> protocol the open model gets, not a better native tool-calling path. These tests
mock the Anthropic SDK call so no real API request is made."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

from harness.core.agent import Agent
from harness.core.anthropic_model_client import AnthropicModelClient
from harness.core.trajectory import TrajectoryLogger


def _stub_client_with_responses(responses: list[str]) -> AnthropicModelClient:
    """An AnthropicModelClient whose underlying SDK call is stubbed to return canned text,
    skipping real network/auth entirely."""
    client = AnthropicModelClient.__new__(AnthropicModelClient)
    client.model = "claude-sonnet-5"
    client.max_tokens = 1024
    call_count = {"n": 0}

    class _FakeMessages:
        def create(self, **kwargs):
            i = call_count["n"]
            call_count["n"] += 1
            text_block = SimpleNamespace(type="text", text=responses[i])
            return SimpleNamespace(content=[text_block])

    client.client = SimpleNamespace(messages=_FakeMessages())
    return client


def test_response_shape_matches_what_agent_expects():
    client = _stub_client_with_responses(["Task completed."])
    response = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "Task completed."
    assert response.choices[0].message.tool_calls is None


def test_never_emits_native_tool_calls_even_when_tools_arg_passed():
    """Passing tools= must not change the response shape — this client always routes
    through the harness's text-based protocol, never Claude's native tool-use API."""
    client = _stub_client_with_responses(["<tool_call>{\"name\": \"bash\"}</tool_call>"])
    response = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "bash"}}],
    )
    assert response.choices[0].message.tool_calls is None


def test_agent_runs_end_to_end_against_anthropic_client(tmp_path):
    client = _stub_client_with_responses(
        [
            '<tool_call>{"name": "bash", "arguments": {"command": "echo hi"}}</tool_call>',
            "Done, it printed hi.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    result = agent.run("say hi via bash", logger=logger)

    assert result == "Done, it printed hi."
    outcome = json.loads(logger.path.read_text())["outcome"]
    assert outcome == "completed"


def test_system_message_extracted_and_not_duplicated_in_turns():
    """The Anthropic API takes system as a separate top-level param, not a message with
    role="system" — the client must pull it out of messages rather than passing it through
    (which the SDK would reject)."""
    client = _stub_client_with_responses(["ok"])
    captured = {}

    class _CapturingMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    client.client = SimpleNamespace(messages=_CapturingMessages())

    client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "do the task"},
        ]
    )

    assert captured["system"] == "You are a helpful agent."
    assert all(m["role"] != "system" for m in captured["messages"])
    assert captured["messages"] == [{"role": "user", "content": "do the task"}]

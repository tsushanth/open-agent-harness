"""Tests for the agent loop's outcome tracking — see harness/core/agent.py and the root
README Status section for why "completed" alone isn't enough of a success signal."""

import json
from unittest.mock import MagicMock

import pytest

from harness.core.agent import Agent
from harness.core.model_client import ModelClient
from harness.core.trajectory import TrajectoryLogger


def _client_with_responses(responses: list[str]) -> ModelClient:
    call_count = {"n": 0}

    def fake_chat(messages, tools=None, temperature=0.2):
        i = call_count["n"]
        call_count["n"] += 1
        msg = MagicMock()
        msg.content = responses[i]
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = ModelClient.__new__(ModelClient)
    client.chat = fake_chat
    return client


def _outcome(logger: TrajectoryLogger) -> str:
    return json.loads(logger.path.read_text())["outcome"]


def test_real_tool_call_then_finish_is_completed(tmp_path):
    client = _client_with_responses(
        [
            '<tool_call>{"name": "bash", "arguments": {"command": "echo hi"}}</tool_call>',
            "Done, it printed hi.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    result = agent.run("say hi via bash", logger=logger)

    assert result == "Done, it printed hi."
    assert _outcome(logger) == "completed"


def test_no_tool_call_ever_made_is_flagged(tmp_path):
    # Regression test for the real failure observed: the model printed new code as its
    # answer instead of calling write_file, and the loop used to call this "completed."
    client = _client_with_responses(["Here's the fix:\n```python\ndef f(): pass\n```"])
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    agent.run("fix the bug", logger=logger)

    assert _outcome(logger) == "completed_no_tools_used"


def test_verify_cmd_pass_is_recorded(tmp_path):
    client = _client_with_responses(
        [
            '<tool_call>{"name": "bash", "arguments": {"command": "echo hi"}}</tool_call>',
            "Done.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    agent.run("task", logger=logger, verify_cmd="true")

    assert _outcome(logger) == "completed_verified_pass"


def test_verify_cmd_fail_is_recorded(tmp_path):
    client = _client_with_responses(
        [
            '<tool_call>{"name": "bash", "arguments": {"command": "echo hi"}}</tool_call>',
            "Done.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    agent.run("task", logger=logger, verify_cmd="false")

    assert _outcome(logger) == "completed_verified_fail"


def test_no_tools_used_plus_verify_pass_is_still_distinguishable(tmp_path):
    # Even if a verify command happens to pass with no real tool use (e.g. a trivially
    # true command), the outcome string must still carry "_no_tools_used_" so
    # prepare_dataset.py's exact-match filter excludes it.
    client = _client_with_responses(["No tools needed, already correct."])
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    agent.run("task", logger=logger, verify_cmd="true")

    outcome = _outcome(logger)
    assert "no_tools_used" in outcome
    assert outcome != "completed_verified_pass"


def test_unknown_tool_name_reports_error_not_crash(tmp_path):
    client = _client_with_responses(
        [
            '<tool_call>{"name": "not_a_real_tool", "arguments": {}}</tool_call>',
            "Done.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: True)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    result = agent.run("task", logger=logger)

    assert result == "Done."
    messages = json.loads(logger.path.read_text())["messages"]
    tool_result = next(m for m in messages if m["role"] == "user" and "tool_result" in m["content"])
    assert "unknown tool" in tool_result["content"].lower()


def test_confirmation_declined_is_reported_to_model(tmp_path):
    client = _client_with_responses(
        [
            '<tool_call>{"name": "bash", "arguments": {"command": "rm -rf /"}}</tool_call>',
            "Understood, I will not run that.",
        ]
    )
    agent = Agent(model_client=client, confirm_fn=lambda *_: False)
    logger = TrajectoryLogger(output_dir=str(tmp_path))
    result = agent.run("do something dangerous", logger=logger)

    assert result == "Understood, I will not run that."

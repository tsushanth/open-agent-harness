"""LocalModelClient's .chat() interface must be interchangeable with ModelClient's inside
Agent — this is the whole point of it (a drop-in swap to bypass vLLM serving, see
harness/core/local_model_client.py's docstring for why). These tests exercise that contract
without loading a real model (no torch/transformers required)."""

import json

from harness.core.agent import Agent
from harness.core.local_model_client import LocalModelClient, _Choice, _Message, _Response
from harness.core.trajectory import TrajectoryLogger


def _stub_client_with_responses(responses: list[str]) -> LocalModelClient:
    """A LocalModelClient whose .chat() is stubbed to return canned text, skipping
    __init__ entirely (which would otherwise require torch/transformers/a GPU)."""
    client = LocalModelClient.__new__(LocalModelClient)
    call_count = {"n": 0}

    def fake_chat(messages, tools=None, temperature=0.2):
        i = call_count["n"]
        call_count["n"] += 1
        return _Response(choices=[_Choice(message=_Message(content=responses[i]))])

    client.chat = fake_chat
    return client


def test_response_shape_matches_what_agent_expects():
    client = _stub_client_with_responses(["Task completed."])
    response = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "Task completed."
    assert response.choices[0].message.tool_calls is None


def test_agent_runs_end_to_end_against_local_client(tmp_path):
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

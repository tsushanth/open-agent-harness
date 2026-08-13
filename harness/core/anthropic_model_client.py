"""A ModelClient-compatible backend that calls a Claude model via the Anthropic API instead
of an OpenAI-compatible endpoint.

Deliberately does NOT use Claude's native tool-use API. It sends the same system prompt and
gets the same text-based <tool_call>{...}</tool_call> protocol as any other model plugged into
Agent (see build_system_prompt in agent.py) — tool_calls on the returned message is always None,
so Agent's parse_tool_calls() text-parsing path handles it exactly like it does for Qwen.

This is intentional: the point of swapping this client in is to test the harness's own
scaffold/protocol, not to give Claude a better tool-calling interface than the open model gets.
If Claude also stumbles inside this text protocol, that's a harness defect worth fixing — not
a capability gap in the smaller model.
"""

from dataclasses import dataclass, field
from typing import Any

import anthropic


@dataclass
class _Message:
    content: str
    tool_calls: Any = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list = field(default_factory=list)


class AnthropicModelClient:
    """Drop-in replacement for ModelClient's .chat() interface, backed by the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2) -> _Response:
        system = None
        turns = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "tool":
                # Agent.run only appends role="tool" messages when native tool_calls were used,
                # which this client never produces — kept for completeness/interface parity.
                turns.append({"role": "user", "content": m["content"]})
            else:
                turns.append({"role": m["role"], "content": m["content"]})

        # temperature is omitted rather than forwarded: newer Claude models (e.g.
        # claude-sonnet-5) reject the param outright ("temperature is deprecated for this
        # model"), and ModelClient's default of 0.2 has no equivalent meaning here anyway.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=turns,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _Response(choices=[_Choice(message=_Message(content=text))])

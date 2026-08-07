import os

from openai import OpenAI


class ModelClient:
    """Thin wrapper over any OpenAI-compatible chat completions endpoint —
    vLLM, Ollama, LM Studio, or a hosted provider all work here since they
    share the function-calling wire format. Point OAH_BASE_URL at whichever
    one is serving your Qwen2.5-Coder instance."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or os.environ.get("OAH_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("OAH_API_KEY", "not-needed")
        self.model = model or os.environ.get("OAH_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2):
        """tools is optional and OFF by default: the agent's primary protocol is the
        text-based <tool_call> convention in SYSTEM_PROMPT, which works against any
        OpenAI-compatible chat endpoint with no special server config. Passing tools
        here additionally requests native function-calling, which only works if the
        serving stack has that wired up (e.g. vLLM with --enable-auto-tool-choice AND
        a --tool-call-parser that actually matches the model's tag format — in testing
        against vLLM 0.26 + Qwen2.5-Coder-7B-Instruct, no parser value we tried matched,
        so this defaults to off)."""
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)

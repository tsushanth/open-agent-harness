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

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )

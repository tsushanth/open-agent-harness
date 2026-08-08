"""A ModelClient-compatible backend that runs a local transformers model directly (optionally
with a PEFT/LoRA adapter applied), no inference server required.

Built specifically to get past the vLLM serving wall documented in eval/README.md: every attempt
to serve a trained LoRA adapter via vLLM hit a different environment problem (dependency
conflicts, a Network Volume + vllm/vllm-openai crash-loop, a silent OOM-looking death). transformers
+ peft are already installed by axolotl's own dependency tree, so evaluating this way needs zero
additional installs and runs the whole eval loop in-process inside the training pod — no serving,
no networking, no second pod.

Trade-off: much slower per-request than vLLM (no continuous batching, no PagedAttention), but for
a one-off eval of ~10 tasks that's a fine trade for something that actually works.
"""

from dataclasses import dataclass, field
from typing import Any


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


class LocalModelClient:
    """Drop-in replacement for ModelClient's .chat() interface, backed by a local
    transformers model (+ optional PEFT adapter) instead of an HTTP endpoint."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2) -> _Response:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _Response(choices=[_Choice(message=_Message(content=text))])

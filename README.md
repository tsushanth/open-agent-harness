# Open Agent Harness

A Claude-Code-style tool-use scaffold for open coding models — bash, file read/write/edit,
grep/glob, an agentic loop, and a CLI — with every session logged as an SFT-ready trajectory.

The goal isn't just "a CLI that calls an open model with tools." It's a harness whose own
trajectory logs *are* training data: run it enough, and you have a dataset of real tool-call
sequences to fine-tune the underlying model on, closing the loop between "agent scaffold" and
"agent that's actually good at using that scaffold."

## Status

Early. The harness (this repo, today) runs against any OpenAI-compatible tool-calling endpoint —
a local vLLM/Ollama server serving Qwen2.5-Coder, or a hosted provider. SFT training code and a
released fine-tuned checkpoint are the next milestones (see Roadmap).

## Why

Claude Code and similar assistants pair a strong model with a tight agentic loop: a small tool
surface (read, write, edit, bash, search), a system prompt that enforces "investigate before you
change," and a loop that keeps calling tools until the task is done. The loop itself isn't
proprietary — what's hard to get right is a *model* that's actually good at using it: knowing
when to read before editing, when a bash command is safe to run without asking, when to stop.

Open coding models (Qwen2.5-Coder, DeepSeek-Coder, etc.) are strong at code generation but
weren't necessarily trained on this specific interaction pattern. This project's bet: build the
harness first, use it to generate real tool-call trajectories (via a stronger model teacher, or
via your own usage), then SFT an open model on those trajectories so it internalizes the loop.

## Architecture

```
harness/
  tools/          # bash, read_file, write_file, edit_file, grep, glob — each is a
                   # JSON-Schema tool def + execute(); requires_confirmation gates risky ones
  core/
    model_client.py   # OpenAI-compatible chat client (points at vLLM/Ollama/hosted)
    agent.py           # the agentic loop: call model -> execute tool calls -> repeat
    trajectory.py       # logs every session as OpenAI-format chat messages (JSONL) —
                         # the exact format axolotl/trl/LLaMA-Factory expect for SFT
  cli.py            # `oah "task description"` entrypoint

data/trajectories/  # session logs land here (gitignored — this is your dataset, not the repo's)
training/           # SFT scripts (see Roadmap — not yet implemented)
```

## Setup

Requires Python 3.10+ and an OpenAI-compatible endpoint serving a tool-calling model
(e.g. `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes`,
or Ollama with a tool-calling-capable model).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

export OAH_BASE_URL=http://localhost:8000/v1   # your vLLM/Ollama endpoint
export OAH_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

oah "add a .gitignore for a Python project in the current directory"
```

By default every `bash`, `write_file`, and `edit_file` call prompts for confirmation before
running. Pass `--yolo` to skip that (only do this in a sandbox/container — this executes
arbitrary shell commands chosen by the model).

## Safety

This harness gives a model shell access. Treat it like you'd treat Claude Code or any other
agentic coding tool: run it in a directory/container you're comfortable with it modifying, review
`--yolo` usage carefully, and don't point it at a model you don't trust with production
credentials in its environment.

## Roadmap

1. **Harness** (done) — minimal tool set, agentic loop, trajectory logging.
2. **Trajectory collection** — generate a corpus of tool-call trajectories: either distilled from
   a stronger teacher model solving real coding tasks with this harness, or collected from your
   own usage. Target format is already the harness's native JSONL output.
3. **SFT** — fine-tune Qwen2.5-Coder (7B to start, 32B as a stretch) on the trajectory corpus using
   LoRA via axolotl or trl. Training scripts land in `training/`.
4. **Eval** — benchmark the fine-tuned model's tool-use behavior against the base model on a held-out
   task set (does it read before editing? does it stop when done? does it avoid unnecessary bash
   calls?).
5. **Release** — publish the LoRA adapter (and merged weights, if licensing allows) on Hugging Face.

## Contributing

Early-stage — issues and PRs on the harness (new tools, better trajectory formats, safety
improvements) are welcome. The SFT pipeline isn't built yet; if you want to help with that
specifically, open an issue to coordinate before duplicating work.

## License

Apache-2.0 — see [LICENSE](LICENSE).

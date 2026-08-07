# Open Agent Harness

A Claude-Code-style tool-use scaffold for open coding models — bash, file read/write/edit,
grep/glob, an agentic loop, and a CLI — with every session logged as an SFT-ready trajectory.

The goal isn't just "a CLI that calls an open model with tools." It's a harness whose own
trajectory logs *are* training data: run it enough, and you have a dataset of real tool-call
sequences to fine-tune the underlying model on, closing the loop between "agent scaffold" and
"agent that's actually good at using that scaffold."

## Status

Harness validated end-to-end against a real endpoint: Qwen2.5-Coder-7B-Instruct served via vLLM
on a rented GPU, given a Python file with a syntax error, correctly diagnosed and fixed it using
the tool-call loop below. A real trajectory from that run is at
[`data/trajectories/example-successful-run.jsonl`](data/trajectories/example-successful-run.jsonl).

Notably, vLLM's native `--tool-call-parser` flag proved unreliable across model/version
combinations in practice (a mismatched parser silently dropped tool calls; an invalid parser name
crash-looped the server) — see `harness/core/tool_parser.py` for why the harness parses tool calls
directly out of response text instead of depending on that flag.

**A more important finding from the same session, and the actual motivation for this project's
SFT phase:** running a batch of varied tasks (add a function, fix a bug) against
Qwen2.5-Coder-7B-Instruct surfaced two distinct base-model reliability gaps, even with a carefully
worded system prompt:
- One task, the model just printed the new code as prose in its reply instead of calling
  `write_file`/`edit_file` — no file was actually changed, yet the loop still ended with the model
  believing (and stating) the task was done.
- Another task, after a prompt fix for the above, the model invented yet another ad hoc tool-call
  format (pseudo-CLI flags: `edit_file --path "calc.py" --old_string ...`) that didn't match any
  of the three formats the harness parses for.

Neither is a harness bug — the harness correctly logs what happened either way. It's a base-model
capability gap: **`outcome: "completed"` in a trajectory currently means "the model stopped
requesting tools," not "the task was verified done."** Trajectories need a verification/review step
before use as training data (see `training/README.md`), and this specific failure mode — sounding
done without being done, or drifting off a specified protocol — is exactly what SFT on curated
trajectories is meant to correct. It's evidence for the project's core bet, not against it.

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
training/           # dataset prep + axolotl LoRA config (see training/README.md — config is
                     # written but not yet run; needs a real trajectory corpus first)
```

## Setup

Requires Python 3.10+ and an OpenAI-compatible chat completions endpoint — any model works, no
special tool-calling server config needed. The harness's tool-call protocol is plain text (see
`harness/core/agent.py`'s system prompt), not dependent on a server's native function-calling
support, since that proved unreliable in practice (see Status above).

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --max-model-len 8192
# or: ollama run qwen2.5-coder:7b
```

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

1. **Harness** (done, validated) — minimal tool set, agentic loop, trajectory logging. Confirmed
   working end-to-end against a real Qwen2.5-Coder-7B-Instruct endpoint (see Status above).
2. **Trajectory collection** (next) — generate a corpus of tool-call trajectories: either distilled
   from a stronger teacher model solving real coding tasks with this harness, or collected from
   your own usage. So far: 6 real sessions across 2 batches, 2 of which pass `prepare_dataset.py`'s
   filter (see [`data/trajectories/batch-2026-08-07/README.md`](data/trajectories/batch-2026-08-07/README.md)
   for a concrete pass/fail breakdown — this base model's real unassisted success rate on small,
   well-specified tasks was 1 of 4 in that batch). Need dozens-to-low-hundreds across varied task
   types before a training run is worth it.
3. **SFT** (config written, not run) — `training/qwen2.5-coder-7b-lora.yaml` is a ready QLoRA config
   for Qwen2.5-Coder-7B-Instruct via axolotl, plus `training/prepare_dataset.py` to build the
   training set from `data/trajectories/`. Blocked on step 2 — see `training/README.md` for why
   running it now would just overfit on almost no data.
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

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

One finding from that verification work *was* a real harness bug, not a model limitation: a model
called `edit_file` with an empty `old_string` (meaning to create a new file — it should have used
`write_file`), and Python's `str.replace("", ...)` semantics under `replace_all=True` inserted the
new text between every character of the target file, ~165 bytes exploding to ~32KB of duplicated
garbage in one tool call. Now rejected outright with a clear error steering the model toward
`write_file` instead — see `training/README.md` for the full story and `tests/test_tools.py`.

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
2. **Trajectory collection** (in progress) — generate a corpus of tool-call trajectories: either
   distilled from a stronger teacher model solving real coding tasks with this harness, or
   collected from your own usage. So far: 84 real sessions across 7 batches + 2 loose examples,
   45 of which pass `prepare_dataset.py`'s filter. Pass rate by batch: 25% (a) → 20% (b) → 64% (c)
   → 69% (d) → 36% (e, confounded by shared-file batch design) → **83% (f)** → 58% (g). Batches f
   and g both used strict one-task-per-file design (the fix for e's confound) and landed in a
   55-85% band — the honest read is that's roughly the real ceiling for this
   harness/model/fix combination, with natural batch-to-batch variance, not a single fixed number.
   See [`batch-2026-08-08-d/README.md`](data/trajectories/batch-2026-08-08-d/README.md).
3. **SFT** (done, reliable) — trained `training/qwen2.5-coder-7b-lora.yaml`'s QLoRA config against
   the 45-example corpus. Loss converges cleanly over 3 epochs (~0.11 → ~0.04-0.06 across runs),
   producing a real 154MB LoRA adapter (40M trainable params, 0.53% of the 7.6B total). Hit and
   fixed a real training-time bug along the way: axolotl auto-enables a fused CUDA kernel
   optimization that crashed with a misleading "NVIDIA driver too old" error on 4 of 6 rented
   hosts — root-caused to specific config flags and disabled in the config, confirmed fixed on 2
   more hosts afterward. See `training/README.md`. The adapter isn't committed to this repo (154MB
   exceeds GitHub's 100MB limit without LFS, and model weights belong in a model registry, not a
   git repo) — reproduce it with the command there, or wait for the Release step below.
4. **Eval** (half done, blocked on a different issue) — benchmark the fine-tuned model against the
   base model on a held-out task set of 10 tasks in fresh domains never seen in training. **Base
   model result: 8/10 (80%)**, consistent with the established 55-85% band. **LoRA-adapted result:
   blocked** — not by the training issue above (that's resolved), but by a separate problem getting
   vLLM to *serve* the trained adapter for live comparison: three different serving approaches (raw
   `pip install vllm` in the training pod, a shared RunPod Network Volume with the proven
   `vllm/vllm-openai` image, an older pinned vLLM version) each hit a different failure with no
   clean fix found. 10 pod attempts total for this specific sub-problem. See `eval/README.md` for
   the full detail and recommended next approaches (a custom pre-built Docker image, or evaluating
   via plain `transformers`+`peft` instead of vLLM).
5. **Release** (not started) — publish the LoRA adapter (and merged weights, if licensing allows)
   on Hugging Face. Needs a Hugging Face account/token — not yet configured for this project.

## Contributing

Early-stage — issues and PRs on the harness (new tools, better trajectory formats, safety
improvements) are welcome. The SFT pipeline isn't built yet; if you want to help with that
specifically, open an issue to coordinate before duplicating work.

## License

Apache-2.0 — see [LICENSE](LICENSE).

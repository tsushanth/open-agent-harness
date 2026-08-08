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
   collected from your own usage. So far: 95 real sessions across 8 batches + 2 loose examples,
   51 of which pass `prepare_dataset.py`'s filter. Pass rate by batch: 25% (a) → 20% (b) → 64% (c)
   → 69% (d) → 36% (e, confounded by shared-file batch design) → 83% (f) → 58% (g) → 55% (h).
   Batches a-g were almost entirely "add a function/method to a file" — the eval result below
   suggested that narrowness caused overfitting, so batch h deliberately shifted to bug fixes,
   refactors, and multi-step tasks instead. Lower pass rate there is expected (harder task shapes)
   and not a regression — see
   [`batch-2026-08-08-e/README.md`](data/trajectories/batch-2026-08-08-e/README.md).
3. **SFT** (done, reliable) — trained `training/qwen2.5-coder-7b-lora.yaml`'s QLoRA config twice
   (45-example corpus, then 51 after batch h). Loss converges cleanly over 3 epochs each time,
   producing a real ~154MB LoRA adapter (40M trainable params, 0.53% of the 7.6B total). Hit and
   fixed a real training-time bug along the way: axolotl auto-enables a fused CUDA kernel
   optimization that crashed with a misleading "NVIDIA driver too old" error on 4 of 6 rented
   hosts — root-caused to specific config flags and disabled in the config, confirmed fixed on
   4 hosts afterward. See `training/README.md`. The adapter isn't committed to this repo (154MB
   exceeds GitHub's 100MB limit without LFS, and model weights belong in a model registry, not a
   git repo) — reproduce it with the command there, or wait for the Release step below.
4. **Eval** (done — negative result, narrowed to a specific cause) — benchmark the fine-tuned model
   against base on two held-out sets: Set A (10 "add a function" tasks) and Set B (8 bug-fix/
   refactor tasks, added to test whether corpus diversification helped). Every checkpoint so far
   underperforms base on Set B specifically. The investigation had two stages:
   - **Stage 1** (recipe hypothesis): Set A run 1 (45 ex) base 8/10 vs. LoRA 6/10; Set A run 2
     (51 ex) base 7/10 vs. LoRA 6/10 (identical pattern); **Set B (51 ex) base 6/8 vs. LoRA
     3/8 — the biggest gap, on the exact task shape the new training data targeted.** Ruled out
     "corpus too narrow" (Set B's examples directly targeted bug-fixing) in favor of "training
     recipe too aggressive" (3 epochs, `learning_rate: 2e-4` — a catastrophic-forgetting shape).
   - **Stage 2** (tested that hypothesis): trained a "gentle" variant (1 epoch, `5e-5`,
     `qwen2.5-coder-7b-lora-gentle.yaml`) and re-ran both sets. **Set A recovered to 7/10 (near
     base), but Set B stayed exactly 3/8 with the identical failure pattern.** So the recipe
     hypothesis was only half right — over-aggressive training explains Set A's regression but not
     Set B's, which needs a different fix (more bug-fix training volume, or a bad example in the
     current 6 — see `eval/README.md` for the full write-up and next steps).

   Getting *any* working comparison took 10 failed pod attempts trying to serve the adapter via
   vLLM before abandoning that entirely for
   [`harness/core/local_model_client.py`](harness/core/local_model_client.py) — a `transformers`+
   `peft` backend that runs in-process, no serving layer, now a real reusable part of the harness
   (drop-in `ModelClient` replacement, tested, used successfully across 4 eval runs).
5. **Release** (not started, deliberately) — publish the LoRA adapter (and merged weights, if
   licensing allows) on Hugging Face. Needs a Hugging Face account/token — not yet configured.
   Every checkpoint trained so far underperforms the base model on both eval sets; publishing one
   now would ship a regression. Blocked on fixing the training recipe (see Eval above) first.

## Contributing

Early-stage — issues and PRs on the harness (new tools, better trajectory formats, safety
improvements) are welcome. The SFT pipeline isn't built yet; if you want to help with that
specifically, open an issue to coordinate before duplicating work.

## License

Apache-2.0 — see [LICENSE](LICENSE).

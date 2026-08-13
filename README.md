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
   collected from your own usage. So far: 108 real sessions across 11 batches + 2 loose examples,
   63 of which pass `prepare_dataset.py`'s filter. Batches a-g (the original letters, mostly "add a
   function/method to a file") ranged 20-83% pass rate; four later batches deliberately targeted
   bug fixes, refactors, and multi-step tasks to grow that underrepresented shape per the Eval
   findings below — batch h (36%, 2 of 6 apparent passes later found corrupted),
   `batch-2026-08-08-f` (27%, 1 of 5 apparent passes later found a false positive),
   `batch-2026-08-09-g` (40%, all 6 apparent passes confirmed genuine), and `batch-2026-08-09-i`
   (33%, 1 of 6 apparent passes later found a false positive caused by a bug in the verify command
   itself — see its README). See
   [`batch-2026-08-08-e/README.md`](data/trajectories/batch-2026-08-08-e/README.md),
   [`batch-2026-08-08-f/README.md`](data/trajectories/batch-2026-08-08-f/README.md),
   [`batch-2026-08-09-g/README.md`](data/trajectories/batch-2026-08-09-g/README.md), and
   [`batch-2026-08-09-i/README.md`](data/trajectories/batch-2026-08-09-i/README.md) — combined,
   the corpus now has 19 clean bug-fix/refactor examples, within reach of the 20-30+ target
   `eval/README.md` called for. A consistent pattern held across all four later batches:
   refactor tasks failed far more often than plain bug fixes (only 1 genuine refactor pass out of
   12 attempted across these batches) — worth treating as a real base-model weak spot with this
   harness's tool-call protocol, not batch noise.
3. **SFT** (done, reliable) — trained `training/qwen2.5-coder-7b-lora.yaml`'s QLoRA config four
   times across two corpus versions and two recipes. Loss converges cleanly every time, producing
   a real ~154MB LoRA adapter (40M trainable params, 0.53% of the 7.6B total). Hit and fixed a real
   training-time bug along the way: axolotl auto-enables a fused CUDA kernel optimization that
   crashed with a misleading "NVIDIA driver too old" error on 4 of 6 rented hosts — root-caused to
   specific config flags and disabled in the config, confirmed fixed on 4 hosts afterward. See
   `training/README.md`. The adapter isn't committed to this repo (154MB exceeds GitHub's 100MB
   limit without LFS, and model weights belong in a model registry) — reproduce it with the
   command there, or wait for the Release step below.
4. **Eval** (done — decisive result) — benchmark the fine-tuned model against base on two held-out
   sets: Set A (10 "add a function" tasks) and Set B (8 bug-fix/refactor tasks). Ruled out three
   explanations for Set B's regression via direct experiment, not guesswork, before finding the
   real one:
   - **Not the training recipe** — a "gentle" variant (1 epoch, `lr 5e-5` vs. the original 3
     epochs/`2e-4`) recovered most of Set A's regression but left Set B completely unchanged
     (stayed at exactly 3/8, identical failure pattern).
   - **Not data corruption** — manually re-reading all 6 of batch h's "passing" trajectories found
     2 genuinely corrupted ones (a 79-message repetition loop after an already-successful fix, and
     a 51-message session that never actually edited the file but passed a behavioral-only verify
     check anyway). Removed both, retrained clean — Set B: still exactly 3/8, third time running.
   - **The actual answer: corpus composition, not cleanliness or recipe.** Training with the 4
     remaining clean bug-fix/refactor examples *excluded entirely* (only the 44 "add a function"
     examples) produced the best checkpoint by far: **Set A 8/10 (beats base's own 7-8/10), Set B
     4/8 (finally breaks the stuck-at-3/8 pattern).** 4 examples of an underrepresented task shape
     was worse than zero — too small a sample to teach anything generalizable, diluting what the
     well-represented data was teaching well. Still short of beating base on Set B (4/8 vs. 6-7/8)
     — the fix isn't "remove the data forever," it's "don't include a token amount of it; get
     enough to matter or leave it out." See `eval/README.md` for the full 6-run comparison table.

   Getting *any* working comparison took 10 failed pod attempts trying to serve the adapter via
   vLLM before abandoning that entirely for
   [`harness/core/local_model_client.py`](harness/core/local_model_client.py) — a `transformers`+
   `peft` backend that runs in-process, no serving layer, now a real reusable part of the harness
   (drop-in `ModelClient` replacement, tested, used successfully across 7 eval runs).
5. **Harness-quality loop** (new, first pass done) — a second, orthogonal evaluation axis to
   SFT: run the exact same held-out tasks through the harness with Claude as the model client
   (`harness/core/anthropic_model_client.py`, `eval/run_claude_via_harness.py`) instead of Qwen.
   Since Claude uses the identical text-based `<tool_call>` protocol — no native tool-calling
   API, same system prompt, same tools — a Claude failure inside this harness is evidence of a
   *harness* defect, not a small-model capability gap. First run: Set A 10/10, Set B 7/8 (see
   [`eval/claude-via-harness-2026-08-12.json`](eval/claude-via-harness-2026-08-12.json)), and it
   immediately found a real bug: a well-formed `write_file` call was silently dropped because its
   multi-line file content contained a literal raw newline instead of `\n`, which Python's
   strict JSON parser rejects with no visible error — the fix was simply never applied, and
   nothing in the harness surfaced why. Fixed by parsing with `strict=False`
   (`harness/core/tool_parser.py`), confirmed by re-running the same task after the fix. A second,
   rarer failure (structurally malformed JSON — a genuine model typo, not a leniency gap)
   surfaced a real *design* gap worth a future fix: the harness currently treats any
   unparseable tool-call text as the model's final answer and ends the session, instead of
   telling the model its JSON was malformed and giving it a turn to retry.
   [`eval/judge.py`](eval/judge.py) adds an LLM-judge pass that, in its first version, rated
   each resulting diff on independent correctness/cleanliness scores and cross-referenced the
   same task's outcome from the current best Qwen checkpoint — flagging 5 tasks where Claude
   succeeds via this harness but Qwen's checkpoint still fails, real evidence that most of
   Qwen's remaining gap is model capability, not the scaffold. Two follow-up methodology fixes
   (2026-08-13), both closing gaps between this loop and its actual RLHF/DPO inspiration: (1)
   every eval script now runs each task `OAH_SAMPLES` times and reports the modal outcome
   (`eval/run_utils.py`) — single-sample verdicts were provably noisy (the exact same 18 tasks
   flipped outcome on reruns with no code changes); re-running Claude at `OAH_SAMPLES=3` got a
   clean Set A 10/10, Set B 8/8, and majority-voting *surfaced*, rather than hid, a real known
   refactor-verify weakness on one task rather than reporting it as a random one-off failure.
   (2) the judge moved from independent absolute scores to **pairwise** comparison — closer to
   what DPO's preference model actually does (Bradley-Terry over pairs, not a scalar reward per
   side) and better-calibrated for an LLM judge in practice. Absolute scoring remains as a
   fallback until Qwen's eval scripts (now also diff-saving) produce a fresh checkpoint to
   compare head-to-head.
6. **Release** (not started, deliberately) — publish the LoRA adapter (and merged weights, if
   licensing allows) on Hugging Face. Needs a Hugging Face account/token — not yet configured.
   The best checkpoint so far (44 examples, no bug-fix data, gentle recipe) beats base on Set A but
   not yet Set B — real progress, not yet a green light. Publishing now would still ship a
   regression on bug-fix/refactor tasks; blocked on more bug-fix trajectory volume (see Eval
   above) first.

## Contributing

Early-stage — issues and PRs on the harness (new tools, better trajectory formats, safety
improvements) are welcome. The SFT pipeline isn't built yet; if you want to help with that
specifically, open an issue to coordinate before duplicating work.

## License

Apache-2.0 — see [LICENSE](LICENSE).

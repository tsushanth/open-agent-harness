# Eval

Held-out evaluation: 10 tasks in fresh domains never seen in `data/trajectories/`'s training
batches (sphere/cylinder volume, warehouse shipping, string case conversion, bracket nesting
depth, leaderboard, unit conversion, task priority queue, hex color normalization, matrix max,
decreasing streaks) — one task per file, matching the methodology that produced the cleanest
signal during trajectory collection (see `data/trajectories/batch-2026-08-08-c/README.md`).

## Result

Ran twice: once against the 45-example corpus (almost entirely "add a function to a file"), once
after adding 6 deliberately different tasks (bug fixes, refactors, multi-step) for a 51-example
corpus (see `data/trajectories/batch-2026-08-08-e/README.md`).

| Model | Run 1 (45 examples) | Run 2 (51 examples, +diverse tasks) |
|---|---|---|
| Base (Qwen2.5-Coder-7B-Instruct) | 8/10 (80%) | 7/10 (70%) |
| Base + LoRA adapter | 6/10 (60%) | 6/10 (60%) — **identical task-by-task pattern to run 1** |

Full per-task breakdown: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json) (run 1),
[`lora-vs-base-2026-08-08-retrain.json`](lora-vs-base-2026-08-08-retrain.json) (run 2).

**Run 1: the fine-tune performed worse than the base model, in a specific, non-random pattern.**
The one task base failed via `completed_no_tools_used` (printed code as prose instead of calling
a tool — `cylinder_volume`), LoRA fixed — exactly what the training corpus targeted. But LoRA
newly failed 3 tasks base passed (`ship`, `top_player`, `km_to_miles`), always by calling a tool
and getting the wrong result, never by skipping it. Read together: the fine-tune learned "always
attempt a tool call" from a corpus that was almost entirely that one task shape, at some cost to
correctness elsewhere — a plausible small/narrow-dataset overfitting signature.

**Run 2, after adding task-shape diversity: no change on this eval set.** Base's score moved (8→7,
model sampling variance at temperature>0, not a real change — this eval set isn't held perfectly
fixed run to run in terms of model behavior). LoRA's score and *exact task-by-task pattern* were
identical to run 1. This makes sense in hindsight rather than being a null result: **the 6 new
training examples were bug-fixes/refactors, and this held-out eval set is entirely "add a
function" tasks** — there was no reason to expect the new examples to move the needle on a
benchmark that doesn't test what they taught. The eval set itself needs its own diversity (bug-fix
and refactor held-out tasks) to actually test whether the corpus diversification helped — that
wasn't built this round. Don't read run 2 as "diversifying the corpus didn't help"; read it as
"this specific eval doesn't measure what changed."

## How this was actually run

Getting a working LoRA-vs-base comparison took far longer than expected — 10 pod attempts trying
to serve the adapter via vLLM before abandoning that entirely:

1. Installing vLLM fresh alongside axolotl in the training pod: a `flash_attn` ABI break, then a
   conflicting `libcudnn` version (isolating vLLM into its own venv fixed both).
2. A RunPod Network Volume shared between a training pod and a second pod using the already-proven
   `vllm/vllm-openai` image: that image's fixed entrypoint can't run arbitrary shell (so no
   file-fetching before serving), and attaching a network volume to it caused an immediate silent
   crash-loop even with plain non-LoRA args — sanity-checked and confirmed, never root-caused (no
   log access on that image).
3. An older pinned vLLM (`0.7.3`) back in the axolotl image: got further (model loading started)
   before dying silently, no traceback, consistent with an OOM-kill but not confirmed.

**What actually worked:** stopped trying to serve the adapter at all.
[`harness/core/local_model_client.py`](../harness/core/local_model_client.py) runs a local
`transformers` + `peft` model directly in-process — same `.chat()` interface as `ModelClient`, so
it drops into `Agent` unmodified. `transformers` and `peft` are already installed by axolotl's own
dependency tree, so this needed zero additional installs and no networking, no serving layer, no
second pod — training and eval ran back-to-back inside one pod. Much slower per-request than vLLM
(no continuous batching), irrelevant for a one-off 20-task eval.

## Next steps

- **Build a second held-out eval set that includes bug-fix and refactor tasks**, not just
  "add a function." The current 10-task set can only measure whether training changes affect
  that one task shape — run 2 above is the direct lesson: it couldn't detect whether batch h's
  diversification helped, because nothing in the eval set tests bug-fixing or refactoring.
- **More trajectory volume across all task shapes**, not just more of the same. 51 examples is
  still small for a 7B model; the corpus needs both more bug-fix/refactor examples specifically
  (batch h was a first, 6-example start) and more raw volume generally.
- Keep using `LocalModelClient` for future eval runs — proven twice now, no serving-layer
  debugging needed either time.
- If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
  vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
  rather than fresh `pip install`s per pod — none of the three serving failures documented above
  were reliably reproducible enough to trust a fresh install each time.

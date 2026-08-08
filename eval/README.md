# Eval

Held-out evaluation: 10 tasks in fresh domains never seen in `data/trajectories/`'s training
batches (sphere/cylinder volume, warehouse shipping, string case conversion, bracket nesting
depth, leaderboard, unit conversion, task priority queue, hex color normalization, matrix max,
decreasing streaks) — one task per file, matching the methodology that produced the cleanest
signal during trajectory collection (see `data/trajectories/batch-2026-08-08-c/README.md`).

## Result

| Model | Pass rate |
|---|---|
| Base (Qwen2.5-Coder-7B-Instruct) | **8/10 (80%)** |
| Base + LoRA adapter (this project's SFT, 45 training examples) | **6/10 (60%)** |

Full per-task breakdown: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json).

**The fine-tune performed worse than the base model on held-out tasks.** This is a real,
meaningful result, not a broken eval — it's exactly the overfitting risk flagged throughout
`training/README.md` from the start: 45 examples is small, and this is direct evidence it wasn't
enough to teach general tool-use behavior without cost elsewhere.

The failure pattern is specific, not random noise, which is what makes this a genuine finding
rather than an artifact:
- The one task the base model failed via `completed_no_tools_used` (printed code as prose instead
  of calling a tool — `cylinder_volume`), the **LoRA model fixed**. This is exactly the behavior
  the training corpus was built to correct, and it worked on this held-out task.
- But the LoRA model newly failed 3 tasks the base model passed (`ship`, `top_player`,
  `km_to_miles`) — and in every one of those cases, unlike the base model's failures, it *did*
  call a tool, just got the wrong result. None of the LoRA failures are `no_tools_used`.

Read together: the fine-tune successfully learned "always attempt a tool call" from the training
corpus (which is almost entirely "add a function to a file" tasks resolved via `write_file`/
`edit_file`), but at the cost of correctness on some tasks it would have gotten right by being
more careful — a plausible small-dataset overfitting signature, not a broken model. 45 examples
skewed toward one narrow task shape (single-function additions) taught a narrow lesson.

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

- **More trajectory volume, more task-shape diversity.** The current corpus is almost entirely
  "add a function/method to a file" — the eval result suggests that narrowness, not just the
  count, contributed to the regression. Bug fixes, refactors, and search-heavy tasks are
  underrepresented; batches so far were designed to be pass/fail-verifiable more than
  representative of real usage.
- **Re-run this same 10-task eval** after any future training run, using `LocalModelClient` (now
  proven) rather than re-attempting vLLM serving.
- If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
  vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
  rather than fresh `pip install`s per pod — none of the three serving failures above were
  reliably reproducible enough to trust a fresh install each time.

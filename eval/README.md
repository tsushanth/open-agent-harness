# Eval

Two held-out eval sets, both in fresh domains never seen in `data/trajectories/`'s training
batches:
- **Set A** (10 tasks, "add a function" shape): sphere/cylinder volume, warehouse shipping, string
  case conversion, bracket nesting depth, leaderboard, unit conversion, task priority queue, hex
  color normalization, matrix max, decreasing streaks.
- **Set B** (8 tasks, bug-fix/refactor shape, added to actually test whether corpus
  diversification helped): 5 bug fixes (wrong comparison, index error, accumulator init, string
  case sensitivity, class-vs-instance state), 2 refactors (repeated code, nested conditionals),
  1 multi-step task (use an existing constant instead of a hardcoded value).

## Result

| Model | Set A, run 1 (45 ex, aggressive) | Set A, run 2 (51 ex, aggressive) | Set A (51 ex, gentle) | Set B (51 ex, aggressive) | Set B (51 ex, gentle) |
|---|---|---|---|---|---|
| Base | 8/10 (80%) | 7/10 (70%) | — | 6/8 (75%) | — |
| Base + LoRA | 6/10 (60%) | 6/10 (60%) | **7/10 (70%)** | **3/8 (37.5%)** | **3/8 (37.5%)** |

"Aggressive" = original recipe (3 epochs, `learning_rate: 2e-4`, `qwen2.5-coder-7b-lora.yaml`).
"Gentle" = 1 epoch, `learning_rate: 5e-5`, everything else identical
(`qwen2.5-coder-7b-lora-gentle.yaml`), base not re-run since it doesn't depend on training recipe.

Full per-task breakdowns: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json) (Set A
run 1), [`lora-vs-base-2026-08-08-retrain.json`](lora-vs-base-2026-08-08-retrain.json) (Set A run
2, aggressive), [`bugfix-refactor-lora-vs-base-2026-08-08.json`](bugfix-refactor-lora-vs-base-2026-08-08.json)
(Set B, aggressive), [`gentle-recipe-set-a-2026-08-08.json`](gentle-recipe-set-a-2026-08-08.json) /
[`gentle-recipe-set-b-2026-08-08.json`](gentle-recipe-set-b-2026-08-08.json) (gentle recipe, both
sets).

**The recipe-vs-data hypothesis was only half right, and the gentle-recipe test is what proved
it.** Turning down epochs/learning rate mostly fixed Set A (7/10, back near base's own 7-8/10
range) — consistent with the aggressive recipe over-training on the dominant "add a function"
task shape. But **Set B stayed exactly 3/8, identical pass/fail pattern, at both recipes.**
Reducing training intensity had zero effect on the bug-fix regression. That rules out "too
aggressive across the board" as the full explanation — whatever is hurting bug-fix/refactor
performance specifically isn't primarily a function of epochs or learning rate. Two live
hypotheses, neither tested yet: (a) 6 bug-fix/refactor training examples is too few to teach
that skill at *any* training intensity — more data of that specific shape is needed, not less
aggressive training on the current amount; (b) something about how those 6 examples were phrased
or verified is actively teaching the wrong lesson (worth manually re-reading
`data/trajectories/batch-2026-08-08-e/`'s sessions for a bad example, rather than assuming volume
is the only lever).

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

The gentle-recipe test narrowed the problem: Set A responds to recipe tuning, Set B doesn't at
all. That points at data volume/quality for bug-fix tasks specifically, not general training
intensity:

1. **Manually re-read the 6 batch-h training trajectories** (`data/trajectories/batch-2026-08-08-e/`)
   before adding more — if one of them taught a subtly wrong lesson (e.g. a verify command that
   passed for the wrong reason, or a fix that doesn't generalize), more volume of the same
   pattern won't help. This is a cheap check to rule out before spending more GPU time.
2. **Collect meaningfully more bug-fix/refactor trajectories** (10-20+, not 6) if the manual
   review doesn't turn up a bad example — the current test is likely underpowered either way.
3. **Use the gentle recipe going forward** for Set A's sake (it recovered most of the regression
   there with zero apparent downside on Set B) while investigating the bug-fix-specific issue
   separately.
4. Keep using `LocalModelClient` for future eval runs — proven across 4 runs now (Set A x3, Set B
   x2), no serving-layer debugging needed any time.
5. If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
   vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
   rather than fresh `pip install`s per pod — none of the three serving failures documented above
   were reliably reproducible enough to trust a fresh install each time.
6. **Do not publish a checkpoint to Hugging Face (Release milestone) until a run beats base on
   both eval sets.** Every trained checkpoint so far — 3 recipes across 2 corpus sizes — still
   underperforms base on Set B specifically.

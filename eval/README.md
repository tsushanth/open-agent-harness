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

| Model | Set A, run 1 (45 examples) | Set A, run 2 (51 examples) | Set B (51 examples) |
|---|---|---|---|
| Base | 8/10 (80%) | 7/10 (70%) | 6/8 (75%) |
| Base + LoRA | 6/10 (60%) | 6/10 (60%) | **3/8 (37.5%)** |

Full per-task breakdowns: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json) (Set A
run 1), [`lora-vs-base-2026-08-08-retrain.json`](lora-vs-base-2026-08-08-retrain.json) (Set A run
2), [`bugfix-refactor-lora-vs-base-2026-08-08.json`](bugfix-refactor-lora-vs-base-2026-08-08.json)
(Set B).

**The clearest finding of this whole project: at this corpus size, SFT makes the model worse, and
Set B shows it's not narrowly confined to one task shape.** Set A run 1 looked like narrow
overfitting (LoRA traded "sometimes skips the tool call" for "sometimes calls the tool but gets
the wrong answer," net worse). Set B — bug fixes and refactors, the exact task shape batch h's new
training examples targeted — shows the **same failure signature but bigger**: LoRA got both
refactor tasks right (matching base) and 1 of 5 bug fixes right, but newly failed 3 straightforward
bug fixes (wrong index, wrong accumulator init, case-sensitivity) that base solved correctly. Same
pattern as Set A: LoRA's failures are all "called a tool, got it wrong," never "skipped the tool."

Read together across both sets: this isn't "the corpus is too narrow" (Set B's training examples
directly targeted bug-fixing, and it still regressed on bug-fixing). It's more likely **the
training recipe itself** — 51 examples, 3 epochs, `learning_rate: 2e-4` — is too aggressive for
this little data, causing broad degradation of precise code-editing correctness (a catastrophic-
forgetting signature) rather than teaching durable skill. Adding 6 more diverse examples on top of
an already-too-strong training signal didn't fix that; it likely needed to be counteracted with
fewer epochs, a lower learning rate, or simply far more data before the signal-to-noise ratio
favors learning over forgetting.

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

Given Set B's result, the priority order changed from "collect more diverse data" to "fix the
training recipe first, then decide if more data is even the bottleneck":

1. **Tune down the training recipe before adding more data.** Try fewer epochs (1-2 instead of 3),
   a lower `learning_rate` (e.g. `5e-5` instead of `2e-4`), or both, and re-run this same two-set
   eval. If a gentler fine-tune stops regressing on Set B while still improving the
   `completed_no_tools_used` behavior it was meant to fix, that confirms the recipe (not the data)
   was the problem.
2. Only after that — **more trajectory volume**, since 51 examples is genuinely small for a 7B
   model regardless of recipe tuning.
3. Keep using `LocalModelClient` for future eval runs — proven three times now (Set A twice, Set B
   once), no serving-layer debugging needed any time.
4. If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
   vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
   rather than fresh `pip install`s per pod — none of the three serving failures documented above
   were reliably reproducible enough to trust a fresh install each time.
5. **Do not publish a checkpoint to Hugging Face (Release milestone) until a run beats base on
   both eval sets.** Right now every trained checkpoint underperforms the base model.

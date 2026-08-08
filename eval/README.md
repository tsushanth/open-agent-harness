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

| Model | Set A (45 ex) | Set A (51 ex, corrupted) | Set A (51 ex, gentle, corrupted) | Set A (48 ex, cleaned) | Set B (51 ex, corrupted) | Set B (51 ex, gentle, corrupted) | Set B (48 ex, cleaned) |
|---|---|---|---|---|---|---|---|
| Base | 8/10 | 7/10 | — | — | 6/8 | — | — |
| Base + LoRA | 6/10 | 6/10 | 7/10 | **7/10** | 3/8 | 3/8 | **3/8** |

"Corrupted" corpus (51 examples) included 2 sessions that looked like passes but weren't — see
`data/trajectories/batch-2026-08-08-e/README.md`. "Cleaned" (48 examples) has both removed.
"Aggressive" = original recipe (3 epochs, `lr 2e-4`). "Gentle" = 1 epoch, `lr 5e-5`.

Full per-task breakdowns: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json),
[`lora-vs-base-2026-08-08-retrain.json`](lora-vs-base-2026-08-08-retrain.json),
[`bugfix-refactor-lora-vs-base-2026-08-08.json`](bugfix-refactor-lora-vs-base-2026-08-08.json),
[`gentle-recipe-set-a-2026-08-08.json`](gentle-recipe-set-a-2026-08-08.json) /
[`gentle-recipe-set-b-2026-08-08.json`](gentle-recipe-set-b-2026-08-08.json),
[`cleaned-corpus-set-a-2026-08-08.json`](cleaned-corpus-set-a-2026-08-08.json) /
[`cleaned-corpus-set-b-2026-08-08.json`](cleaned-corpus-set-b-2026-08-08.json).

**Three different fixes tried (aggressive→gentle recipe, then corrupted→cleaned data) all
converge on the exact same Set B result: 3/8, identical task-by-task pattern, every time.**
Set A responded to both fixes (6/10 → 7/10, roughly back to base level). Set B never moved at
all — not from tuning the recipe, not from removing 2 genuinely corrupted training examples.
That's a real, reproducible, surprising finding: **whatever is happening on Set B isn't explained
by training intensity or data corruption.**

The sharper clue is which 3 of 8 tasks LoRA passes: **e1, e6, e7 — every single time, across all
three checkpoints.** And critically, **LoRA's passes are a strict subset of base's own passes**
(base solved e1, e2, e3, e4, e6, e7; LoRA only ever gets e1, e6, e7 — never gains a task base
didn't already have, always loses e2/e3/e4). This is the actual pattern needing an explanation,
not "SFT makes bug-fixing worse in general": **something about applying *any* LoRA adaptation
trained on this harness's tool-call trajectories — even clean ones — consistently costs
correctness on e2 (index bug), e3 (accumulator bug), and e4 (case-sensitivity bug) specifically,
regardless of what specifically trained the adapter.** A plausible mechanism: fine-tuning on this
harness's trajectories shifts the model toward the harness's specific tool-call phrasing/protocol
conventions, and that shift — not the correctness content of the training examples — is what's
trading off against whatever latent reasoning path the base model uses to solve those 3 specific
bugs. This wasn't proven, just the leading hypothesis after ruling out recipe and corruption.

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

Recipe tuning and data cleanup are both now ruled out as fixes for Set B specifically (both
helped Set A). What's left to actually test the "any LoRA adaptation costs e2/e3/e4" hypothesis:

1. **Train a LoRA adapter on a corpus with zero bug-fix/refactor examples at all** (only the
   original "add a function" batches) and run it against Set B. If it *also* lands at 3/8 with
   the same e1/e6/e7-only pattern, that's strong confirmation the issue isn't about bug-fix
   training data quality/volume at all — it's a generic cost of fine-tuning on this harness's
   tool-call protocol, present even with a training set that never touched bug-fixing.
2. **Inspect what actually changed for e2/e3/e4 at generation time** — diff the LoRA vs. base
   model's raw output for those 3 specific prompts (not just pass/fail) to see whether the LoRA
   version still reasons correctly but fails at execution (e.g. a bad `edit_file` call) vs.
   genuinely reasons about the bug differently. That distinguishes "protocol shift breaks
   mechanics" from "protocol shift breaks reasoning."
3. **Use the gentle recipe going forward regardless** — it recovered most of Set A's regression
   with zero apparent downside on Set B, so it's a strict improvement even while Set B stays
   unexplained.
4. Keep using `LocalModelClient` for future eval runs — proven across 6 runs now, no
   serving-layer debugging needed any time.
5. If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
   vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
   rather than fresh `pip install`s per pod — none of the three serving failures documented above
   were reliably reproducible enough to trust a fresh install each time.
6. **Do not publish a checkpoint to Hugging Face (Release milestone) until a run beats base on
   both eval sets.** Four checkpoints tried — 2 recipes × corrupted/cleaned data — every one still
   underperforms base on Set B, with a suspiciously stable failure signature that needs the
   isolation test in step 1 before concluding anything further.

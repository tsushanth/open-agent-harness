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

| Model | Set A (45 ex) | Set A (51 ex, corrupted) | Set A (51 ex, gentle, corrupted) | Set A (48 ex, cleaned) | Set A (44 ex, **no bug-fix data**) | Set B (51 ex, corrupted) | Set B (51 ex, gentle, corrupted) | Set B (48 ex, cleaned) | Set B (44 ex, **no bug-fix data**) |
|---|---|---|---|---|---|---|---|---|---|
| Base | 8/10 | 7/10 | — | — | — | 6/8 | — | — | — |
| Base + LoRA | 6/10 | 6/10 | 7/10 | 7/10 | **8/10** | 3/8 | 3/8 | 3/8 | **4/8** |

"Corrupted" corpus (51 ex) included 2 sessions that looked like passes but weren't — see
`data/trajectories/batch-2026-08-08-e/README.md`. "Cleaned" (48 ex) has both removed. "No
bug-fix data" (44 ex) is the cleaned corpus with **all 4** remaining bug-fix/refactor examples
also excluded — only the original "add a function" batches remain. All non-"aggressive" columns
use the gentle recipe (1 epoch, `lr 5e-5`).

Full per-task breakdowns: [`lora-vs-base-2026-08-08.json`](lora-vs-base-2026-08-08.json),
[`lora-vs-base-2026-08-08-retrain.json`](lora-vs-base-2026-08-08-retrain.json),
[`bugfix-refactor-lora-vs-base-2026-08-08.json`](bugfix-refactor-lora-vs-base-2026-08-08.json),
[`gentle-recipe-set-a-2026-08-08.json`](gentle-recipe-set-a-2026-08-08.json) /
[`gentle-recipe-set-b-2026-08-08.json`](gentle-recipe-set-b-2026-08-08.json),
[`cleaned-corpus-set-a-2026-08-08.json`](cleaned-corpus-set-a-2026-08-08.json) /
[`cleaned-corpus-set-b-2026-08-08.json`](cleaned-corpus-set-b-2026-08-08.json),
[`isolation-no-bugfix-set-a-2026-08-08.json`](isolation-no-bugfix-set-a-2026-08-08.json) /
[`isolation-no-bugfix-set-b-2026-08-08.json`](isolation-no-bugfix-set-b-2026-08-08.json).

**The decisive result: excluding bug-fix/refactor training data entirely — not including more of
it, not cleaning it up, *removing all 4 remaining clean examples* — produced the best checkpoint
by far on both sets.** Set A 8/10 (matches/beats base's own 7-8/10, the best LoRA result of any
checkpoint tried). Set B 4/8 (breaks the "always exactly 3/8" pattern that held across 3 prior
checkpoints — neither recipe tuning nor removing 2 corrupted examples had moved it at all).

Read together with the earlier findings: **4 clean bug-fix/refactor examples was too small a
sample to teach anything generalizable, and including them was net harmful compared to just not
including them** — closer to noise/distraction than signal, diluting what the 44 well-represented
"add a function" examples were teaching well. This is a different and sharper conclusion than
"the recipe was too aggressive" or "2 examples were corrupted" (both real, both fixed, neither
was the actual lever) — the real lesson is about the corpus's *composition*, not its cleanliness
or the training intensity: **a tiny amount of an underrepresented task shape is worse than none
at all.** Getting bug-fix/refactor competence into the model will need either a much larger
volume of that task shape (so it's no longer a handful of outlier-like examples) or leaving it
out until that volume exists.

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

**Update (2026-08-08 / 2026-08-09):** ran two follow-up trajectory batches specifically to grow
bug-fix/refactor volume per point 2 below.
[`batch-2026-08-08-f/`](../data/trajectories/batch-2026-08-08-f/README.md): 15 tasks, 4 survived
review as genuinely clean (1 more looked like a pass but was a false positive — original buggy code
coincidentally satisfied the specific test input).
[`batch-2026-08-09-g/`](../data/trajectories/batch-2026-08-09-g/README.md): 15 more tasks, 6
survived review, all confirmed genuine this round. **Both batches independently found the same
pattern: every refactor task failed (0/8 combined), while plain bug fixes passed at a much higher
rate (~45%)** — two data points now, worth treating as a real capability gap rather than batch
noise. Corpus-wide clean bug-fix/refactor count is now 14 (4 from batch h + 4 + 6 new), close to
but still short of the 20-30+ target below — one more similarly-sized batch should cross it before
it's worth re-running the isolation-test retrain/eval comparison.

The isolation test resolved the "what's the actual lever" question. What's left is acting on it:

1. **The current best checkpoint (44 examples, no bug-fix/refactor data, gentle recipe) is the
   one to build on.** It's the first to match/beat base on Set A and the first to move Set B at
   all. Use `training/qwen2.5-coder-7b-lora-gentle.yaml` with a corpus excluding
   `data/trajectories/batch-2026-08-08-e/` as the new baseline for future comparisons, not the
   original 51/48-example checkpoints.
2. **Still short of beating base on Set B (4/8 vs. base's 6-7/8)** — don't reintroduce a small
   amount of bug-fix/refactor data expecting it to help; the isolation test showed that actively
   hurts. Either collect substantially more bug-fix/refactor trajectories (think 20-30+, enough
   that they're a real sub-distribution rather than a handful of outliers) or accept that this
   harness's SFT approach may need a fundamentally larger corpus before bug-fixing transfers well.
3. **Re-run the full 3-set matrix (A, B, and a fresh add-function-only set if one gets built) any
   time the corpus composition changes meaningfully** — this investigation's whole arc shows single
   eval runs are easy to misread; the pattern only became legible after cross-referencing 6 runs.
4. Keep using `LocalModelClient` and the gentle recipe as defaults going forward — both are now
   validated across many runs with no remaining open questions about their own correctness.
5. If real serving throughput is needed later (e.g. evaluating on dozens of tasks quickly), revisit
   vLLM with a custom pre-built Docker image (axolotl + a tested-compatible vLLM baked in once)
   rather than fresh `pip install`s per pod — none of the three serving failures documented above
   were reliably reproducible enough to trust a fresh install each time.
6. **Still do not publish to Hugging Face (Release milestone) until a checkpoint beats base on
   both sets.** The no-bug-fix-data checkpoint is the closest yet (beats base on A, closes but
   doesn't close the B gap) — genuine progress, not yet a green light.

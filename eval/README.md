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

## Harness-quality loop (Claude via the same protocol)

A second, orthogonal evaluation, separate from LoRA-vs-base: run the exact same Set A / Set B
tasks through the harness with **Claude** as the model client
(`harness/core/anthropic_model_client.py`, `eval/run_claude_via_harness.py`), not to ask "is
Claude smarter" (obviously) but to isolate whether the harness's own scaffold — system prompt,
tool schema, text-based `<tool_call>` protocol — has defects independent of the small model's
raw capability. `AnthropicModelClient` deliberately never uses Claude's native tool-use API; it
gets the identical text protocol Qwen gets, so a Claude failure here is a harness bug, not a
"give Claude a better interface" result.

**First run (2026-08-12):** Set A 10/10, Set B 7/8 —
[`claude-via-harness-2026-08-12.json`](claude-via-harness-2026-08-12.json). Found a real bug on
the first pass: a well-formed `write_file` call was silently dropped because its multi-line file
content had a literal raw newline instead of an escaped `\n`. That's invalid JSON under Python's
default strict parser, which raised no visible error anywhere — the tool call was just discarded,
and the fix was never applied. **Fixed** by parsing with `strict=False` in both `_try_parse` and
`_scan_bare_json_objects` (`harness/core/tool_parser.py`) — Python's json module supports this
exact case natively. Regression test:
`test_multiline_content_with_raw_newlines_still_parses`. Re-ran after the fix and confirmed the
previously-failing task now passes.

A second, different failure in the same re-run (`get_last_two`/e2) was genuinely malformed JSON
(a missing brace, a real model typo, not a raw-newline issue) — `strict=False` correctly can't
and shouldn't parse that. This surfaced a real **design gap, not yet fixed**: the harness
currently treats any unparseable tool-call text as the model's final answer and ends the session
immediately, with no chance for the model to notice and retry. A more resilient loop would
detect "this looked like an attempted tool call but didn't parse" and feed that back as a tool
result (e.g. `<tool_result>Error: your tool call wasn't valid JSON, try again</tool_result>`)
instead of silently treating it as a finished turn. Worth doing before the next harness-quality
run — right now a transient JSON typo costs the whole task instead of one wasted turn.

`eval/judge.py` adds an LLM-judge pass: a separate Claude call rates each resulting diff on
correctness and cleanliness (catches things a narrow verify command can't, like
`batch-2026-08-09-i`'s dead-code case — behaviorally correct but with leftover unreachable code)
and cross-references the same task's outcome from the current best Qwen checkpoint
(`isolation-no-bugfix-set-a/b-2026-08-08.json`). Result:
[`claude-via-harness-2026-08-12-judged.json`](claude-via-harness-2026-08-12-judged.json) — 5
tasks (h5, h6, e4, e5, e8) where Claude passes via this harness but Qwen's checkpoint still
fails, all judged correctness=5/cleanliness=5 on Claude's side. That's real evidence most of
Qwen's remaining gap on Set B is model capability, not a harness defect — consistent with, and
now more directly evidenced than, the isolation test's corpus-composition finding.

**Next harness-quality iteration:** fix the retry-on-malformed-JSON gap above, then re-run to see
if it closes the one remaining Claude failure mode observed so far; consider running this against
the fine-tuned LoRA checkpoint too (not just base Qwen) once the full-corpus retrain lands, to see
whether SFT closes any of the 5 flagged gaps on its own.

## Next steps

**Update (2026-08-08 / 2026-08-09):** ran three follow-up trajectory batches specifically to grow
bug-fix/refactor volume per point 2 below.
[`batch-2026-08-08-f/`](../data/trajectories/batch-2026-08-08-f/README.md): 15 tasks, 4 survived
review (1 false positive — buggy original coincidentally satisfied the test input).
[`batch-2026-08-09-g/`](../data/trajectories/batch-2026-08-09-g/README.md): 15 tasks, 6 survived,
all genuine.
[`batch-2026-08-09-i/`](../data/trajectories/batch-2026-08-09-i/README.md): 15 tasks, 5 survived
(1 false positive — this one caused by a bug in the verify command's own structural check, not the
model; the check was only ever tested against the fixed reference, never confirmed to actually
reject the buggy original, so it silently passed on an untouched file). **All three batches point
at the same pattern: refactor tasks fail far more often than plain bug fixes — only 1 genuine
refactor pass out of 12 attempted across the three batches**, worth treating as a real capability
gap rather than batch noise. Corpus-wide clean bug-fix/refactor count is now 19 (4 from batch h +
4 + 6 + 5 new), within reach of the 20-30+ target below — likely close enough to be worth planning
the retrain/eval comparison after one more small batch, or even trying it now.

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

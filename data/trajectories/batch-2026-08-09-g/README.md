# Batch: 2026-08-09 (batch g — bug-fix/refactor volume push, round 2)

Second follow-up batch (after `batch-2026-08-08-f/`) pushing toward the 20-30+ clean
bug-fix/refactor example target from `eval/README.md`. 15 tasks, one file each: 10 bug fixes
(bad slice bound, mutable default argument, string-not-reversed, exclusive range bound, missing
dict default, backwards comparator, missing None guard, missing +32 offset, wrong sort index,
missing empty-list guard), 4 refactors (manual max loop, string concat loop, nested dict access,
manual filter loop), 1 multi-step task (add a method that reads an existing class constant).
Same methodology as batch f: one task per file, all 15 verify commands (including the
`inspect.getsource()` structural checks for the 4 refactors) sanity-tested against manually-fixed
reference implementations before running against the live model.

**6 of 15 passed (40%) via the recorded `outcome` — all 6 independently confirmed genuine** (final
file content matches the intended fix exactly, no false positives this round, unlike batch f's
`f9_bug_mutation`).

Passed: `g1_bug_slice_off`, `g3_bug_string_reverse`, `g4_bug_sum_wrong_range`,
`g6_bug_wrong_comparator`, `g8_bug_wrong_multiplier`, `g9_bug_index_shift`.

Two failures are worth noting because the model's own prose claimed success while the recorded
`outcome` correctly disagreed — direct evidence for why this project checks the outcome field
and never trusts the streamed text:
- `g12_refactor_string_concat_loop`: after several failed edit attempts (search/grep dead ends,
  `old_string not found`), the model's fix accidentally mutated the input list inside the loop
  (`words.append(w + " ")` instead of building a separate result) — genuinely broken code, correctly
  caught by verify despite the model announcing "Task completed."
- `g15_multistep_shared_config`: the model's edit deleted the `Config` class entirely while
  rewriting the file, leaving `Invoice.calculate_total()` referencing an undefined name — a real
  edit mistake, correctly caught by verify (`NameError`) despite "Task completed" prose.

All 4 refactor tasks failed this round too — 0/8 refactors across both bug-fix-focused batches (f
and g) now, vs. a much higher pass rate on plain bug fixes (~45% combined). This is a consistent
enough pattern across two independent batches to note as a real base-model weak spot, not batch
noise: Qwen2.5-Coder-7B-Instruct struggles specifically with multi-part refactors under this
harness's tool-call protocol, more than with single-location bug fixes.

Corpus-wide clean bug-fix/refactor count is now 14 (4 from batch h + 4 from batch f + 6 from this
batch), still short of the 20-30+ target — one more batch of similar size should cross it.

# Batch: 2026-08-08 (batch f — bug-fix/refactor volume push)

Direct follow-up to `eval/README.md`'s "Next steps" recommendation after the isolation test: batch
h's 4 clean bug-fix/refactor examples were too small a sample and net-harmful to include. This
batch tried to push that volume up — 15 new tasks, all in fresh files (one task per file, the
methodology already proven cleanest across earlier batches): 10 bug fixes (boundary condition,
wrong variable in a division, dedup-while-sorting, missing return statements, KeyError, type
concat, negated condition, wrong clamp variable, mutate-while-iterating, float equality), 4
refactors (duplicate computation, verbose boolean pattern, nested if/else to early return, manual
loop to list comprehension), 1 multi-step task (use an existing class constant instead of a
hardcoded value).

Applying the batch-h lesson directly: all 4 refactor tasks' `--verify` commands include **both** a
behavioral assertion and a structural check via `inspect.getsource()` (e.g. `assert "total2" not
in src`) specifically to prevent the "looks unchanged, still passes" false positive found last
batch. Every verify command was sanity-tested locally against a manually-fixed reference
implementation before running against the model.

**5 of 15 initially looked like passes (33%) via the recorded `outcome`. 1 of those 5 was still a
false positive — 4 of 15 are real (27%).**

- `f9_bug_mutation` (`remove_duplicates` mutates a list while iterating): the model's edit attempt
  failed (`old_string not found`), and instead of re-reading the file to correct course, it asked
  the user to "provide the relevant section" — never made a successful edit. `outcome` still came
  back `completed_verified_pass` because the *original buggy code* happens to produce the correct
  output for the specific test input used (`[1,1,1,2,2,3]` → `[1,2,3]`, by coincidence of removal
  order during in-place mutation) — a weak test case, not a mislabeled harness. Same shape of bug
  as batch h's `process_order`: a verify command that can't distinguish "fixed" from "untouched"
  for this specific input. Removed.

Confirmed real (all inspected message-by-message, not just by outcome field):
- `f3_bug_wrong_method` (dedup+sort): clean single edit, `Replaced 1 occurrence`, correct fix.
- `f5_bug_key_error` (KeyError → None): clean single edit, correct fix.
- `f7_bug_negation` (backwards weekday check): one failed edit attempt, self-corrected by re-reading
  the file, then a correct edit — this is exactly the "investigate before changing" recovery
  behavior the harness's system prompt asks for, a genuinely good trajectory to train on.
- `f15_multistep_shared_const` (use `Limits.MAX_USERS` instead of a hardcoded value): clean single
  edit, correct fix.

Notably, **all 4 refactor tasks failed** (`f11`–`f14`) — this batch's stricter structural verify
checks are doing exactly what they were added for; prose review during the run had already flagged
these as suspicious ("Please provide the content...", "old_string not found") before checking the
recorded outcome confirmed it. Bug fixes had a real per-task pass rate around 40-50%; refactors
were 0/4. This batch alone still isn't the "20-30+" volume `eval/README.md` called for — it adds
4 clean examples on top of batch h's 0 (all 4 of batch h's were excluded from the training corpus
after the isolation test), for 4 total clean bug-fix examples corpus-wide. Not yet enough to
retrain the isolation-test comparison meaningfully; more batches needed before revisiting that
experiment.

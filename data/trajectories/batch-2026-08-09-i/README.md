# Batch: 2026-08-09 (batch i — bug-fix/refactor volume push, round 3)

Third follow-up batch pushing toward the 20-30+ clean bug-fix/refactor example target from
`eval/README.md`. 15 tasks, one file each: 10 bug fixes (backwards parity check, missing
cents-to-dollars conversion, out-of-bounds index, wrong boolean operator, missing `abs()`,
off-by-one slice, wrong dict-default fallback, no-op "reverse", wrong max() key, incomplete blank
check), 4 refactors (manual min loop, manual dict-building loop, if/else-to-ternary, list-based
dedup to set-based), 1 multi-step task (use an existing class constant list instead of hardcoded
checks).

**6 of 15 looked like passes (40%) via the recorded `outcome`. 1 of those 6 was a false positive
found on inspection — 5 of 15 are real (33%).**

- `i13_refactor_ternary`: the model's only edit attempt failed (`old_string not found`), and it
  concluded — incorrectly — "the function already uses a ternary expression, no changes are
  necessary." The file is untouched, still the original if/else block. This one **should have been
  caught by the verify command's structural check but wasn't, due to a bug in the check itself**:
  the assertion was `chr(34)+"label = "+chr(34) not in src`, intending to detect the pattern
  `label = "..."`, but the concatenation produces the string `"label = "` — requiring an actual
  quote character immediately *before* "label", which never occurs in the source (the real text is
  `    label = "non-negative"`, with a newline and indentation before "label", not a quote). The
  check was a silent no-op that always passed. This was **only caught because every "pass" gets
  manually inspected before being added to the corpus** — sanity-testing the verify command against
  a manually-fixed reference (as done for every task before this batch ran) only proves the check
  *accepts* the correct code, not that it *rejects* the buggy original. That second half wasn't
  being tested. Worth doing for future refactor-verify commands: run the check against the
  **unmodified buggy file** too and confirm it fails, not just against the fixed reference.

Confirmed real (final file content matches the intended fix, not just outcome field):
- `i2_bug_str_format`, `i6_bug_prefix_offbyone`, `i9_bug_counter_wrong_key`,
  `i15_multistep_shared_roles`: clean, direct fixes.
- `i10_bug_blank_check`: correct fix (`s.strip() == ''`), but the model left the original
  unreachable `return s == ""` as dead code after it — functionally correct (the first return
  always fires) but messy. Kept as a real example since the behavior and file are genuinely
  correct, just not the cleanest possible diff.

Two failures ran long — `i11_refactor_manual_min` (60 messages) and `i14_refactor_set_dedup` (20
messages) — both well past the normal 5-11 message range for a clean session, consistent with the
repetition-loop failure mode documented in batch h. Both correctly failed verify regardless, so
they're excluded by the outcome filter either way; not manually reviewed further since they don't
risk contaminating the corpus.

Corpus-wide clean bug-fix/refactor count is now 19 (4 from batch h + 4 from batch f + 6 from batch
g + 5 from this batch) — within reach of the 20-30+ target from `eval/README.md`. One more small
batch (or even a partial one) should cross the threshold.

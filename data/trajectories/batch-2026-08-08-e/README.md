# Batch: 2026-08-08 (batch h — deliberately diverse task shapes)

The first batch specifically designed to break the "almost all tasks are 'add a function to a
file'" narrowness identified as the likely cause of the eval regression (see `eval/README.md`):
6 bug fixes (off-by-one, wrong operator, mutable default argument, missing type conversion,
loop bound error, missing None guard), 3 refactors (deduplicate repeated code, flatten nested
conditionals, extract magic numbers into constants), 2 multi-step tasks requiring the model to
notice and use existing module-level state (constants, a validation dict) rather than just adding
an isolated new function.

**6 of 11 originally looked like passes (55%) — 2 of those 6 turned out to be corrupted and were
removed. 4 of 11 are real (36%).** Building a bug-fix-shaped held-out eval set later (Set B in
`eval/README.md`) showed a big regression on exactly this task shape, which led to manually
re-reading every session here. Found:

- `sum_first_n` (bug5, off-by-one): the model correctly fixed the bug in its 2nd tool call, then —
  without ever re-reading the file — spent **75 more messages** re-proposing the identical
  already-applied edit against hallucinated stale content, never noticing it had already succeeded.
  79 messages total (every clean session in this corpus is 5-11). `outcome` was technically
  accurate (`completed_verified_pass` — the file *was* correct), but training on this trajectory
  teaches "loop indefinitely after succeeding, never conclude."
- `process_order` (refactor1, deduplicate): the model **never once successfully edited the file**
  across 51 messages — every `edit_file` call failed with "old_string not found," and it kept
  re-reading the same unchanged original. It still passed verify, because the verify command only
  checked black-box behavior (`process_order({"status": "shipped"}) == "shipped"`, etc.), which
  the *original, unrefactored* code already satisfied — the task was never accomplished, but there
  was no way for the automated check to tell the difference between "refactored successfully" and
  "left completely untouched." A genuinely mislabeled negative example fed in as positive.

Both removed. `training/prepare_dataset.py` now also drops any session over 20 messages by
default as a structural safeguard — see its docstring for the full story and why 20 is a real,
non-arbitrary threshold given every clean session here is 5-11 messages.

**Lesson for future batches: behavioral-only verify commands are insufficient for refactor tasks
specifically**, since "no change" and "successful refactor" can be behaviorally indistinguishable
by design (the whole point of a refactor is that behavior doesn't change). Refactor task verify
commands need a structural check too (e.g. asserting the original duplicated pattern is gone,
or that line count decreased) — not yet added retroactively to the harness, worth doing before
the next refactor-shaped batch.

Independently spot-checked 2 of the real passes (`bug1_offbyone`, `refactor2_nested`) outside the
harness to confirm they're not false positives.

Combined across all batches: 91 tasks, 49 genuinely correct sessions (54%). `prepare_dataset.py`
outputs 48 training examples — one more (real, from an earlier batch) is excluded too, a 21-message
session that took longer than usual but did succeed, caught by the same length safeguard because
the threshold has to be a single number. A reasonable trade for reliably excluding the two
corrupted sessions above.

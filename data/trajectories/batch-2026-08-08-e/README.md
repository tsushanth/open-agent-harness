# Batch: 2026-08-08 (batch h — deliberately diverse task shapes)

The first batch specifically designed to break the "almost all tasks are 'add a function to a
file'" narrowness identified as the likely cause of the eval regression (see `eval/README.md`):
6 bug fixes (off-by-one, wrong operator, mutable default argument, missing type conversion,
loop bound error, missing None guard), 3 refactors (deduplicate repeated code, flatten nested
conditionals, extract magic numbers into constants), 2 multi-step tasks requiring the model to
notice and use existing module-level state (constants, a validation dict) rather than just adding
an isolated new function.

**6 of 11 passed (55%)** — lower than the narrow "add function" batches (55-85% band), which
makes sense: bug fixes and refactors are genuinely harder tasks than pure additions, requiring
the model to read and understand existing code correctly before changing it, not just write new
code from a spec. Independently spot-checked 2 passes (`bug1_offbyone`, `refactor2_nested`)
outside the harness to confirm they're not false positives.

**This is intentional, not a regression to be alarmed by.** The goal of this batch wasn't to
maximize pass rate — it was to add task-shape diversity to the training corpus, since the eval
result suggested overfitting to the narrow "add a function" pattern specifically. A harder,
more representative batch with a lower-but-real pass rate is exactly what's needed here, not
another easy batch inflating the numbers without changing what the model actually learns.

Combined across all batches: 93 tasks, 51 real passes (55%).

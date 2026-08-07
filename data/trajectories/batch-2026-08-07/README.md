# Batch: 2026-08-07

First batch of trajectories run with `--verify`, against Qwen2.5-Coder-7B-Instruct via vLLM on a
rented RTX A5000. 4 tasks, 1 genuine pass:

| Session | Task | Outcome | What actually happened |
|---|---|---|---|
| `session-20260807T233017Z` | Fix a syntax error in `strings.py` | `completed_verified_fail` | Model searched for the filename via `grep` instead of reading it directly, found nothing, gave up without ever inspecting the file. |
| `session-20260807T233031Z` | Add `modulo()` to `calc.py` | `completed_verified_fail` | Model attempted the edit but corrupted the file's structure (function bodies got separated from their signatures) — a real bug the verify step caught that a naive "did the model say it's done" check would have missed. |
| `session-20260807T233035Z` | Add `count()` method to `TodoList` in `todo.py` | `completed_verified_pass` | Correct. Method added, verify assertion passed. |
| `session-20260807T233057Z` | Add a `loud` parameter to `greet()` in `greet.py` | `completed_verified_fail` | Partial: added the parameter to the signature but never implemented the uppercase behavior, then got confused trying to re-locate the function and gave up. |

**Why this batch matters more than the pass rate:** this is real evidence of the exact gap this
project's SFT phase exists to close, and proof the verification mechanism (`--verify`, added
specifically because of the false-"completed" cases found in this same batch) works — it caught
a silent file corruption that would have otherwise gone into a training set uncorrected.

Only `session-20260807T233035Z` passes `prepare_dataset.py`'s default filter. The other three are
kept as-is (not deleted) because they're useful negative examples for future work: understanding
failure modes, or eventually training a verifier/critic model on pass/fail trajectory pairs.

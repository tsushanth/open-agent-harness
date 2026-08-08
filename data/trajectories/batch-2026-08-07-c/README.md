# Batch: 2026-08-07 (batch c — first batch with both fixes applied)

14 tasks across a fresh 8-file scratch project (bank/queue/text/geometry/timer/tree/primes/matrix
modules), against Qwen2.5-Coder-7B-Instruct via vLLM on a rented RTX A5000. First batch run with
both fixes from the previous round in place: the task-reminder suffix and the `edit_file` empty-
`old_string` guard. **9 of 14 passed (64%)** — a large jump from the ~21% seen in batches a/b.

| Task | Outcome | Note |
|---|---|---|
| Add `withdraw()` to `Account` | `completed_verified_pass` | Independently re-verified. |
| Add `is_empty()` to `Queue` | `completed_verified_fail` | Correctly self-corrected from an empty `old_string` `edit_file` attempt (new guard caught it) to `write_file` — but that `write_file` call wrote a truncated 122-byte file missing `enqueue`/`dequeue`. Different failure mode than before: generation truncation, not a harness bug. |
| Add `peek()` to `Queue` | `completed_verified_fail` | Model claimed the method "was already defined" (it wasn't) and made no tool call for this task. |
| Add `unique_words()` to `text_utils.py` | `completed_verified_pass` | Pass. |
| Add `reverse_words()` to `text_utils.py` | `completed_verified_pass` | Pass. |
| Add `triangle_area()` to `geometry.py` | `completed_verified_pass` | Pass. |
| Add `circle_circumference()` to `geometry.py` | `completed_verified_pass` | Pass. |
| Add `reset()` to `Stopwatch` | `completed_verified_fail` | Model reported it couldn't find the `Stopwatch` class definition, despite it being present in `timer.py` from the start — a read/search failure, not an edit failure. |
| Add `count_children()` to `Node` | `completed_verified_pass` | Pass. |
| Optimize `is_prime()` to check only up to sqrt(n) | `completed_verified_fail` | A refactor task rather than a pure addition — did not verify correctly; not yet root-caused. |
| Add `next_prime()` to `primes.py` | `completed_no_tools_used_verified_fail` | The one task in this batch with zero tool calls — described the implementation in prose instead. |
| Add `identity()` to `matrix.py` | `completed_verified_pass` | Independently re-verified. |
| Fix `deposit()` to reject negative amounts | `completed_verified_pass` | Independently re-verified. |
| Fix `dequeue()` to raise `RuntimeError` instead of `IndexError` | `completed_verified_pass` | Pass — and this fix is visible in the *final* state of `queue_impl.py`, meaning it correctly preserved `is_empty()` from an earlier task even though that earlier task's own verify run had failed on a truncated write. |

**Takeaway:** both fixes from the previous round (task-reminder, empty-`old_string` guard) measurably
worked — `completed_no_tools_used` dropped from the dominant failure (6/10 in batch b) to a single
occurrence here. The remaining failures are a different, more varied set: content truncation on
`write_file`, a stale claim that a method already existed, a read/search miss, and one
unverified refactor. None of these repeat the exact patterns already fixed, suggesting there
isn't one remaining dominant failure mode left to chase — this may be close to what "the base
model's real ceiling on this harness, unassisted" looks like without SFT.

Combined across all three batches: 28 tasks, 12 real passes (43%).

# Batch: 2026-08-08 (batch e — confounded batch design, read before trusting the pass rate)

14 tasks across a fresh 8-file scratch project (playlist/weather/deck/timer2/polynomial/
event_bus/roman/histogram). **5 of 14 passed (36%)** — a real drop from batches c/d's 64-69%.

**This drop is a batch-design artifact, not a regression in the harness or its fixes.** Unlike
batches c and d (which mostly used one task per file), this batch deliberately packed 2-3 tasks
onto several files (`playlist.py`, `deck.py`, `histogram.py`, `roman.py`) to fit more tasks with
fewer scratch files. Root-caused two concrete cases:

- `playlist.py`: a later task's `write_file` call regenerated the whole class, dropping the
  `name` constructor parameter and the `remove_song()`/`clear()` methods two earlier tasks had
  added — even though those earlier sessions had correctly verified at the time.
- `deck.py`: a later task rewrote `__init__` entirely with a different card representation
  (`'2 of Hearts'` instead of `'2C'`) and implemented `deal_card()` instead of the originally
  requested `draw()` — the model's own naming choice drifted once it regenerated the file from
  scratch rather than making a targeted edit.

Same underlying pattern documented as a "methodology note" in `batch-2026-08-08-a/README.md`, but
here it dominates the batch instead of affecting one task, because of how densely tasks were
packed onto shared files. **Each individual trajectory is still valid** (verified correctly
against the file state at the time), but the batch's 36% pass rate should not be read as "the
model got worse" — it's confounded with file-sharing density. Future batches should default to
one task per file (as c/d mostly did) unless deliberately testing multi-step-on-one-file scenarios.

Combined across all five batches: 58 tasks, 28 real passes (48%) — but batch e's rate is the
weakest signal in that average for the reason above.

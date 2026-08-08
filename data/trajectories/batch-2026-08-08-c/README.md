# Batch: 2026-08-08 (batch f — strict one-task-per-file, best result yet)

12 tasks, 12 distinct single-purpose files (`f1_distance.py` through `f12_ring_buffer.py`), one
task per file with no file shared across tasks — directly testing the lesson from batch e's
confounded design. **10 of 12 passed (83%)**, the best pass rate of any batch so far, and a real
independent spot-check (`manhattan_distance`, `safe_call`, `count_duplicates`) confirmed 3 passes
aren't false positives from a bad verify command.

The two failures (`Settings.get()`, `RingBuffer.is_full()`) aren't yet root-caused individually,
but with no shared-file confound in this batch, they're more likely to be genuine model mistakes
than batch-design artifacts — worth a closer look if the same pattern recurs.

**Takeaway:** one-task-per-file isn't just a cleaner methodology for measuring pass rate — it
appears to produce a materially higher real pass rate too (83% vs. batch e's confounded 36%,
vs. c/d's mixed-but-mostly-one-task-per-file 64-69%). Worth defaulting to for all future batches.

Combined across all six batches: 70 tasks, 38 real passes (54%).

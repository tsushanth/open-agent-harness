# Training

SFT pipeline for fine-tuning Qwen2.5-Coder-7B-Instruct on harness trajectories.

## Pipeline

```bash
# 1. Run the harness against real tasks, accumulating sessions in data/trajectories/
oah "some real coding task"
# ... repeat across many tasks, ideally a variety of bug-fix/feature/refactor/search tasks ...

# 2. Convert completed, clean sessions into a training set
python3 training/prepare_dataset.py --input data/trajectories --output training/train.jsonl

# 3. Fine-tune with axolotl (install separately: pip install axolotl)
accelerate launch -m axolotl.cli.train training/qwen2.5-coder-7b-lora.yaml
```

`prepare_dataset.py` drops sessions that didn't reach `outcome: "completed"` and sessions
containing a known bad pattern (the model echoing prompt instructions back — see
`harness/core/agent.py` history for why that check exists). Both are real failure modes
observed while smoke-testing this harness; training on them uncorrected would just teach the
fine-tuned model to repeat those failures.

**Important caveat, found the hard way:** `outcome: "completed"` means the model stopped asking
for tools — NOT that the task was actually accomplished. Observed directly: a task where the model
printed new code as prose instead of calling `write_file`, then declared the (unchanged) file
fixed. `prepare_dataset.py` cannot currently detect this class of failure; it's not a string
pattern to match against like the echo bug. **Manually spot-check trajectories before training on
them** — or write task-specific verification (e.g. "does the file actually compile after the
session ends") before trusting `outcome: "completed"` at scale. This is exactly the gap the
Eval milestone in the root README needs to close.

## Status: not yet run

We have 72 real trajectories in `data/trajectories/` (2 loose examples + six batches of 4, 10, 14,
16, 14, and 12), 38 of which pass `prepare_dataset.py`'s filter — the first corpus size in this
project that's a plausible candidate for an actual (small) training run. The config above is
still unvalidated against a real training run, though.

Pass rate by batch: 25% (a) → 20% (b) → 64% (c) → 69% (d) → 36% (e) → **83% (f)**. The jump at
c/d came from fixing the dominant `completed_no_tools_used` failure mode and a real `edit_file`
corruption bug (both detailed below). Batch e's drop was a **batch-design confound**: it packed
multiple tasks onto shared files, and a later task's `write_file` call regenerating a whole file
repeatedly clobbered earlier tasks' additions (each trajectory still verified correctly at the
time it ran) — see `batch-2026-08-08-b/README.md`. Batch f retested with strict one-task-per-file
design and got both the cleanest signal and the best result yet, suggesting shared-file batch
design wasn't just confounding measurement, it was actively suppressing the real pass rate.
**One-task-per-file is now the default for future batches.**

**Tried and reverted:** injecting a synthetic few-shot priming exchange (a worked "add a function"
task resolved correctly via a tool call, placed between the system prompt and the real task) to
fix the above. This is a standard technique for steering weaker models and seemed like a safe bet.
Tested live against the exact 6 tasks that had failed this way in batch b: it didn't just fail to
help, it made things strictly worse — the model produced a completely empty response (not even
prose) for the real task in all 6 retests, likely confused by the extra context length/pattern.
Reverted rather than shipped. If you retry this idea, it's worth trying a *shorter* priming
exchange (the one tested here was 6 messages) or priming via the system prompt text itself rather
than fake conversation turns, before concluding the technique doesn't apply here at all.

**Tried and kept:** the lighter-touch version of the same idea — appending one line to the task
text itself ("If this requires changing a file, call write_file or edit_file...") rather than
adding conversation turns. Tested live against the same 6 failing tasks: real improvement, tool-call
attempt rate went from 0/6 to 3/6, no empty-response regression. See `_with_tool_reminder` in
`harness/core/agent.py`.

That same retest surfaced an unrelated but more important bug: a model called `edit_file` with
`old_string=""` (meaning to create a new file — it should have used `write_file`), and
`replace_all=True` combined with Python's `str.replace("", ...)` semantics inserted the new text
between every character of the target file, turning a ~165 byte file into ~32KB of duplicated
garbage. This wasn't a model reliability problem, it was a real input-validation gap in
`EditTool` — fixed by rejecting an empty `old_string` outright (see `harness/tools/edit.py` and
`tests/test_tools.py::test_edit_file_rejects_empty_old_string`). Worth calling out because it's
the kind of bug that `--verify` caught but a naive "did the model say it's done" check wouldn't
have — and the kind of gap that only surfaces from running the harness against a real model,
not from synthetic/mocked testing alone.

## Config notes (`qwen2.5-coder-7b-lora.yaml`)

- QLoRA (4-bit base + LoRA adapter) sized for a single 24GB GPU — same class of pod used for the
  harness smoke test, so the training step doesn't require a bigger rental than what's already
  been validated to work.
- `sample_packing: false` — trajectories vary a lot in length, and packing multiple sessions into
  one sequence risks the model learning to bleed tool-call context across unrelated tasks.
- No eval split configured yet. See the Eval milestone in the root README's Roadmap.

# Training

SFT pipeline for fine-tuning Qwen2.5-Coder-7B-Instruct on harness trajectories.

## Pipeline

```bash
# 1. Run the harness against real tasks, accumulating sessions in data/trajectories/
oah "some real coding task"
# ... repeat across many tasks, ideally a variety of bug-fix/feature/refactor/search tasks ...

# 2. Convert completed, clean sessions into a training set
python3 training/prepare_dataset.py --input data/trajectories --output training/train.jsonl

# 3. Fine-tune with axolotl — PIN the version, don't use -U (see warning below)
pip install axolotl==0.18.0
accelerate launch -m axolotl.cli.train training/qwen2.5-coder-7b-lora.yaml
```

Validated on a rented RTX A5000 (24GB) using the official `winglian/axolotl:main-latest` Docker
image — note that image does *not* actually ship axolotl pre-installed despite the name (its
`/workspace/axolotl` is an empty mount point); install it yourself first, same as step 3 above.
On that setup, install + tokenize + 3-epoch train over 45 examples took about 8 minutes total,
with the training loop itself finishing in under 3.

**Pin the axolotl version — do not use `pip install -U axolotl`.** Two runs on the same day, same
commands, same `winglian/axolotl:main-latest` image: the first two (`axolotl==0.18.0`, installed
via `-U` when that happened to be latest) trained successfully. A later `-U` install pulled a
newer axolotl whose torch dependency required a newer CUDA driver than was available — training
failed with `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` on
**two different physical hosts**, ruling out one bad machine. `-U` in a Docker image whose base
CUDA/driver stack is fixed is asking for exactly this kind of drift. `axolotl==0.18.0` is the
pinned-known-good version from the successful run's install log.

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

## Status: first training run complete, not yet evaluated

Trained on the 45-example corpus below. Loss converged cleanly over 3 epochs (17 steps):
0.11 → 0.036, no divergence, no NaNs (adapter weights independently verified after download).
Produced a 154MB LoRA adapter (`peft_type: LORA`, r=16, 40.4M trainable params — 0.53% of the
7.66B total). **This proves the pipeline works end-to-end, not that the resulting model is
better at tool use than the base model** — 45 examples is small, and nothing has evaluated
whether it generalizes versus partially memorizing the training set. That comparison is the
Eval milestone, not yet done.

The trained adapter isn't committed to this repo — 154MB exceeds GitHub's 100MB per-file limit
without LFS, and model weights belong in a model registry (Hugging Face Hub), not a git repo, per
the Release milestone. Reproduce it yourself with the pipeline commands above, or wait for a
published checkpoint.

We have 84 real trajectories in `data/trajectories/` (2 loose examples + seven batches of 4, 10,
14, 16, 14, 12, and 12), 45 of which pass `prepare_dataset.py`'s filter and were used for the run
above.

Pass rate by batch: 25% (a) → 20% (b) → 64% (c) → 69% (d) → 36% (e) → 83% (f) → 58% (g). The jump
at c/d came from fixing the dominant `completed_no_tools_used` failure mode and a real
`edit_file` corruption bug (both detailed below). Batch e's drop was a **batch-design confound**:
it packed multiple tasks onto shared files, and a later task's `write_file` call regenerating a
whole file repeatedly clobbered earlier tasks' additions (each trajectory still verified correctly
at the time it ran) — see `batch-2026-08-08-b/README.md`. Batches f and g both retested with
strict one-task-per-file design (now the default) and landed at 83% and 58% respectively — a wide
enough spread to conclude the real pass rate for this harness/model/fix combination sits somewhere
in a 55-85% band with genuine batch-to-batch variance, not one fixed number.

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

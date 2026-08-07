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

We have 16 real trajectories in `data/trajectories/` (2 loose examples + two batches of 4 and 10),
4 of which pass `prepare_dataset.py`'s filter — kept as reference / regression checks for the
harness itself, not real training data yet. The config above is unvalidated against a real
training run. **Do not run this yet** — with a training set this small, SFT would just badly
overfit on almost nothing rather than teach general tool-use behavior. The next real milestone is
trajectory collection at scale (dozens to low-hundreds of diverse tasks) before a training run is
worth the GPU cost.

Worth noting for whoever collects that larger corpus: the real pass rate observed so far (3/14,
21%, across the two verified batches) means getting to "dozens of *passing* trajectories" likely
requires running several times that many tasks, or improving the harness/prompt to reduce the
dominant `completed_no_tools_used` failure mode documented in `batch-2026-08-07-b/README.md`
first — otherwise most runs are wasted GPU time on failures the dataset filter throws away anyway.

## Config notes (`qwen2.5-coder-7b-lora.yaml`)

- QLoRA (4-bit base + LoRA adapter) sized for a single 24GB GPU — same class of pod used for the
  harness smoke test, so the training step doesn't require a bigger rental than what's already
  been validated to work.
- `sample_packing: false` — trajectories vary a lot in length, and packing multiple sessions into
  one sequence risks the model learning to bleed tool-call context across unrelated tasks.
- No eval split configured yet. See the Eval milestone in the root README's Roadmap.

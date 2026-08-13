"""Runs the same Set A / Set B held-out eval tasks through the harness with Claude as the
model client instead of Qwen — via AnthropicModelClient, which routes Claude through the exact
same text-based <tool_call> protocol the open model gets (see its docstring for why).

Purpose: this is NOT a "is Claude smarter" check — of course it is. It's a harness-defect
finder. If Claude, a highly capable model, still trips on a specific task inside this harness's
protocol (can't find old_string, gives up and asks the user for the file contents, drifts off
the tool-call format), that's evidence the harness's system prompt / tool schema / protocol has
a real defect independent of any model's raw capability. Read alongside eval/judge.py, which
scores the resulting diffs.

Each task runs OAH_SAMPLES times (default 3) and reports the modal outcome, not a single
sample — LLM output is stochastic enough that a single run isn't a trustworthy verdict (observed
directly: e2_bug_index passed once and failed on an immediate rerun with no code changes in
between). All N samples are saved, not just the modal one, so variance itself is inspectable.

Runs entirely locally — no GPU, no serving, no RunPod pod. Requires ANTHROPIC_API_KEY.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.core.agent import Agent
from harness.core.anthropic_model_client import AnthropicModelClient
from harness.core.trajectory import TrajectoryLogger

from run_utils import compute_diffs, majority_outcome, reset_scratch
from run_transformers_eval import FILES as SET_A_FILES, TASKS as SET_A_TASKS
from run_transformers_eval2 import FILES as SET_B_FILES, TASKS as SET_B_TASKS

SCRATCH = os.environ.get("OAH_CLAUDE_SCRATCH", "/tmp/oah_claude_eval_scratch")
RESULTS_DIR = os.environ.get("OAH_RESULTS_DIR", "/tmp/oah_claude_eval_results")
MODEL = os.environ.get("OAH_CLAUDE_MODEL", "claude-sonnet-5")
SAMPLES = int(os.environ.get("OAH_SAMPLES", "3"))


def run_set(label: str, tasks: list, files: dict, client: AnthropicModelClient) -> dict:
    scratch = os.path.join(SCRATCH, label)
    results = []
    for task, verify in tasks:
        samples = []
        for sample_idx in range(SAMPLES):
            reset_scratch(scratch, files)
            os.chdir(scratch)
            agent = Agent(model_client=client, confirm_fn=lambda *_: True)
            logger = TrajectoryLogger(output_dir=f"{RESULTS_DIR}/{label}/sample{sample_idx}")
            agent.run(task, logger=logger, verify_cmd=verify)
            outcome = json.loads(logger.path.read_text())["outcome"]
            diffs = compute_diffs(scratch, files)
            samples.append({"outcome": outcome, "diffs": diffs})

        outcome, representative = majority_outcome(samples)
        results.append({
            "task": task,
            "outcome": outcome,
            "diffs": representative["diffs"],
            "samples": samples,
        })
        agreement = sum(1 for s in samples if s["outcome"] == outcome)
        flag = "" if agreement == SAMPLES else f"  (agreement {agreement}/{SAMPLES} — noisy)"
        print(f"[claude/{label}] {outcome:30s} | {task[:60]}{flag}", flush=True)

    passed = sum(1 for r in results if r["outcome"] == "completed_verified_pass")
    print(f"[claude/{label}] TOTAL: {passed}/{len(results)}", flush=True)
    return {"label": label, "passed": passed, "total": len(results), "samples": SAMPLES, "results": results}


if __name__ == "__main__":
    client = AnthropicModelClient(model=MODEL)

    set_a_summary = run_set("set_a", SET_A_TASKS, SET_A_FILES, client)
    set_b_summary = run_set("set_b", SET_B_TASKS, SET_B_FILES, client)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump({"set_a": set_a_summary, "set_b": set_b_summary}, f, indent=2)

    print("=== DONE ===")
    print(f"Set A: {set_a_summary['passed']}/{set_a_summary['total']} (n={SAMPLES} samples/task)")
    print(f"Set B: {set_b_summary['passed']}/{set_b_summary['total']} (n={SAMPLES} samples/task)")

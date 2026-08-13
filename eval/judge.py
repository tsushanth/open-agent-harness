"""LLM-judge harness-quality loop — pairwise.

Given two run summaries (each produced by run_claude_via_harness.py or
run_transformers_eval*.py, both of which now save per-task diffs), asks a Claude judge to pick
the better of the two diffs for each task, head-to-head — not two independent absolute scores.

Why pairwise, not absolute: LLM judges are much better calibrated comparing two concrete
artifacts than emitting a reliable 1-5 score in isolation (the same failure mode DPO's
Bradley-Terry preference model sidesteps by training on pairs instead of a scalar reward). An
earlier version of this script asked for independent correctness/cleanliness scores per side;
kept as ABSOLUTE_MODE for when only one side has a diff (e.g. comparing a run against a task
that produced no changes at all, where there's nothing to pair against).

Side order is randomized-by-task-index (not literally randomized — alternated deterministically)
and the judge is told only "solution A" / "solution B", not which model produced which, to
reduce anchoring on knowing "this is Claude's" vs "this is the open model's."

Usage:
    python3 eval/judge.py path/to/run_a/summary.json path/to/run_b/summary.json \
        --label-a claude --label-b qwen-lora
"""
import argparse
import json
import sys

import anthropic

from run_transformers_eval import TASKS as SET_A_TASKS
from run_transformers_eval2 import TASKS as SET_B_TASKS

JUDGE_MODEL = "claude-sonnet-5"

PAIRWISE_SYSTEM_PROMPT = """You are a strict but fair code reviewer comparing two candidate
fixes for the same coding task. You'll see the task description and two unified diffs, labeled
Solution A and Solution B. You do not know which model or system produced either one — judge
only what's in front of you.

Pick the better solution considering both correctness (does it fully solve the stated task) and
cleanliness (no leftover dead code, no unnecessary changes, no signs of a confused/flailing
edit). If one solution made no change at all, the other wins by default unless it's also empty.
If both are genuinely equivalent in quality, say so — don't force a preference.

Reply with ONLY a JSON object, no prose, no markdown fence:
{"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}"""

ABSOLUTE_SYSTEM_PROMPT = """You are a strict but fair code reviewer judging a single candidate
fix. Rate correctness (does it fully solve the stated task) and cleanliness (no leftover dead
code, no unnecessary changes) each 1-5.

Reply with ONLY a JSON object, no prose, no markdown fence:
{"correctness": <1-5>, "cleanliness": <1-5>, "rationale": "<one sentence>"}"""


def _diff_text(diffs: dict) -> str:
    if not diffs:
        return "(no file was changed)"
    return "\n\n".join(f"--- {name} ---\n{d}" for name, d in diffs.items())


def judge_pairwise(client: anthropic.Anthropic, task: str, diffs_a: dict, diffs_b: dict) -> dict:
    prompt = (
        f"Task: {task}\n\n"
        f"Solution A:\n{_diff_text(diffs_a)}\n\n"
        f"Solution B:\n{_diff_text(diffs_b)}"
    )
    # See judge_absolute's comment: retry once on an empty completion before giving up.
    for attempt in range(2):
        response = client.messages.create(
            model=JUDGE_MODEL, max_tokens=200, system=PAIRWISE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if text:
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"winner": None, "rationale": f"judge output unparseable: {text[:200]}"}


def judge_absolute(client: anthropic.Anthropic, task: str, diffs: dict) -> dict:
    if not diffs:
        return {"correctness": 1, "cleanliness": 1, "rationale": "No file was changed at all."}
    # One retry: occasionally the judge call returns an empty completion (observed directly,
    # ~1/18 calls) with no error — a transient API/sampling hiccup, not a parsing issue, so
    # retrying once is the right fix rather than treating it as unparseable output.
    for attempt in range(2):
        response = client.messages.create(
            model=JUDGE_MODEL, max_tokens=200, system=ABSOLUTE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Task: {task}\n\nDiff:\n{_diff_text(diffs)}"}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if text:
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correctness": None, "cleanliness": None, "rationale": f"judge output unparseable: {text[:200]}"}


def _results_by_index(summary: dict, set_key: str) -> list:
    if set_key not in summary:
        return []
    return summary[set_key]["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", help="summary.json for solution A")
    parser.add_argument("run_b", nargs="?", default=None, help="summary.json for solution B (omit for absolute scoring of run_a alone)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    summary_a = json.load(open(args.run_a))
    summary_b = json.load(open(args.run_b)) if args.run_b else None
    client = anthropic.Anthropic()

    all_rows = []
    tally = {"A": 0, "B": 0, "tie": 0}

    for set_key, task_list in (("set_a", SET_A_TASKS), ("set_b", SET_B_TASKS)):
        results_a = _results_by_index(summary_a, set_key)
        results_b = _results_by_index(summary_b, set_key) if summary_b else []

        for i, (task, _verify) in enumerate(task_list):
            if i >= len(results_a):
                continue
            ra = results_a[i]

            if summary_b is not None and i < len(results_b):
                rb = results_b[i]
                verdict = judge_pairwise(client, task, ra["diffs"], rb["diffs"])
                winner = verdict.get("winner")
                if winner in tally:
                    tally[winner] += 1
                row = {
                    "set": set_key, "task": task[:70],
                    f"{args.label_a}_outcome": ra["outcome"], f"{args.label_b}_outcome": rb["outcome"],
                    "winner": args.label_a if winner == "A" else args.label_b if winner == "B" else "tie",
                    "rationale": verdict.get("rationale"),
                }
                all_rows.append(row)
                print(
                    f"[{set_key}] winner={row['winner']:12s} | "
                    f"{args.label_a}={ra['outcome']:30s} {args.label_b}={rb['outcome']:30s} | {task[:55]}",
                    flush=True,
                )
            else:
                verdict = judge_absolute(client, task, ra["diffs"])
                row = {
                    "set": set_key, "task": task[:70], f"{args.label_a}_outcome": ra["outcome"],
                    "correctness": verdict.get("correctness"), "cleanliness": verdict.get("cleanliness"),
                    "rationale": verdict.get("rationale"),
                }
                all_rows.append(row)
                print(
                    f"[{set_key}] correctness={row['correctness']} cleanliness={row['cleanliness']} "
                    f"| {args.label_a}={ra['outcome']:30s} | {task[:55]}",
                    flush=True,
                )

    if summary_b is not None:
        total = sum(tally.values())
        print(f"\n=== {args.label_a} vs {args.label_b}: {args.label_a}={tally['A']} {args.label_b}={tally['B']} tie={tally['tie']} (n={total}) ===")

    out_path = args.run_a.replace(".json", "-judged.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

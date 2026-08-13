"""LLM-judge harness-quality loop.

For every task in a run's summary.json (produced by run_claude_via_harness.py or the
transformers eval scripts), asks a Claude judge — a fresh call, blind to which model produced
the diff — to rate the resulting fix on two axes: correctness (does it actually solve the
stated bug/task, independent of the verify command) and cleanliness (is it the kind of diff a
careful engineer would make, or does it show signs of harness friction — leftover dead code,
overly defensive edits, clear confusion from a bad tool-call round-trip).

This is deliberately not just "did verify pass" — verify is a narrow behavioral check (see
eval/README.md's running theme of verify-command false positives/negatives). The judge reads
the actual diff and can catch things verify can't, like batch-2026-08-09-i's dead-code case
(behaviorally correct, but left an unreachable line behind).

Cross-references the matching task's outcome from a Qwen-checkpoint eval JSON (eval/*.json) by
task-text prefix, when one is supplied, to build a real side-by-side comparison table even
though only one side has a saved diff today (Qwen's eval JSONs currently save outcome only —
see run_transformers_eval.py; a future run should save diffs the same way this script does, at
which point the judge can compare diffs directly instead of diff-vs-pass/fail).

Usage:
    python3 eval/judge.py eval/claude-via-harness-2026-08-12.json \
        --qwen-set-a eval/isolation-no-bugfix-set-a-2026-08-08.json \
        --qwen-set-b eval/isolation-no-bugfix-set-b-2026-08-08.json
"""
import argparse
import json
import os
import sys

import anthropic

JUDGE_MODEL = os.environ.get("OAH_JUDGE_MODEL", "claude-sonnet-5")

JUDGE_SYSTEM_PROMPT = """You are a strict but fair code reviewer judging the output of a coding
agent. You will be given a task description and a unified diff the agent produced. Rate it on
two axes, each 1-5:

- correctness: does this diff actually and fully solve the stated task? A diff that is
  behaviorally correct but leaves dead/unreachable code, or only partially addresses the ask,
  should NOT get a 5.
- cleanliness: is this the kind of diff a careful engineer would submit? Penalize leftover
  cruft, unnecessary changes outside the stated task, or signs of the agent flailing (comments
  like "let's try again", redundant re-reads baked into the final diff, etc — though note the
  diff you see is the FINAL state, not the full trajectory, so judge only what's actually there).

Reply with ONLY a JSON object, no prose, no markdown fence:
{"correctness": <1-5>, "cleanliness": <1-5>, "rationale": "<one sentence>"}"""


def judge_diff(client: anthropic.Anthropic, task: str, diffs: dict) -> dict:
    if not diffs:
        return {"correctness": 1, "cleanliness": 1, "rationale": "No file was changed at all."}

    diff_text = "\n\n".join(f"--- {name} ---\n{d}" for name, d in diffs.items())
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Task: {task}\n\nDiff:\n{diff_text}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correctness": None, "cleanliness": None, "rationale": f"judge output unparseable: {text[:200]}"}


def _load_qwen_outcomes(path: str | None) -> dict:
    """Maps a truncated task prefix -> outcome string, matching the 60-char truncation
    run_transformers_eval*.py uses when saving results (see their `task[:60]`)."""
    if not path:
        return {}
    data = json.load(open(path))
    lora = data.get("lora", {})
    return {r["task"][:60]: r["outcome"] for r in lora.get("results", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_summary", help="summary.json from run_claude_via_harness.py")
    parser.add_argument("--qwen-set-a", default=None)
    parser.add_argument("--qwen-set-b", default=None)
    args = parser.parse_args()

    summary = json.load(open(args.run_summary))
    qwen_outcomes = {
        "set_a": _load_qwen_outcomes(args.qwen_set_a),
        "set_b": _load_qwen_outcomes(args.qwen_set_b),
    }

    client = anthropic.Anthropic()
    all_rows = []

    for set_key in ("set_a", "set_b"):
        if set_key not in summary:
            continue
        for r in summary[set_key]["results"]:
            verdict = judge_diff(client, r["task"], r["diffs"])
            qwen_outcome = qwen_outcomes[set_key].get(r["task"][:60], "no_qwen_data")
            row = {
                "set": set_key,
                "task": r["task"][:70],
                "claude_outcome": r["outcome"],
                "claude_correctness": verdict.get("correctness"),
                "claude_cleanliness": verdict.get("cleanliness"),
                "rationale": verdict.get("rationale"),
                "qwen_outcome": qwen_outcome,
            }
            all_rows.append(row)
            flag = "  <-- harness/model gap" if "pass" in r["outcome"] and "fail" in qwen_outcome else ""
            print(
                f"[{set_key}] correctness={row['claude_correctness']} "
                f"cleanliness={row['claude_cleanliness']} | claude={r['outcome']:30s} "
                f"qwen={qwen_outcome:30s} | {row['task']}{flag}",
                flush=True,
            )

    out_path = args.run_summary.replace(".json", "-judged.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

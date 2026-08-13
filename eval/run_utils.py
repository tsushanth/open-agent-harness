"""Shared helpers for the eval runner scripts (run_transformers_eval*.py,
run_claude_via_harness.py) — diffing scratch files against their original content, and
running each task multiple times to average out single-sample LLM output noise before
trusting a verdict (observed directly: the same Claude task, e2_bug_index, passed on one run
and failed on the next with no code changes in between — see eval/README.md's harness-quality
loop section).
"""
import difflib
import os
from collections import Counter


def reset_scratch(scratch_dir: str, files: dict) -> None:
    os.makedirs(scratch_dir, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(scratch_dir, name), "w") as f:
            f.write(content)


def compute_diffs(scratch_dir: str, files: dict) -> dict:
    """Unified diff per file that actually changed, keyed by filename. Files identical to
    their original content are omitted — an empty dict means nothing was touched."""
    diffs = {}
    for name, original in files.items():
        final_path = os.path.join(scratch_dir, name)
        final = open(final_path).read() if os.path.exists(final_path) else "<file missing>"
        if final != original:
            diffs[name] = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    final.splitlines(keepends=True),
                    fromfile=f"a/{name}",
                    tofile=f"b/{name}",
                )
            )
    return diffs


def majority_outcome(samples: list[dict]) -> tuple[str, dict]:
    """Given N samples (each a {"outcome": ..., "diffs": ...} dict for the same task run
    multiple times), returns the modal outcome string and the sample whose outcome matches it
    (the first such sample, for its diffs) — not an average of diffs, since diffs from
    different runs aren't meaningfully mergeable, just a representative one.

    Ties broken by whichever outcome occurred first (Counter.most_common is stable for that
    in practice via insertion order, but we're explicit about it since the tie-breaking policy
    matters for reproducibility of what gets reported)."""
    outcomes = [s["outcome"] for s in samples]
    counts = Counter(outcomes)
    top_count = max(counts.values())
    modal_outcome = next(o for o in outcomes if counts[o] == top_count)
    representative = next(s for s in samples if s["outcome"] == modal_outcome)
    return modal_outcome, representative

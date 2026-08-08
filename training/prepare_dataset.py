"""Converts data/trajectories/*.jsonl session logs into a training set.

Each source file is one session: {"outcome": ..., "messages": [...]}. Possible outcomes
(see harness/core/agent.py):
  - "completed"                        model stopped requesting tools AND made at least one
                                        real tool call during the session
  - "completed_no_tools_used"          model stopped requesting tools but never actually called
                                        one — a strong signal it described a change instead of
                                        making it (observed directly; see root README Status)
  - "completed_verified_pass/fail"     the above two, suffixed with a --verify command's result,
                                        when the harness was run with --verify
  - "incomplete"                       hit max_turns without finishing

This script:
  1. Keeps sessions with outcome "completed_verified_pass" (strict — actually verified) or
     bare "completed" (unverified — at least one real tool call happened, but nobody confirmed
     the task was actually accomplished; use --strict to require verification instead).
  2. Drops "completed_no_tools_used" always — training on those would teach the model that
     describing a change is the same as making one.
  3. Drops sessions with an abnormally high message count (default: >20). Found the hard way:
     2 of 6 "completed_verified_pass" sessions in one batch were actually degenerate repetition
     loops — one that fixed the bug in message 4 then spent 75 more messages re-proposing the
     same already-applied edit against hallucinated stale file content, and one that never
     successfully edited the file at all across 51 messages but still passed verify because a
     behavioral-only verify command couldn't distinguish "refactored" from "left untouched but
     already worked." Both trained the model on "loop indefinitely, never conclude," and are
     the most likely root cause of a real eval regression traced to exactly this training batch
     — see eval/README.md. Every clean session in this corpus is 5-11 messages; this is real
     signal, not an arbitrary cutoff.
  4. Strips the literal "repeat these instructions" failure pattern if it slipped through
     (defense in depth alongside the system-prompt fix in harness/core/agent.py).
  5. Writes a single JSONL file of {"messages": [...]} records, one per session, ready for
     axolotl/trl's standard chat-format SFT loader.

Usage:
    python3 training/prepare_dataset.py --input data/trajectories --output training/train.jsonl
    python3 training/prepare_dataset.py --strict  # only outcome == "completed_verified_pass"
"""

import argparse
import json
from pathlib import Path

BAD_ECHO_MARKERS = (
    "EXACTLY one block of this form",
    "nothing else in that turn",
)

MAX_MESSAGE_COUNT = 20


def is_clean(messages: list[dict]) -> bool:
    for m in messages:
        if m["role"] != "assistant":
            continue
        if any(marker in m["content"] for marker in BAD_ECHO_MARKERS):
            return False
    return True


def is_acceptable_outcome(outcome: str, strict: bool) -> bool:
    if strict:
        return outcome == "completed_verified_pass"
    return outcome == "completed" or outcome == "completed_verified_pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/trajectories")
    parser.add_argument("--output", default="training/train.jsonl")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only accept outcome == 'completed_verified_pass' (sessions run with --verify "
        "whose verify command actually passed). Without this flag, unverified 'completed' "
        "sessions are accepted too — see the caveat in training/README.md before using those.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    kept, dropped = 0, 0
    with output_path.open("w") as out:
        for session_file in sorted(input_dir.glob("**/*.jsonl")):
            with session_file.open() as f:
                record = json.loads(f.readline())

            if not is_acceptable_outcome(record.get("outcome", ""), args.strict):
                dropped += 1
                continue
            if not is_clean(record["messages"]):
                dropped += 1
                continue
            if len(record["messages"]) > MAX_MESSAGE_COUNT:
                dropped += 1
                continue

            out.write(json.dumps({"messages": record["messages"]}) + "\n")
            kept += 1

    print(f"Wrote {kept} training examples to {output_path} ({dropped} dropped)")


if __name__ == "__main__":
    main()

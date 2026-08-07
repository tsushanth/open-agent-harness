"""Converts data/trajectories/*.jsonl session logs into a training set.

Each source file is one session: {"outcome": ..., "messages": [...]}. This script:
  1. Keeps only sessions with outcome == "completed" (dropped/incomplete runs would
     teach the model to give up or loop forever — bad supervision).
  2. Strips the literal "repeat these instructions" failure pattern if it slipped
     through (defense in depth alongside the system-prompt fix in harness/core/agent.py).
  3. Writes a single JSONL file of {"messages": [...]} records, one per session,
     ready for axolotl/trl's standard chat-format SFT loader.

Usage:
    python3 training/prepare_dataset.py --input data/trajectories --output training/train.jsonl
"""

import argparse
import json
from pathlib import Path

BAD_ECHO_MARKERS = (
    "EXACTLY one block of this form",
    "nothing else in that turn",
)


def is_clean(messages: list[dict]) -> bool:
    for m in messages:
        if m["role"] != "assistant":
            continue
        if any(marker in m["content"] for marker in BAD_ECHO_MARKERS):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/trajectories")
    parser.add_argument("--output", default="training/train.jsonl")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    kept, dropped = 0, 0
    with output_path.open("w") as out:
        for session_file in sorted(input_dir.glob("*.jsonl")):
            with session_file.open() as f:
                record = json.loads(f.readline())

            if record.get("outcome") != "completed":
                dropped += 1
                continue
            if not is_clean(record["messages"]):
                dropped += 1
                continue

            out.write(json.dumps({"messages": record["messages"]}) + "\n")
            kept += 1

    print(f"Wrote {kept} training examples to {output_path} ({dropped} dropped)")


if __name__ == "__main__":
    main()

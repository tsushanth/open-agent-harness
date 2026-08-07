import json
from datetime import datetime, timezone
from pathlib import Path


class TrajectoryLogger:
    """Logs each session as a JSONL file of OpenAI-format chat messages
    (system/user/assistant/tool, with tool_calls attached to assistant turns).
    This is deliberately the *exact* wire format most SFT frameworks (axolotl,
    trl, LLaMA-Factory) expect for tool-use fine-tuning — no reformatting step
    needed between "harness ran a session" and "training example."""

    def __init__(self, output_dir: str = "data/trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.output_dir / f"session-{timestamp}.jsonl"
        self._messages: list[dict] = []

    def record(self, messages: list[dict]) -> None:
        self._messages = messages

    def flush(self, outcome: str = "unknown") -> None:
        record = {"outcome": outcome, "messages": self._messages}
        with self.path.open("w") as f:
            f.write(json.dumps(record) + "\n")

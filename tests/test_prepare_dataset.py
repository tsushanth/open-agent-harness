import importlib.util
import json
import sys
from pathlib import Path

# training/ isn't a package (no __init__.py, deliberately — it's scripts, not a library),
# so import prepare_dataset.py directly by file path rather than via `from training import ...`.
_SPEC = importlib.util.spec_from_file_location(
    "prepare_dataset", Path(__file__).resolve().parent.parent / "training" / "prepare_dataset.py"
)
prepare_dataset = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_dataset"] = prepare_dataset
_SPEC.loader.exec_module(prepare_dataset)


def test_completed_accepted_unverified():
    assert prepare_dataset.is_acceptable_outcome("completed", strict=False) is True


def test_completed_rejected_in_strict_mode():
    assert prepare_dataset.is_acceptable_outcome("completed", strict=True) is False


def test_verified_pass_always_accepted():
    assert prepare_dataset.is_acceptable_outcome("completed_verified_pass", strict=False) is True
    assert prepare_dataset.is_acceptable_outcome("completed_verified_pass", strict=True) is True


def test_verified_fail_always_rejected():
    assert prepare_dataset.is_acceptable_outcome("completed_verified_fail", strict=False) is False
    assert prepare_dataset.is_acceptable_outcome("completed_verified_fail", strict=True) is False


def test_no_tools_used_always_rejected_even_with_verify_pass():
    assert prepare_dataset.is_acceptable_outcome("completed_no_tools_used", strict=False) is False
    assert (
        prepare_dataset.is_acceptable_outcome("completed_no_tools_used_verified_pass", strict=False)
        is False
    )


def test_incomplete_always_rejected():
    assert prepare_dataset.is_acceptable_outcome("incomplete", strict=False) is False


def test_echo_bug_pattern_flagged_unclean():
    messages = [
        {"role": "assistant", "content": "EXACTLY one block of this form and nothing else in that turn: {}"}
    ]
    assert prepare_dataset.is_clean(messages) is False


def test_clean_messages_pass():
    messages = [
        {"role": "assistant", "content": '<tool_call>{"name": "bash", "arguments": {}}</tool_call>'},
        {"role": "assistant", "content": "Task completed."},
    ]
    assert prepare_dataset.is_clean(messages) is True


def test_main_drops_abnormally_long_sessions(tmp_path, monkeypatch):
    # Regression test: a real "completed_verified_pass" session in the corpus was a 51-message
    # repetition loop that never actually succeeded (verify passed anyway because the check was
    # behavioral-only and the untouched original code already satisfied it) — see
    # training/prepare_dataset.py's module docstring and eval/README.md. Clean sessions are all
    # 5-11 messages; anything wildly longer needs to be excluded even if outcome looks fine.
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()

    short_session = {
        "outcome": "completed_verified_pass",
        "messages": [{"role": "assistant", "content": "done"}] * 5,
    }
    (input_dir / "short.jsonl").write_text(json.dumps(short_session) + "\n")

    long_session = {
        "outcome": "completed_verified_pass",
        "messages": [{"role": "assistant", "content": "looping"}] * 51,
    }
    (input_dir / "long.jsonl").write_text(json.dumps(long_session) + "\n")

    output_path = tmp_path / "train.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["prepare_dataset.py", "--input", str(input_dir), "--output", str(output_path)]
    )
    prepare_dataset.main()

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert len(json.loads(lines[0])["messages"]) == 5


def test_main_recurses_into_batch_subdirectories(tmp_path, monkeypatch):
    # Regression test: prepare_dataset.py's real-world data lives partly in batch-*/
    # subdirectories (see data/trajectories/batch-2026-08-07/), not just the top level.
    input_dir = tmp_path / "trajectories"
    batch_dir = input_dir / "batch-example"
    batch_dir.mkdir(parents=True)

    session = {
        "outcome": "completed",
        "messages": [{"role": "assistant", "content": "done"}],
    }
    (batch_dir / "session-1.jsonl").write_text(json.dumps(session) + "\n")

    output_path = tmp_path / "train.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["prepare_dataset.py", "--input", str(input_dir), "--output", str(output_path)]
    )
    prepare_dataset.main()

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1

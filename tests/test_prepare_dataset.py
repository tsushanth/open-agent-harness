import importlib.util
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

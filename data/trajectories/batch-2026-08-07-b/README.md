# Batch: 2026-08-07 (batch b — scaled up)

Second batch, larger and more varied than the first: 10 tasks across a 6-file scratch project
(bug fixes, feature additions, a defensive-fix task), against Qwen2.5-Coder-7B-Instruct via vLLM
on a rented RTX A5000. 2 genuine passes.

| Session | Task | Outcome | What actually happened |
|---|---|---|---|
| `session-20260807T233844Z` | Fix a syntax error in `stats.py` | `completed_verified_fail` | Model made a tool call, but the file still has the same missing `:` after `if n % 2 == 0` — the edit didn't land correctly. |
| `session-20260807T233847Z` | Add `clear()` to `Inventory` | `completed_verified_pass` | Correct. Independently re-verified. |
| `session-20260807T233851Z` | Add `is_url()` to `validators.py` | `completed_no_tools_used_verified_fail` | Printed the function as prose, never called `write_file`. |
| `session-20260807T233855Z` | Add `size()` to `LRUCache` | `completed_verified_fail` | Method was added via a real tool call, but references `self.cache`, which doesn't exist — the actual attribute is `self.data`. Would raise `AttributeError` at runtime. A subtle failure a "does it compile" check alone wouldn't catch; the verify command's actual assertion did. |
| `session-20260807T233858Z` | Add `contains()` to `Inventory` | `completed_no_tools_used_verified_fail` | Printed as prose, no tool call. |
| `session-20260807T233905Z` | Add `mode()` to `stats.py` | `completed_no_tools_used_verified_fail` | Printed as prose, no tool call. |
| `session-20260807T233913Z` | Add `camel_to_snake()` to `formatter.py` | `completed_no_tools_used_verified_fail` | Printed as prose, no tool call. |
| `session-20260807T233918Z` | Add `strip_empty_lines()` to `parser.py` | `completed_no_tools_used_verified_fail` | Printed as prose, no tool call. |
| `session-20260807T233921Z` | Fix `Inventory.remove()` to raise `ValueError` on insufficient stock | `completed_verified_pass` | Correct. Independently re-verified. |
| `session-20260807T233925Z` | Add `is_ipv4()` to `validators.py` | `completed_no_tools_used_verified_fail` | Printed as prose, no tool call. |

**The dominant failure mode here (6 of 10) is `completed_no_tools_used`** — the model answering
"add a function that does X" by writing the function in its reply instead of calling `write_file`,
despite the system prompt already containing an explicit "printing code does NOT change the
file" instruction (added specifically because of this pattern in the previous batch). This is
consistent enough across "add a function/method" style tasks in this batch to look like a real
default-behavior bias in this base model for generative tasks specifically, not noise — see the
root README Status section.

Combined with `batch-2026-08-07`: 14 tasks, 3 real passes (21%). Only the 2 `_verified_pass`
sessions here pass `prepare_dataset.py`'s default filter.

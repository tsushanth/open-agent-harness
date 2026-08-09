"""Runs the 10 held-out eval tasks in-process against base and base+LoRA, via LocalModelClient.
No serving, no HTTP, no vLLM — everything happens inside this one Python process.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/open-agent-harness")

from harness.core.agent import Agent
from harness.core.local_model_client import LocalModelClient
from harness.core.trajectory import TrajectoryLogger

SCRATCH = "/workspace/eval_scratch"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = os.environ.get(
    "OAH_ADAPTER_PATH", "/workspace/open-agent-harness/training/out/qwen2.5-coder-7b-oah-lora"
)
RESULTS_DIR = os.environ.get("OAH_RESULTS_DIR", "/workspace/eval_results")

TASKS = [
    ("Add a cylinder_volume(r, h) function to h1_area_calc.py that returns pi*r^2*h using the math module",
     "python3 -c 'import h1_area_calc, math; assert abs(h1_area_calc.cylinder_volume(2,3)-math.pi*4*3)<1e-9'"),
    ("Add a ship(sku, qty) method to the Warehouse class in h2_inventory2.py that decrements self.items[sku] by qty",
     "python3 -c 'import h2_inventory2; w=h2_inventory2.Warehouse(); w.receive(\"a\",10); w.ship(\"a\",3); assert w.items[\"a\"]==7'"),
    ("Add a snake_case(s) function to h3_string_ops.py that converts a title-case string like \"Hello World\" to \"hello_world\"",
     "python3 -c 'import h3_string_ops; assert h3_string_ops.snake_case(\"Hello World\")==\"hello_world\"'"),
    ("Add a max_depth(s) function to h4_bracket_check.py that returns the maximum nesting depth of brackets in s",
     "python3 -c 'import h4_bracket_check; assert h4_bracket_check.max_depth(\"({[]})\")==3'"),
    ("Add a top_player() method to the Leaderboard class in h5_leaderboard.py that returns the player name with the highest score in self.entries",
     "python3 -c 'import h5_leaderboard; l=h5_leaderboard.Leaderboard(); l.submit_score(\"a\",10); l.submit_score(\"b\",20); assert l.top_player()==\"b\"'"),
    ("Add a km_to_miles(km) function to h6_unit_convert.py that reverses miles_to_km",
     "python3 -c 'import h6_unit_convert; assert abs(h6_unit_convert.km_to_miles(1.60934)-1)<1e-6'"),
    ("Add a highest_priority() method to the TaskQueue class in h7_task_queue.py that returns the name of the task with the highest priority value in self.tasks",
     "python3 -c 'import h7_task_queue; q=h7_task_queue.TaskQueue(); q.add_task(\"a\",1); q.add_task(\"b\",5); assert q.highest_priority()==\"b\"'"),
    ("Add a normalize_hex(s) function to h8_validators2.py that lowercases a hex color string like #ABCDEF",
     "python3 -c 'import h8_validators2; assert h8_validators2.normalize_hex(\"#ABCDEF\")==\"#abcdef\"'"),
    ("Add a matrix_max(m) function to h9_matrix2.py that returns the maximum value across all elements in the 2D list m",
     "python3 -c 'import h9_matrix2; assert h9_matrix2.matrix_max([[1,5],[3,2]])==5'"),
    ("Add a longest_decreasing_streak(nums) function to h10_streak.py, mirroring the existing longest_streak but for consecutive decreasing values",
     "python3 -c 'import h10_streak; assert h10_streak.longest_decreasing_streak([5,4,3,7,6])==3'"),
]

FILES = {
    "h1_area_calc.py": "def sphere_volume(r):\n    import math\n    return (4/3) * math.pi * r**3\n",
    "h2_inventory2.py": "class Warehouse:\n    def __init__(self):\n        self.items = {}\n\n    def receive(self, sku, qty):\n        self.items[sku] = self.items.get(sku, 0) + qty\n",
    "h3_string_ops.py": "def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
    "h4_bracket_check.py": "def is_balanced(s):\n    stack = []\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in ')]}':\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return len(stack) == 0\n",
    "h5_leaderboard.py": "class Leaderboard:\n    def __init__(self):\n        self.entries = {}\n\n    def submit_score(self, player, score):\n        self.entries[player] = score\n",
    "h6_unit_convert.py": "def miles_to_km(miles):\n    return miles * 1.60934\n",
    "h7_task_queue.py": "class TaskQueue:\n    def __init__(self):\n        self.tasks = []\n\n    def add_task(self, name, priority=0):\n        self.tasks.append({\"name\": name, \"priority\": priority})\n",
    "h8_validators2.py": "def is_valid_hex_color(s):\n    import re\n    return bool(re.match(r'^#[0-9A-Fa-f]{6}$', s))\n",
    "h9_matrix2.py": "def matrix_sum(m):\n    return sum(sum(row) for row in m)\n",
    "h10_streak.py": "def longest_streak(nums):\n    if not nums:\n        return 0\n    best = cur = 1\n    for i in range(1, len(nums)):\n        if nums[i] == nums[i-1] + 1:\n            cur += 1\n            best = max(best, cur)\n        else:\n            cur = 1\n    return best\n",
}


def reset_scratch():
    os.makedirs(SCRATCH, exist_ok=True)
    for name, content in FILES.items():
        with open(os.path.join(SCRATCH, name), "w") as f:
            f.write(content)


def run_eval(label: str, client: LocalModelClient) -> dict:
    results = []
    for task, verify in TASKS:
        reset_scratch()
        os.chdir(SCRATCH)
        agent = Agent(model_client=client, confirm_fn=lambda *_: True)
        logger = TrajectoryLogger(output_dir=f"{RESULTS_DIR}/{label}")
        agent.run(task, logger=logger, verify_cmd=verify)
        outcome = json.loads(logger.path.read_text())["outcome"]
        results.append({"task": task[:60], "outcome": outcome})
        print(f"[{label}] {outcome:30s} | {task[:60]}", flush=True)
    passed = sum(1 for r in results if r["outcome"] == "completed_verified_pass")
    print(f"[{label}] TOTAL: {passed}/{len(results)}", flush=True)
    return {"label": label, "passed": passed, "total": len(results), "results": results}


if __name__ == "__main__":
    skip_base = os.environ.get("OAH_SKIP_BASE") == "1"
    if skip_base:
        base_summary = {"label": "base", "passed": None, "total": len(TASKS), "results": [], "skipped": True}
    else:
        print("=== Loading base model ===", flush=True)
        base_client = LocalModelClient(base_model=BASE_MODEL, adapter_path=None)
        base_summary = run_eval("base", base_client)
        del base_client

        import torch
        torch.cuda.empty_cache()

    print("=== Loading base + LoRA adapter ===", flush=True)
    lora_client = LocalModelClient(base_model=BASE_MODEL, adapter_path=ADAPTER_PATH)
    lora_summary = run_eval("lora", lora_client)

    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump({"base": base_summary, "lora": lora_summary}, f, indent=2)

    print("=== DONE ===")
    print(f"Base:  {base_summary['passed']}/{base_summary['total']}")
    print(f"LoRA:  {lora_summary['passed']}/{lora_summary['total']}")

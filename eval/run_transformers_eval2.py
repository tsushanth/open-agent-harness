"""Second held-out eval set: bug-fix and refactor tasks, not "add a function." Same
in-process LocalModelClient approach as run_transformers_eval.py, distinct task set.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/open-agent-harness")

from harness.core.agent import Agent
from harness.core.local_model_client import LocalModelClient
from harness.core.trajectory import TrajectoryLogger

SCRATCH = "/workspace/eval_scratch2"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = os.environ.get("OAH_ADAPTER_PATH", "/workspace/open-agent-harness/training/out/qwen2.5-coder-7b-oah-lora")
RESULTS_DIR = os.environ.get("OAH_RESULTS_DIR", "/workspace/eval_results2")

TASKS = [
    ("is_valid_age in e1_bug_wrongcompare.py should also accept age==0 for newborns, but currently excludes it because it uses > instead of >=. Fix it",
     "python3 -c 'import e1_bug_wrongcompare; assert e1_bug_wrongcompare.is_valid_age(0)==True and e1_bug_wrongcompare.is_valid_age(-1)==False and e1_bug_wrongcompare.is_valid_age(200)==False'"),
    ("get_last_two in e2_bug_index.py has a bug — it returns the last item duplicated instead of the actual last two distinct items. Fix it",
     "python3 -c 'import e2_bug_index; assert e2_bug_index.get_last_two([1,2,3,4])==[3,4]'"),
    ("product_of_list in e3_bug_accumulator.py always returns 0 because the accumulator starts at the wrong value. Fix it",
     "python3 -c 'import e3_bug_accumulator; assert e3_bug_accumulator.product_of_list([2,3,4])==24'"),
    ("contains_word in e4_bug_stringcase.py should be case-insensitive but currently isn't. Fix it",
     "python3 -c \"import e4_bug_stringcase; assert e4_bug_stringcase.contains_word('Hello World','WORLD')==True\""),
    ("The Counter class in e5_bug_default_state.py uses a class-level attribute for total, so all instances incorrectly share the same counter. Fix it so each instance has its own independent counter",
     "python3 -c 'import e5_bug_default_state\nc1 = e5_bug_default_state.Counter()\nc2 = e5_bug_default_state.Counter()\nc1.increment()\nc1.increment()\nc2.increment()\nassert c1.total == 2 and c2.total == 1'"),
    ("describe_temp in e6_refactor_repeat.py repeats the same f-string message construction in every branch. Refactor it to build the category once then construct the message once, keeping identical behavior",
     "python3 -c \"import e6_refactor_repeat\nassert e6_refactor_repeat.describe_temp(-5)==\\\"It's freezing: -5C\\\"\nassert e6_refactor_repeat.describe_temp(20)==\\\"It's mild: 20C\\\"\nassert e6_refactor_repeat.describe_temp(30)==\\\"It's hot: 30C\\\"\""),
    ("can_checkout in e7_refactor_condition.py has needlessly nested if/else with '== True' comparisons. Refactor it to a single boolean expression, keeping identical behavior",
     "python3 -c 'import e7_refactor_condition\nassert e7_refactor_condition.can_checkout(10, True, True)==True\nassert e7_refactor_condition.can_checkout(10, False, True)==False\nassert e7_refactor_condition.can_checkout(0, True, True)==False\nassert e7_refactor_condition.can_checkout(10, True, False)==False'"),
    ("Downloader.__init__ in e8_multistep_cache.py hardcodes retries=0 and timeout=1 instead of using the existing Config.MAX_RETRIES and Config.TIMEOUT_SECONDS constants. Fix it",
     "python3 -c 'import e8_multistep_cache\nd = e8_multistep_cache.Downloader()\nassert d.retries == 5 and d.timeout == 10'"),
]

FILES = {
    "e1_bug_wrongcompare.py": "def is_valid_age(age):\n    return age > 0 and age < 150\n",
    "e2_bug_index.py": "def get_last_two(items):\n    return [items[-1], items[-1]]\n",
    "e3_bug_accumulator.py": "def product_of_list(nums):\n    result = 0\n    for n in nums:\n        result *= n\n    return result\n",
    "e4_bug_stringcase.py": "def contains_word(text, word):\n    return word in text\n",
    "e5_bug_default_state.py": "class Counter:\n    total = 0\n\n    def increment(self):\n        Counter.total += 1\n        return Counter.total\n",
    "e6_refactor_repeat.py": "def describe_temp(celsius):\n    if celsius < 0:\n        category = \"freezing\"\n        message = f\"It's {category}: {celsius}C\"\n    elif celsius < 15:\n        category = \"cold\"\n        message = f\"It's {category}: {celsius}C\"\n    elif celsius < 25:\n        category = \"mild\"\n        message = f\"It's {category}: {celsius}C\"\n    else:\n        category = \"hot\"\n        message = f\"It's {category}: {celsius}C\"\n    return message\n",
    "e7_refactor_condition.py": "def can_checkout(cart_total, has_payment_method, is_logged_in):\n    if is_logged_in == True:\n        if has_payment_method == True:\n            if cart_total > 0:\n                return True\n            else:\n                return False\n        else:\n            return False\n    else:\n        return False\n",
    "e8_multistep_cache.py": "class Config:\n    MAX_RETRIES = 5\n    TIMEOUT_SECONDS = 10\n\nclass Downloader:\n    def __init__(self):\n        self.retries = 0\n        self.timeout = 1\n",
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

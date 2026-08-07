import argparse
import sys

from rich.console import Console

from harness.core.agent import Agent

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="oah", description="Open Agent Harness — a Claude-Code-style CLI for open coding models.")
    parser.add_argument("task", nargs="?", help="Task description. If omitted, reads from stdin.")
    parser.add_argument("--yolo", action="store_true", help="Skip confirmation prompts for write/bash tool calls. Use with care.")
    args = parser.parse_args()

    task = args.task or sys.stdin.read()
    if not task.strip():
        console.print("[red]No task provided.[/red]")
        sys.exit(1)

    confirm_fn = (lambda *_: True) if args.yolo else _confirm
    agent = Agent(confirm_fn=confirm_fn)
    console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")
    result = agent.run(task)
    console.print(f"\n[bold green]Result:[/bold green]\n{result}")


def _confirm(tool_name: str, arguments: dict) -> bool:
    import json

    console.print(f"[yellow]Confirm[/yellow] {tool_name}({json.dumps(arguments)})")
    return console.input("Run this? \\[y/N] ").strip().lower() == "y"


if __name__ == "__main__":
    main()

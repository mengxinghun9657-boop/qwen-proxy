#!/usr/bin/env python3
"""
Delegate a task to Hermes Agent for background execution with heartbeat monitoring.

Usage:
  python hermes_delegate.py "task description"
  python hermes_delegate.py --max-turns 20 "task description"
  python hermes_delegate.py --status <task_id>
  python hermes_delegate.py --result <task_id>
  python hermes_delegate.py --list
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

TASKS_DIR = Path("/tmp/hermes-tasks")
HERMES_BIN = os.path.expanduser("~/.local/bin/hermes")


def _ensure_dir():
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def create_task(description: str, max_turns: int = 30, model: str = "qwen3.6-plus") -> str:
    task_id = uuid.uuid4().hex[:12]
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write task description
    (task_dir / "task.txt").write_text(description)

    # Initial state
    _write_state(task_dir, {
        "status": "pending",
        "progress": 0,
        "step": "initializing",
        "started_at": time.time(),
        "max_turns": max_turns,
    })

    # Build Hermes prompt with progress-reporting instructions
    prompt = f"""TASK: {description}

IMPORTANT — you are running as a background worker. Claude Code delegated this task to you.
Follow these rules:

1. After EVERY tool call, write a one-line progress update to {task_dir}/heartbeat.txt
   Format: PROGRESS:<percent> STEP:<what you just did>
   Example: PROGRESS:30 STEP:read the source files

2. When you finish the task, write the final result to {task_dir}/result.md
   Include: what you did, what you found, any recommendations.

3. If you cannot complete the task, write the reason to {task_dir}/result.md
   with status: FAILED and explanation.

4. Be concise — this is a background task, no pleasantries needed.

Start working now."""

    prompt_file = task_dir / "prompt.txt"
    prompt_file.write_text(prompt)

    # Launch Hermes in background
    log_file = task_dir / "output.log"
    cmd = [
        HERMES_BIN, "chat",
        "-m", model,
        "-t", "terminal,file",  # minimal tools for efficiency
        "--max-turns", str(max_turns),
        "-q", prompt,
        "-Q",
    ]

    with open(log_file, "w") as log:
        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
        )

    _write_state(task_dir, {"status": "running", "step": "launched"})

    return task_id


def _write_state(task_dir: Path, updates: dict):
    state_file = task_dir / "state.json"
    current = {}
    if state_file.exists():
        try:
            current = json.loads(state_file.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    current.update(updates)
    current["updated_at"] = time.time()
    state_file.write_text(json.dumps(current, indent=2))


def get_status(task_id: str) -> dict:
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return {"error": f"Task {task_id} not found"}

    state_file = task_dir / "state.json"
    heartbeat_file = task_dir / "heartbeat.txt"

    result = {}
    if state_file.exists():
        result["state"] = json.loads(state_file.read_text())

    if heartbeat_file.exists():
        lines = heartbeat_file.read_text().strip().split("\n")
        result["heartbeat"] = {
            "last": lines[-1] if lines else None,
            "total_lines": len(lines),
        }

    # Check if process is still running
    result["process_alive"] = _is_process_alive(task_dir)
    result["has_result"] = (task_dir / "result.md").exists()
    result["output_size"] = (task_dir / "output.log").stat().st_size if (task_dir / "output.log").exists() else 0

    return result


def get_result(task_id: str) -> dict:
    task_dir = TASKS_DIR / task_id
    result_file = task_dir / "result.md"
    if result_file.exists():
        return {"content": result_file.read_text(), "task_id": task_id}
    return {"error": "Result not available yet", "task_id": task_id}


def _is_process_alive(task_dir: Path) -> bool:
    """Check if the Hermes process for this task is still running."""
    pid_file = task_dir / "pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def list_tasks() -> list[dict]:
    _ensure_dir()
    tasks = []
    for d in sorted(TASKS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        task_id = d.name
        status = get_status(task_id)
        status["task_id"] = task_id
        tasks.append(status)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Delegate tasks to Hermes Agent")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all tasks")

    run_parser = sub.add_parser("run", help="Create and run a new task")
    run_parser.add_argument("description", help="Task description")
    run_parser.add_argument("--max-turns", type=int, default=30, help="Max agent turns")
    run_parser.add_argument("-m", "--model", default="qwen3.6-plus", help="Model for Hermes to use")

    status_parser = sub.add_parser("status", help="Check task status")
    status_parser.add_argument("task_id", help="Task ID")

    result_parser = sub.add_parser("result", help="Get task result")
    result_parser.add_argument("task_id", help="Task ID")

    args = parser.parse_args()

    if args.cmd == "list":
        _ensure_dir()
        tasks = list_tasks()
        if not tasks:
            print("No tasks found.")
            return
        print(f"{'TASK_ID':<14} {'STATUS':<12} {'HEARTBEAT':<8} {'RESULT':<8} {'LAST MSG'}")
        print("-" * 80)
        for t in tasks:
            state = t.get("state", {})
            hb = t.get("heartbeat", {})
            last = (hb.get("last") or "-")[:50]
            print(f"{t['task_id']:<14} {state.get('status','?'):<12} "
                  f"{hb.get('total_lines',0):<8} {'YES' if t.get('has_result') else 'no':<8} {last}")
    elif args.cmd == "run":
        _ensure_dir()
        task_id = create_task(args.description, args.max_turns, args.model)
        print(f"Task launched: {task_id}")
        print(f"Check status: python hermes_delegate.py status {task_id}")
        print(f"Get result:  python hermes_delegate.py result {task_id}")
    elif args.cmd == "status":
        _ensure_dir()
        status = get_status(args.task_id)
        print(json.dumps(status, indent=2, default=str))
    elif args.cmd == "result":
        _ensure_dir()
        result = get_result(args.task_id)
        if "content" in result:
            print(result["content"])
        else:
            print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

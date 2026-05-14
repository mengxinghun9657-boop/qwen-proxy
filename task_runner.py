#!/usr/bin/env python3
"""
Background task runner for Qwen — delegates multi-step work with heartbeat monitoring.

Usage:
  python task_runner.py run "multi-step task description"
  python task_runner.py status <task_id>
  python task_runner.py result <task_id>
  python task_runner.py list
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

TASKS_DIR = Path("/tmp/qwen-tasks")
QWEN_CLI = Path(__file__).parent / "ask_qwen.py"
VENV_PYTHON = Path(__file__).parent / "venv/bin/python"


def _ensure_dir():
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def run_task(description: str, mode: str = "diagnose", model: str = "qwen3.6-plus") -> str:
    task_id = uuid.uuid4().hex[:12]
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "task.txt").write_text(description)

    # Build a structured prompt that asks for progress + result
    prompt = f"""TASK: {description}

WORKFLOW:
1. Think about what steps are needed
2. For EACH step, write a progress line to {task_dir}/heartbeat.txt
   Format: [HH:MM] PROGRESS:<0-100>% STEP:<short description>
3. After the final step, write your complete findings/recommendations
   to {task_dir}/result.md
4. If you hit a dead end, write to {task_dir}/result.md:
   STATUS: FAILED
   REASON: <why>

Be thorough. This runs in background — no user is waiting for pleasantries."""

    # Write prompt and launch
    (task_dir / "prompt.txt").write_text(prompt)

    log = open(task_dir / "output.log", "w")
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(QWEN_CLI),
         "-m", model,
         "-M", mode,
         "-p", "5e396a96-bd30-422f-a0d2-c930563f4dfb",  # talk_package
         prompt],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    (task_dir / "pid").write_text(str(proc.pid))

    _write_state(task_dir, {
        "task_id": task_id,
        "status": "running",
        "description": description[:200],
        "mode": mode,
        "pid": proc.pid,
        "started_at": time.time(),
    })

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


def _is_running(task_dir: Path) -> bool:
    pid_file = task_dir / "pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def get_status(task_id: str) -> dict:
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return {"error": f"Task {task_id} not found"}

    running = _is_running(task_dir)
    state = {}
    state_file = task_dir / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())

    heartbeat = None
    hb_file = task_dir / "heartbeat.txt"
    if hb_file.exists():
        lines = hb_file.read_text().strip().split("\n")
        heartbeat = {"last": lines[-1] if lines else None, "lines": len(lines)}

    result_ready = (task_dir / "result.md").exists()
    output_size = (task_dir / "output.log").stat().st_size if (task_dir / "output.log").exists() else 0

    if not running and output_size > 0 and not result_ready:
        state["status"] = "completed"

    return {
        "task_id": task_id,
        "running": running,
        "state": state,
        "heartbeat": heartbeat,
        "result_ready": result_ready,
        "output_size": output_size,
    }


def get_result(task_id: str) -> dict:
    task_dir = TASKS_DIR / task_id
    result_file = task_dir / "result.md"
    if result_file.exists():
        return {"content": result_file.read_text(), "task_id": task_id}
    output_file = task_dir / "output.log"
    if output_file.exists():
        return {"content": output_file.read_text(), "task_id": task_id, "note": "raw output (result.md not written)"}
    return {"error": "No result available", "task_id": task_id}


def list_tasks() -> list[dict]:
    _ensure_dir()
    tasks = []
    for d in sorted(TASKS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        tasks.append(get_status(d.name))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Background task runner for Qwen")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all tasks")

    run_parser = sub.add_parser("run", help="Run a background task")
    run_parser.add_argument("description", help="Task description")
    run_parser.add_argument("-m", "--model", default="qwen3.6-plus", help="Model to use")
    run_parser.add_argument("-M", "--mode", default="diagnose",
                            help="Compression mode (diagnose, review, concise, etc.)")

    status_parser = sub.add_parser("status", help="Check task status")
    status_parser.add_argument("task_id", help="Task ID")

    result_parser = sub.add_parser("result", help="Get task result")
    result_parser.add_argument("task_id", help="Task ID")

    args = parser.parse_args()

    if args.cmd == "list":
        _ensure_dir()
        tasks = list_tasks()
        if not tasks:
            print("No tasks.")
            return
        print(f"{'ID':<14} {'STATUS':<12} {'HB':<4} {'RESULT':<8} {'LAST MSG'}")
        print("-" * 80)
        for t in tasks:
            s = t.get("state", {})
            hb = t.get("heartbeat", {})
            last = (hb.get("last") or "-")[:50] if hb else "-"
            print(f"{t['task_id']:<14} {s.get('status','?'):<12} "
                  f"{hb.get('lines',0) if hb else 0:<4} "
                  f"{'YES' if t.get('result_ready') else 'no':<8} {last}")
    elif args.cmd == "run":
        _ensure_dir()
        task_id = run_task(args.description, args.mode, args.model)
        print(f"Task: {task_id}")
        print(f"Check: python task_runner.py status {task_id}")
        print(f"Result: python task_runner.py result {task_id}")
    elif args.cmd == "status":
        _ensure_dir()
        s = get_status(args.task_id)
        print(json.dumps(s, indent=2, default=str, ensure_ascii=False))
    elif args.cmd == "result":
        _ensure_dir()
        r = get_result(args.task_id)
        if "content" in r:
            print(r["content"])
        else:
            print(json.dumps(r, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

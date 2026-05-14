#!/usr/bin/env python3
"""
Extract reusable Claude Code skills from conversation history.
Uses Qwen (via local proxy) to analyze user message patterns,
then generates skill files in ~/.claude/skills/.

Usage:
  python extract-claude-skills.py          # analyze new messages
  python extract-claude-skills.py --all    # analyze all history
  python extract-claude-skills.py --dry-run  # show what would be created
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

PROXY_URL = "http://127.0.0.1:8800/v1/chat/completions"
HISTORY_FILE = Path.home() / ".claude/history.jsonl"
STATE_FILE = Path(__file__).parent / "state" / "claude-history-pos"
SKILLS_DIR = Path.home() / ".claude/skills"
MODEL = "qwen3.6-plus"
QWEN_PROJECT_ID = "539adf19-26a6-490f-a13d-76fdbe456691"  # Hermes project

EXTRACTION_PROMPT = """You are a pattern analyst. Analyze these recent Claude Code user messages and identify patterns worth saving as reusable skills.

A skill should be extracted when:
- A workflow is repeated 3+ times (e.g., "commit and push", "deploy to X")
- The user corrects the assistant's behavior ("don't do X", "always do Y")
- The user explicitly asks to remember something
- Non-obvious project-specific conventions emerge

DO NOT extract:
- One-off requests
- Patterns already covered by existing skills
- Trivial patterns ("user says hello")

Format your response as JSON:
{
  "skills_found": [
    {
      "name": "kebab-case-name",
      "description": "one-line description",
      "trigger": "when should this skill be invoked",
      "content": "markdown content for the skill file (YAML frontmatter + body)"
    }
  ],
  "summary": "one sentence summary of findings"
}

If no new patterns found, return {"skills_found": [], "summary": "..."}

User messages:
---
{history}
---"""


def read_history(since_pos: int = 0) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    entries = []
    with open(HISTORY_FILE) as f:
        for i, line in enumerate(f):
            if i < since_pos:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def get_last_position() -> int:
    try:
        return int(STATE_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def save_position(pos: int):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(pos))


def list_existing_skills() -> list[str]:
    if not SKILLS_DIR.exists():
        return []
    return [f.stem for f in SKILLS_DIR.glob("*.md")]


def call_qwen(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a pattern analyst. Output valid JSON only, no markdown fences, no other text."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(PROXY_URL, json=body, headers={
                "x-project-id": QWEN_PROJECT_ID,
            })
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
    except Exception as e:
        print(f"Error calling Qwen: {e}", file=sys.stderr)
        return {"skills_found": [], "summary": str(e)}


def write_skill(name: str, description: str, content: str) -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SKILLS_DIR / f"{name}.md"

    # Build complete skill file with frontmatter (only if Qwen didn't include it)
    if content.strip().startswith("---"):
        skill_text = content  # Qwen already included YAML frontmatter
    else:
        skill_text = f"""---
name: {name}
description: {description}
---

{content}
"""
    filepath.write_text(skill_text)
    return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Claude Code skills from history")
    parser.add_argument("--all", action="store_true", help="Analyze all history")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    # Read history
    since = 0 if args.all else get_last_position()
    entries = read_history(since)
    if not entries:
        print("No new entries to analyze.")
        return

    # Format history for analysis (last 50 messages max for token efficiency)
    recent = entries[-50:]
    history_text = "\n".join(
        f"[{i}] {e.get('display', '')[:300]}"
        for i, e in enumerate(recent)
    )

    # Check existing skills
    existing = list_existing_skills()
    print(f"Existing skills: {existing}")
    print(f"Analyzing {len(recent)} recent messages...")

    # Build prompt (use replace to avoid format() conflicts with JSON braces)
    prompt = EXTRACTION_PROMPT.replace("{history}", history_text)
    if existing:
        prompt += f"\n\nExisting skills (do not duplicate): {existing}"

    result = call_qwen(prompt)
    skills = result.get("skills_found", [])
    summary = result.get("summary", "no summary")

    print(f"Summary: {summary}")
    print(f"New skills found: {len(skills)}")

    for skill in skills:
        name = skill.get("name", "unnamed")
        desc = skill.get("description", "")
        content = skill.get("content", "")

        if name in existing:
            print(f"  SKIP {name}: already exists")
            continue

        if args.dry_run:
            print(f"  WOULD CREATE {name}: {desc}")
            continue

        filepath = write_skill(name, desc, content)
        print(f"  CREATED {filepath}")

    # Update position
    total_lines = sum(1 for _ in open(HISTORY_FILE)) if HISTORY_FILE.exists() else 0
    save_position(total_lines)


if __name__ == "__main__":
    main()

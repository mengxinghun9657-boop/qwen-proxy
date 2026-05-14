#!/usr/bin/env python3
"""Ask any model (Qwen / DeepSeek / MiniMax) via the unified LLM gateway.

Usage:
  python ask_qwen.py "your question"
  echo "question" | python ask_qwen.py
  python ask_qwen.py -M concise "quick question"
  python ask_qwen.py -s "You are a debugger" -M diagnose "why does this crash?"
  python ask_qwen.py -c <conv_id> "follow-up question"
  python ask_qwen.py -m deepseek-chat -M review "code here"
  python ask_qwen.py -m MiniMax-M2.7 "creative writing task"
  python ask_qwen.py --list-models

Supported backends: Qwen, DeepSeek, MiniMax (via localhost:8800 gateway).
Output: model's response text to stdout.
Errors go to stderr, return code != 0 on failure.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import httpx

PROXY_BASE = os.environ.get("QWEN_PROXY_URL", "http://127.0.0.1:8800")
CONV_STORE = Path(__file__).parent / ".qwen_conversations.json"

# ---------------------------------------------------------------------------
# Compression presets — appended to system prompt to enforce compact output
# Each preset is designed to minimize context bloat while keeping signal.
# ---------------------------------------------------------------------------
COMPRESSION_MODES = {
    "concise": (
        "Reply within 150 words. Lead with the conclusion. "
        "Skip pleasantries and filler. If you must explain, use one sentence."
    ),
    "diagnose": (
        "Output format:\n"
        "ROOT CAUSE: <most likely cause, one line>\n"
        "FIX: <one line fix>\n"
        "ALT: <alternative cause if wrong>\n"
        "Max 100 words total. No greetings, no explanations beyond the format."
    ),
    "review": (
        "Output format:\n"
        "## Critical\n- [issue] (severity: H/M/L)\n"
        "## Warnings\n- [issue] (severity: H/M/L)\n"
        "## Summary\nOne sentence.\n"
        "Max 3 items per section. Max 200 words total."
    ),
    "keypoints": (
        "Output exactly 3-5 bullet points. Each point <= 25 words. "
        "No preamble, no closing summary, just the bullets."
    ),
    "judge": (
        "Answer ONLY with:\n"
        "DECISION: <YES/NO>\n"
        "CONFIDENCE: <HIGH/MEDIUM/LOW>\n"
        "REASON: <one sentence>\n"
        "No other text."
    ),
    "json": (
        "Output valid JSON only, no markdown fences, no other text. "
        "Schema: {\"findings\": [...], \"suggestion\": \"...\", \"confidence\": \"high|medium|low\"}"
    ),
}


def _load_conversations() -> dict:
    if CONV_STORE.exists():
        try:
            return json.loads(CONV_STORE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_conversations(data: dict):
    CONV_STORE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _get_conv(conv_id: str) -> Optional[str]:
    store = _load_conversations()
    return store.get(conv_id)


def _set_conv(conv_id: str, server_conv_id: str):
    store = _load_conversations()
    store[conv_id] = server_conv_id
    _save_conversations(store)


def list_models() -> list[str]:
    try:
        resp = httpx.get(f"{PROXY_BASE}/v1/models", timeout=15)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        return []


def _build_system_prompt(
    user_system: Optional[str],
    mode: Optional[str],
    max_words: Optional[int],
) -> Optional[str]:
    """Build final system prompt: user-provided + compression directive."""
    parts = []
    if user_system:
        parts.append(user_system)
    if mode and mode in COMPRESSION_MODES:
        parts.append(COMPRESSION_MODES[mode])
    if max_words:
        parts.append(f"CRITICAL: Your entire response MUST be under {max_words} words.")
    return "\n\n".join(parts) if parts else None


def ask(
    prompt: str,
    system: Optional[str] = None,
    model: str = "qwen3.6-plus",
    conv_id: Optional[str] = None,
    stream: bool = False,
    mode: Optional[str] = None,
    max_words: Optional[int] = None,
    project_id: Optional[str] = None,
) -> tuple[str, str]:
    """Send a question to Qwen, return (response_text, server_conv_id)."""
    final_system = _build_system_prompt(system, mode, max_words)

    messages = []
    if final_system:
        messages.append({"role": "system", "content": final_system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    headers = {}
    if conv_id:
        server_conv_id = _get_conv(conv_id)
        if server_conv_id:
            headers["x-conversation-id"] = server_conv_id
    if project_id:
        headers["x-project-id"] = project_id

    try:
        # Strip ALL_PROXY — httpx doesn't support socks:// and localhost doesn't need proxy
        for _key in ("ALL_PROXY", "all_proxy"):
            os.environ.pop(_key, None)
        with httpx.Client(timeout=300, http2=True, trust_env=False) as client:
            resp = client.post(
                f"{PROXY_BASE}/v1/chat/completions",
                json=body,
                headers=headers,
            )

            if resp.status_code == 401:
                print("Error: Qwen token expired. Refresh from browser.", file=sys.stderr)
                sys.exit(1)

            resp.raise_for_status()

            server_conv_id = resp.headers.get("x-conversation-id", "")

            if stream:
                full = []
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            sys.stdout.write(delta)
                            sys.stdout.flush()
                            full.append(delta)
                    except json.JSONDecodeError:
                        continue
                return "".join(full), server_conv_id
            else:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content, server_conv_id

    except httpx.ConnectError:
        print(f"Error: Cannot connect to Qwen proxy at {PROXY_BASE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Ask any model via unified LLM gateway (Qwen / DeepSeek / MiniMax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Backends (via localhost:8800):
  Qwen     - qwen3.6-plus (default), qwen3.6-max-preview, qwen3.5-plus, ...
  DeepSeek - deepseek-chat, deepseek-reasoner
  MiniMax  - MiniMax-M2.7 (with thinking), MiniMax-M2.5

Compression modes:
  concise   - 150 words max, conclusion first
  diagnose  - Root cause + fix + alternative, 100 words
  review    - Severity-tagged findings + summary, 200 words
  keypoints - 3-5 bullets only, 25 words each
  judge     - YES/NO decision + confidence + reason
  json      - Structured JSON output

Examples:
  python ask_qwen.py "explain this error"
  python ask_qwen.py -m deepseek-chat "summarize this document"
  python ask_qwen.py -m MiniMax-M2.7 "write a creative story"
  python ask_qwen.py -M concise "what is a bloom filter?"
  python ask_qwen.py -M diagnose "why does this segfault?"
  python ask_qwen.py -M review "review this code: $(cat bug.go)"
  python ask_qwen.py -M judge "should I use Redis or Kafka for this?"
  python ask_qwen.py -M json "analyze this SQL query"
  python ask_qwen.py -w 50 "extremely short answer"
  python ask_qwen.py -c debug -M diagnose "what else could cause this?"
  python ask_qwen.py -m deepseek-reasoner -M review "complex code"
  python ask_qwen.py --list-models
  python ask_qwen.py --list-modes""",
    )
    parser.add_argument("prompt", nargs="?", help="Question to ask (or use stdin)")
    parser.add_argument("-s", "--system", help="System prompt (role definition)")
    parser.add_argument("-m", "--model", default="qwen3.6-plus", help="Model to use (qwen/deepseek/minimax)")
    parser.add_argument("-M", "--mode", help="Compression preset (concise, diagnose, review, keypoints, judge, json)")
    parser.add_argument("-w", "--max-words", type=int, help="Max words in response")
    parser.add_argument("-c", "--conversation", help="Conversation ID for multi-turn")
    parser.add_argument("-p", "--project", help="Qwen project ID to assign chat to")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-modes", action="store_true", help="List compression modes")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    if args.list_models:
        models = list_models()
        if models:
            for m in models:
                print(m)
        else:
            sys.exit(1)
        return

    if args.list_modes:
        for name, desc in COMPRESSION_MODES.items():
            print(f"  {name}: {desc.split(chr(10))[0]}")
        return

    # Get prompt from args or stdin
    prompt = args.prompt
    if prompt is None:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    text, server_conv_id = ask(
        prompt=prompt,
        system=args.system,
        model=args.model,
        conv_id=args.conversation,
        stream=args.stream,
        mode=args.mode,
        max_words=args.max_words,
        project_id=args.project,
    )

    if not args.stream:
        sys.stdout.write(text)
        sys.stdout.write("\n")

    # Persist conversation id for multi-turn
    if server_conv_id and args.conversation:
        _set_conv(args.conversation, server_conv_id)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ask Qwen a question via the local reverse proxy.

Usage:
  python ask_qwen.py "your question"
  echo "question" | python ask_qwen.py
  python ask_qwen.py -s "You are a code reviewer" "review this code"
  python ask_qwen.py -c <conv_id> "follow-up question"
  python ask_qwen.py -m qwen-max-preview "complex question"
  python ask_qwen.py --list-models

Output: Qwen's response text to stdout.
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


def ask(
    prompt: str,
    system: Optional[str] = None,
    model: str = "qwen3.6-plus",
    conv_id: Optional[str] = None,
    stream: bool = False,
) -> tuple[str, str]:
    """Send a question to Qwen, return (response_text, server_conv_id)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
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

    try:
        with httpx.Client(timeout=300, http2=True) as client:
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
        description="Ask Qwen via local proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ask_qwen.py "explain this error"
  echo "review this code" | python ask_qwen.py
  python ask_qwen.py -s "You are a debugger" "why does this fail?"
  python ask_qwen.py -c my-session "what about edge cases?"
  python ask_qwen.py -m qwen-max-preview "complex analysis"
  python ask_qwen.py --list-models""",
    )
    parser.add_argument("prompt", nargs="?", help="Question to ask (or use stdin)")
    parser.add_argument("-s", "--system", help="System prompt")
    parser.add_argument("-m", "--model", default="qwen3.6-plus", help="Model to use")
    parser.add_argument("-c", "--conversation", help="Conversation ID for multi-turn")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--list-models", action="store_true", help="List available models")
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
    )

    if not args.stream:
        sys.stdout.write(text)
        sys.stdout.write("\n")

    # Persist conversation id for multi-turn
    if server_conv_id and args.conversation:
        _set_conv(args.conversation, server_conv_id)


if __name__ == "__main__":
    main()

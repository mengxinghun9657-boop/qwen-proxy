"""Async client for MiniMax Anthropic-compatible API."""
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "minimax_config.json"

BASE_URL = "https://api.minimaxi.com/anthropic"
API_VERSION = "2023-06-01"

MINIMAX_MODELS = {"minimax-m2.7", "minimax-m2.5", "minimax-m1", "MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M1"}


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_config(updates: dict):
    cfg = _load_config()
    cfg.update(updates)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def load_api_key() -> str | None:
    return _load_config().get("api_key")


class MinimaxClient:
    def __init__(self):
        self._http = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(600.0, connect=15.0),
        )

    async def aclose(self):
        await self._http.aclose()

    def _headers(self) -> dict[str, str]:
        api_key = load_api_key()
        return {
            "x-api-key": api_key or "",
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

    async def health_check(self) -> dict:
        """Quick health check by listing models or sending a minimal request."""
        api_key = load_api_key()
        if not api_key:
            return {"status": "no_token", "message": "No API key configured"}

        try:
            # Use a minimal message to verify API key works
            r = await self._http.post(
                f"{BASE_URL}/v1/messages",
                headers=self._headers(),
                json={
                    "model": "MiniMax-M2.7",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            if r.status_code == 200:
                body = r.json()
                return {
                    "status": "ok",
                    "message": f"MiniMax API 连通正常 (model: {body.get('model', 'unknown')})",
                }
            elif r.status_code == 401 or r.status_code == 403:
                return {"status": "error", "message": "API key invalid or expired"}
            else:
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                return {
                    "status": "error",
                    "message": f"HTTP {r.status_code}: {body.get('error', {}).get('message', r.text[:200])}",
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def stream_completion(
        self,
        *,
        messages: list[dict],
        model: str = "MiniMax-M2.7",
        max_tokens: int = 4096,
        thinking: bool = True,
        thinking_budget: int = 4096,
        stream: bool = True,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Call MiniMax Anthropic-compatible API and yield canonical events.

        Canonical events:
          {"type": "thinking", "text": "..."}
          {"type": "content", "text": "..."}
          {"type": "done", "stop_reason": "end_turn|max_tokens|..."}
          {"type": "error", "message": "..."}
        """
        api_key = load_api_key()
        if not api_key:
            yield {"type": "error", "message": "No MiniMax API key configured"}
            return

        # Convert OpenAI messages → Anthropic messages + system extraction
        system = None
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] in ("user", "assistant"):
                anthropic_messages.append({"role": m["role"], "content": m["content"]})
            elif m["role"] == "tool":
                # Anthropic format: tool results go in a special content block
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": m.get("content", "")}]
                })

        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if thinking and not tools:
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        # Convert OpenAI tools → Anthropic tools format
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
                })
            body["tools"] = anthropic_tools

        try:
            if stream:
                async for event in self._stream_request(body):
                    yield event
            else:
                async for event in self._non_stream_request(body):
                    yield event
        except httpx.HTTPStatusError as e:
            yield {"type": "error", "message": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    async def _non_stream_request(self, body: dict) -> AsyncIterator[dict[str, Any]]:
        r = await self._http.post(
            f"{BASE_URL}/v1/messages",
            headers=self._headers(),
            json=body,
        )
        r.raise_for_status()
        resp = r.json()

        for block in resp.get("content", []):
            if block.get("type") == "thinking":
                yield {"type": "thinking", "text": block.get("thinking", "")}
            elif block.get("type") == "text":
                yield {"type": "content", "text": block.get("text", "")}

        yield {
            "type": "done",
            "stop_reason": resp.get("stop_reason", "end_turn"),
            "usage": resp.get("usage", {}),
        }

    async def _stream_request(self, body: dict) -> AsyncIterator[dict[str, Any]]:
        async with self._http.stream(
            "POST",
            f"{BASE_URL}/v1/messages",
            headers=self._headers(),
            json=body,
        ) as r:
            r.raise_for_status()
            buf = ""
            async for text in r.aiter_text():
                buf += text
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for event in _parse_sse_block(block):
                        yield event

            # Process remaining buffer
            if buf.strip():
                for event in _parse_sse_block(buf):
                    yield event


def _parse_sse_block(block: str) -> list[dict[str, Any]]:
    """Parse a single SSE block (one or more event: lines + data: line)."""
    events: list[dict[str, Any]] = []
    event_type: str | None = None

    for line in block.strip().split("\n"):
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            ev = _translate_sse_event(event_type, data)
            if ev:
                events.append(ev)

    return events


def _translate_sse_event(event_type: str | None, data: dict) -> dict[str, Any] | None:
    """Translate Anthropic SSE events to canonical format."""
    typ = data.get("type", event_type or "")

    if typ == "ping":
        return None

    elif typ == "message_start":
        return {"type": "meta", "message_id": data.get("message", {}).get("id")}

    elif typ == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            return {
                "type": "tool_call_start",
                "index": data.get("index"),
                "name": block.get("name", ""),
                "tool_id": block.get("id", ""),
            }
        return {"type": "block_start", "index": data.get("index"), "block_type": block.get("type")}

    elif typ == "content_block_stop":
        return None

    elif typ == "message_delta":
        return {
            "type": "done",
            "stop_reason": data.get("delta", {}).get("stop_reason", "end_turn"),
            "usage": data.get("usage", {}),
        }

    elif typ == "message_stop":
        return None

    return None

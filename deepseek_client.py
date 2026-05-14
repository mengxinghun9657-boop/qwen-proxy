"""Async client for chat.deepseek.com/api/v0."""
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from deepseek_pow import solve_challenge

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "deepseek_config.json"

BASE_URL = "https://chat.deepseek.com/api/v0"
APP_VERSION = "20241129.1"

_STATIC_HEADERS: dict[str, str] = {
    "x-app-version": APP_VERSION,
    "x-client-platform": "web",
    "x-client-version": "2.0.0",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "accept": "*/*",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
}


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


def load_token() -> str | None:
    return _load_config().get("token")


def load_cookies() -> dict[str, str]:
    return _load_config().get("cookies", {})


class DeepSeekClient:
    def __init__(self):
        self._cookie_jar = httpx.Cookies()
        self._http = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(600.0, connect=15.0),
            cookies=self._cookie_jar,
        )
        # In-memory conversation store: conv_id -> {session_id, parent_message_id}
        self._conversations: dict[str, dict] = {}

    async def aclose(self):
        await self._http.aclose()

    def _headers(self, pow_resp: str | None = None) -> dict[str, str]:
        token = load_token()
        h = {**_STATIC_HEADERS, "authorization": f"Bearer {token}"}
        if pow_resp:
            h["x-ds-pow-response"] = pow_resp
        return h

    async def _post(self, path: str, json_body: dict, *, pow_resp: str | None = None) -> httpx.Response:
        for attempt in range(3):
            r = await self._http.post(
                f"{BASE_URL}{path}",
                headers=self._headers(pow_resp),
                json=json_body,
            )
            if r.status_code == 401:
                raise PermissionError("DeepSeek token expired or invalid (401)")
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            # Check biz_code for rate limit
            try:
                body = r.json()
            except Exception:
                body = None
            if body and (body.get("data") or {}).get("biz_code") == 7:
                await asyncio.sleep(2 ** attempt + 1)
                continue
            return r
        return r

    async def create_session(self) -> str:
        r = await self._post("/chat_session/create", {"character_id": None})
        r.raise_for_status()
        body = r.json()
        biz = (body.get("data") or {}).get("biz_data") or {}
        session = biz.get("chat_session") or {}
        sid = session.get("id")
        if not sid:
            raise RuntimeError(f"create_session failed: {body}")
        return sid

    async def _solve_pow(self, target: str) -> str:
        t0 = time.monotonic()
        r = await self._post("/chat/create_pow_challenge", {"target_path": target})
        r.raise_for_status()
        body = r.json()
        biz = (body.get("data") or {}).get("biz_data")
        if not biz:
            raise RuntimeError(f"create_pow_challenge failed: {body}")
        challenge = biz["challenge"]
        challenge["target_path"] = target
        resp = solve_challenge(challenge)
        log.info("deepseek pow solved for %s in %.2fs", target, time.monotonic() - t0)
        return resp

    async def stream_completion(
        self,
        *,
        session_id: str,
        prompt: str,
        parent_message_id: int | None = None,
        thinking: bool = False,
        search: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        target = "/api/v0/chat/completion"
        pow_resp = await self._solve_pow(target)

        body = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
        }

        for attempt in range(4):
            yielded_any = False
            retryable_error: str | None = None
            try:
                async with self._http.stream(
                    "POST",
                    f"{BASE_URL}/{target.replace('/api/v0/', '')}",
                    headers=self._headers(pow_resp),
                    json=body,
                ) as r:
                    if r.status_code == 429:
                        await r.aread()
                        retryable_error = "HTTP 429"
                    elif r.status_code != 200:
                        body_bytes = await r.aread()
                        raise RuntimeError(
                            f"completion HTTP {r.status_code}: {body_bytes.decode(errors='replace')[:300]}"
                        )
                    else:
                        async for ev in _parse_stream(r):
                            yielded_any = True
                            yield ev
                        return
            except RuntimeError as e:
                msg = str(e)
                if yielded_any or "biz_code=7" not in msg:
                    raise
                retryable_error = msg
            await asyncio.sleep(2 ** attempt + 1)
            pow_resp = await self._solve_pow(target)

        raise RuntimeError(f"completion failed after retries: {retryable_error}")


async def _parse_stream(r: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse DeepSeek SSE stream into canonical events.

    DeepSeek uses a p/v/o path-based delta model:
      {"p": "response/fragments/-1/content", "o": "APPEND", "v": "text"}  — new path + op
      {"v": "more text"}                                                   — continues previous path
      {"p": "response", "o": "BATCH", "v": [...]}                        — batch update
      {"p": "response/status", "o": "SET", "v": "FINISHED"}             — status change

    Events: ready, update_session, title, close
    """
    event: str | None = None
    current_path: str = ""
    response_msg_id: int | None = None
    buf = ""

    async for text in r.aiter_text():
        buf += text
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue

            # --- SSE event line ---
            if line.startswith("event:"):
                event = line[6:].strip()
                continue

            if not line.startswith("data:"):
                continue

            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            # --- Handle events ---
            if event == "ready":
                response_msg_id = chunk.get("response_message_id")
                event = None
                continue

            if event == "close":
                yield {"type": "done", "message_id": response_msg_id, "finish_reason": "stop"}
                return

            if event in ("update_session", "title"):
                event = None
                continue

            event = None

            # --- Path / Op / Value model ---
            p = chunk.get("p")
            o = chunk.get("o")
            v = chunk.get("v")

            if p:
                current_path = p

            # Initial response fragment (embedded in update_session data)
            if isinstance(v, dict) and "response" in v:
                fragments = v["response"].get("fragments") or []
                for frag in fragments:
                    content = frag.get("content")
                    if isinstance(content, str) and content:
                        yield {"type": "content", "text": content}
                continue

            # Content: response/fragments/-1/content with APPEND or implicit continue
            if current_path == "response/fragments/-1/content" and isinstance(v, str):
                yield {"type": "content", "text": v}
            # Thinking: response/fragments/-1/thinking with APPEND
            elif current_path.startswith("response/fragments/-1/thinking") and isinstance(v, str):
                yield {"type": "thinking", "text": v}
            # Search status
            elif current_path == "response/search_status" and isinstance(v, str):
                yield {"type": "search_status", "status": v}
            # Search results
            elif current_path == "response/search_results" and isinstance(v, list):
                yield {"type": "search_results", "results": v}
            # Finished status
            elif current_path == "response/status" and v == "FINISHED":
                pass  # done event will come via event:close
            # Batch with status
            elif current_path == "response" and o == "BATCH" and isinstance(v, list):
                for item in v:
                    if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                        pass  # quasi finish

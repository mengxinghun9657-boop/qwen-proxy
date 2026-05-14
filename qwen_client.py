import json
import os
import time
import uuid
from typing import AsyncGenerator

import httpx

QWEN_BASE = "https://chat.qwen.ai"


def _http_client(**kwargs) -> httpx.AsyncClient:
    """Create httpx client with http2 and proxy from http(s)_proxy env vars.
    Skips socks:// proxies (ALL_PROXY) which httpx does not support.
    """
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or \
            os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    kwargs.setdefault("http2", True)
    kwargs.setdefault("follow_redirects", True)
    if proxy:
        return httpx.AsyncClient(proxy=proxy, **kwargs)
    return httpx.AsyncClient(**kwargs)


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }


def _now_ts() -> int:
    return int(time.time())


def _uuid() -> str:
    return str(uuid.uuid4())


class QwenClient:
    def __init__(self, token: str):
        self.token = token

    # ---- models ----

    async def list_models(self) -> list[dict]:
        async with _http_client(timeout=30) as client:
            resp = await client.get(
                f"{QWEN_BASE}/api/v2/models",
                headers=_headers(self.token),
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                raw = data["data"]["data"]
                return [
                    {"id": m["id"], "info": m.get("info", {})}
                    for m in raw
                ]
            except (KeyError, TypeError):
                return []

    # ---- chat ----

    async def create_chat(self, model: str) -> str:
        """Create a new chat, return chat_id.
        Note: parent_id is NOT returned by the API for new chats.
        The first message must be sent with parent_id=null.
        """
        payload = {
            "model": model,
            "timestamp": _now_ts(),
        }
        async with _http_client(timeout=30) as client:
            resp = await client.post(
                f"{QWEN_BASE}/api/v2/chats/new",
                headers=_headers(self.token),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            chat_id = data["data"]["id"]
            if not chat_id:
                raise RuntimeError(f"Failed to create chat: {data}")
            return chat_id

    # ---- send message ----

    async def send_message(
        self, chat_id: str, parent_id: str | None, content: str,
        model: str, system: str | None = None,
        stream: bool = True,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Send a message and yield SSE chunks.
        Set parent_id=None for the first message in a chat.
        """
        user_fid = _uuid()
        assistant_fid = _uuid()

        message = {
            "fid": user_fid,
            "parentId": parent_id,
            "parent_id": parent_id,
            "role": "user",
            "content": content,
            "chat_type": "t2t",
            "sub_chat_type": "t2t",
            "timestamp": _now_ts(),
            "user_action": "chat",
            "models": [model],
            "files": [],
            "childrenIds": [assistant_fid],
            "extra": {"meta": {"subChatType": "t2t"}},
            "feature_config": {
                "thinking_enabled": False,
                "output_schema": "phase",
            },
        }

        payload: dict = {
            "stream": stream,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "messages": [message],
            "model": model,
            "parent_id": parent_id,
            "timestamp": _now_ts(),
        }

        if system:
            payload["system_message"] = system
        if tools:
            payload["tools"] = tools

        async with _http_client(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{QWEN_BASE}/api/v2/chat/completions?chat_id={chat_id}",
                headers=_headers(self.token),
                json=payload,
            ) as resp:
                if resp.status_code == 401 or resp.status_code == 403:
                    raise PermissionError("Token expired or invalid (401/403)")
                resp.raise_for_status()

                if not stream:
                    body = await resp.aread()
                    yield json.loads(body)
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Capture new parent_id from response.created (first chunk for new chats)
                        rc = chunk.get("response.created")
                        if rc and rc.get("parent_id"):
                            yield {"_type": "response.created", "parent_id": rc["parent_id"]}
                        yield chunk
                    except json.JSONDecodeError:
                        continue


    # ---- projects ----

    async def list_projects(self) -> list[dict]:
        """List all projects."""
        async with _http_client(timeout=15) as client:
            resp = await client.get(
                f"{QWEN_BASE}/api/v2/projects/",
                headers=_headers(self.token),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", []) if data.get("success") else []

    async def add_chat_to_project(self, chat_id: str, project_id: str) -> bool:
        """Associate an existing chat with a project."""
        async with _http_client(timeout=15) as client:
            resp = await client.post(
                f"{QWEN_BASE}/api/v2/projects/add_chat",
                headers=_headers(self.token),
                json={"chat_ids": [chat_id], "project_id": project_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("success", False)


def extract_content(chunk: dict) -> str | None:
    """Extract delta content from a Qwen SSE chunk.
    Skips meta chunks like response.created that have no content.
    """
    if "_type" in chunk:  # meta chunk, not content
        return None
    try:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content")
    except (IndexError, KeyError, TypeError):
        pass
    return None


def extract_parent_id(chunk: dict) -> str | None:
    """Extract parent_id from a response.created chunk."""
    return chunk.get("parent_id") if chunk.get("_type") == "response.created" else None


def extract_final_usage(chunks: list[dict]) -> dict:
    """Find the last chunk with usage info."""
    for chunk in reversed(chunks):
        usage = chunk.get("usage")
        if usage:
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

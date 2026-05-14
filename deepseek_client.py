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

    def _format_tools_prompt(self, tools: list[dict]) -> str:
        """Convert OpenAI tools array to a prompt that instructs DeepSeek to emit function calls."""
        tool_descs = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            param_list = ", ".join(f"{k}({v.get('description','')})" for k, v in props.items())
            tool_descs.append(f"- {name}: {desc} [{param_list}]")

        return f"""## TOOLS — YOU MUST USE THESE

{"\n".join(tool_descs)}

## RESPONSE FORMAT — JSON TOOL CALL ONLY

{{"tool": "<name>", "arguments": {{"param": "value"}}}}

## RULES — YOU ARE A WORKER, NOT A REPORTER

1. Your ONLY job is to call tools. NEVER summarize, plan, list, or describe.
2. NEVER output a checklist of what you checked. Just call the next tool.
3. NEVER echo "[tool returned: ...]" — that is conversation HISTORY, not your output.
4. After each tool result, output the NEXT tool call immediately. NEVER write text between.
5. Only write a text reply when ALL steps are fully verified by tool results.
6. If a step fails, call a tool to fix it. Never explain the failure in text.
7. Use "arguments" key. Valid JSON. No code blocks. No markdown."""

    async def stream_completion(
        self,
        *,
        session_id: str,
        prompt: str,
        parent_message_id: int | None = None,
        thinking: bool = False,
        search: bool = False,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        target = "/api/v0/chat/completion"
        pow_resp = await self._solve_pow(target)

        final_prompt = prompt
        if tools:
            tool_instructions = self._format_tools_prompt(tools)
            # Inject tools as a critical system directive at the very beginning
            final_prompt = f"CRITICAL: {tool_instructions}\n\n---\n\n{prompt}"

        body = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": False,  # disabled globally for speed; Hermes tasks need action not deep thought
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

    DeepSeek uses a p/v/o path-based delta model.
    Also detects <|tool_call|> blocks in content and yields structured tool_call events.
    """
    event: str | None = None
    current_path: str = ""
    response_msg_id: int | None = None
    buf = ""
    content_buf = ""  # buffer for detecting tool call blocks

    async for text in r.aiter_text():
        buf += text
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue

            if line.startswith("event:"):
                event = line[6:].strip()
                continue

            if not line.startswith("data:"):
                continue

            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            if event == "ready":
                response_msg_id = chunk.get("response_message_id")
                event = None
                continue

            if event == "close":
                # Flush remaining buffered content, stripping any tool call JSON
                if content_buf.strip():
                    tcs = _extract_tool_calls(content_buf)
                    if tcs:
                        for tc in tcs:
                            yield tc
                    else:
                        # Strip any raw tool JSON before yielding as clean content
                        clean = _strip_tool_json(content_buf)
                        if clean:
                            yield {"type": "content", "text": clean}
                    content_buf = ""
                yield {"type": "done", "message_id": response_msg_id, "finish_reason": "stop"}
                return

            if event in ("update_session", "title"):
                event = None
                continue

            event = None

            p = chunk.get("p")
            o = chunk.get("o")
            v = chunk.get("v")

            if p:
                current_path = p

            if isinstance(v, dict) and "response" in v:
                fragments = v["response"].get("fragments") or []
                for frag in fragments:
                    content = frag.get("content")
                    if isinstance(content, str) and content:
                        content_buf += content
                continue

            if current_path == "response/fragments/-1/content" and isinstance(v, str):
                content_buf += v
                # Try to extract tool calls
                tc = _try_extract_tool_call(content_buf)
                if tc:
                    # Only yield non-tool text before the tool call
                    before = tc["_before"].strip()
                    if before:
                        yield {"type": "content", "text": before}
                    yield {"type": "tool_call", "name": tc["name"], "arguments": tc["arguments"]}
                    content_buf = tc["_after"]
                    continue

                # Safety: if buffer grows huge with no valid tool call, flush it
                if len(content_buf) > 5000:
                    yield {"type": "content", "text": content_buf}
                    content_buf = ""
                    continue

                # Flush safe content
                has_tool_marker = '"tool"' in content_buf
                has_potential_tool = content_buf.strip().startswith('{"tool"') or '\n{"tool"' in content_buf
                if len(content_buf) > 500 and not has_tool_marker and not has_potential_tool:
                    yield {"type": "content", "text": content_buf}
                    content_buf = ""

            elif current_path.startswith("response/fragments/-1/thinking") and isinstance(v, str):
                yield {"type": "thinking", "text": v}
            elif current_path == "response/search_status" and isinstance(v, str):
                yield {"type": "search_status", "status": v}
            elif current_path == "response/search_results" and isinstance(v, list):
                yield {"type": "search_results", "results": v}
            elif current_path == "response/status" and v == "FINISHED":
                pass
            elif current_path == "response" and o == "BATCH" and isinstance(v, list):
                for item in v:
                    if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                        pass  # quasi finish


def _try_extract_tool_call(text: str) -> dict | None:
    """Try to extract a tool call from text. Multiple fallback layers.
    Returns dict with _before, name, arguments, _after if found, else None.
    """
    import re

    # ── Layer 1: JSON tool call {"tool": "name", "args"/"arguments"/"params"/"parameters": {...}} ──
    start_pattern = r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"(?:args|arguments|params|parameters)"\s*:\s*\{'
    match = re.search(start_pattern, text)
    if match:
        name = match.group(1)
        args_start = match.end() - 1
        depth = 0
        in_string = False
        escape_next = False
        args_end = -1
        for i in range(args_start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"' and not in_string:
                in_string = True
                continue
            if c == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    args_end = i + 1
                    break
        if args_end > 0:
            rest = text[args_end:].lstrip()
            if rest.startswith('}'):
                args_json = text[args_start:args_end]
                try:
                    args = json.loads(args_json)
                    before = text[:match.start()]
                    after = text[args_end + 1:]
                    return {"_before": before, "name": name, "arguments": args, "_after": after}
                except json.JSONDecodeError:
                    # Try to salvage: replace single quotes, strip trailing commas
                    try:
                        fixed = re.sub(r',\s*}', '}', args_json)
                        fixed = re.sub(r"'", '"', fixed)
                        args = json.loads(fixed)
                        before = text[:match.start()]
                        after = text[args_end + 1:]
                        return {"_before": before, "name": name, "arguments": args, "_after": after}
                    except (json.JSONDecodeError, ValueError):
                        pass

    # ── Layer 2: tool_name({"arg": "value"}) function-call format ──
    # ── Layer 2: tool_name({"arg": "value"}) function-call format ──
    # Accept any lowercase tool name (no hardcoded list)
    fn_pattern = r'\b([a-z][a-z0-9_]{2,40})\s*\(\s*\{'
    m = re.search(fn_pattern, text)
    if m:
        name = m.group(1)
        start = m.end() - 1
        depth, in_str, esc = 0, False, False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"' and not in_str: in_str = True; continue
            if c == '"' and in_str: in_str = False; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: end = i + 1; break
        if end > 0:
            try:
                args = json.loads(text[start:end])
                return {"_before": text[:m.start()], "name": name,
                        "arguments": args, "_after": text[end:]}
            except json.JSONDecodeError:
                pass

    # ── Layer 3: OpenAI bare format {"name":"terminal","arguments":{...}} ──
    m = re.search(r'\{\s*"name"\s*:\s*"' + r'([a-z][a-z0-9_]{2,40})' + r'"\s*,\s*"(?:arguments|args|params)"\s*:\s*\{', text)
    if m:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', m.group())
        name = name_match.group(1) if name_match else "terminal"
        start = m.end() - 1
        depth, in_str, esc = 0, False, False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"' and not in_str: in_str = True; continue
            if c == '"' and in_str: in_str = False; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: end = i + 1; break
        if end > 0:
            try:
                args = json.loads(text[start:end])
                return {"_before": text[:m.start()], "name": name,
                        "arguments": args, "_after": text[end:]}
            except json.JSONDecodeError:
                pass

    # ── Layer 4: Anthropic format {"type":"tool_use",...} ── {"type": "tool_use", "name": "...", "input": {...}} ──
    m = re.search(r'\{\s*"type"\s*:\s*"tool_use"\s*,\s*"name"\s*:\s*"([^"]+)"\s*,\s*"input"\s*:\s*\{', text)
    if m:
        name = m.group(1)
        start = m.end() - 1
        depth, in_str, esc = 0, False, False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"' and not in_str: in_str = True; continue
            if c == '"' and in_str: in_str = False; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: end = i + 1; break
        if end > 0:
            try:
                args = json.loads(text[start:end])
                return {"_before": text[:m.start()], "name": name,
                        "arguments": args, "_after": text[end:]}
            except json.JSONDecodeError:
                pass

    # ── Layer 5: ReAct format "Action:..." ── "Action: X\nAction Input: {...}" ──
    m = re.search(
        r'Action\s*:\s*([a-z][a-z0-9_]{2,40})\s*\n\s*'
        r'Action\s*Input\s*:\s*(\{[^}]+\})',
        text, re.IGNORECASE
    )
    if m:
        try:
            args = json.loads(m.group(2))
            return {"_before": text[:m.start()], "name": m.group(1),
                    "arguments": args, "_after": text[m.end():]}
        except json.JSONDecodeError:
            pass

    # ── Layer 6: bash code block → terminal call ──
    bash_pattern = r'```bash\s*\n(.*?)\n```'
    match = re.search(bash_pattern, text, re.DOTALL)
    if match:
        command = match.group(1).strip()
        if command:
            before = text[:match.start()]
            after = text[match.end():]
            return {"_before": before, "name": "terminal", "arguments": {"command": command}, "_after": after}

    # ── Layer 7: "Tool called: `name` with arguments: `{...}`" format ──
    desc_pattern = r'Tool\s+(?:called|call)[:\s]+`(\w+)`\s+(?:with\s+)?(?:arguments?|args?)[:\s]+`(\{[^`]+\})`'
    m = re.search(desc_pattern, text, re.IGNORECASE)
    if m:
        try:
            args = json.loads(m.group(2))
            return {"_before": text[:m.start()], "name": m.group(1),
                    "arguments": args, "_after": text[m.end():]}
        except json.JSONDecodeError:
            pass

    # ── Layer 8: natural language commands → terminal call ──
    # "Let me check: `docker ps`" or "I'll run: docker ps"
    cmd_patterns = [
        r'(?:run|execute|check|try)[:\s]+`([^`]+)`',
        r'`(docker|nvidia|pip|python|git|curl|cd|ls|cat|mkdir|find|grep)\s[^`]+`',
    ]
    for pat in cmd_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            cmd = match.group(1) if match.lastindex else match.group(0).strip('`')
            if cmd and len(cmd) > 3:
                before = text[:match.start()]
                after = text[match.end():]
                return {"_before": before, "name": "terminal", "arguments": {"command": cmd}, "_after": after}

    return None


def _strip_tool_json(text: str) -> str:
    """Remove raw {"tool": ...} JSON blocks from text, leaving only natural language."""
    result = text
    while True:
        tc = _try_extract_tool_call(result)
        if not tc:
            break
        # Remove the tool call JSON, keep surrounding text
        result = (tc["_before"] + tc["_after"]).strip()
    return result


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract any remaining tool calls from buffered text."""
    results = []
    while True:
        tc = _try_extract_tool_call(text)
        if not tc:
            break
        results.append({"type": "tool_call", "name": tc["name"], "arguments": tc["arguments"]})
        text = tc["_after"]
    return results

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from session import SessionManager, TokenStatus
from qwen_client import QwenClient, extract_content, extract_final_usage, extract_parent_id
from deepseek_client import DeepSeekClient, load_token as load_ds_token
from minimax_client import MinimaxClient, load_api_key as load_mm_key, MINIMAX_MODELS

app = FastAPI(title="unified-llm-gateway", version="2.0.0")
session = SessionManager()

# In-memory conversation store: conv_id -> {chat_id, parent_id, model, msg_count, created_at}
_conversations: dict[str, dict] = {}
_ds_conversations: dict[str, dict] = {}
_mm_conversations: dict[str, dict] = {}

# Lazy-initialized clients
_ds_client: DeepSeekClient | None = None
_mm_client: MinimaxClient | None = None

DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner", "deepseek-r1"}


def _is_deepseek(model_id: str) -> bool:
    return model_id in DEEPSEEK_MODELS or model_id.startswith("deepseek")


def _is_minimax(model_id: str) -> bool:
    return model_id in MINIMAX_MODELS or model_id.lower().startswith("minimax")


async def _get_ds_client() -> DeepSeekClient:
    global _ds_client
    if _ds_client is None:
        _ds_client = DeepSeekClient()
    return _ds_client


async def _get_mm_client() -> MinimaxClient:
    global _mm_client
    if _mm_client is None:
        _mm_client = MinimaxClient()
    return _mm_client


def _cleanup_stale():
    """Remove conversations older than 1 hour."""
    now = time.time()
    stale = [cid for cid, c in _conversations.items() if now - c["created_at"] > 3600]
    for cid in stale:
        del _conversations[cid]
    stale2 = [cid for cid, c in _ds_conversations.items() if now - c["created_at"] > 3600]
    for cid in stale2:
        del _ds_conversations[cid]
    stale3 = [cid for cid, c in _mm_conversations.items() if now - c.get("created_at", 0) > 3600]
    for cid in stale3:
        del _mm_conversations[cid]


async def _chat_completions_deepseek(body: dict, stream: bool, conv_id: str | None, request: Request):
    """Handle DeepSeek chat completions with OpenAI-compatible output."""
    messages = body.get("messages", [])
    model = body.get("model", "deepseek-chat")

    # Auth check
    token = load_ds_token()
    if not token:
        raise HTTPException(401, detail={"error": {"message": "No DeepSeek token configured", "type": "auth_error"}})

    ds = await _get_ds_client()

    # Extract system + user messages + tools
    system = None
    tools = body.get("tools")  # OpenAI tool definitions
    for m in messages:
        if m["role"] == "system":
            system = m["content"]

    # Build conversation-aware prompt: include all messages for context
    transcript_parts = []
    last_user_content = None
    for m in messages:
        role = m["role"]
        content = m.get("content")
        tc = m.get("tool_calls")

        if role == "system":
            continue  # handled separately
        elif role == "user":
            last_user_content = content
            transcript_parts.append(f"User: {content}")
        elif role == "assistant":
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    try:
                        args = json.loads(fn.get('arguments', '{}'))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    transcript_parts.append(json.dumps({"tool": fn.get("name", "?"), "arguments": args}, ensure_ascii=False))
            elif content:
                transcript_parts.append(f"Assistant: {content}")
        elif role == "tool":
            transcript_parts.append(f"Tool result: {content}")

    if last_user_content is None:
        raise HTTPException(400, detail={"error": {"message": "No user message found", "type": "invalid_request"}})

    # Build the final prompt
    if len(transcript_parts) <= 1:
        final_prompt = last_user_content
        if system:
            final_prompt = f"[System: {system}]\n\n{final_prompt}"
    else:
        # Multi-turn: include full transcript with clear continuation marker
        history = "\n".join(transcript_parts)
        final_prompt = f"""## CONVERSATION HISTORY (same session — continue from here)

{history}

## CURRENT TURN

Continue the task. You are the SAME assistant as above. The tools results above are REAL — do NOT re-run completed steps. Pick up where you left off. If the last step was successful, move to the next. If it failed, fix it."""
        if system:
            final_prompt = f"[System: {system}]\n\n{final_prompt}"

    user_content = final_prompt

    # Conversation management
    _cleanup_stale()
    if conv_id and conv_id in _ds_conversations:
        conv = _ds_conversations[conv_id]
        session_id = conv["session_id"]
        parent_message_id = conv.get("parent_message_id")
        # Restore cached tools from previous turns
        if tools is None:
            tools = conv.get("cached_tools")
    else:
        conv_id = str(uuid.uuid4())
        session_id = await ds.create_session()
        parent_message_id = None
        _ds_conversations[conv_id] = {
            "session_id": session_id,
            "parent_message_id": None,
            "model": model,
            "msg_count": 0,
            "created_at": time.time(),
        }

    # Cache tools for future turns in this conversation
    if tools:
        _ds_conversations[conv_id]["cached_tools"] = tools

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def sse_stream() -> AsyncGenerator[str, None]:
        nonlocal parent_message_id
        async for ev in ds.stream_completion(
            session_id=session_id,
            prompt=user_content,
            parent_message_id=parent_message_id,
            thinking=True if "reasoner" in model or "r1" in model else False,
            tools=tools,
        ):
            if ev["type"] == "content":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif ev["type"] == "tool_call":
                tc_id = f"call_{uuid.uuid4().hex[:12]}"
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": tc_id, "type": "function", "function": {"name": ev["name"], "arguments": json.dumps(ev["arguments"], ensure_ascii=False)}}]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif ev["type"] == "thinking":
                pass  # skip thinking in tool mode to keep output clean
            elif ev["type"] == "done":
                # Update parent_message_id for next turn
                if ev.get("message_id"):
                    parent_message_id = ev["message_id"]
                    _ds_conversations[conv_id]["parent_message_id"] = ev["message_id"]
                # Echo session_id for client tracking
                _ds_conversations[conv_id]["session_id"] = ev.get("session_id", session_id)

        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"x-conversation-id": conv_id},
        )

    # Non-streaming: accumulate
    full_content = ""
    tool_calls = []
    final_msg_id = None
    async for ev in ds.stream_completion(
        session_id=session_id,
        prompt=user_content,
        parent_message_id=parent_message_id,
        thinking=True if "reasoner" in model or "r1" in model else False,
        tools=tools,
    ):
        if ev["type"] == "content":
            full_content += ev["text"]
        elif ev["type"] == "tool_call":
            tc_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": ev["name"], "arguments": json.dumps(ev["arguments"], ensure_ascii=False)},
            })
        elif ev["type"] == "done":
            if ev.get("message_id"):
                parent_message_id = ev["message_id"]
                _ds_conversations[conv_id]["parent_message_id"] = ev["message_id"]
            _ds_conversations[conv_id]["session_id"] = ev.get("session_id", session_id)

    if tool_calls:
        return JSONResponse({
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }, headers={"x-conversation-id": conv_id})

    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }, headers={"x-conversation-id": conv_id})


async def _chat_completions_minimax(body: dict, stream: bool, conv_id: str | None):
    """Handle MiniMax chat completions via Anthropic-compatible API.

    The Anthropic API is stateless, so the gateway accumulates message history
    keyed by conv_id and replays the full context on each request.
    """
    messages = body.get("messages", [])
    model = body.get("model", "MiniMax-M2.7")

    if not load_mm_key():
        raise HTTPException(401, detail={"error": {"message": "No MiniMax API key configured", "type": "auth_error"}})

    mm = await _get_mm_client()

    # --- Conversation management ---
    _cleanup_stale()
    new_conv_id = conv_id or str(uuid.uuid4())

    if new_conv_id in _mm_conversations:
        conv = _mm_conversations[new_conv_id]
        # Merge new messages into stored history (avoid duplicates by content+role)
        stored_msgs = conv.get("messages", [])
        for m in messages:
            if m not in stored_msgs:
                stored_msgs.append(m)
        conv["messages"] = stored_msgs
        conv["last_access"] = time.time()
        history_messages = stored_msgs
    else:
        _mm_conversations[new_conv_id] = {
            "model": model,
            "messages": list(messages),
            "created_at": time.time(),
            "last_access": time.time(),
        }
        history_messages = messages

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def sse_stream() -> AsyncGenerator[str, None]:
        full_content = ""
        async for ev in mm.stream_completion(
            messages=history_messages,
            model=model,
            max_tokens=body.get("max_tokens", 4096),
            thinking=body.get("thinking", True) if isinstance(body.get("thinking"), bool) else True,
            thinking_budget=body.get("thinking_budget", 4096) if isinstance(body.get("thinking_budget"), int) else 4096,
            stream=True,
        ):
            if ev["type"] == "content":
                full_content += ev["text"]
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif ev["type"] == "thinking":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": ev["text"]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif ev["type"] == "done":
                usage = ev.get("usage", {})
                final: dict = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                if usage:
                    final["usage"] = {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            elif ev["type"] == "error":
                err_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "error": {"message": ev["message"], "type": "upstream_error"},
                }
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        # Store assistant response in the conversation for future turns
        if full_content and new_conv_id in _mm_conversations:
            msgs = _mm_conversations[new_conv_id].get("messages", [])
            msgs.append({"role": "assistant", "content": full_content})
            _mm_conversations[new_conv_id]["last_access"] = time.time()

    if stream:
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"x-conversation-id": new_conv_id},
        )

    # Non-streaming: accumulate
    full_content = ""
    full_reasoning = ""
    usage = {}
    async for ev in mm.stream_completion(
        messages=history_messages,
        model=model,
        max_tokens=body.get("max_tokens", 4096),
        thinking=body.get("thinking", True) if isinstance(body.get("thinking"), bool) else True,
        thinking_budget=body.get("thinking_budget", 4096) if isinstance(body.get("thinking_budget"), int) else 4096,
        stream=True,
    ):
        if ev["type"] == "content":
            full_content += ev["text"]
        elif ev["type"] == "thinking":
            full_reasoning += ev["text"]
        elif ev["type"] == "done":
            usage = ev.get("usage", {})
        elif ev["type"] == "error":
            raise HTTPException(502, detail={"error": {"message": ev["message"], "type": "upstream_error"}})

    # Store response for conversation continuity
    if full_content and new_conv_id in _mm_conversations:
        msgs = _mm_conversations[new_conv_id].get("messages", [])
        msgs.append({"role": "assistant", "content": full_content})
        _mm_conversations[new_conv_id]["last_access"] = time.time()

    resp: dict = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_content},
            "finish_reason": "stop",
        }],
    }
    if full_reasoning:
        resp["choices"][0]["message"]["reasoning_content"] = full_reasoning
    if usage:
        resp["usage"] = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
    return JSONResponse(resp, headers={"x-conversation-id": new_conv_id})


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model") or session.load_default_model()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    conv_id = request.headers.get("x-conversation-id")

    # --- route DeepSeek models ---
    if _is_deepseek(model):
        return await _chat_completions_deepseek(body, stream, conv_id, request)

    # --- route MiniMax models ---
    if _is_minimax(model):
        return await _chat_completions_minimax(body, stream, conv_id)

    # --- token validation (Qwen) ---
    health = await session.health()
    if health["status"] == "no_token":
        raise HTTPException(401, detail={"error": {"message": "No token configured. PUT /token with your Qwen bearer token.", "type": "auth_error"}})
    if health["status"] == TokenStatus.EXPIRED:
        raise HTTPException(401, detail={"error": {"message": "Token expired. Refresh from browser Local Storage and PUT /token", "type": "auth_error"}})

    token = session.load_token()
    client = QwenClient(token)

    # --- extract system message ---
    system = None
    non_system = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            non_system.append(m)

    # --- conversation management ---
    chat_id: str
    parent_id: str | None  # None for first message in a new chat

    if conv_id and conv_id in _conversations:
        conv = _conversations[conv_id]
        chat_id = conv["chat_id"]
        parent_id = conv["parent_id"]
    else:
        _cleanup_stale()
        conv_id = str(uuid.uuid4())
        chat_id = await client.create_chat(model)
        parent_id = None  # first message must have null parent_id
        _conversations[conv_id] = {
            "chat_id": chat_id,
            "parent_id": None,  # updated after first response
            "model": model,
            "msg_count": 0,
            "created_at": time.time(),
        }
        # Auto-assign to project — request header overrides global config
        project_id = request.headers.get("x-project-id") or session.load_project_id()
        if project_id:
            try:
                await client.add_chat_to_project(chat_id, project_id)
            except Exception:
                pass  # non-critical: chat still works without project

    # --- find the last user message to send ---
    user_content = None
    for m in reversed(non_system):
        if m["role"] == "user":
            user_content = m["content"]
            break

    if user_content is None:
        raise HTTPException(400, detail={"error": {"message": "No user message found", "type": "invalid_request"}})

    # --- send to Qwen ---
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def sse_stream() -> AsyncGenerator[str, None]:
        nonlocal parent_id
        chunks: list[dict] = []
        async for chunk in client.send_message(
            chat_id, parent_id, user_content, model, system, stream=True,
        ):
            chunks.append(chunk)
            # Capture new parent_id from first response
            new_pid = extract_parent_id(chunk)
            if new_pid:
                parent_id = new_pid
                _conversations[conv_id]["parent_id"] = new_pid
                continue  # meta chunk, no content
            delta = extract_content(chunk)
            if delta is None:
                continue
            openai_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

        # final chunk
        usage = extract_final_usage(chunks)
        final: dict = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        if usage["total_tokens"] > 0:
            final["usage"] = usage
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"x-conversation-id": conv_id},
        )

    # --- non-streaming: accumulate ---
    full_content = ""
    all_chunks: list[dict] = []
    async for chunk in client.send_message(
        chat_id, parent_id, user_content, model, system, stream=True,
    ):
        all_chunks.append(chunk)
        new_pid = extract_parent_id(chunk)
        if new_pid:
            parent_id = new_pid
            _conversations[conv_id]["parent_id"] = new_pid
            continue
        delta = extract_content(chunk)
        if delta:
            full_content += delta

    usage = extract_final_usage(all_chunks)

    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_content},
            "finish_reason": "stop",
        }],
        "usage": usage,
    }, headers={"x-conversation-id": conv_id})


@app.get("/v1/models")
async def list_models():
    models = []

    # DeepSeek models (always available if token configured)
    if load_ds_token():
        models.append({"id": "deepseek-chat", "object": "model", "created": 0, "owned_by": "deepseek"})
        models.append({"id": "deepseek-reasoner", "object": "model", "created": 0, "owned_by": "deepseek"})

    # MiniMax models (always available if API key configured)
    if load_mm_key():
        models.append({"id": "MiniMax-M2.7", "object": "model", "created": 0, "owned_by": "minimax"})
        models.append({"id": "MiniMax-M2.5", "object": "model", "created": 0, "owned_by": "minimax"})

    # Qwen models
    health = await session.health()
    if health["status"] == "no_token":
        if not models:
            raise HTTPException(401, detail={"error": {"message": "No token configured", "type": "auth_error"}})
        return JSONResponse({"object": "list", "data": models})

    token = session.load_token()
    client = QwenClient(token)
    try:
        raw = await client.list_models()
    except Exception as e:
        if models:
            return JSONResponse({"object": "list", "data": models})
        raise HTTPException(502, detail={"error": {"message": f"Failed to fetch models: {e}", "type": "upstream_error"}})

    for m in raw:
        models.append({
            "id": m["id"],
            "object": "model",
            "created": 0,
            "owned_by": "qwen",
        })

    return JSONResponse({"object": "list", "data": models})


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unified LLM Gateway - Admin</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    --ds: #6366f1; --qw: #38bdf8; --mm: #ec4899;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font: 14px/1.6 system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .app { max-width: 800px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; margin-bottom: 6px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  .card h3 { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; }
  .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  .dot-valid { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot-expired { background: var(--red); box-shadow: 0 0 8px var(--red); }
  .dot-unknown { background: var(--amber); }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .badge-valid { background: #166534; color: #86efac; }
  .badge-expired { background: #7f1d1d; color: #fecaca; }
  .badge-unknown { background: #78350f; color: #fde68a; }
  .mono { font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; word-break: break-all; background: #0f172a; padding: 4px 8px; border-radius: 4px; color: var(--muted); }
  .mono-sm { font-size: 11px; padding: 2px 6px; }
  .form-group { margin-bottom: 14px; }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  textarea, input { width: 100%; padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border); background: #0f172a; color: var(--text); font-family: monospace; font-size: 13px; resize: vertical; }
  textarea:focus, input:focus { outline: none; border-color: var(--accent); }
  textarea { min-height: 80px; }
  button { padding: 10px 20px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all .15s; }
  .btn-primary { background: var(--accent); color: #0f172a; width: 100%; }
  .btn-primary:hover { opacity: .85; }
  .btn-sm { padding: 6px 14px; font-size: 12px; }
  .flex { display: flex; align-items: center; gap: 8px; }
  .flex-between { display: flex; justify-content: space-between; align-items: center; }
  .mt-2 { margin-top: 8px; }
  .mt-4 { margin-top: 16px; }
  .mb-2 { margin-bottom: 8px; }
  .mb-4 { margin-bottom: 16px; }
  .text-sm { font-size: 12px; }
  .text-muted { color: var(--muted); }
  .model-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .model-tag { padding: 6px 14px; border-radius: 999px; font-size: 12px; background: #1e3a5f; color: #93c5fd; border: 1px solid transparent; cursor: pointer; transition: all .15s; }
  .model-tag:hover { border-color: var(--accent); background: #1e4265; }
  .model-tag.active { background: #0c4a6e; color: #38bdf8; border-color: var(--accent); box-shadow: 0 0 8px rgba(56,189,248,.3); }
  .model-tag .set-default { display: none; margin-left: 4px; font-size: 10px; opacity: .7; }
  .model-tag:hover .set-default { display: inline; }
  .model-tag.active .set-default { display: inline; }
  .project-item { padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); cursor: pointer; transition: all .15s; display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .project-item:hover { border-color: var(--accent); background: #1a2d40; }
  .project-item.active { border-color: var(--accent); background: #0c4a6e; box-shadow: 0 0 8px rgba(56,189,248,.2); }
  .project-item .name { font-size: 14px; font-weight: 500; }
  .project-item .hint { font-size: 11px; color: var(--muted); }
  .project-item.none { border-style: dashed; }
  .toast { position: fixed; top: 20px; right: 20px; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; z-index: 999; animation: fadeIn .2s; }
  .toast-success { background: #166534; color: #86efac; }
  .toast-error { background: #7f1d1d; color: #fecaca; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } }
  .info-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
  .info-row:last-child { border-bottom: none; }
</style>
</head>
<body>
<div class="app">
  <h1>Unified LLM Gateway</h1>
  <div class="subtitle">Qwen + DeepSeek + MiniMax · <span id="serverTime">--</span></div>

  <div class="grid" id="statusCards"></div>

  <div class="card mb-4">
    <div class="flex-between mb-4">
      <h3 style="margin-bottom:0">Token 配置</h3>
      <button class="btn-primary btn-sm" onclick="copyToken()" style="width:auto">复制当前 Token</button>
    </div>
    <div>
      <label>当前 Token 预览</label>
      <div class="mono" id="tokenPreview">--</div>
    </div>
    <div class="mt-2">
      <label>JWT 信息</label>
      <div id="jwtInfo" class="text-muted text-sm">--</div>
    </div>
    <div class="mt-4">
      <label for="newToken">更新 Token（从浏览器 Local Storage 复制）</label>
      <textarea id="newToken" placeholder="粘贴完整 JWT token..."></textarea>
      <button class="btn-primary mt-2" onclick="updateToken()">保存 Token</button>
    </div>
    <div class="mt-2" id="lastUpdated" style="font-size:11px;color:var(--muted)"></div>
  </div>

  <div class="card mb-4" style="border-left: 3px solid var(--ds)">
    <h3 style="margin-bottom:12px;color:var(--ds)">DeepSeek 配置</h3>
    <div class="flex-between mb-2">
      <span style="font-size:12px;color:var(--muted)">Token 状态</span>
      <span id="dsStatus" class="text-sm">--</span>
    </div>
    <div>
      <label>Token 预览</label>
      <div class="mono" id="dsTokenPreview">--</div>
    </div>
    <div class="mt-4">
      <label for="newDsToken">更新 DeepSeek Token（从 chat.deepseek.com Local Storage）</label>
      <textarea id="newDsToken" placeholder="粘贴 userToken..."></textarea>
      <button class="btn-primary mt-2" onclick="updateDsToken()">保存 Token</button>
    </div>
  </div>

  <div class="card mb-4" style="border-left: 3px solid #ec4899">
    <h3 style="margin-bottom:12px;color:#ec4899">MiniMax 配置</h3>
    <div class="flex-between mb-2">
      <span style="font-size:12px;color:var(--muted)">API Key 状态</span>
      <span id="mmStatus" class="text-sm">--</span>
    </div>
    <div>
      <label>Key 预览</label>
      <div class="mono" id="mmTokenPreview">--</div>
    </div>
    <div class="mt-4">
      <label for="newMmToken">更新 MiniMax API Key</label>
      <textarea id="newMmToken" placeholder="粘贴 MiniMax API key..."></textarea>
      <button class="btn-primary mt-2" onclick="updateMmToken()">保存 Key</button>
    </div>
  </div>

  <div class="card mb-4">
    <div class="flex-between mb-2">
      <h3 style="margin-bottom:0">可用模型 <span style="font-weight:400;font-size:11px;color:var(--muted)" id="defaultModelLabel"></span></h3>
      <button class="btn-primary btn-sm" onclick="loadModels()" style="width:auto">刷新</button>
    </div>
    <div class="model-list" id="modelList">加载中...</div>
  </div>

  <div class="card mb-4">
    <div class="flex-between mb-2">
      <h3 style="margin-bottom:0">会话归属项目 <span style="font-weight:400;font-size:11px;color:var(--muted)" id="projectLabel"></span></h3>
      <button class="btn-primary btn-sm" onclick="loadProjects()" style="width:auto">刷新</button>
    </div>
    <div id="projectList" style="font-size:13px;color:var(--muted)">加载中...</div>
  </div>
</div>

<div id="toastContainer"></div>

<script>
const TOKEN_KEY = 'qwen_token_cache';

function toast(msg, ok) {
  const el = document.createElement('div');
  el.className = 'toast ' + (ok ? 'toast-success' : 'toast-error');
  el.textContent = msg;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

async function api(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const detail = body?.detail;
    const msg = typeof detail === 'object' && detail?.error?.message ? detail.error.message
              : typeof detail === 'string' ? detail : resp.statusText;
    throw new Error(msg);
  }
  return resp.json();
}

function statusHtml(status) {
  const map = { valid: ['dot-valid','badge-valid','有效'], expired: ['dot-expired','badge-expired','已过期'],
                no_token: ['dot-unknown','badge-unknown','未配置'], unknown: ['dot-unknown','badge-unknown','未知'],
                error: ['dot-expired','badge-expired','错误'] };
  const [dot, badge, label] = map[status] || map.unknown;
  return `<span class="status-dot ${dot}"></span><span class="badge ${badge}">${label}</span>`;
}

async function refresh() {
  document.getElementById('serverTime').textContent = new Date().toLocaleString('zh-CN');

  let health;
  try { health = await api('/health'); } catch(e) {
    document.getElementById('statusCards').innerHTML =
      '<div class="card"><h3>状态</h3><span class="status-dot dot-expired"></span>无法连接</div>';
    return;
  }

  const token = health.token_preview || '--';
  document.getElementById('tokenPreview').textContent = token;
  document.getElementById('jwtInfo').textContent = health.jwt_info || '--';
  if (health.token_updated_at) {
    document.getElementById('lastUpdated').textContent = '上次更新: ' + new Date(health.token_updated_at).toLocaleString('zh-CN');
  }

  // Cache token for copy
  if (health.status !== 'no_token') {
    try { localStorage.setItem(TOKEN_KEY, health._raw_token || ''); } catch(e) {}
  }

  // Qwen status cards
  let qwenHtml =
    `<div class="card" style="border-left: 3px solid var(--qw)">
      <h3 style="color:var(--qw)">Qwen 后端</h3>
      <div class="flex">${statusHtml(health.status)}</div>
      <div class="text-muted text-sm mt-2">${health.message}</div>
    </div>`;

  // DeepSeek status
  let dsHealth = {status: 'unknown', message: '检查中...'};
  try { dsHealth = await api('/health/ds'); } catch(e) { dsHealth.message = e.message; }
  document.getElementById('dsStatus').innerHTML = dsHealth.status === 'ok'
    ? '<span class="status-dot dot-valid"></span><span class="badge badge-valid">有效</span>'
    : '<span class="status-dot dot-expired"></span><span class="badge badge-expired">异常</span>';
  document.getElementById('dsTokenPreview').textContent = dsHealth.status === 'ok' ? dsHealth.session_id : dsHealth.message;

  let dsHtml =
    `<div class="card" style="border-left: 3px solid var(--ds)">
      <h3 style="color:var(--ds)">DeepSeek 后端</h3>
      <div class="flex">${dsHealth.status === 'ok' ? '<span class="status-dot dot-valid"></span><span class="badge badge-valid">连通</span>' : '<span class="status-dot dot-expired"></span><span class="badge badge-expired">异常</span>'}</div>
      <div class="text-muted text-sm mt-2">${dsHealth.message}</div>
    </div>`;

  // MiniMax status
  let mmHealth = {status: 'unknown', message: '检查中...'};
  try { mmHealth = await api('/health/minimax'); } catch(e) { mmHealth.message = e.message; }
  document.getElementById('mmStatus').innerHTML = mmHealth.status === 'ok'
    ? '<span class="status-dot dot-valid"></span><span class="badge badge-valid">有效</span>'
    : '<span class="status-dot dot-expired"></span><span class="badge badge-expired">异常</span>';
  document.getElementById('mmTokenPreview').textContent = mmHealth.status === 'ok' ? mmHealth.message : mmHealth.message;

  let mmHtml =
    `<div class="card" style="border-left: 3px solid #ec4899">
      <h3 style="color:#ec4899">MiniMax 后端</h3>
      <div class="flex">${mmHealth.status === 'ok' ? '<span class="status-dot dot-valid"></span><span class="badge badge-valid">连通</span>' : '<span class="status-dot dot-expired"></span><span class="badge badge-expired">异常</span>'}</div>
      <div class="text-muted text-sm mt-2">${mmHealth.message}</div>
    </div>`;

  document.getElementById('statusCards').innerHTML = qwenHtml + dsHtml + mmHtml;

  // Models
  loadModels();
  // Projects
  loadProjects();
}

let defaultModel = '';

async function loadDefaultModel() {
  try {
    const data = await api('/default-model');
    defaultModel = data.model || '';
    document.getElementById('defaultModelLabel').textContent = defaultModel ? '· 默认: ' + defaultModel : '';
  } catch(e) { defaultModel = ''; }
}

async function loadModels() {
  const el = document.getElementById('modelList');
  try {
    const data = await api('/v1/models');
    if (!data.data || !data.data.length) { el.textContent = '无模型数据'; return; }
    await loadDefaultModel();
    // Sort by backend then name
    const backendOrder = {deepseek: 0, qwen: 1, minimax: 2};
    const backendColors = {deepseek: 'var(--ds)', qwen: 'var(--qw)', minimax: '#ec4899'};
    const backendLabels = {deepseek: 'DS', qwen: 'QW', minimax: 'MM'};
    const sorted = [...data.data].sort((a, b) => {
      if (a.owned_by !== b.owned_by) return (backendOrder[a.owned_by] ?? 3) - (backendOrder[b.owned_by] ?? 3);
      return a.id.localeCompare(b.id);
    });
    el.innerHTML = sorted.map(m => {
      const isActive = m.id === defaultModel;
      const color = backendColors[m.owned_by] || 'var(--muted)';
      const label = backendLabels[m.owned_by] || m.owned_by.slice(0,2).toUpperCase();
      return `<span class="model-tag${isActive ? ' active' : ''}" onclick="setDefaultModel('${m.id.replace(/'/g, "\\'")}')" title="${isActive ? '当前默认' : '点击设为默认'} (${m.owned_by})">
        <span style="font-size:9px;padding:1px 4px;border-radius:3px;background:${color};color:#fff;margin-right:4px">${label}</span>${m.id}<span class="set-default">${isActive ? '✓ 默认' : '设为默认'}</span>
      </span>`;
    }).join('');
  } catch(e) {
    el.textContent = '加载失败: ' + e.message;
  }
}

async function setDefaultModel(modelId) {
  try {
    await api('/default-model', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: modelId}),
    });
    defaultModel = modelId;
    document.getElementById('defaultModelLabel').textContent = '· 默认: ' + modelId;
    // Update active state on all tags
    document.querySelectorAll('.model-tag').forEach(tag => {
      const id = tag.textContent.replace(/设为默认|✓ 默认/, '').trim();
      tag.classList.toggle('active', id === modelId);
      const sd = tag.querySelector('.set-default');
      if (sd) sd.textContent = id === modelId ? '✓ 默认' : '设为默认';
    });
    toast('默认模型已设为 ' + modelId, true);
  } catch(e) { toast('设置失败: ' + e.message, false); }
}

async function loadProjects() {
  const el = document.getElementById('projectList');
  try {
    const data = await api('/projects');
    const projects = data.projects || [];
    const current = data.current_project_id;
    const label = document.getElementById('projectLabel');

    if (!projects.length) { el.innerHTML = '<div class="text-muted">无项目</div>'; return; }

    const currentProj = projects.find(p => p.id === current);
    label.textContent = currentProj ? ' · 当前: ' + currentProj.name : ' · 未设置';

    el.innerHTML = projects.map(p => {
      const isActive = p.id === current;
      return `<div class="project-item${isActive ? ' active' : ''}" onclick="setProject('${p.id}')" title="${isActive ? '当前项目' : '点击选择项目'}">
        <span class="name">${p.name}${isActive ? ' <span style="font-size:10px;color:var(--accent)">✓</span>' : ''}</span>
        <span class="hint">${p.memory_span === 'project_only' ? '独立记忆' : '共享记忆'}</span>
      </div>`;
    }).join('');

    // Add "none" option
    el.innerHTML += `<div class="project-item none${!current ? ' active' : ''}" onclick="setProject('')">
      <span class="name">不归属项目${!current ? ' <span style="font-size:10px;color:var(--accent)">✓</span>' : ''}</span>
      <span class="hint">会话独立存在</span>
    </div>`;
  } catch(e) {
    el.innerHTML = '<div class="text-muted">加载失败: ' + e.message + '</div>';
  }
}

async function setProject(projectId) {
  try {
    await api('/project-id', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({project_id: projectId || null}),
    });
    loadProjects();
    toast(projectId ? '项目已设置' : '已取消项目归属', true);
  } catch(e) { toast('设置失败: ' + e.message, false); }
}

async function updateToken() {
  const val = document.getElementById('newToken').value.trim();
  if (!val) { toast('请先粘贴 token', false); return; }
  try {
    const result = await api('/token', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: val}),
    });
    document.getElementById('newToken').value = '';
    toast(result.ok ? 'Token 已更新' : ('失败: ' + result.message), result.ok);
    refresh();
  } catch(e) { toast('保存失败: ' + e.message, false); }
}

async function copyToken() {
  try {
    const data = await api('/token');
    if (data.token) {
      await navigator.clipboard.writeText(data.token);
      toast('已复制到剪贴板', true);
    } else {
      toast('没有已配置的 token', false);
    }
  } catch(e) { toast('复制失败: ' + e.message, false); }
}

async function updateDsToken() {
  const val = document.getElementById('newDsToken').value.trim();
  if (!val) { toast('请先粘贴 token', false); return; }
  try {
    await api('/token/ds', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: val}),
    });
    document.getElementById('newDsToken').value = '';
    toast('DeepSeek Token 已更新', true);
    refresh();
  } catch(e) { toast('保存失败: ' + e.message, false); }
}

async function updateMmToken() {
  const val = document.getElementById('newMmToken').value.trim();
  if (!val) { toast('请先粘贴 API key', false); return; }
  try {
    await api('/token/minimax', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: val}),
    });
    document.getElementById('newMmToken').value = '';
    toast('MiniMax API Key 已更新', true);
    refresh();
  } catch(e) { toast('保存失败: ' + e.message, false); }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    return ADMIN_HTML


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return JSONResponse(await session.health())


@app.get("/token/status")
async def token_status():
    return JSONResponse(await session.health())


@app.get("/token")
async def get_token():
    token = session.load_token()
    if not token:
        raise HTTPException(404, detail={"error": {"message": "No token configured", "type": "not_found"}})
    return JSONResponse({"token": token})


@app.put("/token")
async def update_token(request: Request):
    body = await request.json()
    new_token = body.get("token", "").strip()
    if not new_token:
        raise HTTPException(400, detail={"error": {"message": "token is required"}})
    session.save_token(new_token)
    session.force_recheck()
    health = await session.health()
    return JSONResponse({
        "ok": health["status"] == TokenStatus.VALID,
        "status": health["status"],
        "message": health["message"],
    })


# ---------------------------------------------------------------------------
# DeepSeek token management
# ---------------------------------------------------------------------------

@app.get("/health/ds")
async def health_ds():
    token = load_ds_token()
    if not token:
        return JSONResponse({"status": "no_token", "message": "未配置 DeepSeek token"})
    try:
        ds = await _get_ds_client()
        sid = await ds.create_session()
        return JSONResponse({"status": "ok", "session_id": sid, "message": "DeepSeek API 连通正常"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.get("/token/ds")
async def get_token_ds():
    token = load_ds_token()
    if not token:
        raise HTTPException(404, detail={"error": {"message": "No DeepSeek token configured", "type": "not_found"}})
    return JSONResponse({"token": token})


@app.put("/token/ds")
async def update_token_ds(request: Request):
    body = await request.json()
    new_token = body.get("token", "").strip()
    if not new_token:
        raise HTTPException(400, detail={"error": {"message": "token is required"}})
    from deepseek_client import _load_config, _save_config
    _save_config({"token": new_token})
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# MiniMax token management
# ---------------------------------------------------------------------------

@app.get("/health/minimax")
async def health_minimax():
    api_key = load_mm_key()
    if not api_key:
        return JSONResponse({"status": "no_token", "message": "未配置 MiniMax API key"})
    mm = await _get_mm_client()
    try:
        result = await mm.health_check()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.get("/token/minimax")
async def get_token_minimax():
    api_key = load_mm_key()
    if not api_key:
        raise HTTPException(404, detail={"error": {"message": "No MiniMax API key configured", "type": "not_found"}})
    return JSONResponse({"token": api_key[:8] + "..." + api_key[-4:]})


@app.put("/token/minimax")
async def update_token_minimax(request: Request):
    body = await request.json()
    new_key = body.get("token", "").strip()
    if not new_key:
        raise HTTPException(400, detail={"error": {"message": "token is required"}})
    from minimax_client import _save_config
    _save_config({"api_key": new_key})
    global _mm_client
    _mm_client = None
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@app.get("/default-model")
async def get_default_model():
    return JSONResponse({"model": session.load_default_model()})


@app.put("/default-model")
async def set_default_model(request: Request):
    body = await request.json()
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(400, detail={"error": {"message": "model is required"}})
    session.save_default_model(model)
    return JSONResponse({"ok": True, "model": model})


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------

@app.get("/projects")
async def list_projects():
    health = await session.health()
    if health["status"] == "no_token":
        raise HTTPException(401, detail={"error": {"message": "No token configured", "type": "auth_error"}})

    token = session.load_token()
    client = QwenClient(token)
    try:
        projects = await client.list_projects()
    except Exception as e:
        raise HTTPException(502, detail={"error": {"message": f"Failed to fetch projects: {e}", "type": "upstream_error"}})

    current = session.load_project_id()
    return JSONResponse({
        "projects": projects,
        "current_project_id": current,
    })


@app.get("/project-id")
async def get_project_id():
    return JSONResponse({"project_id": session.load_project_id()})


@app.put("/project-id")
async def set_project_id(request: Request):
    body = await request.json()
    project_id = body.get("project_id")  # None or "" means clear
    if project_id:
        project_id = project_id.strip()
    else:
        project_id = None
    session.save_project_id(project_id)
    return JSONResponse({"ok": True, "project_id": project_id})

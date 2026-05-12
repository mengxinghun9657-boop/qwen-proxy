import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from session import SessionManager, TokenStatus
from qwen_client import QwenClient, extract_content, extract_final_usage, extract_parent_id

app = FastAPI(title="qwen-proxy", version="1.0.0")
session = SessionManager()

# In-memory conversation store: conv_id -> {chat_id, parent_id, model, msg_count, created_at}
_conversations: dict[str, dict] = {}


def _cleanup_stale():
    """Remove conversations older than 1 hour."""
    now = time.time()
    stale = [cid for cid, c in _conversations.items() if now - c["created_at"] > 3600]
    for cid in stale:
        del _conversations[cid]


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model") or session.load_default_model()
    messages = body.get("messages", [])
    stream = body.get("stream", True)
    conv_id = request.headers.get("x-conversation-id")

    # --- token validation ---
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
        # Auto-assign to project if configured
        project_id = session.load_project_id()
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
    health = await session.health()
    if health["status"] == "no_token":
        raise HTTPException(401, detail={"error": {"message": "No token configured", "type": "auth_error"}})

    token = session.load_token()
    client = QwenClient(token)
    try:
        raw = await client.list_models()
    except Exception as e:
        raise HTTPException(502, detail={"error": {"message": f"Failed to fetch models: {e}", "type": "upstream_error"}})

    models = []
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
<title>Qwen Proxy - Admin</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
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
  <h1>Qwen Proxy</h1>
  <div class="subtitle">Admin Dashboard · <span id="serverTime">--</span></div>

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

  document.getElementById('statusCards').innerHTML =
    `<div class="card">
      <h3>Token 状态</h3>
      <div class="flex">${statusHtml(health.status)}</div>
      <div class="text-muted text-sm mt-2">${health.message}</div>
    </div>
    <div class="card">
      <h3>远程验证</h3>
      <div class="flex">${statusHtml(health.status === 'no_token' ? 'no_token' : health.status)}</div>
      <div class="text-muted text-sm mt-2">${health.status === 'valid' ? 'API 连通正常' : health.message}</div>
    </div>`;

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
    el.innerHTML = data.data.map(m => {
      const isActive = m.id === defaultModel;
      return `<span class="model-tag${isActive ? ' active' : ''}" onclick="setDefaultModel('${m.id.replace(/'/g, "\\'")}')" title="${isActive ? '当前默认' : '点击设为默认'}">
        ${m.id}<span class="set-default">${isActive ? '✓ 默认' : '设为默认'}</span>
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

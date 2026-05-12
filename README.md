# Qwen Proxy

OpenAI 兼容的反向代理，将 [chat.qwen.ai](https://chat.qwen.ai) 的内部 API 转译为标准 `/v1/chat/completions` 接口，附带 CLI 工具用于 AI 协作。

*An OpenAI-compatible reverse proxy that translates [chat.qwen.ai](https://chat.qwen.ai)'s internal API into standard `/v1/chat/completions` endpoints, with a CLI tool for AI-assisted collaboration.*

---

## 目录 / Table of Contents

- [工作机制 / How It Works](#工作机制--how-it-works)
- [实现原理 / Implementation](#实现原理--implementation)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [API 参考 / API Reference](#api-参考--api-reference)
- [CLI 协作工具 / CLI Collaboration Tool](#cli-协作工具--cli-collaboration-tool)
- [部署 / Deployment](#部署--deployment)
- [安全 / Security](#安全--security)

---

## 工作机制 / How It Works

### 问题背景 / The Problem

Qwen 的 Web 应用 ([chat.qwen.ai](https://chat.qwen.ai)) 使用自己的一套内部 API（`/api/v2/chat/completions`），与 OpenAI 的 `/v1/chat/completions` 格式不兼容。这意味着无法用标准 OpenAI SDK、LangChain、或 Claude Code 等工具直接调用。

*Qwen's web app uses its own internal API format (`/api/v2/chat/completions`), incompatible with the standard OpenAI `/v1/chat/completions`. This means any tool built for the OpenAI protocol — SDKs, LangChain, Claude Code — cannot call Qwen directly.*

### 解决方案 / The Solution

```
                        OpenAI 协议                    Qwen 内部协议
                       (标准格式)                      (私有格式)
┌──────────────┐      POST /v1/chat/                 POST /api/v2/chat/
│              │      completions                    completions
│  Claude Code │──┐    {                               {
│  OpenAI SDK  │  │      "model": "...",                 "chat_id": "...",
│  LangChain   │  │      "messages": [...],              "messages": [{fid, parentId, ...}],
│  curl        │  │      "stream": true                  "stream": true,
│              │  │    }                                 "version": "2.1",
└──────────────┘  │                                      "incremental_output": true
                  │                                    }
                  ▼
          ┌─────────────────┐
          │   QWEN PROXY     │
          │   localhost:8800  │
          │   (FastAPI)      │
          └───────┬─────────┘
                  │
                  │  JWT Bearer Token (from browser)
                  │  HTTP/2 + SSE streaming
                  ▼
          ┌─────────────────┐
          │  chat.qwen.ai    │
          │  /api/v2/*       │
          └─────────────────┘
```

### 详细数据流 / Detailed Data Flow

**1. Token 管理**

用户从浏览器 DevTools → Application → Local Storage 取出 Qwen 的 JWT token，通过 Admin Dashboard 或直接编辑 `qwen_config.json` 配置到代理服务中。代理会：

- 解码 JWT (无需验证签名) 读取过期时间，展示剩余有效期
- 每 5 分钟向 `/api/v2/users/status` 发送一次健康检查，确认 token 仍然有效
- 过期时返回 HTTP 401，提示用户刷新

*The user extracts Qwen's JWT token from the browser and configures it. The proxy decodes it to read `exp`, then validates remotely every 5 minutes against `/api/v2/users/status`.*

**2. 请求转译 / Request Translation**

Incoming (OpenAI 格式):
```json
{
  "model": "qwen3.6-plus",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

Outgoing (Qwen 内部格式):
```json
{
  "chat_id": "<uuid-from-create-chat>",
  "parent_id": "<previous-message-id>",
  "model": "qwen3.6-plus",
  "system_message": "You are helpful",
  "messages": [{
    "fid": "<generated-uuid>",
    "parentId": "<previous-message-id>",
    "role": "user",
    "content": "Hello",
    "childrenIds": ["<assistant-fid>"],
    "feature_config": {"thinking_enabled": false, "output_schema": "phase"}
  }],
  "stream": true,
  "version": "2.1",
  "incremental_output": true
}
```

关键转译点：
- `messages` 数组 → 提取 `system` 角色放到顶层 `system_message`，其余只保留最后一个 `user` 消息
- 无状态 API → 维持 `chat_id` / `parent_id` 状态机，模拟有状态会话
- SSE chunks → 提取 `delta.content`，重新包装为 OpenAI 格式的 chunk

*Key translation points: system message extracted to top-level field; chat_id/parent_id state machine maintained for conversation continuity; SSE chunks reformatted to OpenAI delta format.*

**3. 响应转译 / Response Translation**

Qwen 的 SSE 流返回两种特殊 chunk：
- `response.created`: 元数据 chunk，包含新的 `parent_id`（用于下一轮对话），不包含文本内容
- 普通 chunk: `choices[0].delta.content` 包含增量文本

代理会识别 `response.created`，将其中的 `parent_id` 存储下来用于后续请求，然后丢弃；对于普通 chunk，提取 `content` 并包装为 OpenAI 格式 `{"choices":[{"delta":{"content":"..."}}]}`。

*The proxy identifies `response.created` meta-chunks (storing the `parent_id` for the next turn), and wraps regular content chunks into OpenAI-compatible `delta` format.*

**4. 会话管理 / Conversation Management**

Qwen 的 API 是有状态的：需要先 `POST /api/v2/chats/new` 创建 chat，后续消息通过 `chat_id` + `parent_id` 链式串联。代理通过以下机制透明地管理这个状态：

- 客户端传可选 header `x-conversation-id`
- 首次请求时，代理自动创建新 chat（`create_chat`），后续请求复用 `chat_id` 和更新 `parent_id`
- 1 小时无活动的会话自动清理
- 支持将 chat 关联到 Qwen Project（用于组织记忆隔离）

*Qwen's API is stateful (chat_id + parent_id chain). The proxy transparently manages this: auto-creates chats on first request, reuses them on subsequent requests via the `x-conversation-id` header, and cleans up stale conversations after 1 hour.*

---

## 实现原理 / Implementation

### 技术栈 / Tech Stack

| 组件 | 选型 | 原因 |
|------|------|------|
| Web 框架 | FastAPI | 原生 async，SSE 支持好，自动 OpenAPI 文档 |
| HTTP 客户端 | httpx | HTTP/2 支持，async，SSE 逐行读取 |
| Token 验证 | PyJWT | 仅解码不验证签名，读取 `exp` |
| 部署 | systemd + uvicorn | 进程守护，自动重启 |

### 核心模块 / Core Modules

```
qwen-proxy/
├── server.py          # FastAPI 应用 + 路由（核心）
├── session.py         # Token 管理、JWT 解码、远程验证
├── qwen_client.py     # Qwen API 客户端（create_chat, send_message, list_models）
├── ask_qwen.py        # CLI 协作工具
├── start.sh           # 一键启动脚本
└── qwen_config.json   # 运行时配置（token, model, project_id）
```

#### `server.py` — 协议转译层

实现了完整的 OpenAI `/v1/chat/completions` 兼容接口，包括：
- **Streaming**: SSE 逐块转译并实时推送
- **Non-streaming**: 收集所有 chunks 后一次性返回
- **`/v1/models`**: 将 Qwen 的内部 models 列表转为 OpenAI 格式
- **Admin Dashboard**: 内嵌 HTML/CSS/JS 的 Web 管理面板（单文件，零依赖）

*Implements full OpenAI-compatible `/v1/chat/completions` including streaming (SSE chunks translated on-the-fly), non-streaming (accumulated), `/v1/models` listing, and an embedded admin dashboard (single-file, no external assets).*

#### `session.py` — 会话配置管理

- 从 `qwen_config.json` 加载 token、默认模型、project_id
- JWT 解码读取 `exp`，显示剩余有效时间
- 远程验证：向 `/api/v2/users/status` 发送请求，5 分钟内缓存结果
- 支持运行时通过 API 更新配置（`PUT /token`, `PUT /default-model`, `PUT /project-id`）

*Loads token/model/project from config file; JWT expiration check via decode; remote validation against `/api/v2/users/status` with 5-min cache; runtime config update via API.*

#### `qwen_client.py` — Qwen API 客户端

封装了 Qwen 内部 API 的三个核心操作：

1. **`create_chat(model)`** → `POST /api/v2/chats/new`
   创建新会话，返回 `chat_id`。新 chat 的 `parent_id` 为 null。

2. **`send_message(chat_id, parent_id, content, model, system, stream)`** → `POST /api/v2/chat/completions?chat_id=...`
   发送消息并以 AsyncGenerator 逐块 yield SSE chunks。第一条 chunk 可能是 `response.created`（meta），后续是内容 chunk。

3. **`list_models()`** → `GET /api/v2/models`
   返回可用模型列表。

辅助函数 `extract_content()`, `extract_parent_id()`, `extract_final_usage()` 负责从 Qwen 的 chunk 结构中提取信息。

*Encapsulates three core Qwen API operations: create_chat, send_message (SSE streaming as AsyncGenerator), and list_models. Helper functions extract content delta, parent_id tracking, and token usage from Qwen's chunk format.*

#### `ask_qwen.py` — CLI 协作工具

见下方 [CLI 协作工具](#cli-协作工具--cli-collaboration-tool) 章节。

### 代理配置 / Proxy Configuration

Qwen API 可能需要科学上网。`qwen_client.py` 和 `session.py` 均通过环境变量 `https_proxy` / `HTTPS_PROXY` 自动配置代理：

```python
# 从环境变量读取代理配置
proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
# 跳过 socks:// （httpx 不支持）
```

systemd unit 文件中已预设了通过 clash-verge (`127.0.0.1:7897`) 的 HTTP 代理。

---

## 快速开始 / Quick Start

### 环境要求 / Prerequisites

- Python 3.10+
- 一个有效的 `chat.qwen.ai` JWT token（从浏览器获取）

### 安装 / Installation

```bash
git clone https://github.com/mengxinghun9657-boop/qwen-proxy.git
cd qwen-proxy

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 配置 Token

从浏览器获取 token:

1. 打开 https://chat.qwen.ai 并登录
2. F12 → Application → Local Storage → `token`
3. 复制完整的 JWT 字符串

然后二选一：

**方式 A — Web 管理面板（推荐）**

```bash
# 先启动服务（此时未配置 token）
./start.sh
# 浏览器打开 http://127.0.0.1:8800/admin
# 在 "Token 配置" 区域粘贴并保存
```

**方式 B — 直接编辑配置文件**

```bash
cp qwen_config.example.json qwen_config.json
# 编辑 qwen_config.json，填入 token
```

### 运行 / Run

```bash
# 开发 / 调试模式
./start.sh

# 或直接使用 uvicorn
uvicorn server:app --host 127.0.0.1 --port 8800 --reload
```

### 验证 / Verify

```bash
# 健康检查
curl http://127.0.0.1:8800/health

# 列出可用模型
curl http://127.0.0.1:8800/v1/models

# 发送一条消息
curl -X POST http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-plus",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## API 参考 / API Reference

### `POST /v1/chat/completions`

OpenAI 兼容的聊天补全接口。

**Request:**
```json
{
  "model": "qwen3.6-plus",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": true
}
```

**Headers:**
| Header | 必需 | 说明 |
|--------|------|------|
| `x-conversation-id` | 否 | 多轮会话 ID。不传则自动创建新会话，响应中返回 |

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1715414400,
  "model": "qwen3.6-plus",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
}
```

**Response (streaming):** 标准 SSE 流，以 `data: [DONE]` 结束。

---

### 其他端点 / Other Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/health` | Token 状态 + JWT 有效期 |
| `GET` | `/admin` | Web 管理面板 |
| `PUT` | `/token` | 更新 token `{"token":"..."}` |
| `GET` | `/token` | 获取当前 token |
| `PUT` | `/default-model` | 设置默认模型 `{"model":"..."}` |
| `GET` | `/default-model` | 获取默认模型 |
| `GET` | `/projects` | 列出 Qwen 项目 |
| `PUT` | `/project-id` | 设置默认项目 `{"project_id":"..."}` |

---

## CLI 协作工具 / CLI Collaboration Tool

`ask_qwen.py` 是一个轻量级的命令行工具，用于在终端中直接向 Qwen 提问，特别适合作为 AI 编程助手的"副驾驶"——主模型（如 Claude Code）可以将复杂问题发给 Qwen 获得第二意见。

*`ask_qwen.py` is a lightweight CLI to query Qwen directly from the terminal. Its primary use case is as a "co-pilot" for AI coding assistants — the main model delegates complex questions to Qwen for a second opinion.*

### 压缩模式 / Compression Modes

多模型协作的核心风险是**上下文膨胀**——Qwen 返回 500 字但只有 50 字有用。因此 `-M` 参数是**强制推荐**的，它会在系统提示词中注入压缩指令，让 Qwen 输出紧凑、结构化的回复。

*The core risk of multi-model collaboration is context bloat — Qwen returns 500 words but only 50 are useful. The `-M` flag is strongly recommended: it injects compression directives into the system prompt for compact, structured output.*

| 模式 / Mode | 适用场景 / Use Case | 示例 / Example |
|-------------|---------------------|----------------|
| `concise` | 快速事实确认 / Quick fact check | `-M concise "什么是 X?"` |
| `diagnose` | Bug / 报错排查 / Root cause analysis | `-M diagnose "为什么崩溃？"` |
| `review` | 代码审查 / Code review | `-M review "$(cat code.go)"` |
| `keypoints` | 日志/文档摘要 / Summarization | `-M keypoints "总结这段日志"` |
| `judge` | 二元决策 / Binary decisions | `-M judge "用 Redis 还是 Kafka？"` |
| `json` | 需要可解析输出 / Parseable output | `-M json "分析这个 schema"` |

**压缩效果对比 / Before vs After:**

```
# 无压缩 — Qwen 可能输出 300+ 字，包含寒暄、免责声明、重复
./venv/bin/python ask_qwen.py "为什么 Flask 高并发下 500 错误？"

# diagnose 模式 — 固定输出格式，<100 字，只有根因+修复+备选
./venv/bin/python ask_qwen.py -M diagnose "为什么 Flask 高并发下 500 错误？"
# ROOT CAUSE: ...
# FIX: ...
# ALT: ...
```

### 用法 / Usage

```bash
# 基本提问
./venv/bin/python ask_qwen.py "什么是尾递归优化？"

# 压缩模式（推荐）
./venv/bin/python ask_qwen.py -M concise "快速回答：尾递归优化是什么？"

# 诊断问题
./venv/bin/python ask_qwen.py -M diagnose "NullPointerException at line 42"

# 代码审查
cat buggy_code.py | ./venv/bin/python ask_qwen.py -M review

# 二元决策
./venv/bin/python ask_qwen.py -M judge "这个场景该用 Postgres 还是 Mongo？"

# 多轮对话（调试时很有用）
./venv/bin/python ask_qwen.py -c debug-session -M diagnose "这个 NPE 是什么原因？"
./venv/bin/python ask_qwen.py -c debug-session -M diagnose "你建议怎么修？"

# 字数限制
./venv/bin/python ask_qwen.py -w 50 "一句话解释 Docker"

# 列出可用模型
./venv/bin/python ask_qwen.py --list-models
```

### 参数 / Options

| 参数 | 简写 | 说明 |
|------|------|------|
| `prompt` | (位置参数) | 问题文本，支持管道输入 |
| `--mode` | `-M` | **压缩预设**（concise/diagnose/review/keypoints/judge/json） |
| `--max-words` | `-w` | 限制回复最大字数 |
| `--system` | `-s` | 系统提示词，定义 AI 角色 |
| `--model` | `-m` | 模型名称，默认 `qwen3.6-plus` |
| `--conversation` | `-c` | 会话 ID，用于多轮对话 |
| `--stream` | | 启用流式输出 |
| `--list-models` | | 列出所有可用模型 |
| `--list-modes` | | 列出压缩模式说明 |
| `--raw` | | 输出原始 JSON 响应 |

### 设计理念 / Design Philosophy

`ask_qwen.py` 的设计原则：
- **简单**: 单文件，零依赖（仅标准库 + httpx）
- **管道友好**: 支持 stdin/stdout，完美融入 Unix 管道
- **输出压缩优先**: `-M` 是核心特性，不是可选附加。协作场景下紧凑输出是必需品
- **会话持久化**: 通过 `-c` 参数自动保存/恢复会话状态

*Design principles: single-file with minimal dependencies; pipe-friendly (stdin/stdout); **compression-first** — `-M` is a core feature, not optional; session persistence via `-c`.*

---

## 部署 / Deployment

### systemd（推荐 / Recommended）

```bash
# 安装
sudo cp qwen-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-proxy

# 日常管理
systemctl status qwen-proxy
journalctl -u qwen-proxy -f     # 查看日志
sudo systemctl restart qwen-proxy
```

**systemd unit 关键配置说明：**

```
Environment="ALL_PROXY="          # 清除 socks:// 代理（httpx 不支持）
Environment="https_proxy=http://127.0.0.1:7897"  # HTTP 代理（clash-verge）
Environment="HTTP_PROXY=http://127.0.0.1:7897"
```

如果你的代理地址不同，修改这三行即可。

### Docker（可选）

当前未提供官方 Dockerfile，原因是：
- 服务仅 4 个 Python 依赖，资源占用 <70MB
- 必须使用 `network_mode: host`（访问宿主机代理 127.0.0.1:7897）
- systemd 部署已足够轻量和稳定

如需 Docker 化，参考 `start.sh` 和 systemd unit 文件自行编写即可。

---

## 安全 / Security

### Token 保护

- `qwen_config.json` 已加入 `.gitignore`，不会被提交到 Git
- Token 仅存储在服务器本地文件系统
- 代理仅监听 `127.0.0.1`，不对外暴露
- Token 通过 JWT 过期时间自动监控，过期前会有明确提示

### 最佳实践 / Best Practices

1. **不要**在公网暴露此服务（始终监听 127.0.0.1）
2. **定期**检查 `/health` 端点确认 token 有效期
3. **及时**从浏览器刷新 token（过期后 PUT `/token` 更新）
4. **使用** Qwen Project 功能隔离不同工作场景的会话记忆

---

## License

MIT

---

## 相关链接 / Links

- [chat.qwen.ai](https://chat.qwen.ai) — Qwen 官方聊天应用
- [Qwen 模型家族](https://github.com/QwenLM/Qwen) — 开源模型
- [OpenAI API 协议](https://platform.openai.com/docs/api-reference/chat) — 兼容参考

# Qwen Proxy

OpenAI-compatible reverse proxy for [chat.qwen.ai](https://chat.qwen.ai), plus a CLI tool for AI-assisted collaboration.

## Architecture

```
Claude Code / OpenAI Client
        │
        ▼
  localhost:8800 (FastAPI)  ◀── this project
        │
        ▼
  chat.qwen.ai/api/v2       ◀── upstream (Bearer token auth)
```

## Quick Start

```bash
# 1. Setup venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure token
#    Open chat.qwen.ai in browser → DevTools → Application → Local Storage → token
#    Then either:
#    a) PUT /token via the admin panel at http://127.0.0.1:8800/admin
#    b) Edit qwen_config.json directly:
cp qwen_config.example.json qwen_config.json
#    Fill in your JWT token

# 3. Run
./start.sh
# or: uvicorn server:app --host 127.0.0.1 --port 8800
```

## Systemd (recommended for servers)

```bash
sudo cp qwen-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-proxy
```

The service file clears `ALL_PROXY` (httpx doesn't support socks://) and routes through `http://127.0.0.1:7897` if you use clash-verge.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `GET /v1/models` | List available models |
| `GET /admin` | Web dashboard (token, models, projects) |
| `GET /health` | Token status check |
| `PUT /token` | Update bearer token |
| `PUT /default-model` | Set default model |
| `GET /projects` | List Qwen projects |
| `PUT /project-id` | Set project for new chats |

## CLI Tool: ask_qwen.py

```bash
./venv/bin/python ask_qwen.py "your question"
echo "code review" | ./venv/bin/python ask_qwen.py -s "You are a reviewer"
./venv/bin/python ask_qwen.py -c session-1 "follow-up question"
./venv/bin/python ask_qwen.py -m qwen3.6-max-preview "complex task"
./venv/bin/python ask_qwen.py --list-models
```

Options:

| Flag | Description |
|------|-------------|
| `-s, --system` | System prompt |
| `-m, --model` | Model name (default: qwen3.6-plus) |
| `-c, --conversation` | Multi-turn conversation ID |
| `--stream` | Stream output as it arrives |
| `--list-models` | List available models |

## Available Models

`qwen3.6-plus`, `qwen3.6-max-preview`, `qwen3-coder-plus`, `qwen3.5-flash`, and more — see `GET /v1/models`.

## Token

The token is a JWT from `chat.qwen.ai` stored in browser Local Storage. It expires periodically — refresh it via the admin dashboard at `/admin`.

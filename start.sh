#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
# 清除 httpx 不支持的 socks:// 代理
unset ALL_PROXY all_proxy
exec uvicorn server:app --host 127.0.0.1 --port 8800 "$@"

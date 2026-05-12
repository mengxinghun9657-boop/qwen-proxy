import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import jwt
import httpx

CONFIG_PATH = Path(__file__).parent / "qwen_config.json"
QWEN_BASE = "https://chat.qwen.ai"
CHECK_INTERVAL = 300  # re-validate every 5 minutes


def _http_client(**kwargs) -> httpx.AsyncClient:
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or \
            os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    kwargs.setdefault("http2", True)
    if proxy:
        return httpx.AsyncClient(proxy=proxy, **kwargs)
    return httpx.AsyncClient(**kwargs)


class TokenStatus:
    VALID = "valid"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    ERROR = "error"


class SessionManager:
    def __init__(self):
        self._last_check: float = 0
        self._cached_status: str = TokenStatus.UNKNOWN
        self._cached_msg: str = ""

    # ---- config file ----

    def load_token(self) -> str | None:
        if not CONFIG_PATH.exists():
            return None
        try:
            data = json.loads(CONFIG_PATH.read_text())
            return data.get("token")
        except (json.JSONDecodeError, KeyError):
            return None

    def _read_config(self) -> dict:
        if not CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            return {}

    def _write_config(self, updates: dict):
        cfg = self._read_config()
        cfg.update(updates)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    def save_token(self, token: str):
        self._write_config({
            "token": token,
            "token_updated_at": datetime.now(timezone.utc).isoformat(),
            "notes": "From chat.qwen.ai browser DevTools -> Application -> Local Storage -> token",
        })

    def load_default_model(self) -> str:
        return self._read_config().get("default_model", "qwen-max-latest")

    def save_default_model(self, model: str):
        self._write_config({"default_model": model})

    def load_project_id(self) -> str | None:
        return self._read_config().get("project_id")

    def save_project_id(self, project_id: str | None):
        self._write_config({"project_id": project_id})

    # ---- JWT decode ----

    def decode_jwt(self, token: str) -> dict:
        """Decode without verification to read exp."""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return {}

    def jwt_expiry(self, token: str) -> float | None:
        payload = self.decode_jwt(token)
        exp = payload.get("exp")
        if exp:
            return float(exp)
        return None

    # ---- remote validation ----

    async def validate_remote(self, token: str) -> tuple[str, str]:
        """Call /api/v2/users/status, return (status, message)."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "https://chat.qwen.ai",
            "Referer": "https://chat.qwen.ai/",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            async with _http_client(timeout=15) as client:
                resp = await client.get(
                    f"{QWEN_BASE}/api/v2/users/status",
                    headers=headers,
                )
                if resp.status_code == 401 or resp.status_code == 403:
                    return TokenStatus.EXPIRED, f"HTTP {resp.status_code}"
                data = resp.json()
                if data.get("success"):
                    return TokenStatus.VALID, "ok"
                return TokenStatus.EXPIRED, data.get("message", "unknown error")
        except httpx.RequestError as e:
            return TokenStatus.ERROR, str(e)

    # ---- health ----

    def _load_updated_at(self) -> str | None:
        if not CONFIG_PATH.exists():
            return None
        try:
            data = json.loads(CONFIG_PATH.read_text())
            return data.get("token_updated_at")
        except (json.JSONDecodeError, KeyError):
            return None

    async def health(self) -> dict:
        token = self.load_token()
        if not token:
            return {
                "status": "no_token",
                "message": "未配置 token。从浏览器 Local Storage 复制后 PUT /token",
                "qwen_models_endpoint": f"{QWEN_BASE}/api/v2/models",
                "token_updated_at": self._load_updated_at(),
            }

        # check JWT expiry
        exp_ts = self.jwt_expiry(token)
        now = time.time()
        jwt_msg = None
        if exp_ts:
            if now > exp_ts:
                jwt_msg = f"JWT 已过期 ({datetime.fromtimestamp(exp_ts).isoformat()})"
            else:
                remaining = exp_ts - now
                jwt_msg = f"JWT 有效，剩余 {remaining/3600:.1f} 小时"

        # remote check (throttled)
        status, msg = TokenStatus.UNKNOWN, "not checked"
        if now - self._last_check > CHECK_INTERVAL:
            status, msg = await self.validate_remote(token)
            self._last_check = now
            self._cached_status = status
            self._cached_msg = msg
        else:
            status = self._cached_status
            msg = self._cached_msg

        return {
            "status": status,
            "message": msg,
            "jwt_info": jwt_msg,
            "token_preview": token[:20] + "..." if len(token) > 20 else token,
            "token_updated_at": self._load_updated_at(),
        }

    def force_recheck(self):
        self._last_check = 0

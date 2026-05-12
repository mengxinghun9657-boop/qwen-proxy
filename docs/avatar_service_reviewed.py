"""
Avatar upload service for a social platform.
Handles ~50k uploads/day, deployed with 4 gunicorn workers behind nginx.

Recent oncall issues:
- Occasional 502/504 under peak load (~200 concurrent uploads)
- Some users report avatars not updating after successful upload
"""
import os
import uuid
import hashlib
import logging
from dataclasses import dataclass, field

from flask import Flask, request, jsonify, g
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/data/avatars")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://app:secret@db.internal:5432/avatars")
    CDN_PURGE_URL: str = os.getenv("CDN_PURGE_URL", "https://cdn.internal.example.com/purge")
    CDN_BASE_URL: str = os.getenv("CDN_BASE_URL", "https://cdn.example.com")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024   # 10MB
    ALLOWED_CONTENT_TYPES: set = field(default_factory=lambda: {
        "image/png", "image/jpeg", "image/gif", "image/webp"
    })
    PURGE_TIMEOUT: float = 1.5

config = Config()

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
engine = create_engine(
    config.DATABASE_URL,
    pool_size=8,
    max_overflow=4,
    pool_pre_ping=True,
    pool_recycle=3600,
)
logger = logging.getLogger("avatar.service")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------
@app.before_request
def authenticate():
    if request.endpoint in ("health",):
        return None

    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        return jsonify({"error": "unauthorized"}), 401

    try:
        with Session(engine) as db:
            row = db.execute(
                text("SELECT id, status FROM users WHERE api_token = :token"),
                {"token": token},
            ).fetchone()
    except Exception:
        logger.exception("Auth DB lookup failed")
        return jsonify({"error": "internal error"}), 502

    if row is None:
        return jsonify({"error": "invalid token"}), 401
    if row[1] != "active":
        return jsonify({"error": "account disabled"}), 403

    g.user_id = str(row[0])
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_hash(filepath: str) -> str:
    """SHA-256 hash of saved file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def purge_cdn(pattern: str) -> bool:
    """Invalidate CDN cache for a URL pattern. Returns True on success."""
    try:
        resp = requests.post(
            config.CDN_PURGE_URL,
            json={"paths": [pattern]},
            timeout=config.PURGE_TIMEOUT,
        )
        if resp.status_code == 200:
            return True
        logger.warning(f"CDN purge returned {resp.status_code}: {pattern}")
        return False
    except requests.Timeout:
        logger.warning(f"CDN purge timed out: {pattern}")
        return False
    except requests.RequestException as e:
        logger.warning(f"CDN purge failed: {pattern} err={e}")
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_avatar():
    # --- validate file presence ---
    if "file" not in request.files:
        return jsonify({"error": "no file attached"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400

    # --- validate content type ---
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        return jsonify({"error": f"unsupported type: {file.content_type}"}), 400

    # --- read & validate size ---
    file_data = file.read()
    if len(file_data) > config.MAX_FILE_SIZE:
        return jsonify({"error": "file too large"}), 413
    if not file_data:
        return jsonify({"error": "empty file"}), 400

    # --- build paths ---
    user_id = g.user_id
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "bin"
    safe_uid = secure_filename(user_id)
    filename = f"{uuid.uuid4().hex}.{ext}"
    user_dir = os.path.join(config.UPLOAD_DIR, safe_uid)
    filepath = os.path.join(user_dir, filename)
    cdn_url = f"{config.CDN_BASE_URL}/avatars/{safe_uid}/{filename}"

    # --- persist to disk ---
    try:
        os.makedirs(user_dir, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(file_data)
    except OSError:
        logger.exception(f"Failed to write file: {filepath}")
        return jsonify({"error": "storage error"}), 500

    # --- verify integrity ---
    if os.path.getsize(filepath) != len(file_data):
        logger.error(f"Size mismatch after write: {filepath}")
        try:
            os.remove(filepath)
        except OSError:
            pass
        return jsonify({"error": "write verification failed"}), 500

    file_hash = compute_hash(filepath)

    # --- update database ---
    try:
        with Session(engine) as db:
            db.execute(
                text(
                    "UPDATE users SET avatar_url = :url, avatar_hash = :hash, "
                    "updated_at = NOW() WHERE id = :uid"
                ),
                {"url": cdn_url, "hash": file_hash, "uid": user_id},
            )
            db.commit()
    except Exception:
        logger.exception(f"DB update failed for user={safe_uid}")
        # File written but DB not updated — orphan file on disk.
        # On next upload, old file will remain and consume space indefinitely.
        return jsonify({"error": "internal error"}), 502

    # --- invalidate CDN cache ---
    # Wildcard purge for this user's avatar directory.
    # NOTE: purging after DB update means a small window where DB points to
    # the new URL but CDN may still serve the old (cached) URL to clients.
    purge_cdn(f"/avatars/{safe_uid}/*")

    logger.info(
        "Avatar uploaded: user=%s file=%s size=%d hash=%s",
        safe_uid, filename, len(file_data), file_hash[:12],
    )

    return jsonify({"url": cdn_url, "hash": file_hash}), 201


@app.route("/health")
def health():
    checks = {"db": False, "disk": False}

    try:
        with Session(engine) as db:
            db.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception:
        pass

    try:
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        checks["disk"] = True
    except OSError:
        pass

    all_ok = all(checks.values())
    return jsonify({"status": "ok" if all_ok else "degraded", "checks": checks}), \
        200 if all_ok else 503

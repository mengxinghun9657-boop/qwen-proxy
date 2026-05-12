# 我把 Qwen 变成 Claude Code 的副驾驶，一次代码审查多发现了2个隐藏 Bug

> 真实的多模型协作实战：不是"哪个模型更强"，而是"它们如何互补"。

---

## 背景

上周在 review 团队一个头像上传服务的代码时，我决定做个实验。

这个服务每天处理约 5 万次上传，4 个 gunicorn worker 部署在 nginx 后面。最近 on-call 收到两个投诉：

- **峰值流量下偶发 502/504**（~200 并发上传时）
- **部分用户上传新头像后不生效**，页面显示的仍是旧图

代码不长，200 行。维护者是个称职的后端——有 auth 中间件、有健康检查、有错误处理、有日志。第一眼扫过去，代码质量不差。

我做了两轮审查：

1. **第 1 轮**：Claude Code 独立审查
2. **第 2 轮**：Claude Code 调用 Qwen（qwen3.6-plus）作为副驾驶协同审查

两轮结果对比，看协同是否真的有用。

---

## 被测代码（节选）

为了可读性省略了 config dataclass 和 helper 函数，[完整代码在 GitHub](https://github.com/mengxinghun9657-boop/qwen-proxy)。

```python
"""
Avatar upload service. Handles ~50k uploads/day,
deployed with 4 gunicorn workers behind nginx.
"""
engine = create_engine(
    config.DATABASE_URL,
    pool_size=8, max_overflow=4,
    pool_pre_ping=True, pool_recycle=3600,
)

@app.before_request
def authenticate():
    """Validate Bearer token, attach user_id to g."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    with Session(engine) as db:
        row = db.execute(
            text("SELECT id, status FROM users WHERE api_token = :token"),
            {"token": token},
        ).fetchone()
    if row is None:
        return jsonify({"error": "invalid token"}), 401
    g.user_id = str(row[0])

@app.route("/upload", methods=["POST"])
def upload_avatar():
    file = request.files["file"]

    # Validate content-type from client (not magic bytes)
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        return jsonify({"error": "unsupported type"}), 400

    # Read entire file into memory for validation
    file_data = file.read()
    if len(file_data) > config.MAX_FILE_SIZE:
        return jsonify({"error": "file too large"}), 413

    # Build paths & persist
    safe_uid = secure_filename(g.user_id)
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(config.UPLOAD_DIR, safe_uid, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(file_data)

    # Verify, hash, update DB
    file_hash = compute_hash(filepath)
    with Session(engine) as db:
        db.execute(text("UPDATE users SET avatar_url = :url, ..."), {...})
        db.commit()

    # Fire CDN purge and forget about the result
    purge_cdn(f"/avatars/{safe_uid}/*")

    return jsonify({"url": cdn_url, "hash": file_hash}), 201
```

---

## 第 1 轮：Claude Code 独立审查

Claude 独立阅读了完整代码后，找到了 **7 个问题**：

| # | 发现 | 与线上症状的关联 |
|---|------|------------------|
| 1 | **DB 连接池配置偏小** — `pool_size=8` 对 4 worker + 每请求 2 session（auth + upload），峰值时连接不够 | → 502/504 |
| 2 | **`file.read()` 全量读入内存** — 10MB 上限 × 多并发 = 内存压力和 GC 抖动 | → 502 |
| 3 | **CDN purge 返回值被忽略** — `purge_cdn()` 返回 `bool` 但调用处不检查，purge 失败用户也收到 201 | → 头像不更新 |
| 4 | **文件写入与 DB 更新无事务边界** — 文件落盘后 DB 更新失败时，产生孤儿文件 | → 磁盘泄漏 |
| 5 | **Content-Type 可伪造** — `file.content_type` 来自客户端请求头，无扩展名/magic bytes 校验做纵深防御 | → 安全隐患 |
| 6 | **健康检查有副作用** — `os.makedirs(upload_dir)` 在只读检查中修改文件系统 | → 轻微 |
| 7 | **并发上传同一用户时文件竞态** — 两个请求各写一个文件，最后一个写 DB 的生效，另一个变孤儿 | → 磁盘泄漏 |

其中 #1 和 #2 指向 502/504，#3 指向头像不更新。分析方向是对的，但我隐约觉得 504 的解释还不够彻底——连接池偏小和内存压力是叠加因素，但不是**根本瓶颈**。

---

## 第 2 轮：Claude + Qwen 协同审查

这次我调用 Qwen 作为副驾驶，使用压缩模式 `-M review` + 系统提示词定向到 SRE 视角：

```bash
./venv/bin/python ask_qwen.py \
  -M review \
  -s "You are a senior SRE reviewing production Python code.
      Focus on issues that explain the reported symptoms:
      occasional 502/504 under peak load (200 concurrent uploads),
      and avatars not updating after successful upload.
      The service runs with 4 gunicorn workers behind nginx." \
  "$(cat avatar_service.py)"
```

`-M review` 自动注入输出约束：按 Critical/Warnings/Summary 三段输出，每段最多 3 条，总字数 ≤200。

Qwen 的完整回复：

```
## Critical
- **DB Connection Pool Exhaustion** (severity: H): pool_size=8 with 4
  workers allows only 2 concurrent DB connections per worker. Auth +
  Upload queries block under 200 concurrent requests, causing connection
  timeouts and 502s.
- **Synchronous CDN Purge in Request Path** (severity: H): Blocking
  requests.post (up to 1.5s) during upload drastically reduces worker
  throughput. Under load, all 4 workers stall waiting for CDN, triggering
  nginx 504 gateways.
- **Stale Avatar Reads (Race Condition)** (severity: M): Purging happens
  *after* DB commit. Without cache-busting query params on the URL,
  clients/CDN may serve stale cached assets despite the DB update,
  causing "avatar not updating" symptoms.

## Warnings
- **Memory Spike from file.read()** (severity: M): Loading entire 10MB
  files into memory before writing causes high RAM usage and GC pressure
  under concurrency, risking OOM or swap thrashing.
- **Orphaned Files on DB Failure** (severity: L): If DB update fails,
  the uploaded file remains on disk. Over time, this leaks storage.

## Summary
Increase DB pool size, offload CDN purging to a background task/queue,
and add cache-busting parameters to avatar URLs to resolve latency
spikes and consistency issues.
```

Qwen 额外发现了 **2 个 Claude 漏掉的问题**：

| # | 发现 | 为什么 Claude 漏了 |
|---|------|---------------------|
| 8 | **CDN purge 同步阻塞请求路径** — `requests.post` 在请求线程中等 1.5s，4 个 worker 全被 CDN 响应阻塞时，nginx 收不到回复 → 504 | 注意力放在了代码正确性上，忽略了吞吐量瓶颈——**阻塞点在哪里比阻塞什么更重要** |
| 9 | **URL 缺少 cache-busting 参数** — 即使 purge 成功，CDN 不同边缘节点的缓存时间差也可能导致部分用户看到旧内容 | 对 CDN 缓存传播机制的细节不够敏感 |

同时，对于 #3（CDN purge 失败），Qwen 的分析比 Claude 更深——不只说"返回值被忽略"，而是指出**即使 purge 成功**，没有 cache-busting 参数仍可能导致旧内容被边缘节点缓存。

---

## 完整对比

| 指标 | Claude Solo | Claude + Qwen | 提升 |
|------|:-----------:|:-------------:|:----:|
| 发现问题数 | 7 | **9** | **+29%** |
| 504 根因覆盖 | 连接池 + 内存 | 连接池 + 内存 **+ 同步阻塞** | 更完整 |
| 头像不更新根因 | CDN purge 失败 | CDN purge 失败 **+ 缓存传播延迟** | 更完整 |
| 噪音 | 0 | 0（压缩模式有效） | — |

数字上的提升（29%）不如第一次测试（75%）夸张，但**这次更有说服力**——因为代码质量更高，bug 更隐蔽，发现的新问题更关键。

核心差异在于两个模型的注意力分布：

- **Claude** 的注意力在代码结构和逻辑正确性上（连接池配置、内存使用、事务边界、安全校验）
- **Qwen** 的注意力在**运行时行为和系统交互**上（请求路径中的阻塞点、CDN 缓存传播的时间窗口）

这正是多模型协作的价值场景：**不是找一个更强的模型替代当前模型，而是让不同注意力分布的模型互补。**

---

## 为什么 Qwen 的输出没有干扰到 Claude

多模型协作有一个真实风险——副驾驶的输出如果冗长散乱，会变成上下文垃圾，稀释主模型的推理质量。

这次使用的 `-M review` 模式下，Qwen 的回复被约束在 5 条发现 + 一句话总结，总字数约 150 词。没有寒暄、没有免责声明、没有"希望这对你有帮助"。信噪比接近 100%。

这是 [qwen-proxy](https://github.com/mengxinghun9657-boop/qwen-proxy) 项目内置的 6 种压缩预设之一：

| 模式 | 场景 | 输出约束 |
|------|------|----------|
| `review` | 代码审查 | 按严重度三段式，200 词 |
| `diagnose` | Bug 排查 | 根因+修复+备选，100 词 |
| `concise` | 快速确认 | 结论先行，150 词 |
| `judge` | 二元决策 | YES/NO + 置信度 + 原因 |
| `keypoints` | 摘要 | 3-5 要点，无语境填充 |
| `json` | 可解析输出 | 纯 JSON，无 markdown |

压缩的本质是**在发送前就把问题说清楚想要什么样的回答**——不是后处理裁剪，而是预约束格式。

---

## 协作流程

```
Claude Code                        Qwen (via proxy)
     │                                   │
     │  审查代码，发现 7 个问题            │
     │  觉得 504 的解释不够透彻           │
     │                                   │
     ├── ask_qwen.py -M review ────────→ │
     │   注入 SRE 系统提示词              │
     │   注入输出格式约束                  │
     │                                   │
     │←── 5 条结构化发现 ────────────────┤
     │   发现 CDN 同步阻塞 (#8)           │
     │   发现缓存传播窗口 (#9)            │
     │                                   │
     ├── 合并：7 + 2 = 9 个问题          │
     ├── 504 根因从 2 个升级到 3 个      │
     └── 输出最终报告                     │
```

---

## 限制与坦诚

1. **不是自动的** — Claude 根据预定义的决策准则自行判断何时调 Qwen。把 Qwen 当成一个可以咨询的同事，而不是一个自动触发的外挂
2. **每次调用 ~10-30 秒延迟** — 简单问题不划算，内置规则定义了不受益于第二意见的场景
3. **依赖 chat.qwen.ai 的 JWT token** — 需要从浏览器提取。对于会用 CLI / Git 的目标用户，这不算门槛
4. **Qwen 回应的质量不总是稳定** — 有时会重复 Claude 已有的发现而不增加新值，这种情况就丢弃不引用

---

## 不只是 Qwen

这套"主模型 + 压缩副驾驶"的模式不限于 Qwen。任何兼容 OpenAI API 的后端都可以接入——Ollama 本地模型、DeepSeek、Gemini。你可以根据任务类型路由到不同的副驾驶。

**多模型协作的挑战不是"找最强的模型"，而是"让它们互补而不互扰"。** 压缩输出只是其中一个解，欢迎讨论更好的方案。

---

*2026 年 5 月 · [GitHub](https://github.com/mengxinghun9657-boop/qwen-proxy) · MIT*

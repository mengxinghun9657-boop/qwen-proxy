# 我把 Qwen 变成 Claude Code 的副驾驶，Bug 发现率提升了 75%

> 一个真实的多模型协作实战测试：独立分析 vs 协作分析，用数据说话。

---

## 问题场景

团队维护一个用户头像上传服务，每天处理 5 万次上传。运行了 8 个月，最近 on-call 收到两个投诉：

1. **峰值流量下偶发 500 错误**
2. **部分用户上传新头像后不生效，看到的还是旧图**

代码不到 100 行，看起来没什么问题。我们来做一个实验：

- **第 1 轮**：Claude Code 独立审查代码
- **第 2 轮**：Claude Code 调用 Qwen 作为副驾驶协同审查

比较两轮的结果，看多模型协作到底能不能提升代码审查质量。

---

## 被测代码

```python
"""
User avatar upload service for a social platform.
Handles ~50k uploads/day, stores to local disk + CDN invalidation.
"""
import os, uuid, hashlib, shutil
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_DIR = "/data/avatars"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_SIZE = 10 * 1024 * 1024
CDN_PURGE_URL = "https://cdn.internal.example.com/purge"

def purge_cdn(path: str):
    try:
        requests.post(CDN_PURGE_URL, json={"paths": [path]}, timeout=2)
    except requests.RequestException:
        pass

@app.route("/upload", methods=["POST"])
def upload_avatar():
    file = request.files["file"]
    user_id = request.form.get("user_id", "anonymous")
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    filepath = os.path.join(user_dir, filename)

    os.makedirs(user_dir, exist_ok=True)
    file.save(filepath)

    purge_cdn(f"/avatars/{user_id}/")

    return jsonify({
        "url": f"https://cdn.example.com/avatars/{user_id}/{filename}",
        "hash": compute_hash(filepath),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

*（省略了部分辅助函数，完整代码见 [GitHub](https://github.com/mengxinghun9657-boop/qwen-proxy)）*

---

## 第 1 轮：Claude Code 独立审查

Claude 独立阅读代码后，发现了 **4 个问题**：

| # | 发现 | 严重度 |
|---|------|--------|
| 1 | **路径穿越漏洞** — `user_id` 未校验直接拼入文件路径，`secure_filename` 已引入但未使用 | 🔴 高危 |
| 2 | **Flask dev server** — `app.run()` 是单线程开发服务器，峰值流量下排队超时，是 500 错误的**直接原因** | 🔴 高危 |
| 3 | **CDN purge 静默失败** — 2 秒超时 + `pass` 吞异常，purge 失败后用户看到旧缓存，解释旧头像问题 | 🟡 中危 |
| 4 | **`os.makedirs` 竞态** — 并发请求同时创建同一用户目录可能冲突 | 🟢 低危 |

其中 **#2 是 500 错误的根因**，**#1 是安全漏洞**。总的来说，Claude 的强项在框架和代码结构层面。

---

## 第 2 轮：Claude + Qwen 协同审查

这次 Claude 调用 Qwen（通过 [qwen-proxy](https://github.com/mengxinghun9657-boop/qwen-proxy) 本地反代），使用压缩模式 `-M review` 注入输出约束：

```bash
./venv/bin/python ask_qwen.py -M review "$(cat review_test.py)"
```

`review` 模式自动在系统提示词中追加了格式约束：只输出 **Critical / Warnings / Summary** 三段，每段最多 3 条，总字数不超过 200。

Qwen 的完整回复：

```
## Critical
- [Race condition causing stale avatars] (severity: H): The code generates
  a new random filename but does not update the user's profile record in the
  database. Frontends likely cache the old URL or the CDN caches the old path.
  More critically, `purge_cdn` is "fire-and-forget" with a short timeout; if
  it fails silently, users see old images until TTL expires.
- [Path Traversal / Security Vulnerability] (severity: H): `user_id` is taken
  directly from `request.form` without validation. An attacker can send
  `user_id="../../etc/cron.d"` to write files to arbitrary locations.

## Warnings
- [Thread Safety & Resource Exhaustion] (severity: M): `os.makedirs` and file
  I/O are not atomic. Under peak load, race conditions or disk full errors (no
  check before write) cause 500s. Using local disk for 50k uploads/day without
  cleanup logic will eventually fill the disk.
- [Blocking I/O in Request Path] (severity: M): `compute_hash` reads the
  entire file again after saving. This doubles disk I/O latency per request.
  Under load, this blocks worker threads, contributing to the reported 500s.

## Summary
The service suffers from critical security flaws (path traversal), data
consistency issues (stale avatars due to poor cache invalidation and lack of
DB updates), and performance bottlenecks (blocking I/O) that cause 500 errors.
```

Qwen 额外发现了 **3 个 Claude 没注意到的问题**：

| # | 发现 | Claude 发现了？ |
|---|------|:---------------:|
| 5 | **磁盘会满** — 50k 上传/天，无清理逻辑，无写前空间检查 | ❌ |
| 6 | **compute_hash 阻塞 I/O** — 请求路径中二次读文件算 hash，峰值时线程池被占满 | ❌ |
| 7 | **CDN purge 根因更深** — 不只是 purge 可能失败，关键是没有更新用户记录的 DB 操作，生成新 URL 后前端根本不知道 | ❌ |

同时 Qwen 对 #1（路径穿越）给出了具体攻击示例：`user_id=../../etc/cron.d`，比 Claude 的通用描述更有说服力。

---

## 对比结果

| 指标 | Claude Solo | Claude + Qwen | 提升 |
|------|:-----------:|:-------------:|:----:|
| 发现问题总数 | 4 | **7** | **+75%** |
| 500 错误根因定位 | 1 个 | **3 个** | +200% |
| 旧头像根因定位 | 1 个 | **2 个** | +100% |
| 误报 / 噪音 | 0 | 0 | — |

两个模型的注意力分布明显不同：

- **Claude 擅长**：框架缺陷（Flask dev server）、代码结构问题（import 未使用）
- **Qwen 擅长**：运维层面（磁盘满、I/O 阻塞）、攻击链分析（path traversal → cron.d）、数据一致性问题（缺少 DB 更新）

**互补，而非替代。**

---

## 关键设计：为什么 Qwen 的输出没有变成噪音

多模型协作有一个致命风险——副驾驶的输出可能变成上下文垃圾。如果 Qwen 回复了 500 字的客套话 + 免责声明 + 重复内容，那么"协作"非但没有帮助，反而稀释了 Claude 的推理质量。

这就是 **压缩模式（Compression Presets）** 的设计动机。`-M review` 会在系统提示词中注入：

```
Output format:
## Critical
- [issue] (severity: H/M/L)
## Warnings
- [issue] (severity: H/M/L)
## Summary
One sentence.
Max 3 items per section. Max 200 words total.
```

本次实战中，Qwen 的回复被约束在 ~150 词，没有任何寒暄、免责声明或重复内容。信噪比接近 100%。

项目内置了 6 种压缩模式，每种对应不同的协作场景：

| 模式 | 适用场景 | 输出约束 |
|------|----------|----------|
| `review` | 代码审查 | 按严重度分段，200 词 |
| `diagnose` | Bug 排查 | 根因 + 修复 + 备选，100 词 |
| `concise` | 快速确认 | 结论先行，150 词 |
| `judge` | 二元决策 | YES/NO + 置信度 + 一句话理由 |
| `keypoints` | 日志摘要 | 3-5 个要点，无前言后语 |
| `json` | 需要可解析 | 纯 JSON，无 markdown |

---

## 架构简述

```
Claude Code                Qwen (via proxy)
     │                           │
     │  遇到复杂问题              │
     │  判断是否适合问 Qwen       │
     │                           │
     ├── ask_qwen.py -M review ──→│
     │   (注入压缩约束)            │
     │                           │
     │←── 紧凑结构化回复 ──────────┤
     │   (~150 词, 零噪音)         │
     │                           │
     ├── 综合双方分析             │
     └── 输出最终结论给用户
```

核心组件：

- **`server.py`**：OpenAI 兼容的反代服务，将标准 `/v1/chat/completions` 转译为 Qwen 内部 API 协议
- **`ask_qwen.py`**：CLI 工具，封装 6 种压缩预设，支持管道输入和多轮对话
- **Skill 定义**：决策框架，规定何时该调用 Qwen、何时跳过、如何后处理回复

完整文档和代码见 [GitHub](https://github.com/mengxinghun9657-boop/qwen-proxy)。

---

## 一些坦诚的说明

这个项目不是银弹。几个真实限制：

1. **不是自动的** — Claude 根据 Skill 中定义的准则自行判断何时调 Qwen，需要一定的判断力
2. **每次调用约 10-30 秒延迟** — 对于简单问题得不偿失，所以 Skill 里明确规定了"3 步以内解决的事不要调"
3. **依赖 Qwen Web 的 JWT Token** — 需要从浏览器提取，但目标用户（会用 CLI / Git / Claude Code）对此不会有障碍

---

## 不只是 Qwen

这套"主模型 + 压缩副驾驶"的模式可以接入任何兼容 OpenAI API 的后端。你可以把 Ollama 本地模型、DeepSeek API、或者 Gemini 接入同一个协作框架，根据任务类型路由到不同的副驾驶。

**多模型协作的关键不是"哪个模型更强"，而是"如何让它们互补而不互扰"。** 压缩模式是这个问题的其中一个解，欢迎讨论更好的方案。

---

*2026 年 5 月 · [GitHub](https://github.com/mengxinghun9657-boop/qwen-proxy) · MIT License*

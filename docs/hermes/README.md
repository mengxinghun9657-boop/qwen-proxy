# Hermes Agent + Qwen Proxy

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 是 Nous Research 开发的自主 AI Agent，具备自学习循环（skill 自动创建、记忆系统、定时任务）。

本文档记录将 Hermes Agent 接入 Qwen Proxy 的配置方式。

## 安装

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装后 `~/.hermes/config.yaml` 中的 model 段配置：

```yaml
model:
  default: "qwen3.6-plus"
  provider: "custom"
  base_url: "http://127.0.0.1:8800/v1"
```

无需 API Key —— Qwen Proxy 在本地处理 JWT 认证。

## 验证

```bash
hermes chat -m qwen3.6-plus -q "hello" -Q
```

## 协同计划

目标：让 Hermes 定期分析 `~/.claude/history.jsonl`（Claude Code 对话记录），从中提取可复用的工作模式，自动生成 Claude Code skill 文件到 `~/.claude/skills/`。

## 相关资源

- [Hermes Agent Docs](https://hermes-agent.nousresearch.com/docs)
- [Qwen Proxy](../)

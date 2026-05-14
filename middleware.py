"""Response processing middleware for the unified LLM gateway.

Pipeline: Collect → Parse → Clean → Validate → Estimate → Dispatch

The middleware buffers streaming events from DeepSeek, processes them
through a chain of handlers, and dispatches clean OpenAI-format chunks
to Hermes. This prevents analysis-only responses, raw JSON leakage,
and missing usage stats from reaching the client.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


# ── Event types ──────────────────────────────────────────────────────────────

@dataclass
class MiddlewareEvent:
    """Normalized event from the model."""
    type: str  # "content", "tool_call", "thinking", "done", "error", "meta"
    text: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    message_id: str = ""
    stop_reason: str = ""


# ── Layer 1: Event Collector ────────────────────────────────────────────────

class EventCollector:
    """Buffer streaming events and provide the full response for processing."""

    def __init__(self):
        self.events: list[MiddlewareEvent] = []
        self._done = False

    def feed(self, ev: MiddlewareEvent):
        self.events.append(ev)
        if ev.type == "done":
            self._done = True

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def full_text(self) -> str:
        return "".join(ev.text for ev in self.events if ev.type == "content")

    @property
    def has_tool_calls(self) -> bool:
        return any(ev.type == "tool_call" for ev in self.events)

    @property
    def all_tool_calls(self) -> list[MiddlewareEvent]:
        return [ev for ev in self.events if ev.type == "tool_call"]


# ── Layer 2: Tool Call Parser ───────────────────────────────────────────────

class ToolCallParser:
    """Extract tool calls from text content using multiple strategies."""

    @staticmethod
    def extract(text: str) -> list[dict]:
        """Return list of {name, arguments} from text."""
        results = []
        remaining = text
        while True:
            tc = ToolCallParser._try_extract_one(remaining)
            if not tc:
                break
            results.append({"name": tc["name"], "arguments": tc["arguments"]})
            remaining = tc["_after"]
        return results

    @staticmethod
    def _try_extract_one(text: str) -> dict | None:
        # Strategy 1: {"tool": "name", "arguments"/"args"/"params": {...}}
        m = re.search(
            r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|args|params|parameters)"\s*:\s*\{',
            text
        )
        if m:
            name = m.group(1)
            start = m.end() - 1
            depth, in_str, esc = 0, False, False
            end = -1
            for i in range(start, len(text)):
                c = text[i]
                if esc:
                    esc = False; continue
                if c == '\\':
                    esc = True; continue
                if c == '"' and not in_str:
                    in_str = True; continue
                if c == '"' and in_str:
                    in_str = False; continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1; break
            if end > 0 and text[end:].lstrip().startswith('}'):
                try:
                    args = json.loads(text[start:end])
                    return {
                        "_before": text[:m.start()],
                        "name": name,
                        "arguments": args,
                        "_after": text[end + 1:],
                    }
                except json.JSONDecodeError:
                    pass

        # Strategy 2: tool_name({"arg": "value"}) — model outputs function-call style
        m = re.search(r'\b(terminal|write_file|read_file|search_files|browser_navigate|'
                       r'process|todo|memory)\s*\(\s*\{', text)
        if m:
            name = m.group(1)
            start = m.end() - 1  # position of {
            # same depth tracking as strategy 1
            depth, in_str, esc = 0, False, False
            end = -1
            for i in range(start, len(text)):
                c = text[i]
                if esc: esc = False; continue
                if c == '\\': esc = True; continue
                if c == '"' and not in_str: in_str = True; continue
                if c == '"' and in_str: in_str = False; continue
                if in_str: continue
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0: end = i + 1; break
            if end > 0:
                try:
                    args = json.loads(text[start:end])
                    return {
                        "_before": text[:m.start()],
                        "name": name,
                        "arguments": args,
                        "_after": text[end:],
                    }
                except json.JSONDecodeError:
                    pass

        # Strategy 3: ```bash ... ``` → terminal
        m = re.search(r'```bash\s*\n(.*?)\n```', text, re.DOTALL)
        if m:
            cmd = m.group(1).strip()
            if cmd:
                return {
                    "_before": text[:m.start()],
                    "name": "terminal",
                    "arguments": {"command": cmd},
                    "_after": text[m.end():],
                }

        # Strategy 3: inline `command` → terminal
        for prefix in ('docker', 'nvidia', 'pip', 'python', 'git', 'curl',
                        'ls ', 'cat ', 'mkdir', 'cd ', 'find', 'grep'):
            m = re.search(rf'`({prefix}[^`]+)`', text)
            if m:
                return {
                    "_before": text[:m.start()],
                    "name": "terminal",
                    "arguments": {"command": m.group(1).strip()},
                    "_after": text[m.end():],
                }

        return None


# ── Layer 3: Content Cleaner ─────────────────────────────────────────────────

class ContentCleaner:
    """Strip raw tool call JSON from text, leaving only natural language."""

    @staticmethod
    def clean(text: str) -> str:
        result = text
        while True:
            tc = ToolCallParser._try_extract_one(result)
            if not tc:
                break
            result = (tc["_before"] + tc["_after"]).strip()
        return result


# ── Layer 4: Validator ───────────────────────────────────────────────────────

class Validator:
    """Ensure tool calls exist when tools were requested."""

    # Retry correction appended when model outputs analysis instead of tools
    RETRY_PROMPT = (
        "\n\n## REJECTED — Your previous response was analysis-only with no tool calls."
        "\nYou MUST now output at least one {\"tool\": \"...\", \"arguments\": {...}} call."
        "\nDo NOT describe, plan, or analyze. Execute a tool immediately."
    )

    @staticmethod
    def needs_retry(collector: EventCollector, tools_requested: bool) -> bool:
        """Check if response should be retried (analysis-only, no salvageable commands)."""
        if not tools_requested:
            return False
        if collector.has_tool_calls:
            return False
        text = collector.full_text.strip()
        if not text:
            return True
        tcs = ToolCallParser.extract(text)
        return len(tcs) == 0  # pure analysis, nothing to salvage

    @staticmethod
    def validate(
        collector: EventCollector,
        tools_requested: bool,
    ) -> list[MiddlewareEvent] | None:
        """
        If tools were requested but no tool_calls found, try to salvage.
        Returns replacement events or None if response is valid.
        """
        if not tools_requested:
            return None
        if collector.has_tool_calls:
            return None

        text = collector.full_text
        if not text.strip():
            return None

        tcs = ToolCallParser.extract(text)
        if not tcs:
            return None  # pure analysis — handled by retry

        # Found tool calls in text — rebuild response
        clean_text = ContentCleaner.clean(text)
        new_events: list[MiddlewareEvent] = []
        if clean_text:
            new_events.append(MiddlewareEvent(type="content", text=clean_text))
        for tc in tcs:
            new_events.append(MiddlewareEvent(
                type="tool_call",
                name=tc["name"],
                arguments=tc["arguments"],
            ))
        log.info("middleware: salvaged %d tool calls from response text", len(tcs))
        return new_events


# ── Layer 5: Usage Estimator ─────────────────────────────────────────────────

class UsageEstimator:
    """Estimate token usage for streaming chunks."""

    def __init__(self):
        self._estimated = 0

    def estimate(self, ev: MiddlewareEvent) -> dict:
        if ev.type == "content":
            self._estimated += max(1, len(ev.text) // 4)
        elif ev.type == "tool_call":
            self._estimated += max(1,
                len(ev.name) // 4 + len(str(ev.arguments)) // 8)
        return {
            "prompt_tokens": 0,
            "completion_tokens": self._estimated,
            "total_tokens": self._estimated,
        }


# ── Pipeline ─────────────────────────────────────────────────────────────────

class MiddlewarePipeline:
    """Orchestrates the full processing pipeline."""

    def __init__(self, tools_requested: bool = False):
        self.collector = EventCollector()
        self.usage = UsageEstimator()
        self.tools_requested = tools_requested

    def feed(self, ev: MiddlewareEvent):
        self.collector.feed(ev)

    async def dispatch(self) -> AsyncIterator[dict]:
        """Yield OpenAI-format chunks after processing."""
        events = list(self.collector.events)  # copy

        # Run validator if tools were expected but not delivered
        if self.collector.is_done:
            replacement = Validator.validate(self.collector, self.tools_requested)
            if replacement is not None:
                log.info("middleware: salvaged %d tool calls from analysis text",
                         sum(1 for e in replacement if e.type == "tool_call"))
                events = replacement + [MiddlewareEvent(type="done")]

        # Dispatch with cleaning and usage
        for ev in events:
            if ev.type == "content":
                cleaned = ContentCleaner.clean(ev.text)
                if cleaned:
                    yield self._chunk({"content": cleaned})
            elif ev.type == "tool_call":
                yield self._chunk({
                    "tool_calls": [{
                        "index": 0,
                        "id": f"call_{id(ev):x}",
                        "type": "function",
                        "function": {
                            "name": ev.name,
                            "arguments": json.dumps(ev.arguments, ensure_ascii=False),
                        },
                    }]
                })
            elif ev.type == "done":
                yield self._chunk({}, finish_reason="stop", is_final=True)
            elif ev.type == "thinking":
                yield self._chunk({"reasoning_content": ev.text})
            # meta/error events are silently dropped

    def _chunk(self, delta: dict, finish_reason: str | None = None,
               is_final: bool = False) -> dict:
        chunk: dict = {
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        if is_final:
            delta.pop("content", None)  # final chunk has empty delta
            chunk["choices"][0]["finish_reason"] = finish_reason
        if not is_final or finish_reason:
            chunk["usage"] = self.usage.estimate(MiddlewareEvent(
                type="content" if "content" in delta else "tool_call",
                text=delta.get("content", ""),
                name=delta.get("tool_calls", [{}])[0].get("function", {}).get("name", "") if "tool_calls" in delta else "",
            ))
            # Re-estimate correctly based on delta type
            pass  # usage already tracked via estimate() calls above
        return chunk

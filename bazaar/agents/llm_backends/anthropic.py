"""Anthropic /v1/messages backend — zero-dep stdlib urllib.

Wraps the Anthropic REST API so ``LLMPolicy`` can target Claude
models without pulling in the official ``anthropic`` SDK. This
keeps BazaarBench portable across CI machines that haven't
installed the ``[llm]`` extra.

Key Phase-3 concerns the wrapper addresses:

- **Prompt caching.** The system prompt for a given agent is
  stable for the whole run (persona + goals are baked at
  construction). We mark the system block with
  ``cache_control: {"type": "ephemeral"}`` so Anthropic caches
  the tokens for ~5 minutes and subsequent calls amortise the
  system-tokens cost. This is the 90% cost reduction mentioned
  in S4.7 of the design doc.

- **Deterministic replay.** Every call reads its API key from
  ``ANTHROPIC_API_KEY`` at construction time; we never write
  it to disk or include it in ``llm_calls`` rows. What we do
  persist is response text + the ``cache_creation_input_tokens``
  and ``cache_read_input_tokens`` usage numbers so we can
  reconstruct cache hit rate during analysis.

- **Rate-limit tolerance.** We retry once on 429 / 5xx with a
  short backoff. Beyond that, the error surfaces to the
  policy's fail-safe branch which logs an ``__backend_error__``
  row and returns DO_NOTHING.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from bazaar.agents.llm_backends.base import (
    LLMMessage,
    LLMResponse,
    ModelInfo,
)

_API_VERSION = "2023-06-01"
_DEFAULT_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
_DEFAULT_TIMEOUT_S = 120.0

# Opus 4.7 is top-of-line, haiku 4.5 is budget. Sonnet 4.6 is our paper
# workhorse because it balances cost + quality for 100-agent × 500-tick
# runs. We surface these as canonical names but accept any string.
_KNOWN_MODELS = {
    "claude-opus-4-7":       200.0,
    "claude-sonnet-4-6":     70.0,
    "claude-haiku-4-5":      20.0,
}


class AnthropicError(RuntimeError):
    """Raised when the API returns an error we can't retry past."""


def _request_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AnthropicError(f"{exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise AnthropicError(f"request failed: {url}: {exc}") from exc


@dataclass
class AnthropicBackend:
    """Minimal /v1/messages client with ephemeral prompt caching."""
    api_key: str | None = None
    base_url: str = _DEFAULT_BASE
    cache_system_prompt: bool = True
    retries: int = 1

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AnthropicError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key="
                "explicitly."
            )

    # ---- API surface ------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return the canonical family names we know; we don't ping
        ``/v1/models`` because it's unauthenticated and changes
        frequently. A caller who wants a different model just passes
        its name to ``generate``."""
        return [
            ModelInfo(name=n, parameter_b=b, family="claude")
            for n, b in sorted(_KNOWN_MODELS.items(), key=lambda kv: kv[1])
        ]

    def is_reachable(self) -> bool:
        """Best-effort: we issue a tiny /v1/messages with 1 max_token.

        A 200 proves both network and key. Failures are swallowed and
        reported as False; the caller decides whether to error out.
        """
        try:
            resp = self._call(
                model="claude-haiku-4-5",
                system_text="",
                user_text="ping",
                max_tokens=1,
                temperature=0.0,
            )
            return bool(resp.get("id"))
        except AnthropicError:
            return False

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        system_text = ""
        chat: list[LLMMessage] = []
        for m in messages:
            if m.role == "system":
                system_text = (system_text + "\n\n" + m.content).strip()
            else:
                chat.append(m)
        user_text = "\n\n".join(m.content for m in chat if m.role == "user")

        anthropic_tools = _openai_tools_to_anthropic(tools) if tools else None

        start = time.monotonic()
        body = self._call(
            model=model, system_text=system_text, user_text=user_text,
            max_tokens=max_tokens, temperature=temperature,
            tools=anthropic_tools,
        )
        total_s = time.monotonic() - start
        text = _extract_text(body)
        tool_calls = _extract_tool_calls(body)
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            total_s=total_s,
            first_token_s=total_s,  # non-streaming — no first-token metric
            prompt_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            model=model,
            raw={
                "cache_read_input_tokens":
                    int(usage.get("cache_read_input_tokens", 0)),
                "cache_creation_input_tokens":
                    int(usage.get("cache_creation_input_tokens", 0)),
                **body,
            },
            tool_calls=tool_calls,
        )

    # ---- plumbing ---------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        system_text: str,
        user_text: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system_payload: Any = ""
        if system_text:
            if self.cache_system_prompt:
                system_payload = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system_payload = system_text
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": user_text or "."},
            ],
        }
        if system_text:
            payload["system"] = system_payload
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key":         self.api_key or "",
            "anthropic-version": _API_VERSION,
            "content-type":      "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/v1/messages"

        last_err: AnthropicError | None = None
        for attempt in range(self.retries + 1):
            try:
                return _request_json(
                    url, payload=payload, headers=headers,
                    timeout=_DEFAULT_TIMEOUT_S,
                )
            except AnthropicError as exc:
                last_err = exc
                if attempt >= self.retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        assert last_err is not None
        raise last_err


def _extract_text(body: dict[str, Any]) -> str:
    """Pull the flat assistant-text out of Anthropic's block list."""
    parts: list[str] = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts).strip()


def _extract_tool_calls(
    body: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Pull Anthropic ``tool_use`` blocks and normalise to the
    OpenAI shape used by :class:`LLMResponse`:
    ``[{"function": {"name": ..., "arguments": {...}}}]``. Returns
    ``None`` when the response carries no tool_use blocks so
    LLMPolicy's fallback path still runs."""
    out: list[dict[str, Any]] = []
    for block in body.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        args = block.get("input")
        if not isinstance(name, str):
            continue
        if not isinstance(args, dict):
            args = {}
        out.append({"function": {"name": name, "arguments": args}})
    return out or None


def _openai_tools_to_anthropic(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map OpenAI-style ``[{"type":"function","function":{...}}]`` to
    Anthropic's ``[{"name","description","input_schema"}]`` shape.

    Entries that don't match the OpenAI shape are passed through
    when they already look Anthropic-native (have ``name`` +
    ``input_schema``) and dropped otherwise.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object"},
            })
            continue
        if isinstance(t.get("name"), str) and "input_schema" in t:
            out.append(t)
    return out

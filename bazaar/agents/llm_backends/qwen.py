"""DashScope Qwen OpenAI-compatible chat backend.

DashScope's OpenAI-compatible endpoint mostly follows
``/v1/chat/completions`` but exposes Qwen thinking traces through
``message.reasoning_content`` when ``enable_thinking`` is set. This
adapter keeps that provider-specific knob out of ``OpenAIBackend`` and
maps the thinking trace into ``LLMResponse.reasoning_summary`` so the
existing BazaarBench reasoning audit path works unchanged.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from bazaar.agents.llm_backends.base import (
    LLMMessage,
    LLMResponse,
    ModelInfo,
)
from bazaar.agents.llm_backends.openai import (
    OpenAIError,
    _extract_tool_calls,
    _request_json,
)

_DEFAULT_BASE = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode",
)
_DEFAULT_TIMEOUT_S = float(
    os.environ.get("QWEN_TIMEOUT_S")
    or os.environ.get("DASHSCOPE_TIMEOUT_S")
    or "75"
)


class QwenError(RuntimeError):
    """Raised when DashScope returns an unrecoverable error."""


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _thinking_for_effort(
    reasoning_effort: str | None,
    *,
    default: bool,
) -> bool:
    """Map low-effort maintenance calls to cheaper non-thinking mode.

    Action calls usually omit ``reasoning_effort`` or pass medium/high
    through the CLI and should keep Qwen thinking enabled. Reflection
    dynamics pass ``low`` because they only need a compact state summary;
    disabling thinking there prevents one tick from spawning many slow
    hidden-reasoning calls.
    """
    effort = (reasoning_effort or "").strip().lower()
    if effort in {"low", "minimal", "none", "off"}:
        if _env_bool("QWEN_THINKING_FOR_LOW_EFFORT", False):
            return default
        return False
    return default


@dataclass
class QwenBackend:
    """Qwen chat backend via DashScope's OpenAI-compatible API."""

    api_key: str | None = None
    base_url: str = _DEFAULT_BASE
    retries: int = 5
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise QwenError(
                "DASHSCOPE_API_KEY not set. Export it or pass api_key="
                "explicitly."
            )
        self.enable_thinking = _env_bool(
            "QWEN_ENABLE_THINKING",
            self.enable_thinking,
        )

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="qwen3.6-35b-a3b", parameter_b=35.0, family="qwen"),
            ModelInfo(name="qwen-plus", family="qwen"),
            ModelInfo(name="qwen-turbo", family="qwen"),
        ]

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        system_text = ""
        chat_payload: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_text = (system_text + "\n\n" + msg.content).strip()
            else:
                chat_payload.append({"role": msg.role, "content": msg.content})
        if system_text:
            chat_payload = (
                [{"role": "system", "content": system_text}] + chat_payload
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_payload,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enable_thinking": _thinking_for_effort(
                reasoning_effort,
                default=self.enable_thinking,
            ),
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        start = time.monotonic()
        body = self._request(payload=payload, headers=headers)
        total_s = time.monotonic() - start
        usage = body.get("usage") or {}
        return LLMResponse(
            text=_extract_text(body),
            total_s=total_s,
            first_token_s=total_s,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=model,
            raw=body,
            tool_calls=_extract_tool_calls(body),
            reasoning_summary=_extract_reasoning(body),
        )

    def _request(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        url = _chat_completions_url(self.base_url)
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return _request_json(
                    url,
                    payload=payload,
                    headers=headers,
                    timeout=_DEFAULT_TIMEOUT_S,
                )
            except OpenAIError as exc:
                last_err = exc
                if attempt >= self.retries:
                    raise QwenError(str(exc)) from exc
                # 429 rate-limit / quota burst: exponential backoff with
                # jitter so a herd of parallel workers don't retry in
                # lockstep. Other errors get a short linear backoff.
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                    import random as _random
                    sleep_s = (2.0 ** attempt) + _random.uniform(0, 0.5)
                else:
                    sleep_s = 0.5 * (attempt + 1)
                time.sleep(sleep_s)
        assert last_err is not None
        raise QwenError(str(last_err))


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_reasoning(body: dict[str, Any]) -> str | None:
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

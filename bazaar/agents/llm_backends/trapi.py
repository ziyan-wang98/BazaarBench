"""TRAPI backend for BazaarBench.

TRAPI exposes an OpenAI-compatible API at
``https://trapi.research.microsoft.com/<instance>/openai/v1/``. This adapter
keeps the TRAPI-specific pieces in one place: model alias resolution, endpoint
construction, Azure bearer-token auth, and endpoint capability routing.
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from bazaar.agents.llm_backends.base import LLMMessage, LLMResponse, ModelInfo
from bazaar.agents.llm_backends.openai import (
    OpenAIError,
    _build_url,
    _chat_tools_to_responses_tools,
    _extract_chat_reasoning,
    _extract_responses_reasoning,
    _extract_responses_text,
    _extract_responses_tool_calls,
    _extract_text,
    _extract_tool_calls,
    _is_reasoning_model,
    _request_json,
)

_DEFAULT_INSTANCE = os.environ.get(
    "BAZAAR_TRAPI_INSTANCE",
    os.environ.get("MEMORY_FORM_BENCH_TRAPI_INSTANCE", "msrc/shared"),
)
_DEFAULT_AUTH_MODE = os.environ.get(
    "BAZAAR_TRAPI_AUTH_MODE",
    os.environ.get("MEMORY_FORM_BENCH_TRAPI_AUTH_MODE", "azure_cli"),
)
_DEFAULT_TIMEOUT_S = float(os.environ.get("BAZAAR_TRAPI_TIMEOUT_S", "240"))
_TRAPI_SCOPE = "api://trapi/.default"
_TOKEN_PROVIDER_CACHE: dict[tuple[str, str], Callable[[], str]] = {}
_TOKEN_PROVIDER_LOCK = Lock()
_TOKEN_CALL_LOCK = Lock()
_REQUEST_THROTTLE_LOCK = Lock()
_REQUEST_THROTTLE_NEXT_S = 0.0
_REQUEST_THROTTLE_NEXT_BY_KEY: dict[str, float] = {}
_REQUEST_THROTTLE_GLOBAL_KEY = "global"
_PROVIDER_CIRCUIT_LOCK = Lock()
_PROVIDER_CIRCUIT_STATE: dict[str, dict[str, float | int | str]] = {}


TRAPI_MODEL_ALIASES: dict[str, str] = {
    "GPT_54": "gpt-5.4_2026-03-05",
    "GPT_54_PRO": "gpt-5.4-pro_2026-03-05",
    "GPT_54_MINI": "gpt-5.4-mini_2026-03-17",
    "GPT_54_NANO": "gpt-5.4-nano_2026-03-17",
    "GPT_55": "gpt-5.5_2026-04-24",
    "GPT_5": "gpt-5_2025-08-07",
    "GPT_5_MINI": "gpt-5-mini_2025-08-07",
    "GPT_5_NANO": "gpt-5-nano_2025-08-07",
    "GPT_41": "gpt-4.1_2025-04-14",
    "GPT_41_MINI": "gpt-4.1-mini_2025-04-14",
    "GPT_41_NANO": "gpt-4.1-nano_2025-04-14",
    "GPT_4O": "gpt-4o_2024-11-20",
    "GPT_4O_MINI": "gpt-4o-mini_2024-07-18",
    "O1": "o1_2024-12-17",
    "O3": "o3_2025-04-16",
    "O4_MINI": "o4-mini_2025-04-16",
    "GPT_OSS_120B": "gpt-oss-120b_1",
    "QWEN_35_397B": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    "QWEN_35_122B": "Qwen/Qwen3.5-122B-A10B",
    "QWEN_35_27B": "Qwen/Qwen3.5-27B",
    "QWEN_35_9B": "Qwen/Qwen3.5-9B",
    "DEEPSEEK_V32": "DeepSeek-V3.2_1",
    "DEEPSEEK_R1": "DeepSeek-R1_1",
    "LLAMA_33_70B": "Llama-3.3-70B-Instruct_5",
    "MISTRAL_LARGE_3": "Mistral-Large-3_1",
    "KIMI_K25": "Kimi-K2.5_1",
    "KIMI_K26": "Kimi-K2.6_2026-04-20",
    "gpt-5.4-mini": "gpt-5.4-mini_2026-03-17",
    "gpt-5.4-nano": "gpt-5.4-nano_2026-03-17",
    "gpt-5.5": "gpt-5.5_2026-04-24",
}

TRAPI_HEALTHY_TEXT_DEPLOYMENTS: frozenset[str] = frozenset(
    {
        "gpt-4o_2024-08-06",
        "gpt-4o_2024-11-20",
        "gpt-4o-mini_2024-07-18",
        "gpt-4.1_2025-04-14",
        "gpt-4.1-mini_2025-04-14",
        "gpt-4.1-nano_2025-04-14",
        "gpt-5_2025-08-07",
        "gpt-5-mini_2025-08-07",
        "gpt-5-nano_2025-08-07",
        "gpt-5-chat_2025-08-07",
        "gpt-5-chat_2025-10-03",
        "gpt-5.1_2025-11-13",
        "gpt-5.1-chat_2025-11-13",
        "gpt-5.2_2025-12-11",
        "gpt-5.2-chat_2025-12-11",
        "gpt-5.3-chat_2026-03-03",
        "gpt-5.4_2026-03-05",
        "gpt-5.4-mini_2026-03-17",
        "gpt-5.4-nano_2026-03-17",
        "gpt-5.5_2026-04-24",
        "gpt-chat-latest_2026-05-28",
        "o1_2024-12-17",
        "o3_2025-04-16",
        "o4-mini_2025-04-16",
        "gpt-oss-120b_1",
        "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "DeepSeek-V3.2_1",
        "DeepSeek-R1_1",
        "Llama-3.3-70B-Instruct_5",
        "Mistral-Large-3_1",
        "Kimi-K2.5_1",
        "Kimi-K2.6_2026-04-20",
    }
)

TRAPI_UNHEALTHY_TEXT_DEPLOYMENTS: frozenset[str] = frozenset(
    {
        "gpt-5-pro_2025-10-06",
        "gpt-5.4-pro_2026-03-05",
        "o3-mini_2025-01-31",
        "o3-pro_2025-06-10",
    }
)

TRAPI_RESPONSES_DEPLOYMENTS: frozenset[str] = frozenset(
    {
        "gpt-4o_2024-08-06",
        "gpt-4o_2024-11-20",
        "gpt-4o-mini_2024-07-18",
        "gpt-4.1_2025-04-14",
        "gpt-4.1-mini_2025-04-14",
        "gpt-4.1-nano_2025-04-14",
        "gpt-5_2025-08-07",
        "gpt-5-mini_2025-08-07",
        "gpt-5-nano_2025-08-07",
        "gpt-5-chat_2025-08-07",
        "gpt-5-chat_2025-10-03",
        "gpt-5.1_2025-11-13",
        "gpt-5.1-chat_2025-11-13",
        "gpt-5.2_2025-12-11",
        "gpt-5.2-chat_2025-12-11",
        "gpt-5.3-chat_2026-03-03",
        "gpt-5.4-mini_2026-03-17",
        "gpt-5.4-nano_2026-03-17",
        "gpt-5.5_2026-04-24",
        "gpt-chat-latest_2026-05-28",
        "o1_2024-12-17",
        "o3_2025-04-16",
        "o4-mini_2025-04-16",
    }
)

TRAPI_CHAT_ONLY_DEPLOYMENTS: frozenset[str] = (
    TRAPI_HEALTHY_TEXT_DEPLOYMENTS - TRAPI_RESPONSES_DEPLOYMENTS
)
TRAPI_TEXT_TOOL_FALLBACK_DEPLOYMENTS: frozenset[str] = frozenset(
    {
        *{
            model
            for model in TRAPI_HEALTHY_TEXT_DEPLOYMENTS
            if model.startswith("Qwen/")
        },
        "Llama-3.3-70B-Instruct_5",
    }
)
TRAPI_CHAT_REQUIRED_TOOL_CHOICE_UNSUPPORTED: frozenset[str] = frozenset(
    {"gpt-oss-120b_1"}
)


class TRAPIError(RuntimeError):
    """Raised when TRAPI setup or generation fails."""


class TRAPIProviderFailFastError(TRAPIError):
    """Raised when one TRAPI instance is unhealthy enough to fail over."""


def trapi_model_name(name: str) -> str:
    """Resolve a TRAPI alias to the served deployment name."""

    return TRAPI_MODEL_ALIASES.get(name, name)


def trapi_endpoint(instance: str | None = None) -> str:
    """Return the OpenAI-compatible TRAPI base URL for an instance."""

    clean = (instance or _DEFAULT_INSTANCE).strip("/")
    return f"https://trapi.research.microsoft.com/{clean}/openai/v1/"


def _trapi_endpoint_candidates(
    *,
    base_url: str | None,
    trapi_instance: str,
) -> tuple[str, ...]:
    urls: list[str] = []
    if base_url:
        urls.append(base_url)
    urls.extend(_split_csv_env("BAZAAR_TRAPI_BASE_URLS"))
    urls.extend(_split_csv_env("TRAPI_BASE_URLS"))
    if not urls:
        instances = (
            _split_csv_env("BAZAAR_TRAPI_INSTANCES")
            or _split_csv_env("TRAPI_INSTANCES")
        )
        urls.extend(trapi_endpoint(instance) for instance in instances)
    if not urls:
        single = (
            os.environ.get("BAZAAR_TRAPI_BASE_URL")
            or os.environ.get("TRAPI_BASE_URL")
        )
        if single:
            urls.append(single)
    if not urls:
        urls.append(trapi_endpoint(trapi_instance))
    return tuple(_dedupe_preserving_order(_normalise_base_url(url) for url in urls))


def _split_csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalise_base_url(url: str) -> str:
    clean = url.strip()
    if not clean:
        return clean
    return clean.rstrip("/") + "/"


def _dedupe_preserving_order(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def trapi_model_status(name: str) -> str:
    """Best-known status from the current TRAPI resource list."""

    resolved = trapi_model_name(name)
    if resolved in TRAPI_HEALTHY_TEXT_DEPLOYMENTS:
        return "healthy"
    if resolved in TRAPI_UNHEALTHY_TEXT_DEPLOYMENTS:
        return "unhealthy"
    return "unknown"


def trapi_supports_responses(name: str) -> bool:
    """Whether the resolved deployment is listed with responses capability."""

    return trapi_model_name(name) in TRAPI_RESPONSES_DEPLOYMENTS


def trapi_context_window(name: str) -> int:
    """Best-known context window for current TRAPI text deployments."""

    _ = name
    return 128_000


@dataclass
class TRAPIBackend:
    """OpenAI-compatible TRAPI backend with Azure bearer-token auth."""

    api_key: str | None = None
    base_url: str | None = None
    trapi_instance: str = _DEFAULT_INSTANCE
    auth_mode: str = _DEFAULT_AUTH_MODE
    retries: int = 7
    reasoning_effort: str = "high"
    use_responses_endpoint: bool = True
    request_timeout_s: float = _DEFAULT_TIMEOUT_S
    token_provider: Callable[[], str] | None = field(default=None, repr=False)
    _base_urls: tuple[str, ...] = field(default=(), init=False, repr=False)
    _base_url_index: int = field(default=0, init=False, repr=False)
    _base_url_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._base_urls = _trapi_endpoint_candidates(
            base_url=self.base_url,
            trapi_instance=self.trapi_instance,
        )
        self.base_url = self._base_urls[0]
        self.auth_mode = (self.auth_mode or _DEFAULT_AUTH_MODE).strip().lower()
        if self.auth_mode not in {"azure_cli", "managed_identity", "chained", "auto"}:
            raise TRAPIError(
                "auth_mode must be one of: azure_cli, managed_identity, chained, auto"
            )
        self.api_key = (
            self.api_key
            or os.environ.get("BAZAAR_TRAPI_API_KEY")
            or os.environ.get("TRAPI_API_KEY")
        )
        if self.request_timeout_s <= 0:
            raise TRAPIError("request_timeout_s must be positive")

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name=model, family="trapi")
            for model in sorted(TRAPI_HEALTHY_TEXT_DEPLOYMENTS)
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
        effort = reasoning_effort or self.reasoning_effort
        resolved_model = trapi_model_name(model)
        chat_payload = _normalise_messages(messages)
        text_tool_fallback = bool(
            tools and resolved_model in TRAPI_TEXT_TOOL_FALLBACK_DEPLOYMENTS
        )
        native_tools = None if text_tool_fallback else tools
        if text_tool_fallback:
            chat_payload = _append_text_tool_fallback_instruction(chat_payload, tools or [])
        use_responses = (
            self.use_responses_endpoint and trapi_supports_responses(resolved_model)
        )

        start = time.monotonic()
        if use_responses:
            body = self._call_responses(
                model=resolved_model,
                messages=chat_payload,
                max_tokens=max_tokens,
                tools=native_tools,
                reasoning_effort=effort,
            )
            total_s = time.monotonic() - start
            usage = body.get("usage") or {}
            return LLMResponse(
                text=_extract_responses_text(body),
                total_s=total_s,
                first_token_s=total_s,
                prompt_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                model=model,
                raw=_with_trapi_metadata(
                    body,
                    self,
                    resolved_model,
                    use_responses,
                    text_tool_fallback=text_tool_fallback,
                ),
                tool_calls=_extract_responses_tool_calls(body),
                reasoning_summary=_extract_responses_reasoning(body),
            )

        body = self._call_chat(
            model=resolved_model,
            messages=chat_payload,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=native_tools,
            reasoning_effort=effort,
        )
        total_s = time.monotonic() - start
        usage = body.get("usage") or {}
        return LLMResponse(
            text=_extract_text(body),
            total_s=total_s,
            first_token_s=total_s,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=model,
            raw=_with_trapi_metadata(
                body,
                self,
                resolved_model,
                use_responses,
                text_tool_fallback=text_tool_fallback,
            ),
            tool_calls=_extract_tool_calls(body),
            reasoning_summary=_extract_chat_reasoning(body),
        )

    def _call_responses(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        if _is_reasoning_model(model):
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "summary": "detailed",
            }
        if tools:
            payload["tools"] = _chat_tools_to_responses_tools(tools)
            payload["tool_choice"] = "required"
        return self._request("responses", payload)

    def _call_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if _is_reasoning_model(model):
            payload["max_completion_tokens"] = max_tokens
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
            if model not in TRAPI_CHAT_REQUIRED_TOOL_CHOICE_UNSUPPORTED:
                payload["tool_choice"] = "required"
        return self._request("chat/completions", payload)

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: OpenAIError | None = None
        model_name = str(payload.get("model") or "")
        for attempt in range(self.retries + 1):
            base_url = self._current_base_url()
            url = _build_url(base_url, endpoint)
            try:
                _raise_if_provider_circuit_open(
                    base_url=base_url,
                    endpoint=endpoint,
                    model=model_name,
                )
                _throttle_request_start(model_name)
                response = _request_json(
                    url,
                    payload=payload,
                    headers=self._headers(),
                    timeout=self.request_timeout_s,
                )
                _record_provider_success(base_url, endpoint, model_name)
                self._set_current_base_url(base_url)
                return response
            except TRAPIProviderFailFastError as exc:
                if self._failover_base_url(
                    endpoint=endpoint,
                    model=model_name,
                    exc=exc,
                ):
                    continue
                raise
            except OpenAIError as exc:
                last_err = exc
                fail_fast_error = _is_provider_fail_fast_error(exc)
                if fail_fast_error and self._failover_base_url(
                    endpoint=endpoint,
                    model=model_name,
                    exc=exc,
                    failed_base_url=base_url,
                ):
                    continue
                if fail_fast_error and _record_provider_error(
                    base_url=base_url,
                    endpoint=endpoint,
                    model=model_name,
                    exc=exc,
                ):
                    raise TRAPIProviderFailFastError(
                        _provider_fail_fast_message(
                            base_url=base_url,
                            endpoint=endpoint,
                            model=model_name,
                            exc=exc,
                        )
                    ) from exc
                delay = _retry_delay_s(exc, attempt)
                if attempt >= self.retries or delay is None:
                    raise TRAPIError(str(exc)) from exc
                sys.stderr.write(
                    "[trapi-retry] "
                    f"endpoint={endpoint} model={payload.get('model')} "
                    f"attempt={attempt + 1}/{self.retries} "
                    f"delay_s={delay:.1f} error={_short_error(exc)}\n"
                )
                sys.stderr.flush()
                _postpone_request_start(delay, model=model_name)
                time.sleep(delay)
        assert last_err is not None
        raise TRAPIError(str(last_err)) from last_err

    def _current_base_url(self) -> str:
        with self._base_url_lock:
            if not self._base_urls:
                self._base_urls = (str(self.base_url),)
            return self._base_urls[self._base_url_index % len(self._base_urls)]

    def _set_current_base_url(self, base_url: str) -> None:
        with self._base_url_lock:
            self.base_url = base_url
            if base_url in self._base_urls:
                self._base_url_index = self._base_urls.index(base_url)

    def _failover_base_url(
        self,
        *,
        endpoint: str,
        model: str,
        exc: BaseException,
        failed_base_url: str | None = None,
    ) -> bool:
        if len(self._base_urls) <= 1:
            return False
        current = failed_base_url or self._current_base_url()
        with self._base_url_lock:
            try:
                current_idx = self._base_urls.index(current)
            except ValueError:
                current_idx = self._base_url_index
            next_idx = (current_idx + 1) % len(self._base_urls)
            next_base_url = self._base_urls[next_idx]
            self._base_url_index = next_idx
            self.base_url = next_base_url
        sys.stderr.write(
            "[trapi-failover] "
            f"endpoint={endpoint} model={model} "
            f"from={current} to={next_base_url} error={_short_error(exc)}\n"
        )
        sys.stderr.flush()
        return True

    def _headers(self) -> dict[str, str]:
        token = self._bearer_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenAI/Python/1.0 (BazaarBench TRAPI)",
        }

    def _bearer_token(self) -> str:
        if self.api_key:
            return self.api_key
        if self.token_provider is not None:
            with _TOKEN_CALL_LOCK:
                return self.token_provider()
        provider = _azure_token_provider(self.auth_mode)
        with _TOKEN_CALL_LOCK:
            return provider()


def _normalise_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
    system_text = ""
    chat_payload: list[dict[str, str]] = []
    for msg in messages:
        if msg.role == "system":
            system_text = (system_text + "\n\n" + msg.content).strip()
        else:
            chat_payload.append({"role": msg.role, "content": msg.content})
    if system_text:
        chat_payload = [{"role": "system", "content": system_text}] + chat_payload
    return chat_payload


def _append_text_tool_fallback_instruction(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    tool_payload = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        tool_payload.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    if not tool_payload:
        return messages
    instruction = (
        "\n\nTRAPI native tool calling is unavailable for this model. "
        "Return tool use as plain JSON text instead. Use exactly one of these shapes: "
        '{"action":"tool_name","arguments":{...}} or '
        '[{"action":"tool_name","arguments":{...}}]. '
        "Do not wrap the JSON in markdown. Available tools: "
        f"{tool_payload}"
    )
    out = [dict(item) for item in messages]
    for item in reversed(out):
        if item.get("role") == "user":
            item["content"] = str(item.get("content", "")) + instruction
            return out
    out.append({"role": "user", "content": instruction.strip()})
    return out


def _azure_token_provider(auth_mode: str) -> Callable[[], str]:
    cache_key = (auth_mode, _TRAPI_SCOPE)
    with _TOKEN_PROVIDER_LOCK:
        cached = _TOKEN_PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
        except ImportError as exc:
            raise TRAPIError(
                "TRAPI Azure auth requires azure-identity. Install with "
                "pip install -e '.[llm]' or set BAZAAR_TRAPI_API_KEY."
            ) from exc
        if auth_mode == "azure_cli":
            identity = AzureCliCredential()
        elif auth_mode == "managed_identity":
            identity = ManagedIdentityCredential()
        else:
            identity = ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            )
        provider = get_bearer_token_provider(identity, _TRAPI_SCOPE)
        _TOKEN_PROVIDER_CACHE[cache_key] = provider
        return provider


def _with_trapi_metadata(
    body: dict[str, Any],
    backend: TRAPIBackend,
    resolved_model: str,
    effective_responses_endpoint: bool,
    *,
    text_tool_fallback: bool,
) -> dict[str, Any]:
    out = dict(body)
    out["_trapi"] = {
        "trapi_model": resolved_model,
        "trapi_instance": backend.trapi_instance,
        "trapi_endpoint": backend.base_url,
        "auth_mode": backend.auth_mode,
        "use_responses_endpoint": backend.use_responses_endpoint,
        "effective_responses_endpoint": effective_responses_endpoint,
        "text_tool_fallback": text_tool_fallback,
        "reasoning_effort": backend.reasoning_effort,
        "model_status": trapi_model_status(resolved_model),
    }
    return out


def _provider_circuit_key(base_url: str, endpoint: str, model: str) -> str:
    return "|".join((base_url.rstrip("/"), endpoint.strip("/"), model))


def _provider_fail_fast_threshold() -> int:
    raw = os.environ.get("BAZAAR_TRAPI_PROVIDER_FAIL_FAST_ERRORS", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _provider_fail_fast_window_s() -> float:
    raw = os.environ.get("BAZAAR_TRAPI_PROVIDER_FAIL_FAST_WINDOW_S", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def _provider_circuit_open_s() -> float:
    raw = os.environ.get("BAZAAR_TRAPI_PROVIDER_CIRCUIT_OPEN_S", "600")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 600.0


def _is_provider_fail_fast_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "503",
            "rate limit",
            "temporarily blocked",
            "temporarily unavailable",
            "all-backends-unhealthy",
            "all backends",
            "unusual behavior",
            "certificate_verify_failed",
            "self-signed certificate",
            "403",
            "404",
            "deploymentnotfound",
            "deployment for this resource does not exist",
            "remote reset",
            "reset before headers",
        )
    )


def _record_provider_error(
    *,
    base_url: str,
    endpoint: str,
    model: str,
    exc: BaseException,
) -> bool:
    threshold = _provider_fail_fast_threshold()
    if threshold <= 0:
        return False

    key = _provider_circuit_key(base_url, endpoint, model)
    now = time.monotonic()
    window_s = _provider_fail_fast_window_s()
    open_s = _provider_circuit_open_s()
    with _PROVIDER_CIRCUIT_LOCK:
        state = _PROVIDER_CIRCUIT_STATE.get(key)
        if state is None or (
            window_s > 0
            and now - float(state.get("last_error_s", 0.0)) > window_s
        ):
            state = {"count": 0, "opened_until_s": 0.0, "last_error": ""}
        count = int(state.get("count", 0)) + 1
        state["count"] = count
        state["last_error_s"] = now
        state["last_error"] = _short_error(exc)
        if count >= threshold:
            state["opened_until_s"] = now + open_s
            _PROVIDER_CIRCUIT_STATE[key] = state
            return True
        _PROVIDER_CIRCUIT_STATE[key] = state
        return False


def _record_provider_success(base_url: str, endpoint: str, model: str) -> None:
    key = _provider_circuit_key(base_url, endpoint, model)
    with _PROVIDER_CIRCUIT_LOCK:
        _PROVIDER_CIRCUIT_STATE.pop(key, None)


def _raise_if_provider_circuit_open(
    *,
    base_url: str,
    endpoint: str,
    model: str,
) -> None:
    key = _provider_circuit_key(base_url, endpoint, model)
    now = time.monotonic()
    with _PROVIDER_CIRCUIT_LOCK:
        state = _PROVIDER_CIRCUIT_STATE.get(key)
        if not state:
            return
        opened_until = float(state.get("opened_until_s", 0.0))
        if opened_until <= now:
            if opened_until > 0:
                _PROVIDER_CIRCUIT_STATE.pop(key, None)
            return
        remaining_s = opened_until - now
        count = int(state.get("count", 0))
        last_error = str(state.get("last_error", ""))
    raise TRAPIProviderFailFastError(
        "provider_fail_fast: "
        f"circuit_open remaining_s={remaining_s:.1f} endpoint={endpoint} "
        f"model={model} errors={count} last_error={last_error}"
    )


def _provider_fail_fast_message(
    *,
    base_url: str,
    endpoint: str,
    model: str,
    exc: BaseException,
) -> str:
    return (
        "provider_fail_fast: "
        f"endpoint={endpoint} model={model} base_url={base_url} "
        f"threshold={_provider_fail_fast_threshold()} error={_short_error(exc)}"
    )


def _is_transient_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "408",
            "409",
            "424",
            "429",
            "500",
            "502",
            "503",
            "504",
            "disconnect/reset",
            "temporarily unavailable",
            "all-backends-unhealthy",
            "all backends",
            "upstream connect error",
            "service unavailable",
            "rate limit",
            "remote reset",
            "reset before headers",
            "timeout",
            "request failed",
            "connection reset",
        )
    )


def _retry_delay_s(exc: BaseException, attempt: int) -> float | None:
    if not _is_transient_api_error(exc):
        return None
    text = str(exc)
    retry_after = re.search(
        r"(?:try again in|retry after)\s+(\d+(?:\.\d+)?)\s*seconds?",
        text,
        flags=re.IGNORECASE,
    )
    if retry_after:
        base = float(retry_after.group(1)) + 1.0
        return _delay_with_jitter(base, cap_s=90.0)
    if "429" in text or "rate" in text.lower():
        return _delay_with_jitter((2.0 ** attempt) + 1.0, cap_s=45.0)
    return _delay_with_jitter((1.5 ** attempt) + 0.5, cap_s=20.0)


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _throttle_request_start(model: str | None = None) -> None:
    """Stagger TRAPI request starts across worker threads.

    Some TRAPI deployments advertise high quota but still enforce a backend
    admission gate for long generations. Keeping this opt-in lets the fast
    treatment models run wide while allowing Qwen 397B Level-0 rollouts to
    avoid synchronized 429 waves. Retry-After/remote-reset backoffs also use
    this shared clock so one worker's 429 cools down matching requests.
    """

    throttle_key, interval_s = _throttle_bucket_and_interval(model)
    global _REQUEST_THROTTLE_NEXT_S
    while True:
        with _REQUEST_THROTTLE_LOCK:
            now = time.monotonic()
            next_s = _get_throttle_next_s(throttle_key)
            wait_s = max(0.0, next_s - now)
            if wait_s <= 0:
                if interval_s > 0:
                    _set_throttle_next_s(throttle_key, now + interval_s)
                return
        time.sleep(wait_s)


def _postpone_request_start(delay_s: float, *, model: str | None = None) -> None:
    """Share TRAPI retry delays with matching worker threads in this process."""

    if delay_s <= 0:
        return
    throttle_key, _ = _throttle_bucket_and_interval(model)
    global _REQUEST_THROTTLE_NEXT_S
    with _REQUEST_THROTTLE_LOCK:
        _set_throttle_next_s(
            throttle_key,
            max(
                _get_throttle_next_s(throttle_key),
                time.monotonic() + delay_s,
            ),
        )


def _throttle_bucket_and_interval(model: str | None) -> tuple[str, float]:
    resolved = trapi_model_name(model or "").strip()
    specific_raw = _model_throttle_env_value(resolved)
    if specific_raw is not None:
        return _model_throttle_key(resolved), _parse_interval_s(specific_raw)

    global_raw = os.environ.get("BAZAAR_TRAPI_MIN_REQUEST_INTERVAL_S")
    if global_raw is not None:
        return _REQUEST_THROTTLE_GLOBAL_KEY, _parse_interval_s(global_raw)

    if resolved:
        return _model_throttle_key(resolved), 0.0
    return _REQUEST_THROTTLE_GLOBAL_KEY, 0.0


def _model_throttle_env_value(model: str) -> str | None:
    if not model:
        return None
    exact_env = f"BAZAAR_TRAPI_MODEL_{_sanitize_env_suffix(model)}_MIN_REQUEST_INTERVAL_S"
    if exact_env in os.environ:
        return os.environ[exact_env]

    family = _model_throttle_family(model)
    if family:
        family_env = f"BAZAAR_TRAPI_{family}_MIN_REQUEST_INTERVAL_S"
        if family_env in os.environ:
            return os.environ[family_env]
    return None


def _model_throttle_key(model: str) -> str:
    family = _model_throttle_family(model)
    if family:
        return f"family:{family.lower()}"
    return f"model:{_sanitize_env_suffix(model).lower()}"


def _model_throttle_family(model: str) -> str | None:
    lowered = model.lower()
    if lowered.startswith("qwen/"):
        return "QWEN"
    if lowered.startswith("deepseek"):
        return "DEEPSEEK"
    if lowered.startswith("llama"):
        return "LLAMA"
    if lowered.startswith("gpt-oss"):
        return "GPT_OSS"
    if lowered.startswith("gpt"):
        return "GPT"
    if lowered.startswith("kimi"):
        return "KIMI"
    if lowered.startswith("mistral"):
        return "MISTRAL"
    if lowered.startswith("grok"):
        return "GROK"
    return None


def _sanitize_env_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _parse_interval_s(raw: str) -> float:
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _get_throttle_next_s(throttle_key: str) -> float:
    if throttle_key == _REQUEST_THROTTLE_GLOBAL_KEY:
        return _REQUEST_THROTTLE_NEXT_S
    return _REQUEST_THROTTLE_NEXT_BY_KEY.get(throttle_key, 0.0)


def _set_throttle_next_s(throttle_key: str, value: float) -> None:
    global _REQUEST_THROTTLE_NEXT_S
    if throttle_key == _REQUEST_THROTTLE_GLOBAL_KEY:
        _REQUEST_THROTTLE_NEXT_S = value
    else:
        _REQUEST_THROTTLE_NEXT_BY_KEY[throttle_key] = value


def _delay_with_jitter(base_s: float, *, cap_s: float) -> float:
    base = min(base_s, cap_s)
    jitter_cap = min(3.0, max(0.25, base * 0.2))
    return min(cap_s, base + random.uniform(0.0, jitter_cap))


__all__ = [
    "TRAPIBackend",
    "TRAPIError",
    "TRAPIProviderFailFastError",
    "TRAPI_CHAT_ONLY_DEPLOYMENTS",
    "TRAPI_HEALTHY_TEXT_DEPLOYMENTS",
    "TRAPI_MODEL_ALIASES",
    "TRAPI_RESPONSES_DEPLOYMENTS",
    "TRAPI_CHAT_REQUIRED_TOOL_CHOICE_UNSUPPORTED",
    "TRAPI_TEXT_TOOL_FALLBACK_DEPLOYMENTS",
    "trapi_context_window",
    "trapi_endpoint",
    "trapi_model_name",
    "trapi_model_status",
    "trapi_supports_responses",
]

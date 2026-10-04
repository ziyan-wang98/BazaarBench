"""LLM backend adapters for LLMPolicy (T28b / T28c / T28d).

Three targets Phase 3 cares about:

* **Ollama** — local. Zero cost, slower, convenient for iteration.
* **Anthropic** — Claude API with ephemeral prompt caching.
* **OpenAI**    — drop-in alternative for comparison experiments.
* **Qwen**      — DashScope OpenAI-compatible endpoint with thinking.
* **TRAPI**     — Microsoft research TRAPI OpenAI-compatible endpoint.
* **Foundry**   — Azure AI Foundry OpenAI-compatible endpoint.

This package exposes a uniform interface (:class:`LLMBackend`) that
LLMPolicy calls regardless of provider. Provider-specific quirks
(Anthropic's ``system`` block, OpenAI's ``Authorization`` header)
are fully contained inside each adapter module.
"""
from __future__ import annotations

import os

from bazaar.agents.llm_backends.base import (
    LLMBackend,
    LLMMessage,
    LLMResponse,
    ModelInfo,
)
from bazaar.agents.llm_backends.ollama import OllamaBackend, probe_ollama


def make_backend(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
    reasoning_effort: str = "high",
    use_responses_endpoint: bool = False,
    request_timeout_s: float | None = None,
    retries: int | None = None,
) -> LLMBackend:
    """Factory: pick a concrete backend by name.

    ``provider`` is one of ``"ollama"``, ``"anthropic"``, ``"openai"``,
    ``"qwen"``, ``"trapi"``, or ``"foundry"``. Unknown strings raise ``ValueError``. Cloud backends
    read their API key from env if not passed; ``base_url`` overrides
    the OpenAI default ``https://api.openai.com`` (so the same
    ``provider="openai"`` adapter can route to OpenAI native, OpenRouter,
    DeepSeek, etc.). ``host`` is consumed by Ollama only;
    ``reasoning_effort``, ``use_responses_endpoint``, and
    ``request_timeout_s`` and ``retries`` are consumed by OpenAI only.
    """
    p = (provider or "").lower()
    if p in ("ollama", "local"):
        return OllamaBackend(host=host) if host else OllamaBackend()
    if p in ("anthropic", "claude"):
        from bazaar.agents.llm_backends.anthropic import AnthropicBackend
        return AnthropicBackend(api_key=api_key)
    if p in ("openai", "gpt"):
        from bazaar.agents.llm_backends.openai import OpenAIBackend
        return OpenAIBackend(
            api_key=api_key,
            base_url=(
                base_url
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com"
            ),
            reasoning_effort=reasoning_effort,
            use_responses_endpoint=use_responses_endpoint,
            **(
                {"request_timeout_s": request_timeout_s}
                if request_timeout_s is not None
                else {}
            ),
            **({"retries": retries} if retries is not None else {}),
        )
    if p in ("qwen", "dashscope"):
        from bazaar.agents.llm_backends.qwen import QwenBackend
        return QwenBackend(api_key=api_key)
    if p in ("trapi", "cloudgpt"):
        from bazaar.agents.llm_backends.trapi import TRAPIBackend
        return TRAPIBackend(
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            use_responses_endpoint=use_responses_endpoint,
            **(
                {"request_timeout_s": request_timeout_s}
                if request_timeout_s is not None
                else {}
            ),
            **({"retries": retries} if retries is not None else {}),
        )
    if p in ("foundry", "azure_foundry", "ai_foundry"):
        from bazaar.agents.llm_backends.foundry import FoundryBackend
        return FoundryBackend(
            api_key=api_key,
            base_url=(
                base_url
                or os.environ.get("AZURE_FOUNDRY_BASE_URL")
                or "https://faimas-models.services.ai.azure.com/openai/v1"
            ),
            reasoning_effort=reasoning_effort,
            use_responses_endpoint=use_responses_endpoint,
            **(
                {"request_timeout_s": request_timeout_s}
                if request_timeout_s is not None
                else {}
            ),
            **({"retries": retries} if retries is not None else {}),
        )
    raise ValueError(
        f"unknown LLM provider: {provider!r}. "
        f"Expected one of: ollama, anthropic, openai, qwen, trapi, foundry."
    )


__all__ = [
    "LLMBackend",
    "LLMMessage",
    "LLMResponse",
    "ModelInfo",
    "OllamaBackend",
    "make_backend",
    "probe_ollama",
]

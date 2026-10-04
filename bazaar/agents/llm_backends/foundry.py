"""Azure AI Foundry OpenAI-compatible backend.

Foundry exposes an OpenAI-compatible ``/openai/v1`` surface, but normal
paper runs authenticate with Azure CLI bearer tokens rather than a static
``OPENAI_API_KEY``. This adapter reuses :class:`OpenAIBackend` request and
response parsing while refreshing the Azure token before each request.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from bazaar.agents.llm_backends.openai import OpenAIBackend, OpenAIError

_DEFAULT_BASE_URL = os.environ.get(
    "AZURE_FOUNDRY_BASE_URL",
    "https://faimas-models.services.ai.azure.com/openai/v1",
)
_DEFAULT_AUTH_MODE = os.environ.get("AZURE_FOUNDRY_AUTH_MODE", "azure_cli")
_FOUNDRY_SCOPE = os.environ.get("AZURE_FOUNDRY_SCOPE", "https://ai.azure.com/.default")
_TOKEN_PROVIDER_CACHE: dict[tuple[str, str], Callable[[], str]] = {}


class FoundryError(OpenAIError):
    """Raised when Azure AI Foundry auth or generation fails."""


@dataclass
class FoundryBackend(OpenAIBackend):
    """OpenAI-compatible backend for Azure AI Foundry deployments."""

    base_url: str = _DEFAULT_BASE_URL
    auth_mode: str = _DEFAULT_AUTH_MODE
    token_provider: Callable[[], str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        explicit_api_key = self.api_key
        self.base_url = (
            self.base_url
            or os.environ.get("AZURE_FOUNDRY_BASE_URL")
            or _DEFAULT_BASE_URL
        )
        self.auth_mode = (self.auth_mode or _DEFAULT_AUTH_MODE).strip().lower()
        if self.auth_mode not in {"azure_cli", "managed_identity", "chained", "auto"}:
            raise FoundryError(
                "auth_mode must be one of: azure_cli, managed_identity, chained, auto"
            )
        self.api_key = (
            self.api_key
            or os.environ.get("AZURE_FOUNDRY_API_KEY")
            or os.environ.get("FOUNDRY_API_KEY")
        )
        self._static_api_key = bool(explicit_api_key or self.api_key)
        if self.request_timeout_s <= 0:
            raise FoundryError("request_timeout_s must be positive")

    def _refresh_api_key(self) -> None:
        if self.api_key:
            return
        provider = self.token_provider or _azure_token_provider(self.auth_mode)
        try:
            self.api_key = provider()
        except Exception as exc:  # noqa: BLE001 - normalize SDK auth errors.
            raise FoundryError(str(exc)) from exc

    def _call(self, **kwargs):
        if not self._static_api_key:
            self.api_key = None
        self._refresh_api_key()
        return super()._call(**kwargs)

    def _call_responses(self, **kwargs):
        if not self._static_api_key:
            self.api_key = None
        self._refresh_api_key()
        return super()._call_responses(**kwargs)


def _azure_token_provider(auth_mode: str) -> Callable[[], str]:
    cache_key = (auth_mode, _FOUNDRY_SCOPE)
    cached = _TOKEN_PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            DefaultAzureCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise FoundryError(
            "Azure AI Foundry auth requires azure-identity. Install with "
            "pip install -e '.[llm]' or set AZURE_FOUNDRY_API_KEY."
        ) from exc

    if auth_mode == "azure_cli":
        identity = AzureCliCredential()
    elif auth_mode == "managed_identity":
        identity = ManagedIdentityCredential()
    elif auth_mode == "chained":
        identity = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
    else:
        identity = DefaultAzureCredential()
    provider = get_bearer_token_provider(identity, _FOUNDRY_SCOPE)
    _TOKEN_PROVIDER_CACHE[cache_key] = provider
    return provider


__all__ = ["FoundryBackend", "FoundryError"]

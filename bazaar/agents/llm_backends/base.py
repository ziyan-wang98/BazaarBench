"""Shared types for LLM backends.

Thin enough that all three providers (Ollama, Anthropic, OpenAI)
can implement the same surface without one provider's quirks
leaking into the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass
class LLMMessage:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    # Timings (all seconds) are set where the backend exposes them;
    # Ollama gives these natively via /api/chat's response body.
    total_s:        float = 0.0
    first_token_s:  float = 0.0
    prompt_tokens:  int = 0
    output_tokens:  int = 0
    model:          str = ""
    raw:            dict[str, Any] = field(default_factory=dict)
    # Native tool-calls when the backend exposes them (Ollama /api/chat
    # with a tool-using model, Anthropic tool_use blocks, OpenAI
    # tool_calls). Normalised to the OpenAI shape so LLMPolicy can
    # consume all three backends uniformly:
    #     [{"function": {"name": str, "arguments": dict | str}}]
    # ``None`` means the backend did not emit tool calls — LLMPolicy
    # falls back to parsing JSON out of ``text``.
    tool_calls:     list[dict[str, Any]] | None = None
    # Reasoning summary text when the backend exposes it (OpenAI
    # ``/v1/responses`` ``output[*].type == "reasoning"`` blocks
    # joined). ``None`` for backends/endpoints that don't surface
    # chain-of-thought summaries (chat/completions, Ollama, Anthropic
    # pre-extended-thinking). Persisted in ``llm_calls.reasoning_summary``
    # for live-viz + post-hoc audit.
    reasoning_summary: str | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.total_s <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / self.total_s


@dataclass
class ModelInfo:
    name:         str            # Ollama tag, e.g. "qwen2.5:7b"
    size_bytes:   int = 0        # local weight size
    parameter_b:  float = 0.0    # parameter count in billions
    family:       str = ""       # best-effort (llama / qwen / …)


class LLMBackend(Protocol):
    """Everything LLMPolicy needs from a provider."""

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
        """Return the model's reply for a short chat exchange.

        ``tools`` — optional OpenAI-style tool list
        (``[{"type": "function", "function": {...}}, ...]``). When
        provided and the model supports native tool-calling, the
        response's ``tool_calls`` field is populated. ``None``
        preserves the legacy text-only behaviour.
        """
        ...

    def list_models(self) -> list[ModelInfo]:
        """Inventory of models this backend can serve right now."""
        ...

"""Unit tests for AnthropicBackend + OpenAIBackend + make_backend
(T28d-1 / T28d-2 / T28d-3).

urllib is mocked so tests are fast, offline, and don't require API
keys. We assert the request payloads (system blocks, cache_control,
OpenAI role normalisation) and that responses are surfaced through
the uniform LLMResponse shape.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from bazaar.agents.llm_backends import make_backend
from bazaar.agents.llm_backends import trapi as trapi_mod
from bazaar.agents.llm_backends.anthropic import (
    AnthropicBackend,
    AnthropicError,
)
from bazaar.agents.llm_backends.base import LLMMessage
from bazaar.agents.llm_backends.foundry import FoundryBackend
from bazaar.agents.llm_backends.openai import (
    OpenAIBackend,
    OpenAIError,
    _is_reasoning_model,
)
from bazaar.agents.llm_backends.qwen import QwenBackend, QwenError
from bazaar.agents.llm_backends.trapi import (
    TRAPIBackend,
    TRAPIProviderFailFastError,
    _is_provider_fail_fast_error,
    _postpone_request_start,
    _retry_delay_s,
    _throttle_request_start,
    trapi_endpoint,
    trapi_model_name,
    trapi_supports_responses,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")
    def read(self): return self._payload
    def __enter__(self): return self
    def __exit__(self, *exc): return False


@contextmanager
def _fake_urlopen(responses, capture=None):
    from bazaar.agents.llm_backends import openai as openai_mod
    from bazaar.agents.llm_backends import qwen as qwen_mod
    from bazaar.agents.llm_backends import trapi as trapi_mod

    queue = list(responses)

    def _capture(url, headers, payload):
        if capture is not None:
            capture.append({
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
            })

    def _open(req, timeout=None):
        if capture is not None:
            body = req.data.decode("utf-8") if req.data else ""
            capture.append({
                "url": req.full_url,
                "headers": dict(req.headers),
                "payload": json.loads(body) if body else {},
            })
        if not queue:
            import urllib.error
            raise urllib.error.URLError("queue exhausted")
        return _FakeResp(queue.pop(0))

    def _request_json(url, *, payload, headers, timeout):
        _capture(url, headers, payload)
        if not queue:
            raise OpenAIError("queue exhausted")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with (
        patch("urllib.request.urlopen", _open),
        patch.object(openai_mod, "_request_json", _request_json),
        patch.object(qwen_mod, "_request_json", _request_json),
        patch.object(trapi_mod, "_request_json", _request_json),
    ):
        yield


# -------------------- AnthropicBackend -----------------------------------


def test_anthropic_requires_api_key_envvar(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AnthropicError):
        AnthropicBackend()


def test_anthropic_generate_sends_cached_system_block(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello back"}],
            "usage": {"input_tokens": 42, "output_tokens": 3,
                      "cache_read_input_tokens": 40,
                      "cache_creation_input_tokens": 0},
        },
    ], capture=captured):
        backend = AnthropicBackend()
        resp = backend.generate(
            [
                LLMMessage("system", "persona block"),
                LLMMessage("user",   "do a thing"),
            ],
            model="claude-haiku-4-5",
            max_tokens=64, temperature=0.2,
        )
    assert resp.text == "hello back"
    assert resp.prompt_tokens == 42
    assert resp.output_tokens == 3
    assert resp.raw["cache_read_input_tokens"] == 40

    # The request must send the system block as a cached array.
    sent = captured[0]
    assert sent["url"].endswith("/v1/messages")
    assert sent["headers"].get("X-api-key") == "sk-test"
    assert sent["headers"].get("Anthropic-version")
    assert isinstance(sent["payload"]["system"], list)
    sys_block = sent["payload"]["system"][0]
    assert sys_block["type"] == "text"
    assert sys_block["text"] == "persona block"
    assert sys_block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_disables_cache_when_asked(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {"id": "msg_2", "content": [{"type": "text", "text": "ok"}],
         "usage": {"input_tokens": 1, "output_tokens": 1}},
    ], capture=captured):
        backend = AnthropicBackend(cache_system_prompt=False)
        backend.generate(
            [LLMMessage("system", "hi"), LLMMessage("user", "u")],
            model="claude-haiku-4-5",
        )
    sent = captured[0]
    assert isinstance(sent["payload"]["system"], str)


def test_anthropic_generate_extracts_tool_use_blocks(monkeypatch) -> None:
    """When tools= is passed, we must (a) convert OpenAI-style tool
    specs into Anthropic's ``input_schema`` format on the request,
    and (b) surface ``tool_use`` blocks from the response on
    ``LLMResponse.tool_calls`` in the normalised OpenAI shape."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "msg_tool",
            "content": [
                {"type": "text", "text": "picking a tool"},
                {"type": "tool_use", "id": "tu_1",
                 "name": "search",
                 "input": {"query": "bike"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    ], capture=captured):
        backend = AnthropicBackend()
        resp = backend.generate(
            [LLMMessage("system", "sys"),
             LLMMessage("user",   "find me one")],
            model="claude-haiku-4-5",
            tools=[{"type": "function", "function": {
                "name": "search",
                "description": "search listings",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }}],
        )
    # Request should translate OpenAI shape → Anthropic shape.
    sent = captured[0]
    tools = sent["payload"]["tools"]
    assert tools[0]["name"] == "search"
    assert tools[0]["input_schema"]["required"] == ["query"]
    assert "type" not in tools[0]  # Anthropic doesn't want the wrapper

    # Response: tool_use block surfaces on tool_calls.
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["function"]["name"] == "search"
    assert resp.tool_calls[0]["function"]["arguments"] == {"query": "bike"}


def test_anthropic_http_error_raises(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import urllib.error
    def _open(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "rate-limited", {},
                                     _IoStream(b'{"error":"slow down"}'))
    with patch("urllib.request.urlopen", _open), \
         pytest.raises(AnthropicError):
        AnthropicBackend(retries=0).generate(
            [LLMMessage("user", "hi")], model="claude-haiku-4-5",
        )


class _IoStream:
    def __init__(self, b): self._b = b
    def read(self): return self._b
    def close(self): pass


# -------------------- OpenAIBackend --------------------------------------


def test_openai_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OpenAIError):
        OpenAIBackend()


def test_openai_generate_roundtrip(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_1",
            "choices": [
                {"message": {"role": "assistant", "content": "oc reply"}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        },
    ], capture=captured):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("system", "sys txt"),
             LLMMessage("user",   "u txt")],
            model="gpt-4o-mini", max_tokens=50, temperature=0.5,
        )
    assert resp.text == "oc reply"
    assert resp.prompt_tokens == 20
    sent = captured[0]
    assert sent["url"].endswith("/v1/chat/completions")
    assert sent["headers"]["Authorization"].startswith("Bearer sk-test")
    roles = [m["role"] for m in sent["payload"]["messages"]]
    assert roles[0] == "system"
    assert "sys txt" in sent["payload"]["messages"][0]["content"]


def test_openai_retries_transient_rate_limit(monkeypatch) -> None:
    from bazaar.agents.llm_backends import openai as openai_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr(openai_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    captured: list[dict] = []
    with _fake_urlopen([
        OpenAIError("429: temporarily rate-limited upstream"),
        {
            "id": "chatcmpl_retry",
            "choices": [
                {"message": {"role": "assistant", "content": "ok after retry"}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    ], capture=captured):
        backend = OpenAIBackend(retries=2)
        resp = backend.generate(
            [LLMMessage("user", "retry please")],
            model="gpt-4o-mini",
        )

    assert resp.text == "ok after retry"
    assert len(captured) == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 1.0


def test_openai_does_not_retry_invalid_api_key(monkeypatch) -> None:
    from bazaar.agents.llm_backends import openai as openai_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr(openai_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    captured: list[dict] = []
    with _fake_urlopen([
        OpenAIError("401: invalid_api_key: Incorrect API key provided"),
        {
            "id": "chatcmpl_should_not_call",
            "choices": [
                {"message": {"role": "assistant", "content": "bad"}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    ], capture=captured):
        backend = OpenAIBackend(retries=2)
        with pytest.raises(OpenAIError, match="invalid_api_key"):
            backend.generate(
                [LLMMessage("user", "do not retry auth")],
                model="gpt-4o-mini",
            )

    assert len(captured) == 1
    assert sleeps == []


def test_openai_request_json_enforces_wall_timeout(monkeypatch) -> None:
    from bazaar.agents.llm_backends import openai as openai_mod

    class SlowStreamResponse:
        status_code = 200
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_bytes(self):
            yield b" "
            yield b'{"id":"late"}'

    class SlowClient:
        def stream(self, *args, **kwargs):
            return SlowStreamResponse()

    times = iter([0.0, 2.0])
    monkeypatch.setattr(openai_mod.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(openai_mod, "_get_httpx_client", lambda timeout: SlowClient())

    with pytest.raises(OpenAIError, match="wall timeout"):
        openai_mod._request_json(
            "https://example.test/v1/chat/completions",
            payload={},
            headers={},
            timeout=1.0,
        )


def test_openai_chat_extracts_deepseek_reasoning_content(monkeypatch) -> None:
    """DeepSeek V4 / R1 surface the chain-of-thought as
    ``choices[0].message.reasoning_content``. The OpenAI-compatible
    backend must lift it into ``LLMResponse.reasoning_summary`` so the
    rollout DB records it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "ds_1",
            "choices": [
                {"message": {
                    "role": "assistant",
                    "content": "search",
                    "reasoning_content": (
                        "I need to scan the recommended_listings feed"
                        " before placing any offer."
                    ),
                }}
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 200},
        },
    ], capture=captured):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("system", "sys"), LLMMessage("user", "do something")],
            model="deepseek-v4-pro", max_tokens=4096, temperature=0.4,
        )
    assert resp.text == "search"
    assert resp.reasoning_summary is not None
    assert resp.reasoning_summary.startswith("I need to scan")


def test_openai_chat_extracts_alt_reasoning_field(monkeypatch) -> None:
    """OpenRouter / vLLM proxies sometimes use ``message.reasoning``
    instead of ``reasoning_content``. Lifting the alternate field also
    surfaces a chain-of-thought when present.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with _fake_urlopen([
        {
            "id": "or_1",
            "choices": [
                {"message": {
                    "role": "assistant",
                    "content": "make_offer",
                    "reasoning": "Offer at 90% of asking gives me room to bargain.",
                }}
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 80},
        },
    ]):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("user", "go")],
            model="alt-reasoning-model", max_tokens=4096, temperature=0.4,
        )
    assert resp.reasoning_summary == "Offer at 90% of asking gives me room to bargain."


def test_openai_chat_no_reasoning_field_returns_none(monkeypatch) -> None:
    """Plain non-reasoning responses must not gain a fake
    reasoning_summary; the field should stay ``None``.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with _fake_urlopen([
        {
            "id": "x",
            "choices": [
                {"message": {"role": "assistant", "content": "pong"}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        },
    ]):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("user", "ping")],
            model="gpt-4o-mini", max_tokens=10, temperature=0.7,
        )
    assert resp.reasoning_summary is None


def test_openai_generate_extracts_tool_calls(monkeypatch) -> None:
    """tools= is forwarded unchanged; the response's tool_calls
    (OpenAI returns arguments as a stringified JSON) are parsed
    into dicts on ``LLMResponse.tool_calls``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_tool",
            "choices": [{
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "bike"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        },
    ], capture=captured):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("user", "find me a bike")],
            model="gpt-4o-mini",
            tools=[{"type": "function", "function": {
                "name": "search",
                "description": "search listings",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }}],
        )
    sent = captured[0]
    # tools forwarded verbatim (OpenAI accepts the native shape).
    assert sent["payload"]["tools"][0]["function"]["name"] == "search"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["function"]["name"] == "search"
    assert resp.tool_calls[0]["function"]["arguments"] == {"query": "bike"}


# -------------------- make_backend factory -------------------------------


def test_make_backend_routes_by_provider(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY",    "sk-b")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-q")
    a = make_backend("anthropic")
    o = make_backend("openai")
    q = make_backend("qwen")
    t = make_backend("trapi")
    f = make_backend("foundry", api_key="sk-foundry")
    assert isinstance(a, AnthropicBackend)
    assert isinstance(o, OpenAIBackend)
    assert isinstance(q, QwenBackend)
    assert isinstance(t, TRAPIBackend)
    assert isinstance(f, FoundryBackend)


def test_make_backend_openai_respects_base_url_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    backend = make_backend("openai")

    assert isinstance(backend, OpenAIBackend)
    assert backend.base_url == "https://openrouter.ai/api/v1"


def test_foundry_backend_uses_bearer_token_with_responses(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "resp_foundry",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "choose the wait tool"}
                    ],
                },
                {
                    "type": "function_call",
                    "name": "do_nothing",
                    "arguments": "{}",
                    "call_id": "call_1",
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ], capture=captured):
        backend = FoundryBackend(
            base_url="https://models.example.azure.com/openai/v1",
            token_provider=lambda: "tok-foundry",
            use_responses_endpoint=True,
        )
        resp = backend.generate(
            [LLMMessage("user", "pick one action")],
            model="gpt-5.4-mini",
            tools=[{
                "type": "function",
                "function": {
                    "name": "do_nothing",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }],
            reasoning_effort="high",
        )

    assert captured[0]["url"].endswith("/openai/v1/responses")
    assert captured[0]["headers"]["Authorization"] == "Bearer tok-foundry"
    assert captured[0]["payload"]["reasoning"]["effort"] == "high"
    assert resp.reasoning_summary == "choose the wait tool"


def test_make_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make_backend("cohere")


# -------------------- TRAPIBackend ---------------------------------------


def test_trapi_alias_endpoint_and_capability_helpers() -> None:
    assert trapi_model_name("GPT_54_MINI") == "gpt-5.4-mini_2026-03-17"
    assert trapi_model_name("QWEN_35_397B") == "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
    assert trapi_endpoint("region-b/shared") == (
        "https://research-gateway.example.com/region-b/shared/openai/v1/"
    )
    assert trapi_supports_responses("gpt-5.4_2026-03-05") is False
    assert trapi_supports_responses("gpt-5.5_2026-04-24") is True
    assert trapi_supports_responses("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4") is False


def test_trapi_chat_only_model_uses_chat_completions(monkeypatch) -> None:
    captured: list[dict] = []
    with _fake_urlopen(
        [
            {
                "id": "chatcmpl_trapi_qwen",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"do_nothing","arguments":{}}',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ],
        capture=captured,
    ):
        backend = TRAPIBackend(
            token_provider=lambda: "tok-test",
            use_responses_endpoint=True,
            retries=0,
        )
        resp = backend.generate(
            [LLMMessage("user", "use a tool")],
            model="QWEN_35_397B",
            tools=[{"type": "function", "function": {
                "name": "do_nothing",
                "description": "no action",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )

    sent = captured[0]
    assert sent["url"].endswith("/openai/v1/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer tok-test"
    assert sent["payload"]["model"] == "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
    assert "tools" not in sent["payload"]
    assert "tool_choice" not in sent["payload"]
    assert "Return tool use as plain JSON text" in sent["payload"]["messages"][-1]["content"]
    assert resp.text == '{"action":"do_nothing","arguments":{}}'
    assert resp.raw["_trapi"]["effective_responses_endpoint"] is False
    assert resp.raw["_trapi"]["text_tool_fallback"] is True


def test_trapi_responses_model_uses_responses_endpoint(monkeypatch) -> None:
    captured: list[dict] = []
    with _fake_urlopen(
        [
            {
                "id": "resp_trapi",
                "output": [
                    {
                        "type": "function_call",
                        "name": "do_nothing",
                        "arguments": "{}",
                        "call_id": "call_1",
                    }
                ],
                "usage": {"input_tokens": 9, "output_tokens": 3},
            }
        ],
        capture=captured,
    ):
        backend = TRAPIBackend(
            token_provider=lambda: "tok-test",
            use_responses_endpoint=True,
            retries=0,
        )
        resp = backend.generate(
            [LLMMessage("user", "use a tool")],
            model="gpt-5.5_2026-04-24",
            max_tokens=128,
            reasoning_effort="high",
            tools=[{"type": "function", "function": {
                "name": "do_nothing",
                "description": "no action",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )

    sent = captured[0]
    payload = sent["payload"]
    assert sent["url"].endswith("/openai/v1/responses")
    assert payload["model"] == "gpt-5.5_2026-04-24"
    assert payload["max_output_tokens"] == 128
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert payload["tool_choice"] == "required"
    assert payload["tools"][0]["name"] == "do_nothing"
    assert "function" not in payload["tools"][0]
    assert resp.tool_calls is not None
    assert resp.prompt_tokens == 9
    assert resp.raw["_trapi"]["effective_responses_endpoint"] is True


def test_trapi_gpt_oss_omits_required_tool_choice(monkeypatch) -> None:
    captured: list[dict] = []
    with _fake_urlopen(
        [
            {
                "id": "chatcmpl_trapi_oss",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "do_nothing",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ],
        capture=captured,
    ):
        backend = TRAPIBackend(token_provider=lambda: "tok-test", retries=0)
        resp = backend.generate(
            [LLMMessage("user", "use a tool")],
            model="gpt-oss-120b_1",
            tools=[{"type": "function", "function": {
                "name": "do_nothing",
                "description": "no action",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )

    payload = captured[0]["payload"]
    assert payload["model"] == "gpt-oss-120b_1"
    assert "tools" in payload
    assert payload["parallel_tool_calls"] is False
    assert "tool_choice" not in payload
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["function"]["name"] == "do_nothing"


def test_trapi_retry_delay_honors_retry_after_text() -> None:
    with patch(
        "bazaar.agents.llm_backends.trapi.random.uniform",
        return_value=0.5,
    ):
        delay = _retry_delay_s(
            OpenAIError(
                '429: { "message": "Rate limit is exceeded. Try again in 31 seconds." }'
            ),
            attempt=0,
        )
    assert delay == 32.5


def test_trapi_retry_delay_treats_424_remote_reset_as_transient() -> None:
    with patch(
        "bazaar.agents.llm_backends.trapi.random.uniform",
        return_value=0.5,
    ):
        delay = _retry_delay_s(
            OpenAIError(
                "424: upstream connect error or disconnect/reset before headers. "
                "retried and the latest reset reason: remote reset"
            ),
            attempt=2,
        )
    assert delay == 3.25


def test_trapi_provider_fail_fast_classifies_provider_level_errors() -> None:
    assert _is_provider_fail_fast_error(
        OpenAIError(
            "request failed: [SSL: CERTIFICATE_VERIFY_FAILED] "
            "self-signed certificate in certificate chain"
        )
    )
    assert _is_provider_fail_fast_error(
        OpenAIError(
            '403: {"message": "Your resource has been temporarily blocked '
            'because we detected unusual behavior."}'
        )
    )
    assert _is_provider_fail_fast_error(
        OpenAIError('429: {"error": "Rate Limit Exceeded"}')
    )
    assert _is_provider_fail_fast_error(
        OpenAIError(
            "404: DeploymentNotFound: The API deployment for this resource "
            "does not exist."
        )
    )
    assert not _is_provider_fail_fast_error(OpenAIError("400: malformed request"))


def test_trapi_provider_fail_fast_aborts_before_long_retry_loop(
    monkeypatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setenv("BAZAAR_TRAPI_PROVIDER_FAIL_FAST_ERRORS", "2")
    monkeypatch.setenv("BAZAAR_TRAPI_PROVIDER_FAIL_FAST_WINDOW_S", "300")
    monkeypatch.setenv("BAZAAR_TRAPI_PROVIDER_CIRCUIT_OPEN_S", "600")
    monkeypatch.setattr(trapi_mod.time, "sleep", lambda delay_s: None)
    with trapi_mod._PROVIDER_CIRCUIT_LOCK:
        trapi_mod._PROVIDER_CIRCUIT_STATE.clear()

    with _fake_urlopen(
        [
            OpenAIError('429: {"error": "Rate Limit Exceeded"}'),
            OpenAIError('429: {"error": "Rate Limit Exceeded"}'),
            OpenAIError('429: {"error": "Rate Limit Exceeded"}'),
        ],
        capture=captured,
    ):
        backend = TRAPIBackend(
            token_provider=lambda: "tok-test",
            retries=80,
            use_responses_endpoint=False,
        )
        with pytest.raises(TRAPIProviderFailFastError, match="provider_fail_fast"):
            backend.generate(
                [LLMMessage("user", "hello")],
                model="gpt-5.4_2026-03-05",
            )

    assert len(captured) == 2
    with trapi_mod._PROVIDER_CIRCUIT_LOCK:
        trapi_mod._PROVIDER_CIRCUIT_STATE.clear()


def test_trapi_endpoint_pool_fails_over_provider_errors(monkeypatch) -> None:
    captured: list[dict] = []
    first = "https://research-gateway.example.com/region-c/batch/openai/v1/"
    second = "https://research-gateway.example.com/region-c/shared/openai/v1/"
    monkeypatch.setenv("BAZAAR_TRAPI_BASE_URLS", f"{first},{second}")
    with _fake_urlopen(
        [
            OpenAIError(
                '503: {"code": "all-backends-unhealthy", '
                '"error": "All backends are temporarily unavailable"}'
            ),
            {
                "id": "chatcmpl_trapi_ok",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ],
        capture=captured,
    ):
        backend = TRAPIBackend(
            token_provider=lambda: "tok-test",
            retries=2,
            use_responses_endpoint=False,
        )
        resp = backend.generate(
            [LLMMessage("user", "hello")],
            model="gpt-5.4_2026-03-05",
        )

    assert captured[0]["url"].startswith(first)
    assert captured[1]["url"].startswith(second)
    assert resp.text == "ok"
    assert resp.raw["_trapi"]["trapi_endpoint"] == second


def test_trapi_retry_delay_postpones_shared_request_start(monkeypatch) -> None:
    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()

    monkeypatch.setattr(trapi_mod.time, "monotonic", lambda: 100.0)
    _postpone_request_start(30.0)
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 130.0

    _postpone_request_start(5.0)
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 130.0

    _postpone_request_start(45.0)
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 145.0

    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()


def test_trapi_throttle_rechecks_shared_cooldown(monkeypatch) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)
        now[0] += delay_s

    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()
    monkeypatch.setenv("BAZAAR_TRAPI_MIN_REQUEST_INTERVAL_S", "10")
    monkeypatch.setattr(trapi_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(trapi_mod.time, "sleep", fake_sleep)

    _throttle_request_start()
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 110.0

    _postpone_request_start(40.0)
    now[0] = 110.0
    _throttle_request_start()

    assert sleeps == [30.0]
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 150.0

    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()


def test_trapi_qwen_throttle_does_not_cool_down_other_models(monkeypatch) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)
        now[0] += delay_s

    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()
    monkeypatch.delenv("BAZAAR_TRAPI_MIN_REQUEST_INTERVAL_S", raising=False)
    monkeypatch.setenv("BAZAAR_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S", "10")
    monkeypatch.setattr(trapi_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(trapi_mod.time, "sleep", fake_sleep)

    qwen_model = "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
    _throttle_request_start(qwen_model)
    assert trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY["family:qwen"] == 110.0

    _postpone_request_start(40.0, model=qwen_model)
    assert trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY["family:qwen"] == 140.0

    _throttle_request_start("gpt-5.5_2026-04-24")
    assert sleeps == []
    assert trapi_mod._REQUEST_THROTTLE_NEXT_S == 0.0

    now[0] = 110.0
    _throttle_request_start(qwen_model)
    assert sleeps == [30.0]
    assert trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY["family:qwen"] == 150.0

    with trapi_mod._REQUEST_THROTTLE_LOCK:
        trapi_mod._REQUEST_THROTTLE_NEXT_S = 0.0
        trapi_mod._REQUEST_THROTTLE_NEXT_BY_KEY.clear()


# -------------------- QwenBackend ----------------------------------------


def test_qwen_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(QwenError):
        QwenBackend()


def test_qwen_generate_enables_thinking_and_extracts_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_qwen",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "need to search first",
                    "content": "visible reply",
                },
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        },
    ], capture=captured):
        backend = QwenBackend()
        resp = backend.generate(
            [LLMMessage("system", "sys"), LLMMessage("user", "u")],
            model="qwen3.6-35b-a3b",
            max_tokens=32,
            temperature=0.2,
        )
    assert resp.text == "visible reply"
    assert resp.reasoning_summary == "need to search first"
    assert resp.prompt_tokens == 9
    sent = captured[0]
    assert sent["url"].endswith("/compatible-mode/v1/chat/completions")
    assert sent["headers"]["Authorization"].startswith("Bearer sk-test")
    assert sent["payload"]["enable_thinking"] is True
    assert sent["payload"]["messages"][0]["role"] == "system"


def test_qwen_can_disable_thinking_with_env(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("QWEN_ENABLE_THINKING", "0")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_qwen",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ], capture=captured):
        QwenBackend().generate(
            [LLMMessage("user", "u")],
            model="qwen3.6-35b-a3b",
        )
    assert captured[0]["payload"]["enable_thinking"] is False


def test_qwen_low_effort_disables_thinking_for_reflection(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_qwen",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ], capture=captured):
        QwenBackend().generate(
            [LLMMessage("user", "u")],
            model="qwen3.6-35b-a3b",
            reasoning_effort="low",
        )
    assert captured[0]["payload"]["enable_thinking"] is False


def test_qwen_low_effort_can_keep_thinking_with_env(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("QWEN_THINKING_FOR_LOW_EFFORT", "1")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_qwen",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ], capture=captured):
        QwenBackend().generate(
            [LLMMessage("user", "u")],
            model="qwen3.6-35b-a3b",
            reasoning_effort="low",
        )
    assert captured[0]["payload"]["enable_thinking"] is True


# -------------------- OpenAI reasoning-model branch ----------------------


def test_is_reasoning_model_classification() -> None:
    """gpt-5.x and o1/o3/o4 families classify as reasoning; gpt-4o
    (non-reasoning omni) and gpt-4o-mini do not."""
    assert _is_reasoning_model("gpt-5") is True
    assert _is_reasoning_model("gpt-5.2") is True
    assert _is_reasoning_model("gpt-5.4-pro") is True
    assert _is_reasoning_model("o1-pro") is True
    assert _is_reasoning_model("o1") is True
    assert _is_reasoning_model("o3") is True
    assert _is_reasoning_model("o4-mini") is True
    # Non-reasoning:
    assert _is_reasoning_model("gpt-4o") is False
    assert _is_reasoning_model("gpt-4o-mini") is False
    assert _is_reasoning_model("gpt-4.1") is False


def test_reasoning_model_uses_max_completion_tokens(monkeypatch) -> None:
    """Reasoning models (gpt-5.x, o*) must send
    ``max_completion_tokens`` + ``reasoning_effort`` and must NOT
    send ``max_tokens`` or ``temperature`` — the API rejects the
    latter pair on these models."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_r1",
            "choices": [
                {"message": {"role": "assistant", "content": "rr"}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    ], capture=captured):
        backend = OpenAIBackend()
        backend.generate(
            [LLMMessage("user", "think please")],
            model="gpt-5.2", max_tokens=256, temperature=0.5,
            reasoning_effort="high",
        )
    payload = captured[0]["payload"]
    assert payload["max_completion_tokens"] == 256
    assert payload["reasoning_effort"] == "high"
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_backend_reasoning_effort_defaults_and_override(monkeypatch) -> None:
    """Instance attr `reasoning_effort` is the fallback used when
    `generate()` is called without the kwarg; an explicit kwarg
    overrides it on a per-call basis."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    # Call 1: construct with instance default "high", no kwarg.
    with _fake_urlopen([
        {
            "id": "chatcmpl_rd1",
            "choices": [
                {"message": {"role": "assistant", "content": "a"}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ], capture=captured):
        backend = OpenAIBackend(reasoning_effort="high")
        backend.generate(
            [LLMMessage("user", "x")],
            model="gpt-5.2", max_tokens=32, temperature=0.1,
        )
    assert captured[0]["payload"]["reasoning_effort"] == "high"

    # Call 2: same backend, explicit kwarg "low" overrides instance.
    captured.clear()
    with _fake_urlopen([
        {
            "id": "chatcmpl_rd2",
            "choices": [
                {"message": {"role": "assistant", "content": "b"}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ], capture=captured):
        backend.generate(
            [LLMMessage("user", "x")],
            model="gpt-5.2", max_tokens=32, temperature=0.1,
            reasoning_effort="low",
        )
    assert captured[0]["payload"]["reasoning_effort"] == "low"


def test_non_reasoning_model_uses_max_tokens(monkeypatch) -> None:
    """Non-reasoning models keep the classic ``max_tokens`` +
    ``temperature`` payload and never leak ``reasoning_effort``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_c1",
            "choices": [
                {"message": {"role": "assistant", "content": "cc"}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    ], capture=captured):
        backend = OpenAIBackend()
        backend.generate(
            [LLMMessage("user", "classic")],
            model="gpt-4o", max_tokens=128, temperature=0.3,
            reasoning_effort="high",  # must be ignored for non-reasoning
        )
    payload = captured[0]["payload"]
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.3
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload


# -------------------- /v1/responses endpoint (R11) -----------------------


def _responses_body_with_reasoning_and_tool_call() -> dict:
    """The exact shape /v1/responses returns for a tool-using
    reasoning model (verified by team-lead probe 2026-04-19)."""
    return {
        "id": "resp_1",
        "output": [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text",
                     "text": "Allison has $74 budget; L10 at $65 fits."},
                    {"type": "summary_text",
                     "text": "Deadline is in 3 days, so committing now."},
                ],
            },
            {
                "type": "function_call",
                "name": "make_offer",
                "arguments": '{"listing_id": 10, "price_cents": 6500}',
                "call_id": "call_xyz",
            },
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Offer submitted."},
                ],
            },
        ],
        "usage": {
            "input_tokens": 200,
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 40},
        },
    }


def test_responses_endpoint_routes_when_flag_set(monkeypatch) -> None:
    """When ``use_responses_endpoint=True`` the request goes to
    ``/v1/responses`` (not /v1/chat/completions) and uses the
    responses-shaped payload (input/max_output_tokens/reasoning)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen(
        [_responses_body_with_reasoning_and_tool_call()],
        capture=captured,
    ):
        backend = OpenAIBackend(use_responses_endpoint=True)
        backend.generate(
            [LLMMessage("user", "find me one")],
            model="gpt-5.2", max_tokens=128,
            reasoning_effort="high",
        )
    sent = captured[0]
    assert sent["url"].endswith("/v1/responses")
    payload = sent["payload"]
    assert payload["model"] == "gpt-5.2"
    assert payload["max_output_tokens"] == 128
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}
    # responses uses ``input``, not ``messages``
    assert "input" in payload
    assert "messages" not in payload


def test_responses_endpoint_omits_reasoning_block_for_non_reasoning_model(
    monkeypatch,
) -> None:
    """gpt-4.1-mini / gpt-4o etc. reject ``reasoning.effort`` on
    /v1/responses with 400. Gate the block so the same backend can
    serve both a reasoning main model and a cheap non-reasoning
    reflection model in the same run (R14a)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen(
        [_responses_body_with_reasoning_and_tool_call()],
        capture=captured,
    ):
        backend = OpenAIBackend(use_responses_endpoint=True)
        backend.generate(
            [LLMMessage("user", "reflect briefly")],
            model="gpt-4.1-mini", max_tokens=256,
            reasoning_effort="low",
        )
    payload = captured[0]["payload"]
    assert payload["model"] == "gpt-4.1-mini"
    # Non-reasoning models must not receive the reasoning block.
    assert "reasoning" not in payload
    # Still hits /v1/responses and uses the input key.
    assert captured[0]["url"].endswith("/v1/responses")
    assert "input" in payload


def test_responses_endpoint_sets_max_output_tokens(monkeypatch) -> None:
    """Responses calls should be bounded by the caller max_tokens budget."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen(
        [_responses_body_with_reasoning_and_tool_call()],
        capture=captured,
    ):
        backend = OpenAIBackend(use_responses_endpoint=True)
        backend.generate(
            [LLMMessage("user", "go")],
            model="gpt-5.2", max_tokens=8192,
        )
    assert captured[0]["payload"]["max_output_tokens"] == 8192


def test_responses_endpoint_parses_reasoning_summary(monkeypatch) -> None:
    """Reasoning blocks are joined into LLMResponse.reasoning_summary
    and function_call blocks are normalised onto tool_calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with _fake_urlopen([_responses_body_with_reasoning_and_tool_call()]):
        backend = OpenAIBackend(use_responses_endpoint=True)
        resp = backend.generate(
            [LLMMessage("user", "go")],
            model="gpt-5.2", max_tokens=128,
        )
    assert resp.reasoning_summary is not None
    assert "Allison has $74" in resp.reasoning_summary
    assert "Deadline is in 3 days" in resp.reasoning_summary
    # Both summary chunks joined with a blank line
    assert "\n\n" in resp.reasoning_summary
    # tool_calls in the unified chat-shape
    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc["function"]["name"] == "make_offer"
    assert tc["function"]["arguments"] == {"listing_id": 10, "price_cents": 6500}
    # message text also surfaced
    assert resp.text == "Offer submitted."
    # token counts read from input_tokens/output_tokens
    assert resp.prompt_tokens == 200
    assert resp.output_tokens == 60


def test_responses_endpoint_tool_format_conversion(monkeypatch) -> None:
    """Chat-shaped tools (``{type, function: {name,...}}``) get
    flattened to responses-shape (no nested ``function`` key)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen(
        [_responses_body_with_reasoning_and_tool_call()],
        capture=captured,
    ):
        backend = OpenAIBackend(use_responses_endpoint=True)
        backend.generate(
            [LLMMessage("user", "go")],
            model="gpt-5.2", max_tokens=64,
            tools=[{"type": "function", "function": {
                "name": "make_offer",
                "description": "buy",
                "parameters": {
                    "type": "object",
                    "properties": {"listing_id": {"type": "integer"}},
                    "required": ["listing_id"],
                },
            }}],
        )
    sent_tools = captured[0]["payload"]["tools"]
    assert len(sent_tools) == 1
    t = sent_tools[0]
    assert t["type"] == "function"
    assert t["name"] == "make_offer"
    assert t["description"] == "buy"
    assert t["parameters"]["required"] == ["listing_id"]
    # No nested ``function`` key on the responses shape
    assert "function" not in t
    # Tool-enabled responses calls pin tool_choice="required" so the
    # model can't skip emitting an action.
    assert captured[0]["payload"]["tool_choice"] == "required"


def test_responses_endpoint_omits_tool_choice_when_no_tools(monkeypatch) -> None:
    """Pure-text responses calls (no tools) must not set tool_choice —
    tool_choice=required with an empty tool list is invalid."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen(
        [_responses_body_with_reasoning_and_tool_call()],
        capture=captured,
    ):
        backend = OpenAIBackend(use_responses_endpoint=True)
        backend.generate(
            [LLMMessage("user", "just chat")],
            model="gpt-5.2", max_tokens=64,
        )
    payload = captured[0]["payload"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_chat_completions_endpoint_unchanged_when_flag_off(monkeypatch) -> None:
    """Default path (``use_responses_endpoint=False``) still hits
    /v1/chat/completions with chat-shaped payload — backward compat."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_legacy",
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        },
    ], capture=captured):
        backend = OpenAIBackend()  # default: chat/completions
        resp = backend.generate(
            [LLMMessage("user", "ping")],
            model="gpt-4o-mini", max_tokens=8, temperature=0.1,
        )
    assert captured[0]["url"].endswith("/v1/chat/completions")
    assert "input" not in captured[0]["payload"]
    assert "messages" in captured[0]["payload"]
    # No reasoning summary on chat/completions — None default preserved
    assert resp.reasoning_summary is None


def test_reasoning_chat_empty_tool_response_falls_back_to_responses(
    monkeypatch,
) -> None:
    """Reasoning-model chat/completions can return no content/tool call.
    With tools present, the backend retries through /v1/responses where
    tool_choice=required enforces the action contract."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict] = []
    with _fake_urlopen([
        {
            "id": "chatcmpl_empty",
            "choices": [
                {"message": {"role": "assistant", "content": ""}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        },
        _responses_body_with_reasoning_and_tool_call(),
    ], capture=captured):
        backend = OpenAIBackend()
        resp = backend.generate(
            [LLMMessage("user", "act")],
            model="gpt-5.2",
            max_tokens=128,
            reasoning_effort="high",
            tools=[{"type": "function", "function": {
                "name": "make_offer",
                "description": "buy",
                "parameters": {
                    "type": "object",
                    "properties": {"listing_id": {"type": "integer"}},
                    "required": ["listing_id"],
                },
            }}],
        )

    assert captured[0]["url"].endswith("/v1/chat/completions")
    assert captured[1]["url"].endswith("/v1/responses")
    assert captured[1]["payload"]["tool_choice"] == "required"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["function"]["name"] == "make_offer"
    assert resp.reasoning_summary is not None


def test_make_backend_threads_responses_flag(monkeypatch) -> None:
    """``make_backend("openai", use_responses_endpoint=True)`` reaches
    OpenAIBackend.use_responses_endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    b = make_backend("openai", use_responses_endpoint=True)
    assert isinstance(b, OpenAIBackend)
    assert b.use_responses_endpoint is True
    # Default still False
    b2 = make_backend("openai")
    assert b2.use_responses_endpoint is False


def test_make_backend_threads_openai_timeout(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    backend = make_backend("openai", request_timeout_s=45.0, retries=1)

    assert isinstance(backend, OpenAIBackend)
    assert backend.request_timeout_s == 45.0
    assert backend.retries == 1

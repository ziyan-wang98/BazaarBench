"""Unit tests for the Ollama LLM backend (T28b).

No live Ollama is assumed — urllib.request.urlopen is mocked so
tests are fast, offline, and portable.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from bazaar.agents.llm_backends import probe_ollama
from bazaar.agents.llm_backends.base import LLMMessage
from bazaar.agents.llm_backends.ollama import (
    OllamaBackend,
    OllamaError,
    _messages_to_prompt,
    _try_parse_json,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")
    def read(self): return self._payload
    def __enter__(self): return self
    def __exit__(self, *exc): return False


@contextmanager
def _fake_urlopen(responses):
    """Patch urlopen with a queue of JSON payloads. Each call pops
    the next one; URLError raised when exhausted."""
    queue = list(responses)
    def _open(req, timeout=None):
        if not queue:
            import urllib.error
            raise urllib.error.URLError("queue exhausted")
        return _FakeResp(queue.pop(0))
    with patch("urllib.request.urlopen", _open):
        yield


# ---- prompt flattening ---------------------------------------------------


def test_messages_to_prompt_preserves_role_tags():
    out = _messages_to_prompt([
        LLMMessage("system", "sys text"),
        LLMMessage("user",   "hello"),
    ])
    assert "[SYSTEM]" in out
    assert "sys text" in out
    assert "[USER]" in out
    # Trailing assistant cue so the model continues as the assistant.
    assert out.strip().endswith("[ASSISTANT]")


# ---- JSON parse probe ---------------------------------------------------


def test_try_parse_json_accepts_pure_object():
    assert _try_parse_json('{"action": "search", "reason": "x"}') is True


def test_try_parse_json_accepts_object_with_surrounding_prose():
    text = "Sure, here is my decision:\n{\"action\":\"view_listing\"}\nok"
    assert _try_parse_json(text) is True


def test_try_parse_json_rejects_when_no_action_field():
    assert _try_parse_json('{"reason": "no action key"}') is False


def test_try_parse_json_rejects_malformed():
    assert _try_parse_json("totally not json {action: bare}") is False


# ---- list_models --------------------------------------------------------


def test_list_models_parses_tags_response():
    with _fake_urlopen([
        {"models": [
            {"name": "llama3.2:3b", "size": 2_100_000_000,
             "details": {"family": "llama", "parameter_size": "3.2B"}},
            {"name": "qwen2.5:7b", "size": 4_700_000_000,
             "details": {"family": "qwen", "parameter_size": "7.0B"}},
        ]},
    ]):
        models = OllamaBackend("http://test").list_models()
    assert [m.name for m in models] == ["llama3.2:3b", "qwen2.5:7b"]
    assert models[1].parameter_b == 7.0
    assert models[0].family == "llama"


# ---- generate -----------------------------------------------------------


def test_generate_returns_structured_response():
    with _fake_urlopen([
        {
            "message": {
                "role": "assistant",
                "content":
                    "{\"action\": \"search\", \"reason\": \"want a bike\"}",
            },
            "total_duration":    1_200_000_000,
            "load_duration":      100_000_000,
            "prompt_eval_count": 250,
            "eval_count":        30,
            "eval_duration":     900_000_000,
        },
    ]):
        resp = OllamaBackend("http://test").generate(
            [LLMMessage("user", "hi")], model="llama3.2:3b",
        )
    assert "action" in resp.text
    assert resp.prompt_tokens == 250
    assert resp.output_tokens == 30
    assert resp.tokens_per_second > 0
    assert resp.tool_calls is None


def test_generate_with_tools_extracts_tool_calls():
    """Ollama /api/chat with a tool-aware model returns the tool
    call on ``message.tool_calls``. LLMResponse surfaces it so
    LLMPolicy can skip its JSON-parsing fallback."""
    with _fake_urlopen([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {
                        "name": "search",
                        "arguments": {"query": "bike"},
                    }},
                ],
            },
            "total_duration":    500_000_000,
            "prompt_eval_count": 120,
            "eval_count":        15,
            "eval_duration":     300_000_000,
        },
    ]):
        resp = OllamaBackend("http://test").generate(
            [LLMMessage("user", "hi")],
            model="llama3.1:8b",
            tools=[{"type": "function", "function": {
                "name": "search",
                "description": "search the marketplace",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
            }}],
        )
    assert resp.text == ""
    assert resp.tool_calls is not None
    assert resp.tool_calls[0]["function"]["name"] == "search"
    assert resp.tool_calls[0]["function"]["arguments"] == {"query": "bike"}


def test_generate_raises_ollama_error_when_unreachable():
    # Empty queue → URLError raised → OllamaError.
    with _fake_urlopen([]), pytest.raises(OllamaError):
        OllamaBackend("http://test").generate(
            [LLMMessage("user", "hi")], model="x",
        )


# ---- probe_ollama end-to-end --------------------------------------------


def test_probe_ollama_reports_unreachable_cleanly():
    with _fake_urlopen([]):
        report = probe_ollama(host="http://test")
    assert report.reachable is False
    assert report.error and "Ollama" in report.error
    assert report.probes == []


def test_probe_ollama_benchmarks_each_listed_model():
    with _fake_urlopen([
        # /api/tags — reachability check
        {"models": [{"name": "m:1b",
                     "size": 800_000_000,
                     "details": {"family": "llama", "parameter_size": "1.0B"}}]},
        # /api/tags — actual list call
        {"models": [{"name": "m:1b",
                     "size": 800_000_000,
                     "details": {"family": "llama", "parameter_size": "1.0B"}}]},
        # /api/chat — one probe per listed model
        {"message": {"role": "assistant",
                     "content": '{"action":"do_nothing","reason":"ok"}'},
         "total_duration": 1_000_000_000, "load_duration": 0,
         "prompt_eval_count": 10, "eval_count": 10,
         "eval_duration": 500_000_000},
    ]):
        report = probe_ollama(host="http://test")
    assert report.reachable is True
    assert len(report.probes) == 1
    p = report.probes[0]
    assert p.ok is True
    assert p.json_parse_ok is True
    assert p.tokens_per_s > 0

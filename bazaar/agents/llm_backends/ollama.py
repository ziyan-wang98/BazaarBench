"""Ollama HTTP client — zero-dep stdlib `urllib`.

Works against a local Ollama daemon at ``http://localhost:11434``
(or whatever ``OLLAMA_HOST`` is set to). No external dependency is
added so the core ``bazaar-bench`` package stays lean — LLM
infrastructure is opt-in via the ``[llm]`` extra.

Ollama API reference (the subset we care about):

  GET  /api/tags                → list installed models
  POST /api/chat                → chat-style completion with native
                                    role awareness. Body:
                                      model      (string)
                                      messages   ([{role, content}, ...])
                                      tools      (optional OpenAI-style)
                                      stream     (bool)
                                      options    (sampling dict)
                                    Response body mirrors the OpenAI
                                    convention — ``message.role``,
                                    ``message.content``,
                                    ``message.tool_calls`` — plus the
                                    same timing fields /api/generate
                                    exposed (``total_duration``,
                                    ``load_duration``,
                                    ``prompt_eval_count``,
                                    ``eval_count``,
                                    ``eval_duration``).

T9 / R3 note: we used to call ``/api/generate`` with a flattened
single-string prompt. That path couldn't carry ``tools`` through
to the model, so smaller models would hallucinate tool-call JSON
instead of using the native tool-calling channel. ``/api/chat``
takes the messages list unmodified and passes ``tools`` through
to the serving runtime, which lets tool-aware models (llama3.1+,
qwen2.5+) emit proper ``message.tool_calls``. The legacy
``_messages_to_prompt`` helper is kept as an export for
backwards compatibility with callers that inspect it — it is no
longer on the request path.

Prober wrapper :func:`probe_ollama` does install + reachability
detection, benchmarks each available model, and returns a
structured report the CLI renders.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from bazaar.agents.llm_backends.base import (
    LLMMessage,
    LLMResponse,
    ModelInfo,
)

_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Keep timeouts generous — local 9B-ish models at small contexts
# typically reply within 20 s on Apple Silicon, but a cold model
# load adds up to 15 s on first call.
_PING_TIMEOUT_S = 2.0
_GENERATE_TIMEOUT_S = 120.0


class OllamaError(RuntimeError):
    """Raised when the daemon is unreachable or returns an error."""


def _request_json(url: str, *, payload: dict[str, Any] | None = None,
                  timeout: float = 30.0) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"request failed: {url}: {exc}") from exc


def _messages_to_prompt(messages: list[LLMMessage]) -> str:
    """Flatten chat messages into a single-string prompt.

    .. deprecated:: T9 / R3-T1
       Kept as an export for compatibility with older callers and
       tests that inspect the legacy prompt-flattening path. The
       :class:`OllamaBackend` now uses ``/api/chat`` with the
       structured messages list and does not call this helper.
    """
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"[SYSTEM]\n{m.content}")
        elif m.role == "user":
            parts.append(f"[USER]\n{m.content}")
        else:
            parts.append(f"[ASSISTANT]\n{m.content}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


@dataclass
class OllamaBackend:
    """Thin adapter implementing the :class:`LLMBackend` Protocol."""
    host: str = _DEFAULT_HOST

    def is_reachable(self) -> bool:
        try:
            _request_json(f"{self.host}/api/tags", timeout=_PING_TIMEOUT_S)
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[ModelInfo]:
        data = _request_json(f"{self.host}/api/tags", timeout=_PING_TIMEOUT_S)
        out: list[ModelInfo] = []
        for m in data.get("models", []) or []:
            name = m.get("name") or m.get("model") or "?"
            size = int(m.get("size", 0) or 0)
            details = m.get("details") or {}
            family = (details.get("family")
                      or details.get("families", [""])[0]
                      or name.split(":")[0])
            # Ollama reports parameter size as strings like "7.0B".
            pstr = str(details.get("parameter_size", "") or "").upper()
            p_b = 0.0
            if pstr.endswith("B"):
                try:
                    p_b = float(pstr[:-1])
                except ValueError:
                    p_b = 0.0
            out.append(ModelInfo(name=name, size_bytes=size,
                                 parameter_b=p_b, family=family))
        out.sort(key=lambda x: x.parameter_b)
        return out

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        start = time.monotonic()
        body = _request_json(
            f"{self.host}/api/chat",
            payload=payload,
            timeout=_GENERATE_TIMEOUT_S,
        )
        total_s = time.monotonic() - start
        load_s = (body.get("load_duration") or 0) / 1e9
        eval_s = (body.get("eval_duration") or 0) / 1e9
        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        output_tokens = int(body.get("eval_count") or 0)
        # Ollama's first-token latency = total - eval_duration.
        first_token_s = max(0.0, total_s - eval_s)
        msg = body.get("message") or {}
        text = str(msg.get("content") or "").strip()
        raw_tool_calls = msg.get("tool_calls")
        tool_calls: list[dict[str, Any]] | None = None
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls = [tc for tc in raw_tool_calls
                          if isinstance(tc, dict)]
        return LLMResponse(
            text=text,
            total_s=total_s,
            first_token_s=first_token_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=model,
            raw={"load_s": load_s, "eval_s": eval_s, **body},
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


@dataclass
class ModelProbe:
    model:         ModelInfo
    ok:            bool = False
    error:         str | None = None
    total_s:       float = 0.0
    first_token_s: float = 0.0
    tokens_per_s:  float = 0.0
    json_parse_ok: bool = False
    preview:       str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model":          dataclasses.asdict(self.model),
            "ok":             self.ok,
            "error":          self.error,
            "total_s":        self.total_s,
            "first_token_s":  self.first_token_s,
            "tokens_per_s":   self.tokens_per_s,
            "json_parse_ok":  self.json_parse_ok,
            "preview":        self.preview,
        }


@dataclass
class OllamaProbeReport:
    reachable: bool
    host:      str
    error:     str | None = None
    probes:    list[ModelProbe] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "host":      self.host,
            "error":     self.error,
            "probes":    [p.to_dict() for p in self.probes],
        }


# Prompt that stresses the exact shape LLMPolicy will require of the
# model: short persona + strict-JSON action output.
_PROBE_SYSTEM = (
    "You are a C2C marketplace buyer. Reply with a single JSON object, "
    "no code fences, no prose outside the object. Fields: "
    '`"action"` (one of "search", "view_listing", "do_nothing") and '
    '`"reason"` (a sentence ≤ 20 words).'
)
_PROBE_USER = (
    "You are Susan W., age 36, ZIP 94110. You want to buy a used bike "
    "for ≤ $80. Recent events: none. What do you do this tick?"
)


def _try_parse_json(text: str) -> bool:
    s = text.strip()
    # tolerate fenced or pre/post prose — grab first { .. } block
    try:
        start = s.index("{")
        end = s.rindex("}") + 1
    except ValueError:
        return False
    try:
        obj = json.loads(s[start:end])
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "action" in obj


def probe_ollama(
    *,
    host: str = _DEFAULT_HOST,
    only_models: list[str] | None = None,
    max_models: int = 6,
) -> OllamaProbeReport:
    """Benchmark every installed Ollama model (or a subset).

    For each model we run one small JSON-output generation and
    record: total time, first-token latency, tokens/second (= Ollama-
    reported eval_count / eval_duration), and whether the output
    parses as a single JSON object with the required fields.

    Cheap by design — one probe per model, ~10-30 s per model on a
    warm cache. Cold runs pay the one-time load cost; Ollama splits
    that out as ``load_s`` in the raw response.
    """
    report = OllamaProbeReport(reachable=False, host=host)
    backend = OllamaBackend(host=host)
    if not backend.is_reachable():
        report.error = (
            f"Ollama not reachable at {host}. Install from "
            "https://ollama.com/download and run `ollama serve`. "
            "Then `ollama pull llama3.2:3b` to get a starter model."
        )
        return report
    report.reachable = True
    try:
        models = backend.list_models()
    except OllamaError as exc:
        report.error = f"list_models failed: {exc}"
        return report

    wanted = only_models or [m.name for m in models[:max_models]]
    for m in models:
        if m.name not in wanted:
            continue
        probe = ModelProbe(model=m)
        try:
            resp = backend.generate(
                [
                    LLMMessage("system", _PROBE_SYSTEM),
                    LLMMessage("user",   _PROBE_USER),
                ],
                model=m.name,
                max_tokens=120,
                temperature=0.3,
            )
            probe.ok = True
            probe.total_s = resp.total_s
            probe.first_token_s = resp.first_token_s
            probe.tokens_per_s = resp.tokens_per_second
            probe.json_parse_ok = _try_parse_json(resp.text)
            probe.preview = resp.text[:200]
        except OllamaError as exc:
            probe.error = str(exc)
        report.probes.append(probe)
    return report

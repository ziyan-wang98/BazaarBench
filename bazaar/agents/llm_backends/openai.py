"""OpenAI /v1/chat/completions backend — zero-dep stdlib urllib.

Matches the shape of :mod:`bazaar.agents.llm_backends.anthropic` so
LLMPolicy can swap providers with a one-line config change. No SDK
is imported — the official ``openai`` package is installable via
the ``[llm]`` extra but not a hard dep.

Design notes
------------
- OpenAI does not expose automatic prompt caching the way Anthropic
  does. We don't try to emulate it here; the policy's in-process
  response cache (keyed on ``prompt_hash``) already prevents
  duplicate calls for identical prompts within a single run.
- ``generate`` collapses all system-role messages into a single
  ``{"role": "system", ...}`` entry, then passes the rest through
  unchanged. This matches the Anthropic backend's behaviour so
  replay-from-events produces byte-identical request payloads
  across the two providers (minus the provider-specific fields
  like ``anthropic-version``).
- Token usage numbers are exposed through ``raw`` exactly as the
  API returns them. ``llm_calls.sampling_params`` is the
  authoritative source for replay; ``raw`` is for analysis only.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from bazaar.agents.llm_backends.base import (
    LLMMessage,
    LLMResponse,
    ModelInfo,
)

_DEFAULT_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_RETRIES = 4

_KNOWN_MODELS = {
    "gpt-4o":         180.0,
    "gpt-4o-mini":    8.0,
    "gpt-4.1":        220.0,
    "o4-mini":        20.0,
}


def _is_reasoning_model(model: str) -> bool:
    """Whether ``model`` is an OpenAI reasoning model.

    Reasoning models (gpt-5 family, o1/o3/o4 families) reject the
    non-reasoning ``max_tokens`` field and non-default ``temperature``
    on ``/v1/chat/completions`` — they want ``max_completion_tokens``
    and ``reasoning_effort`` instead. Matching is liberal on the
    ``o``-prefix families so future ``o5-*`` / ``gpt-5.x`` variants
    don't need a code update. Note: ``gpt-4o`` / ``gpt-4o-mini`` are
    NOT reasoning models despite the ``4o`` suffix — they're the
    non-reasoning "omni" multi-modal family.
    """
    m = model.lower()
    if m.startswith("gpt-4o"):
        return False  # gpt-4o family is non-reasoning
    if m.startswith("gpt-5"):
        return True
    if m == "o1" or m.startswith("o1-") or m.startswith("o1"):
        return True
    if m == "o3" or m.startswith("o3-") or m.startswith("o3"):
        return True
    if m == "o4" or m.startswith("o4-") or m.startswith("o4"):
        return True
    return False


class OpenAIError(RuntimeError):
    """Raised when the API returns an error we can't retry past."""


def _retry_delay_s(exc: OpenAIError, attempt: int) -> float | None:
    """Return retry delay for a transient provider error, or None.

    Authentication and malformed-request errors should fail
    immediately. Rate limits, upstream 5xxs, dropped connections, and
    read timeouts are transient in long parallel rollouts, so they get
    jittered backoff.
    """
    msg = str(exc).lower()
    if (
        "invalid_api_key" in msg
        or "incorrect api key" in msg
        or "401:" in msg
        or "403:" in msg
    ):
        return None
    if "400:" in msg and "rate" not in msg and "quota" not in msg:
        return None
    if "429" in msg or "rate" in msg or "quota" in msg:
        return min(30.0, (2.0 ** attempt) + random.uniform(0.0, 0.5))
    if (
        "timeout" in msg
        or "request failed" in msg
        or "incompleteread" in msg
        or "connection reset" in msg
        or "empty response" in msg
        or " 5" in msg
        or msg.startswith("5")
    ):
        return min(15.0, (1.5 ** attempt) + random.uniform(0.0, 0.5))
    return 0.5 * (attempt + 1)


def _build_url(base_url: str, endpoint: str) -> str:
    """Construct a ``/v1/<endpoint>`` URL from a base URL that may or
    may not already include the ``/v1`` prefix.

    Native OpenAI sets ``base_url="https://api.openai.com"`` (no v1),
    so callers need ``"/v1/" + endpoint``. OpenRouter sets
    ``base_url="https://openrouter.ai/api/v1"`` (already includes v1),
    so a naive ``"/v1/" + endpoint`` collides as
    ``/api/v1/v1/chat/completions`` and lands on the marketing
    HTML page. This helper detects an existing trailing ``/v1`` and
    only appends ``/<endpoint>``.
    """
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{endpoint}"
    return f"{base}/v1/{endpoint}"


_HTTPX_CLIENT_LOCK = None
_HTTPX_CLIENTS: dict[float, Any] = {}


def _get_httpx_client(timeout: float):
    """Module-level httpx.Client cache keyed by per-call read timeout.

    The earlier "with httpx.Client(...) as client: ..." pattern closed
    the connection pool after every API call, leaving the underlying
    TCP socket in TIME_WAIT for ~60 s. With 100 parallel workers
    across 5 processes that drained the macOS ephemeral-port range
    (49152-65535) within minutes — bazaar would still run, but the
    host could not open any new TCP sockets, so even ``git push``
    failed with "Can't assign requested address". We now share one
    Client per (read-timeout) bucket process-wide; httpx's pool
    keeps the underlying connections in CLOSE_WAIT/keep-alive instead
    of churning new sockets.
    """
    import threading

    import httpx
    global _HTTPX_CLIENT_LOCK
    if _HTTPX_CLIENT_LOCK is None:
        _HTTPX_CLIENT_LOCK = threading.Lock()
    with _HTTPX_CLIENT_LOCK:
        cli = _HTTPX_CLIENTS.get(timeout)
        if cli is None:
            cli = httpx.Client(
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=timeout,
                    write=30.0,
                    pool=30.0,
                ),
                http2=False,
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=200,
                    keepalive_expiry=120.0,
                ),
            )
            _HTTPX_CLIENTS[timeout] = cli
        return cli


def _request_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    # urllib's `timeout=` is per-socket-read, so a server that drips
    # keep-alive bytes (DeepSeek/OpenRouter can do this during long
    # generations) keeps the read timer alive and the call hangs
    # forever. httpx gives us robust socket timeouts and connection
    # pooling, but its sync timeout is still per operation, not a total
    # wall-clock deadline. Stream the body and enforce our own deadline
    # so heartbeat-drip responses cannot stall paid rollouts.
    import httpx
    try:
        client = _get_httpx_client(timeout)
        deadline = time.monotonic() + timeout
        chunks: list[bytes] = []
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            for chunk in resp.iter_bytes():
                if time.monotonic() > deadline:
                    raise OpenAIError(
                        f"request exceeded wall timeout after {timeout}s: {url}"
                    )
                chunks.append(chunk)
            raw = b"".join(chunks).decode(
                resp.encoding or "utf-8",
                errors="replace",
            )
            if resp.status_code >= 400:
                raise OpenAIError(f"{resp.status_code}: {raw[:500]}")
    except httpx.TimeoutException as exc:
        raise OpenAIError(
            f"request timed out after {timeout}s: {url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise OpenAIError(f"request failed: {url}: {exc}") from exc

    # Some OpenAI-compatible aggregators (OpenRouter most notably)
    # interleave the response body with keep-alive whitespace
    # heartbeats — even on non-streaming requests — so the body looks
    # like ``"\n   \n   \n   {real json}"``. ``json.loads`` accepts
    # leading whitespace already, but in practice the heartbeat lines
    # also occasionally contain stray ``data: `` SSE prefixes. Strip
    # whitespace first and, if that still does not parse, try to
    # locate the first balanced ``{...}`` payload as a fallback so the
    # caller does not see a spurious JSON error on otherwise-valid
    # content.
    stripped = raw.strip()
    if not stripped:
        raise OpenAIError(f"empty response body from {url}")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as parse_exc:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError as inner_exc:
                raise OpenAIError(
                    f"unparseable response body (first 500 chars): "
                    f"{stripped[:500]!r}"
                ) from inner_exc
        raise OpenAIError(
            f"unparseable response body (first 500 chars): "
            f"{stripped[:500]!r}"
        ) from parse_exc


@dataclass
class OpenAIBackend:
    """Minimal /v1/chat/completions client.

    When ``use_responses_endpoint=True`` the backend talks to
    ``/v1/responses`` instead, which is the only OpenAI endpoint that
    surfaces reasoning-summary text alongside the function calls. The
    ``LLMResponse`` shape is identical either way — callers don't need
    to branch on which endpoint produced the result.
    """
    api_key: str | None = None
    base_url: str = _DEFAULT_BASE
    organization: str | None = None
    retries: int = _DEFAULT_RETRIES
    reasoning_effort: str = "medium"
    use_responses_endpoint: bool = False
    request_timeout_s: float = _DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAIError(
                "OPENAI_API_KEY not set. Export it or pass api_key="
                "explicitly."
            )
        self.organization = (self.organization
                             or os.environ.get("OPENAI_ORG"))
        if self.request_timeout_s <= 0:
            raise OpenAIError("request_timeout_s must be positive")

    # ---- API surface ------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Known canonical models. ``generate`` accepts any string."""
        return [
            ModelInfo(name=n, parameter_b=b, family="openai")
            for n, b in sorted(_KNOWN_MODELS.items(), key=lambda kv: kv[1])
        ]

    def is_reachable(self) -> bool:
        try:
            resp = self._call(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(resp.get("id"))
        except OpenAIError:
            return False

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
        # Explicit kwarg (non-None) wins over the instance default.
        effort = reasoning_effort if reasoning_effort else self.reasoning_effort
        system_text = ""
        chat_payload: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_text = (system_text + "\n\n" + m.content).strip()
            else:
                chat_payload.append({"role": m.role, "content": m.content})
        if system_text:
            chat_payload = (
                [{"role": "system", "content": system_text}] + chat_payload
            )

        start = time.monotonic()
        if self.use_responses_endpoint:
            body = self._call_responses(
                model=model, messages=chat_payload,
                max_tokens=max_tokens, tools=tools,
                reasoning_effort=effort,
            )
            total_s = time.monotonic() - start
            text = _extract_responses_text(body)
            tool_calls = _extract_responses_tool_calls(body)
            reasoning = _extract_responses_reasoning(body)
            usage = body.get("usage") or {}
            return LLMResponse(
                text=text,
                total_s=total_s,
                first_token_s=total_s,
                prompt_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                model=model,
                raw=body,
                tool_calls=tool_calls,
                reasoning_summary=reasoning,
            )

        body = self._call(
            model=model, messages=chat_payload,
            max_tokens=max_tokens, temperature=temperature,
            tools=tools, reasoning_effort=effort,
        )
        total_s = time.monotonic() - start
        text = _extract_text(body)
        tool_calls = _extract_tool_calls(body)
        if (
            _is_reasoning_model(model)
            and tools
            and not text
            and not tool_calls
        ):
            # Some reasoning-family chat/completions responses can
            # spend the small completion budget on hidden reasoning and
            # return no assistant text or tool call. In an action
            # simulator that silently degenerates into DO_NOTHING. Retry
            # through /v1/responses, where tool_choice="required" is
            # available and reasoning summaries are surfaced.
            body = self._call_responses(
                model=model, messages=chat_payload,
                max_tokens=max_tokens, tools=tools,
                reasoning_effort=effort,
            )
            total_s = time.monotonic() - start
            text = _extract_responses_text(body)
            tool_calls = _extract_responses_tool_calls(body)
            reasoning = _extract_responses_reasoning(body)
            usage = body.get("usage") or {}
            return LLMResponse(
                text=text,
                total_s=total_s,
                first_token_s=total_s,
                prompt_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                model=model,
                raw=body,
                tool_calls=tool_calls,
                reasoning_summary=reasoning,
            )
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            total_s=total_s,
            first_token_s=total_s,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=model,
            raw=body,
            tool_calls=tool_calls,
            reasoning_summary=_extract_chat_reasoning(body),
        )

    # ---- plumbing ---------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str = "medium",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if _is_reasoning_model(model):
            # gpt-5.x / o1 / o3 / o4 family: chat/completions rejects
            # ``max_tokens`` and non-default ``temperature``. Pass
            # ``max_completion_tokens`` + ``reasoning_effort`` instead.
            payload["max_completion_tokens"] = max_tokens
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        # OpenRouter (and a handful of OpenAI-compatible aggregators)
        # only surface a chain-of-thought trace when the request asks
        # for it via ``"reasoning": {"enabled": True}``. The native
        # OpenAI endpoint ignores this extra field, so it is safe to
        # send unconditionally when the caller asks for any non-zero
        # reasoning effort. The corresponding response surfaces the
        # trace on ``choices[0].message.reasoning`` (handled by
        # ``_extract_chat_reasoning``).
        if "openrouter.ai" in (self.base_url or "") and reasoning_effort and reasoning_effort != "none":
            payload["reasoning"] = {"enabled": True}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            # Force JSON + name ourselves with a regular User-Agent.
            # Some OpenAI-compatible aggregators (OpenRouter in
            # particular) will silently fall back to a marketing /
            # rate-limit HTML page when the Accept header is ``*/*``
            # AND the client identifies itself as ``python-urllib/3.x``,
            # which Python's default opener does. Asking explicitly
            # for application/json plus an OpenAI-Python-style UA
            # keeps OpenRouter on the JSON branch.
            "Accept":        "application/json",
            "User-Agent":    "OpenAI/Python/1.0 (BazaarBench)",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        url = _build_url(self.base_url, "chat/completions")

        last_err: OpenAIError | None = None
        for attempt in range(self.retries + 1):
            try:
                return _request_json(
                    url, payload=payload, headers=headers,
                    timeout=self.request_timeout_s,
                )
            except OpenAIError as exc:
                last_err = exc
                delay = _retry_delay_s(exc, attempt)
                if attempt >= self.retries or delay is None:
                    raise
                time.sleep(delay)
        assert last_err is not None
        raise last_err

    def _call_responses(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str = "medium",
    ) -> dict[str, Any]:
        """POST to ``/v1/responses``.

        ``messages`` is the chat-format list this backend already
        normalises (system + alternating user/assistant). The
        ``/v1/responses`` API takes the same role-tagged list under
        the ``input`` key, so no further translation is needed for
        text. Tools, however, use a different shape — see
        ``_chat_tools_to_responses_tools``. Reasoning summary
        requested via ``reasoning.summary = "detailed"``.
        """
        # Bound responses calls. Earlier 512-token ceilings truncated
        # reasoning-family tool calls, but omitting the ceiling entirely
        # lets provider-side long reasoning/heartbeat responses stall a
        # paid rollout for many minutes. Paper-facing L2/L3 runs pass
        # 4096 here, which is generous for one required function call
        # while still bounding latency and spend.
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        # /v1/responses accepts the ``reasoning`` block ONLY for
        # reasoning-family models (gpt-5.x, o1/o3/o4). Sending it to a
        # non-reasoning model like gpt-4.1-mini yields
        # ``400: Unsupported parameter: 'reasoning.effort'``. Gate the
        # block on the model family so the same OpenAIBackend instance
        # can serve both a reasoning main-action model and a non-
        # reasoning reflection model in the same run (R14a's
        # ``--reflection-model gpt-4.1-mini`` use case).
        if _is_reasoning_model(model):
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "summary": "detailed",
            }
        if tools:
            payload["tools"] = _chat_tools_to_responses_tools(tools)
            # Force a tool call every turn. With reasoning=high + summary
            # detailed, the default ``tool_choice="auto"`` routinely lets
            # the model treat its reasoning block as the complete answer
            # and emit no function_call at all (measured: 53% of R11
            # smoke turns). Bazaar is an action simulation — every tick
            # demands an action (or the ``do_nothing`` tool, which is a
            # valid action too). ``required`` restores the per-turn
            # action contract.
            payload["tool_choice"] = "required"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        url = _build_url(self.base_url, "responses")

        last_err: OpenAIError | None = None
        for attempt in range(self.retries + 1):
            try:
                return _request_json(
                    url, payload=payload, headers=headers,
                    timeout=self.request_timeout_s,
                )
            except OpenAIError as exc:
                last_err = exc
                delay = _retry_delay_s(exc, attempt)
                if attempt >= self.retries or delay is None:
                    raise
                time.sleep(delay)
        assert last_err is not None
        raise last_err


def _chat_tools_to_responses_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate chat/completions tool dicts to /v1/responses shape.

    Chat shape:  ``{"type": "function", "function": {name, description, parameters}}``
    Responses:   ``{"type": "function", "name": ..., "description": ..., "parameters": ...}``
    (no nested ``function`` key — the fields hang directly off the dict.)
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict) or "name" not in fn:
            # Already in responses shape, or malformed — pass through
            # the keys that look right.
            if "name" in t:
                out.append({
                    "type":        "function",
                    "name":        t.get("name"),
                    "description": t.get("description", ""),
                    "parameters":  t.get("parameters", {}),
                })
            continue
        out.append({
            "type":        "function",
            "name":        fn["name"],
            "description": fn.get("description", ""),
            "parameters":  fn.get("parameters", {}),
        })
    return out


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_chat_reasoning(body: dict[str, Any]) -> str | None:
    """Extract a reasoning trace from a chat-completions response.

    Recognises three formats:
    * DeepSeek V4 / DeepSeek-R1 surface reasoning as
      ``choices[0].message.reasoning_content``;
    * Some OpenRouter / vLLM proxies surface it as
      ``choices[0].message.reasoning``;
    * Native OpenAI reasoning models on ``/v1/chat/completions`` either
      do not surface a chain-of-thought summary at all or place it on
      ``message.reasoning`` when ``reasoning_effort`` is high. We only
      need any of the three to land in ``reasoning_summary``.

    Returns ``None`` when no field is populated.
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    for key in ("reasoning_content", "reasoning"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _extract_tool_calls(
    body: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Surface ``choices[0].message.tool_calls`` in the shape
    LLMResponse expects: ``[{"function": {"name", "arguments"}}]``.

    OpenAI returns ``arguments`` as a JSON-encoded string; we parse
    it into a dict here so downstream consumers don't have to. Any
    call whose arguments aren't parseable is kept with an empty
    dict — the dispatcher will then reject it cleanly via pydantic
    validation. Returns ``None`` when the response has no tool
    calls so LLMPolicy's fallback path stays in charge.
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    raw = msg.get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if not isinstance(name, str):
            continue
        out.append({"function": {"name": name, "arguments": args}})
    return out or None


# ---- /v1/responses parsers ---------------------------------------------


def _extract_responses_text(body: dict[str, Any]) -> str:
    """Concatenate every ``output_text`` chunk inside ``message`` blocks.

    /v1/responses returns ``output: list`` mixing ``reasoning``,
    ``message``, and ``function_call`` block types. Plain assistant
    text lives only on ``message`` blocks under
    ``content[*].text`` for items where ``type == "output_text"``.
    """
    out_blocks = body.get("output") or []
    parts: list[str] = []
    for block in out_blocks:
        if not isinstance(block, dict) or block.get("type") != "message":
            continue
        for chunk in block.get("content") or []:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "output_text":
                t = chunk.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
    return "".join(parts).strip()


def _extract_responses_tool_calls(
    body: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Translate ``function_call`` blocks → chat-shape tool_calls.

    /v1/responses delivers each tool call as a top-level output
    item with ``type=="function_call"`` carrying ``name``,
    ``arguments`` (JSON string), and ``call_id``. We re-shape to
    ``[{"function": {"name", "arguments": dict}}]`` so the LLMPolicy
    consumer is endpoint-agnostic.
    """
    out_blocks = body.get("output") or []
    out: list[dict[str, Any]] = []
    for block in out_blocks:
        if not isinstance(block, dict) or block.get("type") != "function_call":
            continue
        name = block.get("name")
        args = block.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if not isinstance(name, str):
            continue
        out.append({"function": {"name": name, "arguments": args}})
    return out or None


def _extract_responses_reasoning(body: dict[str, Any]) -> str | None:
    """Concatenate every ``summary_text`` text from ``reasoning`` blocks.

    With ``reasoning.summary == "detailed"`` requested, /v1/responses
    emits one or more ``reasoning`` output items, each carrying a
    ``summary`` list of ``{type: "summary_text", text: ...}``
    chunks. We join everything we find — multiple reasoning blocks
    are separated by a blank line. ``None`` when no summary text
    came back (silent reasoning, or reasoning_effort "none").
    """
    out_blocks = body.get("output") or []
    parts: list[str] = []
    for block in out_blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        for chunk in block.get("summary") or []:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "summary_text":
                t = chunk.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
    if not parts:
        return None
    return "\n\n".join(parts).strip() or None

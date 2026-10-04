"""LLM-backed dynamics: D11 memory consolidation + D12 self-portrait.

These were deferred in Phase-2 (`dynamics/__init__.py` docstring)
because they genuinely need a language model to produce anything
interesting. Now that T28c shipped the LLMPolicy infrastructure,
we can wire them up — the backend lives in the registry as a
captured closure, so the dynamics remain byte-identical across
reruns given the same backend + same cache state.

Both callbacks carry a safe fallback path when no backend is
available or when a backend call fails. This keeps the default
registry usable in offline CI runs that don't want paid API calls.

Event-log discipline: every LLM-driven
write is accompanied by a platform-side event row so Phase-4
replay can reconstruct the action.
"""
from __future__ import annotations

import inspect
import json
import sqlite3
from typing import Any

from bazaar.agents.llm_backends.base import LLMBackend, LLMMessage
from bazaar.core.event_log import log_event, log_llm_call
from bazaar.memory.ledger import build_ledger_context
from bazaar.memory.narrative import HashEncoder, NarrativeStore

# Number of oldest narrative rows to fold into one consolidated entry
# per agent per D11 firing. Keeping this small bounds the token cost
# so a 100-agent × 500-tick run stays under a reasonable LLM budget.
_CONSOLIDATE_BATCH = 6

# Max output characters for each LLM call. Narratives are short
# impressions, not essays.
_MAX_NARRATIVE_CHARS = 400


def make_d11_memory_consolidation(
    backend: LLMBackend | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    agent_ids: set[int] | None = None,
) -> Any:
    """D11 callback factory: consolidate oldest narrative memories.

    Every firing walks every agent with ≥ ``_CONSOLIDATE_BATCH + 1``
    non-decayed narrative rows, folds the oldest batch into a single
    new entry, and marks the originals ``decayed=1`` so recall won't
    retrieve duplicates.

    When ``backend`` is None, the fallback produces a pipe-joined
    concatenation of the originals — still a valid narrative, just
    not summarised prose. This path lets D11 run in CI without a
    real LLM.

    Returns the number of consolidation events logged.
    """
    agent_id_filter = frozenset(agent_ids) if agent_ids is not None else None

    def _callback(conn: sqlite3.Connection, *, tick: int, rng: Any) -> int:
        return _consolidate(conn, tick=tick, backend=backend, model=model,
                            temperature=temperature,
                            agent_ids=agent_id_filter)
    _callback.__name__ = "D11_memory_consolidation"
    return _callback


def make_d14_agent_summary(
    backend: LLMBackend | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    agent_ids: set[int] | None = None,
) -> Any:
    """D14 callback factory: rolling per-agent self-summary.

    R14a Layer-1. Every firing walks every active agent, collects a
    small window of recent structured context (ledger entries, recent
    own thoughts, previous summary, current goals) and calls the
    reflection LLM to produce a compact paragraph describing "what am
    I doing now, what have I learned, what do I want next". The
    result is appended to ``agent_summary``; the most recent row per
    agent is what ``PromptBuilder`` injects as ``## MY CURRENT STATE``
    into every LLMPolicy invocation.

    The reflection call is *separate* from the per-tick action call
    so the action prompt stays cheap and each agent's self-narrative
    can be refreshed every tick (``default_registry`` sets
    ``d14_interval=1``). Long runs can throttle by passing a larger
    interval at registration time.

    Fallback: with no backend, a deterministic template built from
    ledger counts + goals is written instead, tagged ``source =
    'fallback'``. That path keeps the table populated in CI runs.

    Returns the number of summaries written.
    """
    agent_id_filter = frozenset(agent_ids) if agent_ids is not None else None

    def _callback(conn: sqlite3.Connection, *, tick: int, rng: Any) -> int:
        return _write_agent_summaries(
            conn, tick=tick, backend=backend, model=model,
            temperature=temperature, agent_ids=agent_id_filter,
        )
    _callback.__name__ = "D14_agent_summary"
    return _callback


def make_d12_self_portrait(
    backend: LLMBackend | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.6,
    agent_ids: set[int] | None = None,
) -> Any:
    """D12 callback factory: every agent writes a 3-sentence self-portrait.

    Stored in ``self_portraits``. When ``backend`` is None, a fallback
    template based on the persona + ledger counts is used so the
    row still exists (identity-drift analysis just needs *something*
    to diff against a later portrait).

    Returns the number of self-portrait events logged.
    """
    agent_id_filter = frozenset(agent_ids) if agent_ids is not None else None

    def _callback(conn: sqlite3.Connection, *, tick: int, rng: Any) -> int:
        return _write_self_portraits(conn, tick=tick, backend=backend,
                                     model=model, temperature=temperature,
                                     agent_ids=agent_id_filter)
    _callback.__name__ = "D12_self_portrait"
    return _callback


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _active_agent_rows(
    conn: sqlite3.Connection,
    *,
    agent_ids: frozenset[int] | None,
    columns: str,
) -> list[Any]:
    """Return active agent rows, optionally narrowed to a target subset."""
    clauses = ["status = 'active'"]
    params: list[int] = []
    if agent_ids is not None:
        if not agent_ids:
            return []
        placeholders = ",".join("?" * len(agent_ids))
        clauses.append(f"agent_id IN ({placeholders})")
        params.extend(sorted(agent_ids))
    return conn.execute(
        f"SELECT {columns} FROM agents WHERE {' AND '.join(clauses)} "
        "ORDER BY agent_id",
        params,
    ).fetchall()


def _consolidate(
    conn: sqlite3.Connection,
    *,
    tick: int,
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
    agent_ids: frozenset[int] | None,
) -> int:
    agent_rows = _active_agent_rows(
        conn, agent_ids=agent_ids, columns="agent_id",
    )
    n_events = 0
    store = _get_or_make_store(conn)
    for row in agent_rows:
        agent_id = int(row[0])
        entries = conn.execute(
            """
            SELECT memory_id, scope, scope_ref_id, content
            FROM narrative_memories
            WHERE agent_id = ? AND decayed = 0
            ORDER BY created_tick ASC, memory_id ASC
            """,
            (agent_id,),
        ).fetchall()
        if len(entries) <= _CONSOLIDATE_BATCH:
            continue

        batch = [dict(
            memory_id=int(r[0]),
            scope=r[1],
            scope_ref_id=int(r[2]) if r[2] is not None else None,
            content=r[3],
        ) for r in entries[:_CONSOLIDATE_BATCH]]

        summary = _summarise(
            agent_id=agent_id, tick=tick, batch=batch,
            backend=backend, model=model, temperature=temperature,
            conn=conn,
        )

        # Write the new consolidated entry + mark sources decayed.
        store.add(
            agent_id=agent_id,
            scope="self",
            scope_ref_id=None,
            content=summary,
            tick=tick,
        )
        ids_csv = ",".join(str(b["memory_id"]) for b in batch)
        conn.execute(
            f"UPDATE narrative_memories SET decayed = 1 "
            f"WHERE memory_id IN ({ids_csv})"
        )
        log_event(
            conn,
            tick=tick,
            agent_id=None,
            action_type="D11_memory_consolidation",
            payload={
                "agent_id": agent_id,
                "consolidated_memory_ids": [b["memory_id"] for b in batch],
            },
            result_status="ok",
            result_payload={"summary_chars": len(summary)},
        )
        n_events += 1
    return n_events


def _write_self_portraits(
    conn: sqlite3.Connection,
    *,
    tick: int,
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
    agent_ids: frozenset[int] | None,
) -> int:
    rows = _active_agent_rows(
        conn, agent_ids=agent_ids, columns="agent_id, persona_json",
    )
    n_events = 0
    for row in rows:
        agent_id = int(row[0])
        persona_json = row[1] or "{}"
        try:
            persona = json.loads(persona_json)
        except Exception:
            persona = {}
        portrait = _portrait_text(
            agent_id=agent_id, tick=tick, persona=persona,
            conn=conn, backend=backend, model=model,
            temperature=temperature,
        )
        conn.execute(
            "INSERT INTO self_portraits (agent_id, tick, content, embedding) "
            "VALUES (?, ?, ?, NULL)",
            (agent_id, tick, portrait),
        )
        log_event(
            conn,
            tick=tick,
            agent_id=None,
            action_type="D12_self_portrait",
            payload={"agent_id": agent_id, "portrait_chars": len(portrait)},
            result_status="ok",
            result_payload=None,
        )
        n_events += 1
    return n_events


# Advisory length for a D14 rolling summary — passed to the reflection
# prompt as guidance. The returned text is stored verbatim even if it
# overruns; truncation would silently drop state the agent relied on.
_D14_ADVISED_CHARS = 2000


def _write_agent_summaries(
    conn: sqlite3.Connection,
    *,
    tick: int,
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
    agent_ids: frozenset[int] | None,
) -> int:
    """D14 implementation: one rolling summary per active agent.

    For each agent we pull (a) the previous summary (if any) for
    continuity, (b) the last 8 ledger entries, (c) the agent's most
    recent ``llm_calls`` reasoning/response so the reflection "sees"
    what the agent was just thinking, and (d) the persona goals. The
    reflection LLM folds these into a new paragraph which is appended
    to ``agent_summary``.
    """
    agent_rows = _active_agent_rows(
        conn, agent_ids=agent_ids, columns="agent_id, persona_json",
    )
    n_events = 0
    for row in agent_rows:
        agent_id = int(row[0])
        persona_json = row[1] or "{}"
        try:
            persona = json.loads(persona_json)
        except Exception:
            persona = {}

        previous = _fetch_previous_summary(conn, agent_id, tick)
        ledger_entries = build_ledger_context(
            conn, agent_id=agent_id, k=8, up_to_tick=tick,
        )
        recent_thought = _fetch_recent_thought(conn, agent_id, tick)

        summary, source = _reflect(
            agent_id=agent_id, tick=tick, persona=persona,
            previous=previous, ledger=ledger_entries,
            recent_thought=recent_thought,
            backend=backend, model=model, temperature=temperature,
            conn=conn,
        )
        conn.execute(
            "INSERT INTO agent_summary (agent_id, tick, content, source) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, tick, summary, source),
        )
        log_event(
            conn,
            tick=tick,
            agent_id=None,
            action_type="D14_agent_summary",
            payload={
                "agent_id": agent_id,
                "summary_chars": len(summary),
                "source": source,
            },
            result_status="ok",
            result_payload=None,
        )
        n_events += 1
    return n_events


def _fetch_previous_summary(
    conn: sqlite3.Connection, agent_id: int, tick: int,
) -> str | None:
    row = conn.execute(
        "SELECT content FROM agent_summary "
        "WHERE agent_id = ? AND tick < ? "
        "ORDER BY tick DESC, summary_id DESC LIMIT 1",
        (agent_id, tick),
    ).fetchone()
    return (row[0] if row else None)


def _fetch_recent_thought(
    conn: sqlite3.Connection, agent_id: int, tick: int,
) -> str | None:
    row = conn.execute(
        """
        SELECT COALESCE(reasoning_summary, response_text, '') AS content
          FROM llm_calls
         WHERE agent_id = ? AND tick < ?
         ORDER BY tick DESC, call_id DESC LIMIT 1
        """,
        (agent_id, tick),
    ).fetchone()
    if row is None:
        return None
    content = (row[0] or "").strip()
    return content or None


def _reflect(
    *,
    agent_id: int,
    tick: int,
    persona: dict[str, Any],
    previous: str | None,
    ledger: list[Any],
    recent_thought: str | None,
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
    conn: sqlite3.Connection,
) -> tuple[str, str]:
    """Return ``(summary_text, source)``.

    ``source`` is ``'D14'`` when the reflection LLM produced the text
    and ``'fallback'`` when we hit the template path.
    """
    display = persona.get("display_name") or f"agent#{agent_id}"
    goals = persona.get("goals") or {}
    buyer = goals.get("buyer") or {}
    seller = goals.get("seller") or {}
    ledger_text = (
        "\n".join(f"- [t={e.tick}] {e.kind}: {e.summary}" for e in ledger)
        or "(no structured ledger entries yet)"
    )
    goal_text_parts = []
    if buyer:
        dollars = int(buyer.get("max_price_cents", 0)) / 100
        goal_text_parts.append(
            f"Buyer side: want a {buyer.get('want_category', 'item')} "
            f"for <= ${dollars:.0f} (urgency: {buyer.get('urgency')})."
        )
    if seller:
        goal_text_parts.append(
            f"Seller side: hold at >= "
            f"{int(float(seller.get('min_price_fraction') or 0) * 100)}% "
            f"of asking; "
            f"{seller.get('target_listings_count', 0)} items to post."
        )
    goal_text = "\n".join(goal_text_parts) or "(no explicit goals yet)"

    prev_text = previous if previous else "(this is your first reflection)"
    thought_text = recent_thought if recent_thought else "(no prior thought on record)"

    fallback = (
        f"{display}'s state at tick {tick}: "
        f"{len(ledger)} ledger entries so far. "
        f"Buyer target ${int(buyer.get('max_price_cents', 0))/100:.0f} "
        f"({buyer.get('want_category', 'n/a')}); "
        f"{seller.get('target_listings_count', 0)} items to list on the "
        f"seller side. Continuing from prior plan."
    )

    if backend is None or model is None:
        return fallback, "fallback"

    system = (
        "You maintain a rolling self-summary for a marketplace user. "
        "Each firing you rewrite the summary in first person, aiming "
        f"for under {_D14_ADVISED_CHARS} characters, covering three "
        "beats:\n"
        "1. What I'm currently doing (active threads, pending offers, "
        "what I'm trying to buy or sell).\n"
        "2. What I've learned since the last summary (about my "
        "counterparties, about prices, about what works).\n"
        "3. What I want next (concrete, actionable, grounded in my "
        "goal).\n"
        "Stay factual — ground every claim in the provided ledger or "
        "the previous summary. Do not invent counterparties, prices, "
        "or events. Return plain text, no JSON, no bullet markers."
    )
    user = (
        f"User: {display} (agent #{agent_id})\n"
        f"Tick: {tick}\n\n"
        f"GOALS\n{goal_text}\n\n"
        f"PREVIOUS SUMMARY\n{prev_text}\n\n"
        f"RECENT LEDGER\n{ledger_text}\n\n"
        f"MOST RECENT OWN THOUGHT\n{thought_text}\n\n"
        f"Write the new rolling summary now."
    )
    text = _call_llm_safe(
        backend=backend, model=model, temperature=temperature,
        system_text=system, user_text=user,
        agent_id=agent_id, tick=tick, conn=conn,
        fallback=fallback,
        max_chars=None,
        max_tokens=1500,
        reasoning_effort="low",
    )
    # _call_llm_safe returns ``fallback`` on exception — detect that
    # case by identity match so we tag the row correctly.
    source = "fallback" if text is fallback else "D14"
    return text, source


# ---------------------------------------------------------------------------
# LLM callouts (shared by D11 + D12 + D14)
# ---------------------------------------------------------------------------


def _summarise(
    *,
    agent_id: int,
    tick: int,
    batch: list[dict[str, Any]],
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
    conn: sqlite3.Connection,
) -> str:
    """LLM-backed or fallback summary of ``batch``."""
    fallback = " | ".join(b["content"] for b in batch)[: _MAX_NARRATIVE_CHARS]
    if backend is None or model is None:
        return fallback

    system = (
        "You are condensing a marketplace user's older free-text "
        "impressions into a single, compact, factual summary they can "
        "look back on later. Keep it under 400 characters. Do not add "
        "speculation; preserve the original sentiment and any concrete "
        "counterparty identifiers. Return plain text, no JSON."
    )
    user = (
        f"Agent #{agent_id} is consolidating notes at tick {tick}. "
        f"Source notes (one per line):\n"
        + "\n".join(f"- {b['content']}" for b in batch)
    )
    return _call_llm_safe(
        backend=backend, model=model, temperature=temperature,
        system_text=system, user_text=user,
        agent_id=agent_id, tick=tick, conn=conn,
        fallback=fallback,
        max_chars=_MAX_NARRATIVE_CHARS,
    )


def _portrait_text(
    *,
    agent_id: int,
    tick: int,
    persona: dict[str, Any],
    conn: sqlite3.Connection,
    backend: LLMBackend | None,
    model: str | None,
    temperature: float,
) -> str:
    entries = build_ledger_context(conn, agent_id=agent_id, k=6, up_to_tick=tick)
    history_str = (
        "\n".join(f"- [t={e.tick}] {e.summary}" for e in entries)
        or "(no recent marketplace activity)"
    )
    display = persona.get("display_name") or f"agent#{agent_id}"
    profession = persona.get("profession") or ""
    interests = persona.get("interests") or []

    fallback = (
        f"{display} ({profession}) has been active in the marketplace "
        f"at tick {tick}. Interests: "
        f"{', '.join(interests[:3]) if interests else 'varied'}."
    )[: _MAX_NARRATIVE_CHARS]

    if backend is None or model is None:
        return fallback

    system = (
        "Write a short, first-person self-description (3 sentences max) "
        "for this marketplace user, covering: who they are, what they "
        "have been doing lately, and what their current marketplace "
        "goals feel like. Stay grounded in the provided history. "
        "Do not speculate beyond what the data supports."
    )
    user = (
        f"User: {display}\n"
        f"Profession: {profession}\n"
        f"Interests: {', '.join(interests)}\n"
        f"Recent marketplace history:\n{history_str}\n"
        f"Tick: {tick}."
    )
    return _call_llm_safe(
        backend=backend, model=model, temperature=temperature,
        system_text=system, user_text=user,
        agent_id=agent_id, tick=tick, conn=conn,
        fallback=fallback,
        max_chars=_MAX_NARRATIVE_CHARS,
    )


def _backend_accepts(backend: Any, name: str) -> bool:
    """True if ``backend.generate`` accepts ``name`` as a kwarg.

    OpenAIBackend lists ``reasoning_effort`` explicitly; Anthropic /
    Ollama don't, and they also don't absorb **kwargs — so passing the
    kwarg to them raises TypeError. Probe the signature once and only
    pass the kwarg when it's safe.
    """
    try:
        params = inspect.signature(backend.generate).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _call_llm_safe(
    *,
    backend: LLMBackend,
    model: str,
    temperature: float,
    system_text: str,
    user_text: str,
    agent_id: int,
    tick: int,
    conn: sqlite3.Connection,
    fallback: str,
    max_chars: int | None = None,
    max_tokens: int = 300,
    reasoning_effort: str | None = None,
) -> str:
    """Call the backend with our exception boundary + llm_calls log.

    Failures fall back to ``fallback`` so a dynamic never takes down a
    tick. Every successful call produces one ``llm_calls`` row marked
    with ``backend`` and the provider class name. ``reasoning_effort``
    is forwarded only when the backend's signature accepts it (OpenAI
    gpt-5.x / o-series); other providers silently skip it.
    """
    import hashlib as _hash
    import time as _time

    prompt_hash = _hash.sha256(
        (system_text + "||" + user_text).encode("utf-8")
    ).hexdigest()
    extra_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None and _backend_accepts(
        backend, "reasoning_effort"
    ):
        extra_kwargs["reasoning_effort"] = reasoning_effort

    t0 = _time.monotonic()
    try:
        resp = backend.generate(
            [LLMMessage("system", system_text),
             LLMMessage("user",   user_text)],
            model=model, max_tokens=max_tokens, temperature=temperature,
            **extra_kwargs,
        )
        stripped = resp.text.strip()
        if max_chars is not None:
            stripped = stripped[:max_chars]
        text = stripped or fallback
        latency_ms = int((_time.monotonic() - t0) * 1000)
        sampling_params: dict[str, Any] = {
            "temperature": temperature, "max_tokens": max_tokens,
            "model": model,
        }
        if reasoning_effort is not None:
            sampling_params["reasoning_effort"] = reasoning_effort
        log_llm_call(
            conn,
            tick=tick, agent_id=agent_id, model=model,
            backend=type(backend).__name__,
            prompt_hash=prompt_hash,
            prompt_text=user_text,
            sampling_params=sampling_params,
            response_text=text,
            tool_calls=None,
            seed=None,
            cache_hit=False,
            latency_ms=latency_ms,
        )
        return text
    except Exception as exc:
        sampling_params = {
            "temperature": temperature, "max_tokens": max_tokens,
            "model": model,
        }
        if reasoning_effort is not None:
            sampling_params["reasoning_effort"] = reasoning_effort
        log_llm_call(
            conn,
            tick=tick, agent_id=agent_id, model=model,
            backend=type(backend).__name__,
            prompt_hash=prompt_hash,
            prompt_text=user_text,
            sampling_params=sampling_params,
            response_text=f"__backend_error__: {exc}",
            tool_calls=None,
            seed=None,
            cache_hit=False,
            latency_ms=int((_time.monotonic() - t0) * 1000),
        )
        return fallback


def _get_or_make_store(conn: sqlite3.Connection) -> NarrativeStore:
    """Return the cached NarrativeStore for ``conn`` or build an offline one.

    The dynamics module must not force a MiniLM load in CI — if no
    store has been installed for this connection, we build a
    ``HashEncoder``-backed one. Production runs install a real
    encoder-backed store explicitly via ``bazaar.memory.install_store``.
    """
    from bazaar.memory import get_store, install_store
    try:
        return get_store(conn)
    except Exception:
        store = NarrativeStore(conn, encoder=HashEncoder(dim=32))
        install_store(conn, store)
        return store

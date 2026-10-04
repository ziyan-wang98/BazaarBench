"""R14b Part A — mental-price probe.

Ask the LLM, outside the main action loop, what an agent's *private*
walkaway price for a listing is RIGHT NOW. Persist it to the
``mental_prices`` table + append a narrative memory so future recall
can surface prior probes.

Probe stages
------------
* ``initial``       — buyer just viewed / pinned a listing, or seller
                      just created one
* ``after_chat``    — messages have been exchanged since the last
                      probe for this (agent, listing, role)
* ``after_compare`` — the agent has compared this listing against
                      peers / the market feed
* ``final``         — immediately before ``make_offer`` (buyer) or
                      ``accept_offer`` / ``counter_offer`` (seller)

The drift between ``initial`` → ``final`` — and between mental price
and market baseline — is the paper's H1 signal. The probe is a
*separate* backend call (cheap model, reasoning=low) so it never
pollutes the main-policy trace and can be switched off without
breaking dispatch.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from bazaar.agents.llm_backends.base import LLMBackend, LLMMessage
from bazaar.core.event_log import log_llm_call, require_lastrowid

_STAGES = {"initial", "after_chat", "after_compare", "final"}
_ROLES = {"buyer", "seller"}

# Max number of prior probes for this (agent, listing, role) to include
# in the prompt context. 6 is enough to show trajectory without
# ballooning the prompt on long-running listings.
_MAX_PRIOR_PROBES = 6

# Messages pulled for the ``after_chat`` stage — chronological,
# truncated per-body so no single long-winded message dominates.
_MAX_RECENT_MESSAGES = 10
_MAX_MESSAGE_CHARS = 200


def _backend_accepts(backend: Any, name: str) -> bool:
    """True if ``backend.generate`` takes ``name`` as a kwarg.

    Mirrors the helper in ``dynamics/llm_dynamics.py`` so the probe
    can forward ``reasoning_effort="low"`` safely — OpenAIBackend
    lists it explicitly, Anthropic/Ollama don't and don't absorb
    ``**kwargs``, so a blind forward would raise ``TypeError``.
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


def _listing_row(conn: sqlite3.Connection, listing_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT listing_id, category, title, description, price_cents,
               condition, created_at_tick, status
        FROM listings WHERE listing_id = ?
        """,
        (listing_id,),
    ).fetchone()
    if row is None:
        return None
    keys = ("listing_id", "category", "title", "description",
            "price_cents", "condition", "created_at_tick", "status")
    return dict(zip(keys, row, strict=True))


def _market_baseline_cents(
    conn: sqlite3.Connection, *, category: str, up_to_tick: int,
) -> int | None:
    """Average ``price_cents`` across active listings in ``category``.

    Returns ``None`` when no comparable listings exist. This mirrors
    what Part C's ``_compute_market_baselines`` will expose wholesale
    for prompt injection; probe uses it inline so the probe row has a
    denormalised baseline recorded at probe time.
    """
    row = conn.execute(
        """
        SELECT AVG(price_cents) FROM listings
        WHERE category = ?
          AND status IN ('active', 'bumped', 'committed')
          AND created_at_tick <= ?
        """,
        (category, up_to_tick),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _prior_probes(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    listing_id: int,
    role: str,
) -> list[dict[str, Any]]:
    """Return up to ``_MAX_PRIOR_PROBES`` earlier probes, oldest first."""
    rows = conn.execute(
        """
        SELECT tick, stage, mental_price_cents, rationale
        FROM mental_prices
        WHERE agent_id = ? AND listing_id = ? AND role = ?
        ORDER BY tick ASC, entry_id ASC
        """,
        (agent_id, listing_id, role),
    ).fetchall()
    if not rows:
        return []
    rows = rows[-_MAX_PRIOR_PROBES:]
    return [
        {"tick": int(r[0]), "stage": r[1],
         "mental_price_cents": int(r[2]), "rationale": r[3] or ""}
        for r in rows
    ]


def _recent_messages(
    conn: sqlite3.Connection, *, listing_id: int, agent_id: int,
) -> list[dict[str, Any]]:
    """Last ``_MAX_RECENT_MESSAGES`` messages in threads this agent
    participates in for ``listing_id``, chronological, body truncated
    at ``_MAX_MESSAGE_CHARS`` each.

    No cap on *number* of threads returned from — if the agent runs
    multiple threads on the same listing we want every recent line
    they've seen.
    """
    rows = conn.execute(
        """
        SELECT m.tick, m.sender_agent_id, m.body
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE t.listing_id = ?
          AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
        ORDER BY m.tick DESC, m.message_id DESC
        LIMIT ?
        """,
        (listing_id, agent_id, agent_id, _MAX_RECENT_MESSAGES),
    ).fetchall()
    rows = list(reversed(rows))  # chronological
    return [
        {"tick": int(r[0]), "sender_agent_id": int(r[1]),
         "body": (r[2] or "")[:_MAX_MESSAGE_CHARS]}
        for r in rows
    ]


def _persona_block(conn: sqlite3.Connection, agent_id: int) -> dict[str, Any]:
    """Return the small persona slice the probe needs.

    Imports ``_load_persona`` lazily — handlers.py imports this module
    indirectly in its event wiring, so a module-level import would
    cycle.
    """
    from bazaar.actions.handlers import _load_persona
    p = _load_persona(conn, agent_id)
    goal_desc = None
    if p.goals is not None and p.goals.buyer is not None:
        goal_desc = (
            f"{p.goals.buyer.description} "
            f"(target category {p.goals.buyer.want_category}, "
            f"max price ${p.goals.buyer.max_price_cents / 100:.0f})"
        )
    deadline_line = None
    if p.deadline is not None:
        deadline_line = (
            f"Deadline: {p.deadline.reason} "
            f"at tick {p.deadline.deadline_tick}."
        )
    fs_line = None
    if p.financial_stress is not None:
        fs = p.financial_stress
        fs_line = (
            f"Financial stress: in {fs.bill_due_tick} ticks, "
            f"${fs.bill_amount_cents / 100:.0f} bill is due — "
            f"{fs.consequence}. On hand: "
            f"${fs.current_cash_cents / 100:.0f}; shortfall "
            f"${fs.shortfall_cents / 100:.0f}."
        )
    return {
        "display_name": p.display_name,
        "profession": p.profession,
        "goal_description": goal_desc,
        "deadline_line": deadline_line,
        "financial_stress_line": fs_line,
    }


_SYSTEM_PROMPT = (
    "You are helping an agent think privately about their walkaway "
    "price for a marketplace listing. Respond in JSON only, no prose "
    "outside the JSON object."
)


def _build_user_prompt(
    *,
    persona: dict[str, Any],
    listing: dict[str, Any],
    market_baseline_cents: int | None,
    prior_probes: list[dict[str, Any]],
    recent_messages: list[dict[str, Any]],
    role: str,
    stage: str,
) -> str:
    lines: list[str] = []
    lines.append(
        f"You are {persona['display_name']}, {persona['profession']}."
    )
    lines.append("")
    if persona["goal_description"]:
        lines.append(f"Goal: {persona['goal_description']}")
    else:
        lines.append("Goal: (no explicit buyer goal on this persona)")
    if persona["deadline_line"]:
        lines.append(persona["deadline_line"])
    if persona["financial_stress_line"]:
        lines.append(persona["financial_stress_line"])
    lines.append("")
    lines.append(f"Listing (listing_id={listing['listing_id']}):")
    lines.append(f"  Title: {listing['title']}")
    lines.append(f"  Description: {listing['description']}")
    lines.append(f"  Asking price: ${listing['price_cents'] / 100:.2f}")
    lines.append(f"  Category: {listing['category']}")
    if market_baseline_cents is not None:
        lines.append(
            f"  Market baseline (category avg): "
            f"${market_baseline_cents / 100:.2f}"
        )
    else:
        lines.append("  Market baseline (category avg): (no comparable listings)")
    lines.append("")
    lines.append("Prior mental price for this listing (yours):")
    if prior_probes:
        for p in prior_probes:
            lines.append(
                f"  - t={p['tick']}, stage={p['stage']}: "
                f"${p['mental_price_cents'] / 100:.2f} — "
                f"\"{p['rationale']}\""
            )
    else:
        lines.append("  (this is your first probe for this listing)")
    lines.append("")
    if stage == "after_chat" and recent_messages:
        lines.append("Recent messages in the thread:")
        for m in recent_messages:
            lines.append(
                f"  [t={m['tick']} from agent {m['sender_agent_id']}] "
                f"{m['body']}"
            )
        lines.append("")
    if role == "buyer":
        lines.append(
            "Now answer: what is your current private walkaway price "
            "for buying this listing — the most you'd pay before "
            "walking away, in cents. Also explain in 1-2 sentences "
            "why, referencing your goal, constraints, and any "
            "relevant context."
        )
    else:
        lines.append(
            "Now answer: what is your current private walkaway price "
            "for selling this listing — the least you'd accept before "
            "walking away, in cents. Also explain in 1-2 sentences "
            "why, referencing your goal, constraints, and any "
            "relevant context."
        )
    lines.append("")
    lines.append(
        'Respond with JSON: '
        '{"mental_price_cents": int, "rationale": "1-2 sentences"}'
    )
    return "\n".join(lines)


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_probe_response(text: str) -> tuple[int | None, str]:
    """Extract ``(mental_price_cents, rationale)`` from an LLM reply.

    The system prompt asks for strict JSON, but reasoning models
    sometimes emit a preamble or trailing prose. We find the first
    valid JSON object in the reply and coerce the required fields.
    """
    if not text:
        return None, ""
    text = text.strip()
    # Strip fenced code blocks defensively; reasoning models love them.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    candidates.extend(_JSON_OBJ_RE.findall(text))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        raw_price = obj.get("mental_price_cents")
        if raw_price is None:
            continue
        try:
            price = int(raw_price)
        except (TypeError, ValueError):
            continue
        if price < 0:
            continue
        rationale = obj.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        return price, rationale
    return None, ""


def _insert_row(
    conn: sqlite3.Connection,
    *,
    listing_id: int,
    agent_id: int,
    role: str,
    stage: str,
    mental_price_cents: int,
    market_baseline_cents: int | None,
    rationale: str,
    tick: int,
) -> int:
    wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO mental_prices
            (listing_id, agent_id, role, stage, mental_price_cents,
             market_baseline_cents, rationale, tick, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (listing_id, agent_id, role, stage, mental_price_cents,
         market_baseline_cents, rationale, tick, wall),
    )
    return require_lastrowid(cur, table="mental_prices")


def probe_mental_price(
    conn: sqlite3.Connection,
    *,
    backend: LLMBackend,
    model: str,
    agent_id: int,
    listing_id: int,
    role: str,
    stage: str,
    tick: int,
    temperature: float = 0.3,
    max_tokens: int = 180,
    reasoning_effort: str | None = "low",
) -> int | None:
    """Ask the LLM what this agent's walkaway price is RIGHT NOW.

    Returns the mental price in cents, or ``None`` on any failure
    (missing listing, backend error, unparseable JSON). The failure
    path never raises — the caller (``LLMPolicy.decide``) keeps
    looping.

    On success, one row is written to ``mental_prices`` and a best-
    effort narrative memory (``scope='self'``, ``scope_ref_id=listing_id``)
    is appended so the agent can later recall their own price
    trajectory.
    """
    if role not in _ROLES:
        raise ValueError(f"invalid role {role!r}; expected one of {_ROLES}")
    if stage not in _STAGES:
        raise ValueError(f"invalid stage {stage!r}; expected one of {_STAGES}")

    listing = _listing_row(conn, listing_id)
    if listing is None:
        return None
    persona = _persona_block(conn, agent_id)
    market_baseline = _market_baseline_cents(
        conn, category=listing["category"], up_to_tick=tick,
    )
    priors = _prior_probes(
        conn, agent_id=agent_id, listing_id=listing_id, role=role,
    )
    recent = (
        _recent_messages(conn, listing_id=listing_id, agent_id=agent_id)
        if stage == "after_chat" else []
    )

    user_prompt = _build_user_prompt(
        persona=persona, listing=listing,
        market_baseline_cents=market_baseline,
        prior_probes=priors, recent_messages=recent,
        role=role, stage=stage,
    )
    prompt_hash = hashlib.sha256(
        (_SYSTEM_PROMPT + "||" + user_prompt).encode("utf-8")
    ).hexdigest()
    sampling: dict[str, Any] = {
        "temperature": temperature, "max_tokens": max_tokens, "model": model,
    }
    extra: dict[str, Any] = {}
    if reasoning_effort is not None and _backend_accepts(
        backend, "reasoning_effort",
    ):
        extra["reasoning_effort"] = reasoning_effort
        sampling["reasoning_effort"] = reasoning_effort

    t0 = time.monotonic()
    try:
        resp = backend.generate(
            [LLMMessage("system", _SYSTEM_PROMPT),
             LLMMessage("user", user_prompt)],
            model=model, max_tokens=max_tokens, temperature=temperature,
            **extra,
        )
        response_text = resp.text or ""
        latency_ms = int((time.monotonic() - t0) * 1000)
        log_llm_call(
            conn, tick=tick, agent_id=agent_id, model=model,
            backend=type(backend).__name__,
            prompt_hash=prompt_hash, prompt_text=user_prompt,
            sampling_params=sampling, response_text=response_text,
            tool_calls=None, seed=None, cache_hit=False,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        log_llm_call(
            conn, tick=tick, agent_id=agent_id, model=model,
            backend=type(backend).__name__,
            prompt_hash=prompt_hash, prompt_text=user_prompt,
            sampling_params=sampling,
            response_text=f"__probe_error__: {exc}",
            tool_calls=None, seed=None, cache_hit=False,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return None

    price, rationale = _parse_probe_response(response_text)
    if price is None:
        return None

    _insert_row(
        conn, listing_id=listing_id, agent_id=agent_id, role=role,
        stage=stage, mental_price_cents=price,
        market_baseline_cents=market_baseline, rationale=rationale,
        tick=tick,
    )

    try:
        from bazaar.memory import get_store
        store = get_store(conn)
        content = (
            f"Mental price probe ({role}, {stage}) on listing "
            f"{listing_id} at tick {tick}: "
            f"${price / 100:.2f}. {rationale}"
        ).strip()
        store.add(
            agent_id=agent_id, scope="self", scope_ref_id=listing_id,
            content=content, tick=tick,
        )
    except Exception:
        pass

    return price

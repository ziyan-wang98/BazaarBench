"""R14b Part A — mental-price probe unit tests.

Covers the live-path obligations of
``bazaar.memory.mental_price.probe_mental_price``:

* happy path — a well-formed JSON reply writes one ``mental_prices``
  row, returns the price, and appends one narrative memory (best-
  effort) scoped to ``self`` on the listing.
* failure paths — missing listing, backend exception, unparseable
  reply, negative price — all return ``None`` without raising and
  without writing a ``mental_prices`` row.
* prompt construction — market baseline, prior probes, and recent
  messages thread through the user text so the LLM sees the full
  context; ``after_chat`` stage injects messages, others don't.
* ``reasoning_effort`` is forwarded only when the backend's
  ``generate`` signature accepts it (OpenAI) and silently skipped
  otherwise (Anthropic/Ollama). The probe row is written in both
  cases.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from bazaar.agents.llm_backends.base import LLMMessage, LLMResponse
from bazaar.agents.persona import generate_persona
from bazaar.core.schema import initialize_db
from bazaar.memory import HashEncoder, NarrativeStore, install_store
from bazaar.memory.mental_price import (
    _build_user_prompt,
    _parse_probe_response,
    probe_mental_price,
)

# ---- backends ---------------------------------------------------------------


class _ScriptedBackend:
    """Replies with a canned text body. No ``reasoning_effort`` kwarg."""

    def __init__(self, text: str = '{"mental_price_cents": 4200, '
                                   '"rationale": "fair for condition"}') -> None:
        self.text = text
        self.calls = 0
        self.last_kwargs: dict = {}

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.last_kwargs = {
            "model": model, "max_tokens": max_tokens,
            "temperature": temperature, "tools": tools,
        }
        return LLMResponse(
            text=self.text, total_s=0.01, first_token_s=0.005,
            prompt_tokens=10, output_tokens=5, model=model,
        )

    def list_models(self) -> list:
        return []


class _ReasoningBackend:
    """Advertises ``reasoning_effort`` in its signature (OpenAI-style)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.captured_reasoning: str | None = None

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        self.captured_reasoning = reasoning_effort
        return LLMResponse(
            text=self.text, total_s=0.01, model=model,
        )

    def list_models(self) -> list:
        return []


class _RaisingBackend:
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools: list | None = None,
    ) -> LLMResponse:
        raise RuntimeError("backend is down")

    def list_models(self) -> list:
        return []


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = pathlib.Path(tmp) / "t.db"
        conn = initialize_db(db_path)
        install_store(conn, NarrativeStore(conn, encoder=HashEncoder(dim=16)))
        yield conn
        conn.close()


def _seed_agent(conn, agent_id: int = 1, *, seed: int = 42):
    p = generate_persona(agent_id, seed=seed)
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat,
             home_lng, activity_rate, privacy_awareness, device,
             persona_json, parent_agent_id, created_at_tick, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 'active')
        """,
        (p.agent_id, p.user_name, p.display_name, p.home_zip,
         p.home_lat, p.home_lng, p.activity_rate,
         p.privacy_awareness, p.device,
         json.dumps(p.to_dict(), ensure_ascii=False, default=str)),
    )
    conn.commit()
    return p


def _seed_listing(
    conn, *, listing_id: int = 100, owner_agent_id: int | None = None,
    category: str = "electronics-cameras", price_cents: int = 5000,
    title: str = "Used camera", desc: str = "works",
    created_at_tick: int = 0,
) -> int:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat,
             location_lng, is_phantom, view_count, save_count,
             inquiry_count, created_at_tick, status)
        VALUES (?, ?, ?, ?, ?, ?, 'good', '94110', 0.0, 0.0,
                0, 0, 0, 0, ?, 'active')
        """,
        (listing_id, owner_agent_id, category, title, desc,
         price_cents, created_at_tick),
    )
    conn.commit()
    return listing_id


# ---- happy path -------------------------------------------------------------


def test_probe_writes_row_on_success(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend(
        text='{"mental_price_cents": 3500, '
             '"rationale": "cheaper used examples on the market"}'
    )
    price = probe_mental_price(
        conn, backend=backend, model="gpt-4.1-mini",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=5,
    )
    assert price == 3500
    assert backend.calls == 1
    row = conn.execute(
        "SELECT listing_id, agent_id, role, stage, mental_price_cents, "
        "market_baseline_cents, rationale, tick "
        "FROM mental_prices WHERE listing_id = 100"
    ).fetchone()
    assert row is not None
    assert row[0:5] == (100, 1, "buyer", "initial", 3500)
    assert row[5] == 5000  # baseline = sole listing's price
    assert "cheaper" in row[6]
    assert row[7] == 5


def test_probe_returns_int_and_logs_llm_call(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend()
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=7,
    )
    calls = conn.execute(
        "SELECT agent_id, model, backend, tick FROM llm_calls"
    ).fetchall()
    assert len(calls) == 1
    assert calls[0][0] == 1
    assert calls[0][1] == "m"
    assert calls[0][3] == 7


def test_probe_appends_narrative_memory(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend(
        text='{"mental_price_cents": 2100, "rationale": "stretch budget"}'
    )
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=1,
    )
    rows = conn.execute(
        "SELECT scope, scope_ref_id, content FROM narrative_memories "
        "WHERE agent_id = 1"
    ).fetchall()
    assert rows, "probe should append at least one narrative memory"
    assert any(r[0] == "self" and r[1] == 100 for r in rows)
    assert any("stretch budget" in r[2] for r in rows)


# ---- failure paths ----------------------------------------------------------


def test_probe_returns_none_when_listing_missing(conn):
    _seed_agent(conn)
    backend = _ScriptedBackend()
    result = probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=999, role="buyer",
        stage="initial", tick=0,
    )
    assert result is None
    assert backend.calls == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM mental_prices"
    ).fetchone()[0] == 0


def test_probe_returns_none_on_backend_exception(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    result = probe_mental_price(
        conn, backend=_RaisingBackend(), model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=0,
    )
    assert result is None
    assert conn.execute(
        "SELECT COUNT(*) FROM mental_prices"
    ).fetchone()[0] == 0
    row = conn.execute(
        "SELECT response_text FROM llm_calls"
    ).fetchone()
    assert row is not None
    assert "__probe_error__" in row[0]


def test_probe_returns_none_on_unparseable_reply(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend(text="I would pay about forty bucks I guess")
    result = probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=0,
    )
    assert result is None
    assert conn.execute(
        "SELECT COUNT(*) FROM mental_prices"
    ).fetchone()[0] == 0


def test_probe_returns_none_on_negative_price(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend(
        text='{"mental_price_cents": -10, "rationale": "nope"}'
    )
    result = probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=0,
    )
    assert result is None


def test_probe_rejects_invalid_role_or_stage(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend()
    with pytest.raises(ValueError):
        probe_mental_price(
            conn, backend=backend, model="m", agent_id=1,
            listing_id=100, role="observer", stage="initial", tick=0,
        )
    with pytest.raises(ValueError):
        probe_mental_price(
            conn, backend=backend, model="m", agent_id=1,
            listing_id=100, role="buyer", stage="lurking", tick=0,
        )


# ---- JSON parser tolerance --------------------------------------------------


def test_parse_response_accepts_fenced_json():
    price, rat = _parse_probe_response(
        '```json\n{"mental_price_cents": 4200, "rationale": "ok"}\n```'
    )
    assert price == 4200
    assert rat == "ok"


def test_parse_response_accepts_prose_preamble():
    price, rat = _parse_probe_response(
        'Reasoning: I think 5000 is fair.\n'
        '{"mental_price_cents": 5000, "rationale": "market-anchored"}'
    )
    assert price == 5000
    assert "market-anchored" in rat


def test_parse_response_rejects_missing_price():
    price, rat = _parse_probe_response(
        '{"rationale": "missing price field"}'
    )
    assert price is None


# ---- prompt composition -----------------------------------------------------


def test_prompt_mentions_market_baseline_and_priors_in_after_chat(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    # A prior probe so the prompt shows a trajectory.
    conn.execute(
        "INSERT INTO mental_prices (listing_id, agent_id, role, stage, "
        "mental_price_cents, market_baseline_cents, rationale, tick, "
        "created_at) VALUES (100, 1, 'buyer', 'initial', 4000, 5000, "
        "'starting anchor', 0, 'now')"
    )
    # A message on a thread so after_chat has something to render.
    # Agent 2 has to be created before the thread references it.
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES "
        "(2, 'u2', 'U2', '94110', 0.0, 0.0, '{}')"
    )
    conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, status) VALUES "
        "(10, 100, 1, 2, 0, 'open')"
    )
    conn.execute(
        "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
        "content_hash) VALUES (10, 2, 2, 'My floor is 4800', 'h')"
    )
    conn.commit()

    backend = _ScriptedBackend()
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="after_chat", tick=3,
    )
    prompt_text = conn.execute(
        "SELECT prompt_text FROM llm_calls"
    ).fetchone()[0]
    assert "Market baseline" in prompt_text
    assert "$50.00" in prompt_text  # 5000 cents
    assert "stage=initial" in prompt_text
    assert "$40.00" in prompt_text  # prior probe dollars
    assert "Recent messages in the thread:" in prompt_text
    assert "My floor is 4800" in prompt_text


def test_prompt_omits_messages_when_stage_not_after_chat(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, status) VALUES "
        "(10, 100, 1, 1, 0, 'open')"
    )
    conn.execute(
        "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
        "content_hash) VALUES (10, 1, 0, 'self-note', 'h')"
    )
    conn.commit()

    backend = _ScriptedBackend()
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=1,
    )
    prompt_text = conn.execute(
        "SELECT prompt_text FROM llm_calls"
    ).fetchone()[0]
    assert "Recent messages in the thread:" not in prompt_text


def test_prompt_uses_seller_wording_for_seller_role(conn):
    _seed_agent(conn)
    _seed_listing(conn, owner_agent_id=1)
    backend = _ScriptedBackend()
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=1, listing_id=100, role="seller",
        stage="initial", tick=0,
    )
    prompt_text = conn.execute(
        "SELECT prompt_text FROM llm_calls"
    ).fetchone()[0]
    assert "selling" in prompt_text
    assert "least you'd accept" in prompt_text


def test_prompt_includes_financial_stress_when_present(conn):
    # Walk seeds to find a stressed persona.
    p = None
    for seed in range(60):
        cand = generate_persona(seed + 1, seed=seed)
        if cand.financial_stress is not None:
            p = cand
            break
    assert p is not None
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat,
             home_lng, activity_rate, privacy_awareness, device,
             persona_json, parent_agent_id, created_at_tick, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 'active')
        """,
        (p.agent_id, p.user_name, p.display_name, p.home_zip,
         p.home_lat, p.home_lng, p.activity_rate,
         p.privacy_awareness, p.device,
         json.dumps(p.to_dict(), ensure_ascii=False, default=str)),
    )
    _seed_listing(conn, listing_id=200)
    conn.commit()

    backend = _ScriptedBackend()
    probe_mental_price(
        conn, backend=backend, model="m",
        agent_id=p.agent_id, listing_id=200, role="buyer",
        stage="initial", tick=0,
    )
    prompt_text = conn.execute(
        "SELECT prompt_text FROM llm_calls"
    ).fetchone()[0]
    assert "Financial stress" in prompt_text
    assert p.financial_stress.consequence in prompt_text


# ---- reasoning_effort forwarding --------------------------------------------


def test_reasoning_effort_forwarded_to_openai_style_backend(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ReasoningBackend(
        text='{"mental_price_cents": 4200, "rationale": "ok"}',
    )
    probe_mental_price(
        conn, backend=backend, model="gpt-5.2-mini",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=0, reasoning_effort="low",
    )
    assert backend.captured_reasoning == "low"


def test_reasoning_effort_skipped_for_non_reasoning_backend(conn):
    _seed_agent(conn)
    _seed_listing(conn)
    backend = _ScriptedBackend()
    price = probe_mental_price(
        conn, backend=backend, model="llama3.2:3b",
        agent_id=1, listing_id=100, role="buyer",
        stage="initial", tick=0, reasoning_effort="low",
    )
    assert price == 4200
    # Sampling params still get logged without the kwarg being
    # forwarded to the backend — check logged sampling_params lacks
    # reasoning_effort.
    row = conn.execute(
        "SELECT sampling_params FROM llm_calls"
    ).fetchone()[0]
    assert "reasoning_effort" not in row


# ---- build_user_prompt direct-call sanity ----------------------------------


def test_build_user_prompt_shape():
    persona = {
        "display_name": "Jane D.",
        "profession": "graphic designer",
        "goal_description": "a cheap lens (target electronics-cameras)",
        "deadline_line": None,
        "financial_stress_line": None,
    }
    listing = {
        "listing_id": 42, "category": "electronics-cameras",
        "title": "Canon lens", "description": "EF 50mm",
        "price_cents": 5000,
    }
    text = _build_user_prompt(
        persona=persona, listing=listing, market_baseline_cents=6000,
        prior_probes=[], recent_messages=[],
        role="buyer", stage="initial",
    )
    assert "Jane D." in text
    assert "Canon lens" in text
    assert "$60.00" in text  # baseline
    assert "Goal: a cheap lens" in text
    assert "(this is your first probe for this listing)" in text
    assert 'mental_price_cents' in text


# ---------------------------------------------------------------------------
# R15 Part 4 — buyer_final probe fires BEFORE make_offer dispatch
# ---------------------------------------------------------------------------


def test_probe_one_tool_call_writes_buyer_final_for_make_offer(tmp_path):
    """``LLMPolicy._probe_one_tool_call`` fires a buyer ``final``
    probe for a ``make_offer`` call synchronously. That's the
    ordering contract ``decide()`` now depends on — the primary is
    probed here BEFORE the env dispatches it, so the
    ``mental_prices`` row is always present before the ``offers``
    row lands.
    """
    from bazaar.agents.market_agent import MarketAgent
    from bazaar.agents.policies import LLMPolicy

    db_path = tmp_path / "ordering.db"
    conn = initialize_db(db_path)
    import sqlite3 as _sqlite
    conn.row_factory = _sqlite.Row

    install_store(conn, NarrativeStore(conn, encoder=HashEncoder(dim=16)))

    # Seed: agent 1 (buyer) + seller-less listing.
    persona = generate_persona(1, seed=3)
    conn.execute(
        """
        INSERT INTO agents (agent_id, user_name, display_name,
                            home_zip, home_lat, home_lng, persona_json)
        VALUES (1, ?, ?, '94110', 0.0, 0.0, ?)
        """,
        (persona.user_name, persona.display_name,
         json.dumps(persona.to_dict(), default=str)),
    )
    conn.execute(
        """
        INSERT INTO listings (listing_id, owner_agent_id, category,
                              title, description, price_cents,
                              condition, location_zip, location_lat,
                              location_lng, is_phantom, created_at_tick,
                              status)
        VALUES (7, NULL, 'books', 'Book', '', 4000, 'good',
                '94110', 0.0, 0.0, 1, 0, 'active')
        """,
    )
    conn.commit()

    backend = _ScriptedBackend(
        '{"mental_price_cents": 3500, "rationale": "tight budget"}'
    )
    policy = LLMPolicy(
        backend=backend, model="fake-model", seed=0,
        probe_backend=backend, probe_model="fake-model",
        probe_enabled=True,
    )
    agent = MarketAgent(persona=persona, policy=policy)

    # Before the probe, no mental_prices row exists.
    assert conn.execute(
        "SELECT COUNT(*) FROM mental_prices WHERE agent_id=1"
    ).fetchone()[0] == 0

    policy._probe_one_tool_call(
        conn, agent, tick=5,
        call={"name": "make_offer",
              "arguments": {"listing_id": 7, "price_cents": 3500}},
    )

    rows = conn.execute(
        "SELECT agent_id, listing_id, role, stage, mental_price_cents, tick "
        "FROM mental_prices WHERE agent_id=1"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["listing_id"] == 7
    assert row["role"] == "buyer"
    assert row["stage"] == "final"
    assert row["mental_price_cents"] == 3500
    assert row["tick"] == 5

    # The offers table is still empty — dispatch happens outside
    # the probe path, so this is the "before" half of "probe before
    # dispatch".
    assert conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0] == 0
    conn.close()

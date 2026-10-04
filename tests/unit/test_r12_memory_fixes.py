"""R12 memory-system fixes — tests covering the four gaps.

Covers:

* Gap 1: D11 memory consolidation auto-registers when the default
  registry is built with an LLM backend.
* Gap 2: ``LLMPolicy._recall_for_tick`` derives its query from the
  current observation (focus counterparty) rather than the past
  ledger trail.
* Gap 3: ``_render_recent_thoughts`` pulls the agent's last ``n``
  ticks' LLM reasoning/response text and the ``PromptBuilder``
  injects it as a RECENT THOUGHTS block when present.
* Gap 4 covered in tests/unit/test_narrative.py (scope filter).
"""
from __future__ import annotations

import sqlite3

import pytest

from bazaar.agents.persona import generate_persona
from bazaar.agents.prompt import (
    RECENT_THOUGHTS_HEADER,
    PromptBuilder,
    _render_recent_thoughts,
)
from bazaar.core.event_log import log_llm_call
from bazaar.core.schema import initialize_db
from bazaar.core.tick_clock import TICKS_PER_DAY


@pytest.fixture
def conn(tmp_path):
    c = initialize_db(tmp_path / "r12.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def _seed_agent(conn: sqlite3.Connection, agent_id: int) -> None:
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (agent_id, f"u{agent_id}", f"U{agent_id}", "00000",
         0.0, 0.0, "{}"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Gap 1 — default_registry wires D11/D12 when a backend is supplied
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal backend stand-in; D11 wiring only checks it's non-None."""

    def generate(self, *args, **kwargs):  # pragma: no cover - not called
        raise NotImplementedError


def test_default_registry_without_backend_omits_d11():
    from bazaar.dynamics import default_registry
    reg = default_registry()
    names = {s.name for s in reg.specs}
    assert "D11_memory_consolidation" not in names
    assert "D12_self_portrait" not in names


def test_default_registry_with_backend_registers_d11_and_d12():
    """R12 Gap 1: LLM runs must get D11 + D12 out of the box."""
    from bazaar.dynamics import default_registry
    reg = default_registry(
        llm_backend=_FakeBackend(), llm_model="fake-model",
    )
    names = {s.name for s in reg.specs}
    assert "D11_memory_consolidation" in names
    assert "D12_self_portrait" in names


def test_default_registry_d11_interval_is_one_day_by_default():
    """Once per simulated day — matches the core tick clock."""
    from bazaar.dynamics import default_registry
    reg = default_registry(
        llm_backend=_FakeBackend(), llm_model="fake-model",
    )
    d11 = next(s for s in reg.specs if s.name == "D11_memory_consolidation")
    assert d11.interval == TICKS_PER_DAY


def test_default_registry_can_defer_initial_llm_dynamics():
    """Scale-up runs can avoid all-agent reflection at tick 0."""
    from bazaar.dynamics import default_registry
    reg = default_registry(
        llm_backend=_FakeBackend(), llm_model="fake-model",
        d11_interval=TICKS_PER_DAY,
        d12_interval=4 * TICKS_PER_DAY,
        d14_interval=TICKS_PER_DAY,
        d11_start_tick=TICKS_PER_DAY,
        d12_start_tick=4 * TICKS_PER_DAY,
        d14_start_tick=TICKS_PER_DAY,
    )
    by_name = {s.name: s for s in reg.specs}
    assert by_name["D11_memory_consolidation"].should_fire(0) is False
    assert by_name["D11_memory_consolidation"].should_fire(TICKS_PER_DAY) is True
    assert by_name["D14_agent_summary"].should_fire(0) is False
    assert by_name["D14_agent_summary"].should_fire(TICKS_PER_DAY) is True


def test_default_registry_backend_without_model_skips_llm_dynamics():
    """Symmetric gate: both backend + model must be present to register."""
    from bazaar.dynamics import default_registry
    reg = default_registry(llm_backend=_FakeBackend(), llm_model=None)
    names = {s.name for s in reg.specs}
    assert "D11_memory_consolidation" not in names


def test_d11_fires_on_schedule_within_72_tick_sim():
    """Smoke-level end-to-end: run 13 ticks with D11 registered at
    one-day intervals and verify the callback actually fires (no crashes,
    event log contains the consolidation row)."""
    from pathlib import Path

    from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy
    from bazaar.dynamics import default_registry
    from bazaar.memory import HashEncoder, NarrativeStore, install_store

    tmp = Path("/tmp/bazaar_r12_d11.db")
    if tmp.exists():
        tmp.unlink()

    reg = default_registry(
        llm_backend=_FakeBackend(), llm_model="fake-model",
    )
    env = BazaarEnv(db_path=tmp, dynamics=reg)
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    for i in range(2):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=1),
            policy=RandomBenignPolicy(seed=i),
        ))
    env.reset()

    # Seed a handful of narrative rows so D11 has candidates to fold.
    from bazaar.memory import get_store
    store = get_store(env.platform.conn)
    for i in range(8):
        store.add(agent_id=1, scope="self",
                  content=f"self note {i}", tick=i)
    env.platform.conn.commit()

    # Advance 13 ticks. D11 fires at tick 0, 12 — should observe at
    # least one consolidation event (tick 12) given the batch size.
    reports = env.step_many(TICKS_PER_DAY + 1)
    fired_d11 = sum(
        r.dynamics_fired.get("D11_memory_consolidation", 0)
        for r in reports
    )
    assert fired_d11 > 0, "D11 consolidation never fired"
    env.close()
    if tmp.exists():
        tmp.unlink()


# ---------------------------------------------------------------------------
# Gap 2 — LLMPolicy._recall_for_tick uses focus counterparty when set
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Test double for NarrativeStore that captures recall args."""

    def __init__(self):
        self.calls: list[dict] = []

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        return []


def test_recall_query_prefers_focus_counterparty(monkeypatch, conn):
    """R12 Gap 2: when focus has counterparty_id, recall should be
    anchored on that agent and scoped to counterparty memories."""
    from bazaar.agents.policies import LLMPolicy

    _seed_agent(conn, 1)
    _seed_agent(conn, 42)
    # threads.listing_id is NOT NULL — seed a listing owned by 42.
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick) VALUES "
        "(?, ?, 'bikes', 't', 'd', 100, 'good', '00000', 0, 0, 0)",
        (7, 42),
    )
    # Open a thread so _infer_focus returns counterparty_id=42.
    conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, status, created_at_tick, last_msg_tick) "
        "VALUES (?, ?, ?, ?, 'open', 0, 0)",
        (1, 7, 1, 42),
    )
    conn.commit()

    recording = _RecordingStore()

    import bazaar.memory as mem_pkg
    monkeypatch.setattr(mem_pkg, "get_store", lambda _conn: recording)

    policy = LLMPolicy(backend=_FakeBackend(), model="fake-model",
                       recall_k=3, seed=0)
    policy._recall_for_tick(conn, agent_id=1, tick=5)

    # At least one call with scope='counterparty' + scope_ref_id=42.
    cp_calls = [c for c in recording.calls
                if c.get("scope") == "counterparty"
                and c.get("scope_ref_id") == 42]
    assert cp_calls, (
        "Gap 2 broken: focus counterparty#42 but no "
        "scope=counterparty recall was issued."
    )
    # Query references the counterparty, not just ledger history.
    assert "agent#42" in cp_calls[0]["query"]


def test_recall_query_falls_back_to_ledger_without_focus(monkeypatch, conn):
    """When no focus counterparty exists the recall still proceeds
    via the old ledger-context path, so agents early in a run aren't
    left with an empty PRIOR IMPRESSIONS block."""
    from bazaar.agents.policies import LLMPolicy
    from bazaar.memory.ledger import LedgerEntry, record_ledger_entry

    _seed_agent(conn, 1)
    _seed_agent(conn, 99)
    # No thread → _infer_focus returns {}. But a ledger entry from a
    # past interaction should still drive the fallback recall.
    record_ledger_entry(conn, LedgerEntry(
        agent_id=1, kind="rating", counterparty_id=99,
        ref_table="ratings", ref_id=1,
        summary="received 5-star rating from agent#99", tick=2,
    ))
    conn.commit()

    recording = _RecordingStore()
    import bazaar.memory as mem_pkg
    monkeypatch.setattr(mem_pkg, "get_store", lambda _conn: recording)

    policy = LLMPolicy(backend=_FakeBackend(), model="fake-model", seed=0)
    policy._recall_for_tick(conn, agent_id=1, tick=5)

    # Fallback branch: a recall without scope gets issued.
    unscoped = [c for c in recording.calls if c.get("scope") is None]
    assert unscoped, "fallback ledger-context recall should fire"


# ---------------------------------------------------------------------------
# Gap 3 — RECENT THOUGHTS verbatim block
# ---------------------------------------------------------------------------


def test_render_recent_thoughts_empty_when_no_llm_calls(conn):
    _seed_agent(conn, 1)
    assert _render_recent_thoughts(conn, agent_id=1, tick=10, n=2) == ""


def test_render_recent_thoughts_returns_last_n_in_desc_order(conn):
    """R12 Gap 3: helper returns the two most-recent ticks' reasoning."""
    _seed_agent(conn, 1)
    for t, summary in [(10, "first plan"), (11, "revised plan"),
                       (12, "latest plan")]:
        log_llm_call(
            conn, tick=t, agent_id=1, model="x", backend="Fake",
            prompt_hash=f"h{t}", prompt_text=None,
            sampling_params={}, response_text="",
            tool_calls=None, reasoning_summary=summary,
        )
    conn.commit()

    out = _render_recent_thoughts(conn, agent_id=1, tick=13, n=2)
    assert RECENT_THOUGHTS_HEADER.format(n=2) in out
    assert "latest plan" in out
    assert "revised plan" in out
    # Most recent first — tick 12 appears before tick 11.
    assert out.index("t=12") < out.index("t=11")
    # Older tick (10) falls outside the n=2 window.
    assert "first plan" not in out


def test_render_recent_thoughts_prefers_reasoning_falls_back_to_text(conn):
    """Gap 3: COALESCE(reasoning_summary, response_text) — when reasoning
    is NULL the response_text shows up instead."""
    _seed_agent(conn, 1)
    log_llm_call(
        conn, tick=5, agent_id=1, model="x", backend="Fake",
        prompt_hash="h5", prompt_text=None,
        sampling_params={}, response_text="visible assistant output",
        tool_calls=None,
    )
    conn.commit()
    out = _render_recent_thoughts(conn, agent_id=1, tick=6, n=2)
    assert "visible assistant output" in out


def test_render_recent_thoughts_prefers_external_decision_note(conn):
    _seed_agent(conn, 1)
    log_llm_call(
        conn, tick=5, agent_id=1, model="x", backend="Fake",
        prompt_hash="h5", prompt_text=None,
        sampling_params={}, response_text=(
            "Decision note: I already agreed to on-platform shipment, "
            "so I should schedule shipping instead of a meetup."
        ),
        tool_calls=None,
        reasoning_summary="Internal summary should not be shown first.",
    )
    conn.commit()

    out = _render_recent_thoughts(conn, agent_id=1, tick=6, n=2)

    assert "Decision note:" in out
    assert "schedule shipping" in out
    assert "Internal summary" not in out


def test_render_recent_thoughts_falls_back_to_tool_calls(conn):
    """Tool-call-only chat completions still provide working memory."""
    _seed_agent(conn, 1)
    log_llm_call(
        conn, tick=5, agent_id=1, model="x", backend="Fake",
        prompt_hash="h5", prompt_text=None,
        sampling_params={}, response_text="",
        tool_calls=[{
            "name": "make_offer",
            "arguments": {
                "listing_id": 77,
                "price_cents": 5500,
                "terms": {
                    "fulfillment": "shipment",
                    "payment_method": "on_platform",
                },
            },
        }],
        reasoning_summary=None,
    )
    conn.commit()

    out = _render_recent_thoughts(conn, agent_id=1, tick=6, n=2)

    assert "Tool action memory" in out
    assert "make_offer" in out
    assert "listing_id=77" in out
    assert "shipment" in out


def test_render_recent_thoughts_skips_rows_after_current_tick(conn):
    """Rows at or after `tick` are future — must not leak in."""
    _seed_agent(conn, 1)
    log_llm_call(
        conn, tick=10, agent_id=1, model="x", backend="Fake",
        prompt_hash="h10", prompt_text=None, sampling_params={},
        response_text="past", tool_calls=None,
    )
    log_llm_call(
        conn, tick=20, agent_id=1, model="x", backend="Fake",
        prompt_hash="h20", prompt_text=None, sampling_params={},
        response_text="future", tool_calls=None,
    )
    conn.commit()
    out = _render_recent_thoughts(conn, agent_id=1, tick=15, n=5)
    assert "past" in out
    assert "future" not in out


def test_render_recent_thoughts_skips_whitespace_only_rows(conn):
    _seed_agent(conn, 1)
    log_llm_call(
        conn, tick=1, agent_id=1, model="x", backend="Fake",
        prompt_hash="h1", prompt_text=None, sampling_params={},
        response_text="   ", tool_calls=None, reasoning_summary=None,
    )
    log_llm_call(
        conn, tick=2, agent_id=1, model="x", backend="Fake",
        prompt_hash="h2", prompt_text=None, sampling_params={},
        response_text="real content", tool_calls=None,
    )
    conn.commit()
    out = _render_recent_thoughts(conn, agent_id=1, tick=5, n=5)
    assert "real content" in out
    assert "t=1" not in out


def test_prompt_includes_recent_thoughts_block_when_llm_calls_exist(conn):
    """End-to-end: PromptBuilder.build surfaces RECENT THOUGHTS when
    there are prior llm_calls for this agent."""
    p = generate_persona(1, seed=42)
    _seed_agent(conn, 1)
    # Fix persona.agent_id to match the seeded row.
    object.__setattr__(p, "agent_id", 1)
    log_llm_call(
        conn, tick=3, agent_id=1, model="x", backend="Fake",
        prompt_hash="h", prompt_text=None, sampling_params={},
        response_text="", tool_calls=None,
        reasoning_summary="Remember: the bike at listing 10 fits my budget.",
    )
    conn.commit()

    built = PromptBuilder(persona=p).build(conn=conn, tick=5)
    assert "RECENT THOUGHTS" in built.user_text
    assert "bike at listing 10" in built.user_text
    # Block sits between PRIOR IMPRESSIONS and SITUATION.
    prior_at = built.user_text.find("# PRIOR IMPRESSIONS")
    recent_at = built.user_text.find("## RECENT THOUGHTS")
    situation_at = built.user_text.find("## SITUATION")
    assert prior_at < recent_at < situation_at


def test_prompt_omits_recent_thoughts_block_when_no_llm_calls(conn):
    """When the agent has no prior calls (fresh agent), the RECENT
    THOUGHTS header should NOT be rendered — we don't want an empty
    block cluttering the prompt."""
    p = generate_persona(1, seed=42)
    _seed_agent(conn, 1)
    object.__setattr__(p, "agent_id", 1)
    built = PromptBuilder(persona=p).build(conn=conn, tick=5)
    assert "RECENT THOUGHTS" not in built.user_text

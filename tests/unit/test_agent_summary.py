"""R14a Part A — D14 rolling self-summary + prompt injection.

These tests exercise the schema, the dynamic callback (both offline
fallback and backend-driven paths with a fake backend), and the
``## MY CURRENT STATE`` block in the prompt output. They stay offline
by default — no real LLM is called — but the ``FakeBackend`` drives
the backend-path branch so we can assert that successful calls tag
rows ``source='D14'`` while failures tag ``source='fallback'``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bazaar import (
    BazaarEnv,
    MarketAgent,
    RandomBenignPolicy,
    generate_persona,
)
from bazaar.agents.llm_backends.base import (
    LLMMessage,
    LLMResponse,
)
from bazaar.agents.prompt import (
    FOCUS_HEADER,
    MY_CURRENT_STATE_HEADER,
    PromptBuilder,
    _fetch_latest_summary,
)
from bazaar.dynamics import DynamicRegistry
from bazaar.dynamics.llm_dynamics import make_d14_agent_summary


class _FakeBackend:
    """Canned-response backend that keeps a call log.

    Implements the minimal ``generate`` surface so D14 can call it
    without pulling in a real provider. Useful for asserting that the
    reflection loop is actually invoked and what it returns ends up
    in ``agent_summary``.
    """

    def __init__(self, response_text: str = "I'm in the middle of negotiating."):
        self.calls: list[list[LLMMessage]] = []
        self.response_text = response_text

    def generate(
        self, messages: list[LLMMessage], *, model: str,
        max_tokens: int = 300, temperature: float = 0.4,
        **_: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(text=self.response_text)


class _FailingBackend:
    def generate(self, *a: Any, **kw: Any) -> LLMResponse:
        raise RuntimeError("simulated backend outage")


def _fresh_env(tmp_path: Path, *, n_agents: int = 2) -> BazaarEnv:
    env = BazaarEnv(
        db_path=tmp_path / "d14.db",
        dynamics=DynamicRegistry(),  # empty — no auto dynamics
    )
    for i in range(n_agents):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=42 + i),
            policy=RandomBenignPolicy(seed=i),
        ))
    env.reset()
    env.step_many(2)  # so agents exist with a tick history
    return env


def test_agent_summary_table_exists(tmp_path: Path):
    """Schema v2 has the agent_summary table after init."""
    env = _fresh_env(tmp_path)
    row = env.platform.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_summary'"
    ).fetchone()
    assert row is not None
    env.close()


def test_d14_fallback_writes_one_row_per_agent(tmp_path: Path):
    """Without a backend, D14 still writes a fallback summary per agent
    so the prompt block is never empty in CI runs."""
    env = _fresh_env(tmp_path, n_agents=3)
    cb = make_d14_agent_summary(backend=None, model=None)
    import random as _r
    written = cb(env.platform.conn, tick=5, rng=_r.Random(0))
    assert written == 3
    rows = env.platform.conn.execute(
        "SELECT agent_id, source FROM agent_summary ORDER BY agent_id"
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]
    assert all(r[1] == "fallback" for r in rows)
    env.close()


def test_d14_backend_path_tags_source_d14(tmp_path: Path):
    """Given a working backend, D14 calls it and tags rows 'D14'."""
    env = _fresh_env(tmp_path, n_agents=1)
    backend = _FakeBackend(
        response_text="Quietly watching the bike listings."
    )
    cb = make_d14_agent_summary(backend=backend, model="fake-model")
    import random as _r
    n = cb(env.platform.conn, tick=5, rng=_r.Random(0))
    assert n == 1
    row = env.platform.conn.execute(
        "SELECT content, source FROM agent_summary WHERE agent_id = 1"
    ).fetchone()
    assert row[0] == "Quietly watching the bike listings."
    assert row[1] == "D14"
    # One call was routed through the backend.
    assert len(backend.calls) == 1
    env.close()


def test_d14_backend_failure_falls_back(tmp_path: Path):
    """If the backend raises, D14 must not crash — row tagged 'fallback'."""
    env = _fresh_env(tmp_path, n_agents=1)
    cb = make_d14_agent_summary(
        backend=_FailingBackend(), model="fake-model",
    )
    import random as _r
    n = cb(env.platform.conn, tick=5, rng=_r.Random(0))
    assert n == 1
    row = env.platform.conn.execute(
        "SELECT source FROM agent_summary WHERE agent_id = 1"
    ).fetchone()
    assert row[0] == "fallback"
    env.close()


def test_fetch_latest_summary_returns_most_recent(tmp_path: Path):
    """``_fetch_latest_summary`` picks the latest row ≤ tick."""
    env = _fresh_env(tmp_path, n_agents=1)
    env.platform.conn.execute(
        "INSERT INTO agent_summary (agent_id, tick, content, source) "
        "VALUES (1, 10, 'older state', 'fallback')"
    )
    env.platform.conn.execute(
        "INSERT INTO agent_summary (agent_id, tick, content, source) "
        "VALUES (1, 20, 'newer state', 'D14')"
    )
    env.platform.conn.commit()
    assert "newer state" in _fetch_latest_summary(env.platform.conn, 1, tick=25)
    assert "older state" in _fetch_latest_summary(env.platform.conn, 1, tick=15)
    # At tick=5 (before any summary), block is empty.
    assert _fetch_latest_summary(env.platform.conn, 1, tick=5) == ""
    env.close()


def test_prompt_builder_injects_my_current_state_block(tmp_path: Path):
    """After D14 writes a summary, PromptBuilder embeds ## MY CURRENT
    STATE verbatim above the retrieval blocks."""
    env = _fresh_env(tmp_path, n_agents=1)
    env.platform.conn.execute(
        "INSERT INTO agent_summary (agent_id, tick, content, source) "
        "VALUES (1, 1, 'Currently haggling with agent#4 over the drill.', 'D14')"
    )
    env.platform.conn.commit()

    # Install a narrative store so PromptBuilder.build works.
    from bazaar.memory import HashEncoder, NarrativeStore, install_store
    install_store(env.platform.conn, NarrativeStore(
        env.platform.conn, encoder=HashEncoder(dim=32),
    ))
    agent = env.agents[0]
    builder = PromptBuilder(persona=agent.persona)
    built = builder.build(conn=env.platform.conn, tick=3)
    assert MY_CURRENT_STATE_HEADER in built.user_text
    assert "Currently haggling with agent#4" in built.user_text
    # Summary block appears BEFORE PRIOR IMPRESSIONS block.
    idx_state = built.user_text.index(MY_CURRENT_STATE_HEADER)
    idx_prior = built.user_text.index("# PRIOR IMPRESSIONS")
    assert idx_state < idx_prior
    env.close()


def test_focus_header_above_prior_when_thread_exists(tmp_path: Path):
    """When the agent has a focus thread, the ## FOCUS block sits
    between MY CURRENT STATE and PRIOR IMPRESSIONS."""
    env = _fresh_env(tmp_path, n_agents=2)
    # Seed a listing + thread + message so focus has something to pull.
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick) VALUES "
        "(1, 2, 'tools', 'drill', 'd', 100, 'good', '00000', 0, 0, 0)"
    )
    env.platform.conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (7, 1, 1, 2, 0, 1, 'open')"
    )
    env.platform.conn.execute(
        "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
        "content_hash) VALUES (7, 2, 1, 'still available', 'h1')"
    )
    env.platform.conn.execute(
        "INSERT INTO agent_summary (agent_id, tick, content, source) "
        "VALUES (1, 1, 'On it.', 'D14')"
    )
    env.platform.conn.commit()

    from bazaar.memory import HashEncoder, NarrativeStore, install_store
    install_store(env.platform.conn, NarrativeStore(
        env.platform.conn, encoder=HashEncoder(dim=32),
    ))
    agent = env.agents[0]
    builder = PromptBuilder(persona=agent.persona)
    built = builder.build(
        conn=env.platform.conn, tick=3,
        focus={"thread_id": 7, "listing_id": 1, "counterparty_id": 2},
    )
    assert FOCUS_HEADER in built.user_text
    idx_state = built.user_text.index(MY_CURRENT_STATE_HEADER)
    idx_focus = built.user_text.index(FOCUS_HEADER)
    idx_prior = built.user_text.index("# PRIOR IMPRESSIONS")
    assert idx_state < idx_focus < idx_prior
    # Message body leaks through verbatim.
    assert "still available" in built.user_text
    env.close()


def test_d14_registers_in_default_registry(tmp_path: Path):
    """default_registry wires D14 even in offline mode (fallback writes)."""
    from bazaar.dynamics import default_registry
    reg = default_registry()
    assert any(s.name == "D14_agent_summary" for s in reg.specs)


def test_d14_uses_previous_summary_for_continuity(tmp_path: Path):
    """Subsequent firings receive the prior summary in the prompt so
    the new summary can be a delta, not a restart."""
    env = _fresh_env(tmp_path, n_agents=1)
    backend = _FakeBackend(response_text="summary #2")
    cb = make_d14_agent_summary(backend=backend, model="fake-model")

    import random as _r
    cb(env.platform.conn, tick=5, rng=_r.Random(0))
    backend.response_text = "summary #2 (new)"
    cb(env.platform.conn, tick=10, rng=_r.Random(0))

    # Second call's prompt must mention the first summary's content.
    assert len(backend.calls) == 2
    second_user_msg = backend.calls[1][1].content
    assert "summary #2" in second_user_msg
    env.close()

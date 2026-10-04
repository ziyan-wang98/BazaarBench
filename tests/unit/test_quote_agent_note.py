"""Tests for the QUOTE_AGENT_NOTE action + gating (T28c-5)."""
from __future__ import annotations

import pytest

from bazaar.actions.dispatch import dispatch
from bazaar.actions.types import ActionType
from bazaar.core.env import BazaarEnv
from bazaar.memory import HashEncoder, NarrativeStore, install_store


def _seed_agents(conn) -> None:
    conn.executemany(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "a1", "Alice", "00001", 0.0, 0.0, "{}"),
            (2, "a2", "Bob",   "00002", 0.0, 0.0, "{}"),
        ],
    )
    conn.commit()


@pytest.fixture
def env_gated_off(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "g.db",
                    allow_cross_agent_notes=False)
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=HashEncoder(dim=16)))
    _seed_agents(env.platform.conn)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def env_gated_on(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "g.db",
                    allow_cross_agent_notes=True)
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=HashEncoder(dim=16)))
    _seed_agents(env.platform.conn)
    try:
        yield env
    finally:
        env.close()


def test_gated_off_returns_blocked(env_gated_off) -> None:
    result = dispatch(
        env_gated_off.platform.conn,
        agent_id=1, action=ActionType.QUOTE_AGENT_NOTE,
        raw_args={
            "source_agent_id": 2,
            "content": "Bob said the seller was quick.",
        }, tick=1,
    )
    assert result.status == "blocked"
    assert result.payload["reason"] == "cross_agent_notes_disabled"
    # No row written.
    n = env_gated_off.platform.conn.execute(
        "SELECT COUNT(*) FROM narrative_memories"
    ).fetchone()[0]
    assert n == 0


def test_gated_on_writes_with_provenance(env_gated_on) -> None:
    result = dispatch(
        env_gated_on.platform.conn,
        agent_id=1, action=ActionType.QUOTE_AGENT_NOTE,
        raw_args={
            "source_agent_id": 2,
            "content": "Bob said the seller was quick.",
            "scope": "counterparty",
            "scope_ref_id": 2,
        }, tick=5,
    )
    assert result.status == "ok"
    assert result.payload["provenance"] == 2
    row = env_gated_on.platform.conn.execute(
        "SELECT agent_id, provenance, content, scope FROM narrative_memories"
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == 2
    assert "Bob said" in row[2]
    assert row[3] == "counterparty"


def test_self_source_blocked(env_gated_on) -> None:
    result = dispatch(
        env_gated_on.platform.conn,
        agent_id=1, action=ActionType.QUOTE_AGENT_NOTE,
        raw_args={
            "source_agent_id": 1,
            "content": "I think I said this.",
        }, tick=1,
    )
    assert result.status == "blocked"
    assert result.payload["reason"] == "source_is_self"


def test_missing_source_returns_error(env_gated_on) -> None:
    result = dispatch(
        env_gated_on.platform.conn,
        agent_id=1, action=ActionType.QUOTE_AGENT_NOTE,
        raw_args={
            "source_agent_id": 99,
            "content": "Ghost said something.",
        }, tick=1,
    )
    assert result.status == "error"
    assert result.payload["reason"] == "source_agent_not_found"

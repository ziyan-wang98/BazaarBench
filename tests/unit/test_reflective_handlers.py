"""Tests for T13 — SUMMARIZE_SESSION and RECALL handlers.

All tests install a FakeEncoder-backed NarrativeStore on the env's
connection before dispatching any reflective action, so they never
touch sentence-transformers. The real-model path is covered by the
slow test in ``test_narrative.py``.
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.memory import NarrativeStore, clear_stores, install_store

from .test_narrative import FakeEncoder


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=20 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    # Install a fast, deterministic store BEFORE any handler runs so
    # SUMMARIZE_SESSION / RECALL never lazy-load the real MiniLM model.
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=FakeEncoder(dim=16)))
    yield env
    env.close()
    clear_stores()


# ---- SUMMARIZE_SESSION ------------------------------------------------------


def test_summarize_session_persists_narrative_memory(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SUMMARIZE_SESSION,
        raw_args={"scope": "self", "scope_ref_id": None,
                  "content": "I prefer cash-only local deals"},
        tick=3,
    )
    assert r.status == "ok"
    assert r.payload["scope"] == "self"
    assert r.payload["narrative_count"] == 1

    row = env.platform.conn.execute(
        "SELECT agent_id, scope, content, created_tick "
        "FROM narrative_memories WHERE memory_id = ?",
        (r.payload["memory_id"],),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "self"
    assert row[2] == "I prefer cash-only local deals"
    assert row[3] == 3


def test_summarize_session_scope_counterparty_with_ref(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SUMMARIZE_SESSION,
        raw_args={"scope": "counterparty", "scope_ref_id": 2,
                  "content": "agent#2 drove a fair bargain"},
        tick=7,
    )
    assert r.status == "ok"
    row = env.platform.conn.execute(
        "SELECT scope, scope_ref_id FROM narrative_memories "
        "WHERE memory_id = ?",
        (r.payload["memory_id"],),
    ).fetchone()
    assert row[0] == "counterparty"
    assert row[1] == 2


def test_summarize_session_rejects_invalid_scope(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SUMMARIZE_SESSION,
        raw_args={"scope": "galaxy", "scope_ref_id": None, "content": "x"},
        tick=0,
    )
    assert r.status == "error"
    assert r.payload["error"] == "validation"


# ---- RECALL -----------------------------------------------------------------


def test_recall_returns_ranked_hits(env):
    # Seed three memories for agent 1.
    for scope, ref, content, tick in (
        ("self", None, "I only meet buyers at coffee shops", 1),
        ("counterparty", 2, "agent#2 once tried to lowball me", 5),
        ("self", None, "Tuesdays feel slow for listings", 10),
    ):
        dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SUMMARIZE_SESSION,
                 raw_args={"scope": scope, "scope_ref_id": ref,
                           "content": content},
                 tick=tick)

    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.RECALL,
        raw_args={"query": "lowball buyers", "top_k": 2},
        tick=12,
    )
    assert r.status == "ok"
    assert len(r.payload["hits"]) == 2
    # Top hit contains 'lowball'.
    assert "lowball" in r.payload["hits"][0]["content"]
    # Scores are ordered descending.
    assert r.payload["hits"][0]["score"] >= r.payload["hits"][1]["score"]


def test_recall_is_scoped_per_agent(env):
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "self", "scope_ref_id": None,
                       "content": "agent#1 secret note"},
             tick=1)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "self", "scope_ref_id": None,
                       "content": "agent#2 secret note"},
             tick=1)

    r1 = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.RECALL,
        raw_args={"query": "secret note", "top_k": 5}, tick=2,
    )
    r2 = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.RECALL,
        raw_args={"query": "secret note", "top_k": 5}, tick=2,
    )
    c1 = [h["content"] for h in r1.payload["hits"]]
    c2 = [h["content"] for h in r2.payload["hits"]]
    assert all("agent#1" in c for c in c1)
    assert all("agent#2" in c for c in c2)


def test_recall_respects_tick_horizon(env):
    """recall() uses tick as an up_to_tick gate, so a memory created
    at tick=20 is not visible to a RECALL dispatched at tick=10."""
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "self", "scope_ref_id": None,
                       "content": "first memory"},
             tick=5)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "self", "scope_ref_id": None,
                       "content": "memory from the future"},
             tick=20)

    early = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.RECALL,
        raw_args={"query": "memory", "top_k": 5}, tick=10,
    )
    contents = [h["content"] for h in early.payload["hits"]]
    assert "first memory" in contents
    assert "memory from the future" not in contents


def test_recall_empty_store_returns_no_hits(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.RECALL,
        raw_args={"query": "anything", "top_k": 5}, tick=1,
    )
    assert r.status == "ok"
    assert r.payload["hits"] == []

"""Unit tests for T12 — narrative memory.

Fast tests run against a deterministic ``FakeEncoder`` that maps
keywords to stable unit vectors. A single ``@pytest.mark.slow`` test
exercises the real ``MiniLMEncoder`` end-to-end, gated so CI's fast
lane doesn't pay the ~80 MB download / model-load tax.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.memory import NarrativeStore, render_narrative_context


class FakeEncoder:
    """Deterministic bag-of-hashed-words encoder for fast tests.

    Same words → same coordinates → same similarity, so tests can
    make precise assertions about which memories a query retrieves.
    ``hashlib.md5`` is used instead of ``hash()`` because Python's
    built-in ``hash`` is salted per process.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in text.lower().split():
                h = int.from_bytes(
                    hashlib.md5(w.encode("utf-8")).digest()[:4],
                    "little",
                ) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=10 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


@pytest.fixture
def store(env):
    return NarrativeStore(env.platform.conn, encoder=FakeEncoder(dim=16))


# ---- Write path -------------------------------------------------------------


def test_add_persists_row_with_embedding_blob(store, env):
    mid = store.add(
        agent_id=1, scope="self",
        content="Sarah is a trustworthy buyer", tick=5,
    )
    row = env.platform.conn.execute(
        "SELECT agent_id, scope, content, created_tick, decayed, "
        "LENGTH(embedding) FROM narrative_memories WHERE memory_id = ?",
        (mid,),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "self"
    assert row[2] == "Sarah is a trustworthy buyer"
    assert row[3] == 5
    assert row[4] == 0  # not decayed
    # 16-dim float32 = 64 bytes.
    assert row[5] == 16 * 4


def test_add_many_batches_embeddings(store):
    ids = store.add_many([
        {"agent_id": 1, "scope": "self", "content": "first", "tick": 1},
        {"agent_id": 1, "scope": "self", "content": "second", "tick": 2},
        {"agent_id": 2, "scope": "counterparty",
         "content": "third", "scope_ref_id": 99, "tick": 3},
    ])
    assert len(ids) == 3
    assert store.count(agent_id=1) == 2
    assert store.count(agent_id=2) == 1


def test_add_many_empty_is_noop(store):
    assert store.add_many([]) == []


# ---- Read path --------------------------------------------------------------


def test_all_for_agent_orders_recent_first(store):
    store.add(agent_id=1, scope="self", content="old", tick=1)
    store.add(agent_id=1, scope="self", content="newer", tick=10)
    store.add(agent_id=1, scope="self", content="newest", tick=20)

    ms = store.all_for_agent(1)
    assert [m.created_tick for m in ms] == [20, 10, 1]


def test_decayed_rows_excluded_by_default(store):
    m1 = store.add(agent_id=1, scope="self", content="hello", tick=1)
    store.add(agent_id=1, scope="self", content="world", tick=2)
    store.mark_decayed(m1)

    assert store.count(agent_id=1) == 1
    assert store.count(agent_id=1, include_decayed=True) == 2


def test_up_to_tick_filters_all_and_recall(store):
    store.add(agent_id=1, scope="self", content="early event", tick=5)
    store.add(agent_id=1, scope="self", content="late event", tick=50)

    ms = store.all_for_agent(1, up_to_tick=10)
    assert len(ms) == 1
    assert ms[0].created_tick == 5

    # recall also respects the gate
    hits = store.recall(agent_id=1, query="event", top_k=5, up_to_tick=10)
    assert len(hits) == 1
    assert hits[0].content == "early event"


# ---- Recall semantics -------------------------------------------------------


def test_recall_prefers_lexically_matching_memory(store):
    """The deterministic bag-of-words encoder gives exact-match memories
    a strictly higher score than unrelated ones."""
    store.add(agent_id=1, scope="self",
              content="Sarah is a trustworthy buyer", tick=1)
    store.add(agent_id=1, scope="self",
              content="I love my red bicycle", tick=2)
    store.add(agent_id=1, scope="self",
              content="weather is nice today", tick=3)

    hits = store.recall(agent_id=1, query="trustworthy buyer", top_k=2)
    assert len(hits) == 2
    assert "trustworthy" in hits[0].content
    # Top hit's score strictly greater than second.
    assert hits[0].score > hits[1].score


def test_recall_is_scoped_per_agent(store):
    store.add(agent_id=1, scope="self", content="secret note for 1", tick=1)
    store.add(agent_id=2, scope="self", content="secret note for 2", tick=1)

    hits1 = store.recall(agent_id=1, query="secret note", top_k=5)
    hits2 = store.recall(agent_id=2, query="secret note", top_k=5)
    assert [h.memory_id for h in hits1] != [h.memory_id for h in hits2]
    assert all(h.agent_id == 1 for h in hits1)
    assert all(h.agent_id == 2 for h in hits2)


def test_recall_empty_store_returns_empty(store):
    assert store.recall(agent_id=1, query="anything", top_k=5) == []


def test_recall_top_k_caps_result_count(store):
    for i in range(10):
        store.add(agent_id=1, scope="self", content=f"note {i}", tick=i)
    assert len(store.recall(agent_id=1, query="note", top_k=3)) == 3


# ---- R12 Gap 4: scope-aware recall -----------------------------------------


def test_recall_scope_filter_narrows_to_scope(store):
    """R12 Gap 4: scope=counterparty should exclude self-scope memories."""
    store.add(agent_id=1, scope="self",
              content="I am feeling restless today", tick=1)
    store.add(agent_id=1, scope="self",
              content="self reminder to bump listing", tick=2)
    store.add(agent_id=1, scope="self",
              content="my plan is to buy a bike", tick=3)
    store.add(agent_id=1, scope="counterparty", scope_ref_id=42,
              content="agent#42 haggled hard", tick=4)
    store.add(agent_id=1, scope="counterparty", scope_ref_id=42,
              content="agent#42 sent stock photo", tick=5)

    # No scope → all five candidates in the pool.
    all_hits = store.recall(agent_id=1, query="agent#42", top_k=10)
    assert len(all_hits) == 5

    # scope=counterparty → only the two counterparty rows survive.
    cp_hits = store.recall(
        agent_id=1, query="agent#42", top_k=10, scope="counterparty",
    )
    assert len(cp_hits) == 2
    assert all(h.scope == "counterparty" for h in cp_hits)


def test_recall_scope_ref_id_filter_isolates_counterparty(store):
    """R12 Gap 4: scope_ref_id pins the lookup to one counterparty."""
    store.add(agent_id=1, scope="counterparty", scope_ref_id=42,
              content="agent#42 haggled hard", tick=1)
    store.add(agent_id=1, scope="counterparty", scope_ref_id=42,
              content="agent#42 sent stock photo", tick=2)
    store.add(agent_id=1, scope="counterparty", scope_ref_id=99,
              content="agent#99 was chatty", tick=3)

    hits = store.recall(
        agent_id=1, query="counterparty agent", top_k=10,
        scope="counterparty", scope_ref_id=42,
    )
    assert len(hits) == 2
    assert all(h.scope_ref_id == 42 for h in hits)


def test_recall_scope_ref_id_alone_without_scope(store):
    """scope_ref_id should filter independently of scope being set."""
    store.add(agent_id=1, scope="counterparty", scope_ref_id=7,
              content="agent#7 was helpful", tick=1)
    store.add(agent_id=1, scope="thread", scope_ref_id=7,
              content="thread#7 went smoothly", tick=2)
    store.add(agent_id=1, scope="self", content="unrelated", tick=3)

    hits = store.recall(agent_id=1, query="agent thread", top_k=10,
                        scope_ref_id=7)
    assert len(hits) == 2
    assert all(h.scope_ref_id == 7 for h in hits)


# ---- Rendering --------------------------------------------------------------


def test_render_empty_is_stable_shape():
    txt = render_narrative_context([])
    assert txt.startswith("Your own impressions")
    assert "(none)" in txt


def test_render_includes_similarity_and_scope(store):
    store.add(agent_id=1, scope="counterparty",
              content="Kim drives hard bargains", tick=7, scope_ref_id=42)
    hits = store.recall(agent_id=1, query="Kim bargains", top_k=3)
    txt = render_narrative_context(hits)
    assert "counterparty#42" in txt
    assert "sim=" in txt
    assert "Kim drives hard bargains" in txt


# ---- Divergence-friendly check (precursor to T14) ---------------------------


def test_narrative_can_conflict_with_ledger(store, env):
    """Load-bearing for H2/inherited-drift: narrative may assert
    something that the ledger contradicts. This test only verifies
    the two stores are independently writable; T14 will log divergence.
    """
    from bazaar.actions import ActionType
    from bazaar.actions.dispatch import dispatch
    from bazaar.memory.ledger import (
        auto_populate_from_events,
        build_ledger_context,
    )

    # Ledger side: agent 1 blocks agent 2.
    dispatch(env.platform.conn, agent_id=1, action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=3)
    auto_populate_from_events(env.platform.conn)
    ledger = build_ledger_context(env.platform.conn, agent_id=1)
    assert ledger[0].kind == "block"

    # Narrative side: agent 1 nonetheless remembers agent 2 positively.
    store.add(agent_id=1, scope="counterparty",
              content="agent#2 is helpful and honest",
              scope_ref_id=2, tick=4)
    hits = store.recall(agent_id=1, query="agent#2 honest", top_k=1)
    assert "honest" in hits[0].content


# ---- Slow integration test: real model --------------------------------------
#
# The real MiniLMEncoder loads sentence-transformers, which pulls in
# transformers + torch. When torch is imported after faiss in the same
# process, some platforms segfault on exit. So we gate this test behind
# the BAZAAR_SLOW_TESTS env var and run it in its own pytest process:
#
#     BAZAAR_SLOW_TESTS=1 pytest tests/unit/test_narrative.py::test_minilm_encoder_end_to_end


@pytest.mark.slow
def test_minilm_encoder_end_to_end(env):
    import os
    if os.environ.get("BAZAAR_SLOW_TESTS") != "1":
        pytest.skip("set BAZAAR_SLOW_TESTS=1 to run real-model tests")

    from bazaar.memory import MiniLMEncoder
    store = NarrativeStore(env.platform.conn, encoder=MiniLMEncoder())
    assert store.dim == 384

    store.add(agent_id=1, scope="self", content="I enjoy cycling", tick=1)
    store.add(agent_id=1, scope="self",
              content="The weather today is overcast", tick=2)

    hits = store.recall(agent_id=1, query="bicycle riding", top_k=1)
    # Semantic match: "cycling" ≈ "bicycle riding"
    assert "cycling" in hits[0].content

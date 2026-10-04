"""Tests for T14 — ledger/narrative divergence detection.

Uses the FakeEncoder from test_narrative so the divergence detector
is exercised with deterministic embeddings.
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.memory import (
    NarrativeStore,
    auto_populate_from_events,
    clear_stores,
    count_divergences,
    detect_divergences,
    install_store,
    scan_and_log,
)

from .test_narrative import FakeEncoder


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=30 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=FakeEncoder(dim=16)),
    )
    yield env
    env.close()
    clear_stores()


# ---- Positive cases ---------------------------------------------------------


def test_positive_narrative_vs_block_flags_divergence(env):
    # Agent 1 blocks agent 2 at tick 3.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=3)
    auto_populate_from_events(env.platform.conn)

    # Agent 1 later writes a positive narrative about agent 2.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 was honest and friendly"},
             tick=10)

    divs = detect_divergences(env.platform.conn, agent_id=1)
    assert len(divs) == 1
    d = divs[0]
    assert d.conflict_kind == "positive_vs_block"
    assert d.counterparty_id == 2
    assert "honest" in d.narrative_content
    assert "blocked" in d.ledger_summary


def test_positive_narrative_vs_bad_rating_flags_divergence(env):
    # Agent 1 gives agent 2 a 1-star rating at tick 5.
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, body, tick) "
        "VALUES (1, 2, 1, 'unresponsive', 5)",
    )
    env.platform.conn.commit()
    auto_populate_from_events(env.platform.conn)

    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 is reliable and kind"},
             tick=12)

    divs = detect_divergences(env.platform.conn, agent_id=1)
    kinds = [d.conflict_kind for d in divs]
    assert "positive_vs_bad_rating" in kinds


def test_negative_narrative_vs_good_rating_flags_divergence(env):
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, body, tick) "
        "VALUES (1, 2, 5, 'great buyer', 2)",
    )
    env.platform.conn.commit()
    auto_populate_from_events(env.platform.conn)

    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 turned out to be a scam"},
             tick=9)

    divs = detect_divergences(env.platform.conn, agent_id=1)
    kinds = [d.conflict_kind for d in divs]
    assert "negative_vs_good_rating" in kinds


# ---- Negative / null cases --------------------------------------------------


def test_neutral_narrative_is_never_flagged(env):
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=3)
    auto_populate_from_events(env.platform.conn)

    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 asked about meetup logistics"},
             tick=10)
    assert detect_divergences(env.platform.conn, agent_id=1) == []


def test_self_scope_narrative_is_ignored(env):
    # Self-summaries can't contradict ledger-about-counterparty.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=1)
    auto_populate_from_events(env.platform.conn)

    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "self", "scope_ref_id": None,
                       "content": "I am a trustworthy seller"},
             tick=5)
    assert detect_divergences(env.platform.conn, agent_id=1) == []


def test_positive_narrative_without_ledger_entry_is_not_a_divergence(env):
    # No ledger entry ⇒ nothing to contradict.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 is fair and honest"},
             tick=3)
    assert detect_divergences(env.platform.conn, agent_id=1) == []


def test_rating_by_other_rater_does_not_apply_to_this_agent(env):
    """Only the agent's *own* rating of the counterparty counts —
    a third party's rating isn't the agent's ledger evidence."""
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, body, tick) "
        "VALUES (3, 2, 1, '', 4)",  # agent 3 rates agent 2 badly
    )
    env.platform.conn.commit()
    auto_populate_from_events(env.platform.conn)

    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 is trustworthy"},
             tick=9)
    # Agent 1 received no direct signal — no divergence for agent 1.
    assert detect_divergences(env.platform.conn, agent_id=1) == []


# ---- Event logging ---------------------------------------------------------


def test_scan_and_log_is_idempotent(env):
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=1)
    auto_populate_from_events(env.platform.conn)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 is trustworthy"},
             tick=5)

    n1 = scan_and_log(env.platform.conn, tick=6)
    n2 = scan_and_log(env.platform.conn, tick=7)
    assert n1 == 1
    assert n2 == 0
    assert count_divergences(env.platform.conn) == 1


def test_scan_and_log_sets_agent_id_to_affected_agent(env):
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=1)
    auto_populate_from_events(env.platform.conn)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 reliable and kind"},
             tick=5)
    scan_and_log(env.platform.conn, tick=6)

    row = env.platform.conn.execute(
        "SELECT agent_id, action_type, payload FROM events "
        "WHERE action_type = 'memory_divergence'"
    ).fetchone()
    assert row[0] == 1  # the drifting agent
    assert row[1] == "memory_divergence"
    # Per-agent count aligns with payload.
    assert count_divergences(env.platform.conn, agent_id=1) == 1
    assert count_divergences(env.platform.conn, agent_id=2) == 0


def test_up_to_tick_filters_detection(env):
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=1)
    auto_populate_from_events(env.platform.conn)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SUMMARIZE_SESSION,
             raw_args={"scope": "counterparty", "scope_ref_id": 2,
                       "content": "agent#2 is trustworthy"},
             tick=20)

    # At tick 10 the narrative hasn't been written yet → no divergence.
    assert detect_divergences(
        env.platform.conn, agent_id=1, up_to_tick=10,
    ) == []
    assert len(detect_divergences(
        env.platform.conn, agent_id=1, up_to_tick=25,
    )) == 1

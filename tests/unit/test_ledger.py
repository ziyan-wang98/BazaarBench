"""Unit tests for T11 — structured ledger.

Covers:
- empty event log → empty ledger
- block via real handler → one 'block' entry for the blocker, none
  for the blocked agent
- manually-inserted rating → two entries (rater 'gave', ratee 'received')
- completed meetup → two entries (buyer 'bought from', seller 'sold to')
- report of a listing resolves counterparty via listings.owner_agent_id
- idempotent ingestion (calling twice doesn't duplicate)
- build_ledger_context ordering (tick desc, kind priority asc) and cap
- render produces stable prompt text including a header even when empty
- up_to_tick gate on ingestion and on context
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.memory import (
    LedgerEntry,
    auto_populate_from_events,
    build_ledger_context,
    record_ledger_entry,
    render_ledger_context,
)
from bazaar.memory.ledger import get_entry_counts


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


# ---- Baseline --------------------------------------------------------------


def test_empty_env_yields_empty_ledger(env):
    n = auto_populate_from_events(env.platform.conn)
    assert n == 0
    ctx = build_ledger_context(env.platform.conn, agent_id=1)
    assert ctx == []


def test_render_empty_context_has_stable_shape(env):
    text = render_ledger_context([])
    assert text.startswith("Your verified marketplace history")
    assert "(none)" in text


# ---- Block via real handler ------------------------------------------------


def test_block_creates_entry_only_for_blocker(env):
    dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.BLOCK_USER,
        raw_args={"user_agent_id": 2}, tick=5,
    )
    n = auto_populate_from_events(env.platform.conn)
    assert n == 1
    ctx1 = build_ledger_context(env.platform.conn, agent_id=1)
    ctx2 = build_ledger_context(env.platform.conn, agent_id=2)
    assert len(ctx1) == 1
    assert len(ctx2) == 0
    e = ctx1[0]
    assert e.kind == "block"
    assert e.counterparty_id == 2
    assert e.tick == 5
    assert "agent#2" in e.summary


# ---- Rating ingestion ------------------------------------------------------


def test_rating_creates_two_entries_one_per_perspective(env):
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, body, tick) "
        "VALUES (1, 2, 5, 'Great buyer!', 10)",
    )
    env.platform.conn.commit()

    n = auto_populate_from_events(env.platform.conn)
    assert n == 2

    c1 = get_entry_counts(env.platform.conn, agent_id=1)
    c2 = get_entry_counts(env.platform.conn, agent_id=2)
    assert c1["rating"] == 1 and c2["rating"] == 1

    e_rater = build_ledger_context(env.platform.conn, agent_id=1)[0]
    e_ratee = build_ledger_context(env.platform.conn, agent_id=2)[0]
    assert "gave 5-star" in e_rater.summary
    assert "received 5-star" in e_ratee.summary
    assert "Great buyer!" in e_rater.summary


# ---- Report ingestion ------------------------------------------------------


def test_report_of_listing_resolves_owner_as_counterparty(env):
    # Agent 2 owns listing L.  Agent 1 reports it.
    created = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "Dodgy listing",
                  "description": "", "price_cents": 1, "condition": "fair"},
        tick=0,
    )
    listing_id = created.payload["listing_id"]

    env.platform.conn.execute(
        """
        INSERT INTO reports (reporter_id, target_kind, target_id, reason, tick)
        VALUES (1, 'listing', ?, 'looks scammy', 3)
        """,
        (listing_id,),
    )
    env.platform.conn.commit()

    auto_populate_from_events(env.platform.conn)

    e = build_ledger_context(env.platform.conn, agent_id=1)[0]
    assert e.kind == "report"
    # The reporter's counterparty is the listing's owner.
    assert e.counterparty_id == 2
    assert "scammy" in e.summary


def test_report_of_phantom_listing_leaves_counterparty_null(env):
    # Phantom listings already exist? platform.seed_phantom_listings seeds
    # with owner_agent_id=NULL.  Seed one and report it.
    env.platform.seed_phantom_listings(count=1)
    phantom_id = env.platform.conn.execute(
        "SELECT listing_id FROM listings WHERE is_phantom = 1"
    ).fetchone()[0]
    env.platform.conn.execute(
        """
        INSERT INTO reports (reporter_id, target_kind, target_id, reason, tick)
        VALUES (1, 'listing', ?, 'bait', 2)
        """,
        (phantom_id,),
    )
    env.platform.conn.commit()

    auto_populate_from_events(env.platform.conn)
    e = build_ledger_context(env.platform.conn, agent_id=1)[0]
    assert e.counterparty_id is None


# ---- Transaction ingestion -------------------------------------------------


def test_completed_meetup_creates_entry_for_both_sides(env):
    # Seed a listing + thread so the meetup FK is satisfiable.
    created = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "A book", "description": "",
                  "price_cents": 200, "condition": "good"},
        tick=0,
    )
    lid = created.payload["listing_id"]
    offered = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 180, "terms": {}},
        tick=1,
    )
    tid = offered.payload["thread_id"]
    env.platform.conn.execute(
        """
        INSERT INTO meetups (thread_id, scheduled_tick, location_desc,
                              payment_method, status)
        VALUES (?, 10, 'coffee shop', 'cash', 'completed')
        """,
        (tid,),
    )
    env.platform.conn.commit()

    n = auto_populate_from_events(env.platform.conn)
    assert n == 2

    c1 = build_ledger_context(env.platform.conn, agent_id=1)
    c2 = build_ledger_context(env.platform.conn, agent_id=2)
    assert len(c1) == 1 and len(c2) == 1
    # Agent 1 was the seller, agent 2 was the buyer.
    assert c1[0].kind == "transaction"
    assert c2[0].kind == "transaction"
    assert "sold to agent#2" in c1[0].summary
    assert "bought from agent#1" in c2[0].summary


# ---- Idempotence -----------------------------------------------------------


def test_auto_populate_is_idempotent(env):
    dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.BLOCK_USER,
        raw_args={"user_agent_id": 2}, tick=5,
    )
    n1 = auto_populate_from_events(env.platform.conn)
    n2 = auto_populate_from_events(env.platform.conn)
    assert n1 == 1
    assert n2 == 0
    assert len(build_ledger_context(env.platform.conn, agent_id=1)) == 1


# ---- Ordering & cap --------------------------------------------------------


def test_build_ledger_context_orders_recent_first_and_caps_at_k(env):
    # Manually insert 5 entries at ticks 1..5 for agent 1.  counterparty
    # must be a real agent to satisfy the FK; agent 2 exists in the fixture.
    for t in range(1, 6):
        record_ledger_entry(env.platform.conn, LedgerEntry(
            agent_id=1, kind="block", counterparty_id=2,
            ref_table="blocks", ref_id=t, summary=f"block {t}", tick=t,
        ))
    env.platform.conn.commit()

    top3 = build_ledger_context(env.platform.conn, agent_id=1, k=3)
    assert [e.tick for e in top3] == [5, 4, 3]


def test_build_ledger_context_kind_priority_breaks_tick_ties(env):
    # Three entries at the same tick, one of each relevant kind.
    for kind, ref in (("report", 1), ("block", 2), ("rating", 3)):
        record_ledger_entry(env.platform.conn, LedgerEntry(
            agent_id=1, kind=kind, counterparty_id=None,   # type: ignore[arg-type]
            ref_table=kind, ref_id=ref, summary=f"{kind} {ref}", tick=7,
        ))
    env.platform.conn.commit()
    kinds = [e.kind for e in build_ledger_context(env.platform.conn, agent_id=1)]
    # kind priority: transaction < rating < block < report
    assert kinds == ["rating", "block", "report"]


def test_up_to_tick_gate_filters_future_ingestion_and_context(env):
    # Two blocks, at ticks 5 and 15.
    dispatch(env.platform.conn, agent_id=1, action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 2}, tick=5)
    dispatch(env.platform.conn, agent_id=1, action=ActionType.BLOCK_USER,
             raw_args={"user_agent_id": 3}, tick=15)

    auto_populate_from_events(env.platform.conn, up_to_tick=10)
    ctx_t10 = build_ledger_context(env.platform.conn, agent_id=1, up_to_tick=10)
    assert [e.tick for e in ctx_t10] == [5]

    # Now ingest the rest.
    auto_populate_from_events(env.platform.conn)
    ctx_full = build_ledger_context(env.platform.conn, agent_id=1)
    assert [e.tick for e in ctx_full] == [15, 5]


# ---- Rendering -------------------------------------------------------------


def test_render_contains_every_summary_line(env):
    entries = [
        LedgerEntry(agent_id=1, kind="block", counterparty_id=2,
                    ref_table="blocks", ref_id=1, summary="blocked agent#2",
                    tick=5),
        LedgerEntry(agent_id=1, kind="rating", counterparty_id=3,
                    ref_table="ratings", ref_id=7, summary="gave 4-star to agent#3",
                    tick=6),
    ]
    text = render_ledger_context(entries)
    assert "[t=5] block: blocked agent#2" in text
    assert "[t=6] rating: gave 4-star to agent#3" in text

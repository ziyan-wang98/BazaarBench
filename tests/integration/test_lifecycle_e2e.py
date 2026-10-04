"""End-to-end lifecycle integration test (P5 of T23).

Scripts a single marketplace dyad all the way through the
transaction chain and verifies every downstream side effect. If
any link in the chain regresses, this test pinpoints which one.
"""
from __future__ import annotations

import json

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.memory import (
    HashEncoder,
    NarrativeStore,
    auto_populate_from_events,
    install_store,
)


def _set_inventory(conn, agent_id: int, inventory: list[dict]) -> None:
    """Overwrite the persona's declared inventory so R15 speculative
    tagging classifies the test's listing as authentic (no fraud
    auto-rating)."""
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["inventory_items"] = inventory
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona), agent_id),
    )
    conn.commit()


@pytest.fixture
def env(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "lifecycle.db")
    for i in range(4):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=1500 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    yield env
    env.close()


@pytest.mark.integration
def test_full_lifecycle_two_buyers_one_winner(env):
    """Agent 1 lists. Agents 2 and 3 both make offers on separate
    threads. Seller counters agent 2's offer; agent 2 re-counters;
    seller accepts round 3. Seller schedules meetup; both sides
    confirm; transaction completes; both rate each other. Agent 3's
    losing thread must end up cancelled with their offer rejected.
    """
    conn = env.platform.conn
    # R15 Part 2: ensure the listing classifies authentic so the R15
    # Part 3 fraud hook doesn't inject a 3rd auto-rating.
    _set_inventory(
        conn, 1,
        inventory=[{"category": "electronics", "title": "Like-new speaker"}],
    )

    # --- 0. List --------------------------------------------------------
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "electronics", "title": "Like-new speaker",
                           "description": "", "price_cents": 5000,
                           "condition": "like_new"}, tick=0)
    assert r.status == "ok"
    lid = r.payload["listing_id"]

    # --- 1. Buyer A makes an offer; buyer B makes a competing offer ----
    r = dispatch(conn, agent_id=2, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 4000, "terms": {}},
                 tick=1)
    assert r.status == "ok"
    oa_id = r.payload["offer_id"]
    ta_id = r.payload["thread_id"]

    r = dispatch(conn, agent_id=3, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 4200, "terms": {}},
                 tick=2)
    assert r.status == "ok"
    ob_id = r.payload["offer_id"]
    tb_id = r.payload["thread_id"]
    assert ta_id != tb_id

    # --- 2. Seller counters A; A counters back; seller accepts ---------
    r = dispatch(conn, agent_id=1, action=ActionType.COUNTER_OFFER,
                 raw_args={"offer_id": oa_id, "price_cents": 4800, "terms": {}},
                 tick=3)
    assert r.status == "ok"
    oa2_id = r.payload["offer_id"]
    assert r.payload["round"] == 2

    r = dispatch(conn, agent_id=2, action=ActionType.COUNTER_OFFER,
                 raw_args={"offer_id": oa2_id, "price_cents": 4500, "terms": {}},
                 tick=4)
    assert r.status == "ok"
    oa3_id = r.payload["offer_id"]

    r = dispatch(conn, agent_id=1, action=ActionType.ACCEPT_OFFER,
                 raw_args={"offer_id": oa3_id}, tick=5)
    assert r.status == "ok"

    # Thread A is now 'committed'. Every earlier round in thread A is
    # 'countered' (intermediate rounds) or 'accepted' (the final).
    status_a = conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (ta_id,),
    ).fetchone()[0]
    assert status_a == "committed"

    # --- 3. Seller schedules meetup ------------------------------------
    r = dispatch(conn, agent_id=1, action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": ta_id, "location_desc": "coffee shop",
                           "scheduled_tick": 10, "payment_method": "cash"},
                 tick=6)
    assert r.status == "ok"
    mid = r.payload["meetup_id"]

    # Schedule is blocked while one is already scheduled.
    r = dispatch(conn, agent_id=2, action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": ta_id, "location_desc": "park",
                           "scheduled_tick": 15, "payment_method": "venmo"},
                 tick=7)
    assert r.status == "blocked"
    assert r.payload["error"] == "meetup_already_scheduled"

    # --- 4. Both sides confirm; on the second the sale finalises ------
    # v2: the buyer must inspect_at_meetup before completing on a
    # meetup-mode meetup. Inspect at the scheduled tick.
    r = dispatch(conn, agent_id=2, action=ActionType.INSPECT_AT_MEETUP,
                 raw_args={"meetup_id": mid}, tick=10)
    assert r.status == "ok"
    r = dispatch(conn, agent_id=2, action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=11)
    assert r.status == "ok" and r.payload["completed"] is False

    r = dispatch(conn, agent_id=1, action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=12)
    assert r.status == "ok" and r.payload["completed"] is True

    # Downstream side-effect cascade:
    lstatus, tstatus_a, mstatus = conn.execute(
        """
        SELECT (SELECT status FROM listings WHERE listing_id = ?),
               (SELECT status FROM threads  WHERE thread_id  = ?),
               (SELECT status FROM meetups  WHERE meetup_id  = ?)
        """,
        (lid, ta_id, mid),
    ).fetchone()
    assert lstatus == "sold"
    assert tstatus_a == "completed"
    assert mstatus == "completed"

    # Sister thread B should be cancelled; its pending offer rejected.
    tstatus_b = conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (tb_id,),
    ).fetchone()[0]
    assert tstatus_b == "cancelled"
    ob_status = conn.execute(
        "SELECT status FROM offers WHERE offer_id = ?", (ob_id,),
    ).fetchone()[0]
    assert ob_status == "rejected"

    # --- 5. Both rate each other ----------------------------------------
    r = dispatch(conn, agent_id=2, action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 5,
                           "body": "smooth seller", "thread_id": ta_id},
                 tick=13)
    assert r.status == "ok"
    r = dispatch(conn, agent_id=1, action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 2, "stars": 5,
                           "body": "easy buyer", "thread_id": ta_id},
                 tick=14)
    assert r.status == "ok"
    n_ratings = conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE thread_id = ?", (ta_id,),
    ).fetchone()[0]
    assert n_ratings == 2

    # --- 6. Ledger populates from event log ----------------------------
    inserted = auto_populate_from_events(conn)
    assert inserted >= 4  # 2 per rating + 2 per transaction

    # Agent 1 sees a 'transaction' entry AND a 'rating' entry
    # (two of each if we count both received + gave, but we dedupe
    # by ref_id+kind per-agent).
    rows = conn.execute(
        """
        SELECT kind, COUNT(*) FROM ledger_entries
        WHERE agent_id = 1 GROUP BY kind
        """
    ).fetchall()
    by_kind = {r[0]: r[1] for r in rows}
    assert by_kind.get("transaction", 0) >= 1
    assert by_kind.get("rating", 0) >= 1

    # --- 7. Event log has no errors or blocks --------------------------
    n_err, n_block = conn.execute(
        """
        SELECT
          SUM(CASE WHEN result_status = 'error'   THEN 1 ELSE 0 END),
          SUM(CASE WHEN result_status = 'blocked' THEN 1 ELSE 0 END)
        FROM events
        """
    ).fetchone()
    # The one intentional block (duplicate SCHEDULE_MEETUP) is expected.
    assert int(n_err or 0) == 0
    assert int(n_block or 0) == 1


@pytest.mark.integration
def test_ghost_cleans_up_pending_offers_on_abandoned_thread(env):
    """Orthogonal lifecycle path: one buyer walks (GHOST) instead of
    transacting, thread ends up in 'ghosted', pending offer rejected."""
    conn = env.platform.conn
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Book", "description": "",
                           "price_cents": 500, "condition": "good"}, tick=0)
    lid = r.payload["listing_id"]
    r = dispatch(conn, agent_id=2, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
                 tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    r = dispatch(conn, agent_id=2, action=ActionType.GHOST,
                 raw_args={"thread_id": tid}, tick=2)
    assert r.status == "ok"
    status = conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert status == "ghosted"
    offer_status = conn.execute(
        "SELECT status FROM offers WHERE offer_id = ?", (oid,),
    ).fetchone()[0]
    assert offer_status == "rejected"

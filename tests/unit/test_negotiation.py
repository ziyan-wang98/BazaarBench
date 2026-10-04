"""Exhaustive tests for the negotiation lifecycle handlers (P1 of T23).

Covers every state transition and every failure branch of
COUNTER_OFFER, ACCEPT_OFFER, WITHDRAW_OFFER, SCHEDULE_MEETUP,
COMPLETE_TRANSACTION, and CANCEL_MEETUP. Cross-listing cleanup
invariants (sister threads cancelled, pending offers rejected on
sale) have their own tests so future refactors don't regress them.
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(4):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=800 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _create_listing(env, *, owner: int, tick: int = 0) -> int:
    r = dispatch(
        env.platform.conn,
        agent_id=owner, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "Test item",
                  "description": "", "price_cents": 500, "condition": "good"},
        tick=tick,
    )
    assert r.status == "ok"
    return r.payload["listing_id"]


def _make_offer(env, *, buyer: int, listing_id: int, price: int = 400, tick: int = 1):
    r = dispatch(
        env.platform.conn,
        agent_id=buyer, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": listing_id, "price_cents": price, "terms": {}},
        tick=tick,
    )
    assert r.status == "ok"
    return r.payload  # {offer_id, thread_id, round}


# ---- COUNTER_OFFER ---------------------------------------------------------


def test_counter_offer_happy(env):
    lid = _create_listing(env, owner=1)
    o1 = _make_offer(env, buyer=2, listing_id=lid, price=400)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.COUNTER_OFFER,
        raw_args={"offer_id": o1["offer_id"], "price_cents": 450, "terms": {}},
        tick=2,
    )
    assert r.status == "ok"
    assert r.payload["counter_to"] == o1["offer_id"]
    assert r.payload["round"] == 2
    # Original offer is now 'countered'; new offer is 'pending' from seller.
    s_orig, s_new = env.platform.conn.execute(
        "SELECT status FROM offers WHERE offer_id = ? UNION ALL "
        "SELECT status FROM offers WHERE offer_id = ?",
        (o1["offer_id"], r.payload["offer_id"]),
    ).fetchall()
    assert s_orig[0] == "countered"
    assert s_new[0] == "pending"


def test_counter_offer_blocks_missing_offer(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.COUNTER_OFFER,
        raw_args={"offer_id": 99999, "price_cents": 100, "terms": {}},
        tick=2,
    )
    assert r.status == "blocked" and r.payload["error"] == "offer_not_found"


def test_counter_offer_blocks_own_offer(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.COUNTER_OFFER,
        raw_args={"offer_id": o["offer_id"], "price_cents": 410, "terms": {}},
        tick=2,
    )
    assert r.status == "blocked" and r.payload["error"] == "cannot_counter_own_offer"


def test_counter_offer_blocks_when_offer_not_pending(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    # Accept it first.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    # Now counter.
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.COUNTER_OFFER,
        raw_args={"offer_id": o["offer_id"], "price_cents": 410, "terms": {}},
        tick=3,
    )
    assert r.status == "blocked" and r.payload["error"] == "offer_accepted"


def test_counter_offer_blocks_non_participant(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    # Agent 3 isn't in the thread.
    r = dispatch(
        env.platform.conn,
        agent_id=3, action=ActionType.COUNTER_OFFER,
        raw_args={"offer_id": o["offer_id"], "price_cents": 410, "terms": {}},
        tick=2,
    )
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


def test_counter_offer_blocks_on_committed_thread(env):
    """Defensive: even if someone contrives an orphan pending offer
    on a committed thread, counter_offer refuses to extend it."""
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    # Manually plant a stray pending offer on the now-committed thread
    # (simulating a hypothetical race) so we can test the guard.
    env.platform.conn.execute(
        """
        INSERT INTO offers
            (thread_id, proposer_id, round, price_cents, terms_json,
             tick, status)
        VALUES (?, 2, 2, 410, '{}', 3, 'pending')
        """,
        (o["thread_id"],),
    )
    env.platform.conn.commit()
    stray_id = env.platform.conn.execute(
        "SELECT MAX(offer_id) FROM offers"
    ).fetchone()[0]
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.COUNTER_OFFER,
                 raw_args={"offer_id": stray_id, "price_cents": 420, "terms": {}},
                 tick=4)
    assert r.status == "blocked" and r.payload["error"] == "thread_committed"


def test_make_offer_blocks_reuse_of_terminal_thread(env):
    """Regression: make_offer used to reuse the buyer's existing
    thread regardless of status, which left pending offers on
    committed / cancelled / ghosted threads. T23 acceptance
    invariant #4 catches this; test it at the handler level too."""
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid, price=400)
    # Commit the thread via accept.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    # Buyer tries to re-offer on the same listing. Must be blocked —
    # otherwise a new pending offer would appear on a committed thread.
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 450, "terms": {}},
                 tick=3)
    assert r.status == "blocked"
    assert r.payload["error"] == "thread_committed"


# ---- ACCEPT_OFFER ----------------------------------------------------------


def test_accept_offer_commits_thread_and_rejects_other_pending(env):
    lid = _create_listing(env, owner=1)
    o1 = _make_offer(env, buyer=2, listing_id=lid, price=400)
    # Second buyer in a separate thread on same listing.
    o2 = _make_offer(env, buyer=3, listing_id=lid, price=420)
    assert o1["thread_id"] != o2["thread_id"]

    # Seller accepts o1. Thread 1 → committed. Offer 2 is in a different
    # thread, so not directly superseded here — only completion does that.
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.ACCEPT_OFFER,
                 raw_args={"offer_id": o1["offer_id"]}, tick=2)
    assert r.status == "ok"

    thread_status = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?",
        (o1["thread_id"],),
    ).fetchone()[0]
    assert thread_status == "committed"


def test_accept_offer_supersedes_other_pending_in_same_thread(env):
    """Multiple rounds of counters create several pending offers in one
    thread; accepting one rejects any other still-pending ones."""
    lid = _create_listing(env, owner=1)
    o1 = _make_offer(env, buyer=2, listing_id=lid, price=400)
    # Seller counters (this marks o1 'countered' and creates o2 'pending').
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.COUNTER_OFFER,
                 raw_args={"offer_id": o1["offer_id"], "price_cents": 450, "terms": {}},
                 tick=2)
    o2_id = r.payload["offer_id"]
    # Buyer counters again — creates o3 'pending', o2 'countered'.
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.COUNTER_OFFER,
                 raw_args={"offer_id": o2_id, "price_cents": 420, "terms": {}},
                 tick=3)
    # Now seller accepts o3.
    o3_id = r.payload["offer_id"]
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o3_id}, tick=4)
    # No offer should still be 'pending' in this thread.
    n_pending = env.platform.conn.execute(
        "SELECT COUNT(*) FROM offers WHERE thread_id = ? AND status='pending'",
        (o1["thread_id"],),
    ).fetchone()[0]
    assert n_pending == 0


def test_accept_offer_blocks_own_offer(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.ACCEPT_OFFER,
                 raw_args={"offer_id": o["offer_id"]}, tick=2)
    assert r.status == "blocked"
    assert r.payload["error"] == "cannot_accept_own_offer"


def test_accept_offer_blocks_missing(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.ACCEPT_OFFER,
                 raw_args={"offer_id": 99999}, tick=2)
    assert r.status == "blocked" and r.payload["error"] == "offer_not_found"


# ---- WITHDRAW_OFFER --------------------------------------------------------


def test_withdraw_offer_happy(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.WITHDRAW_OFFER,
                 raw_args={"offer_id": o["offer_id"]}, tick=2)
    assert r.status == "ok"
    status = env.platform.conn.execute(
        "SELECT status FROM offers WHERE offer_id = ?", (o["offer_id"],),
    ).fetchone()[0]
    assert status == "withdrawn"


def test_withdraw_offer_blocks_non_proposer(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.WITHDRAW_OFFER,
                 raw_args={"offer_id": o["offer_id"]}, tick=2)
    assert r.status == "blocked" and r.payload["error"] == "not_proposer"


def test_withdraw_offer_blocks_when_not_pending(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.WITHDRAW_OFFER,
                 raw_args={"offer_id": o["offer_id"]}, tick=3)
    assert r.status == "blocked" and r.payload["error"] == "offer_accepted"


# ---- SCHEDULE_MEETUP -------------------------------------------------------


def _commit_thread(env, *, seller: int, buyer: int):
    lid = _create_listing(env, owner=seller)
    o = _make_offer(env, buyer=buyer, listing_id=lid)
    dispatch(env.platform.conn, agent_id=seller,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    return lid, o["thread_id"]


def test_schedule_meetup_happy(env):
    _lid, tid = _commit_thread(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "coffee shop",
                           "scheduled_tick": 20, "payment_method": "cash"},
                 tick=3)
    assert r.status == "ok"
    assert r.payload["scheduled_tick"] == 20
    row = env.platform.conn.execute(
        "SELECT status, payment_method, location_desc, scheduled_tick "
        "FROM meetups WHERE meetup_id = ?",
        (r.payload["meetup_id"],),
    ).fetchone()
    assert row[0] == "scheduled"
    assert row[1] == "cash"
    assert row[2] == "coffee shop"
    assert row[3] == 20


def test_schedule_meetup_blocks_uncommitted_thread(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    # Skip accept — thread is 'open', not 'committed'.
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": o["thread_id"],
                           "location_desc": "x", "scheduled_tick": 20,
                           "payment_method": "cash"}, tick=3)
    assert r.status == "blocked"
    assert r.payload["error"] == "thread_not_committed_open"


def test_schedule_meetup_blocks_duplicate(env):
    _lid, tid = _commit_thread(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.SCHEDULE_MEETUP,
             raw_args={"thread_id": tid, "location_desc": "a",
                       "scheduled_tick": 20, "payment_method": "cash"},
             tick=3)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "b",
                           "scheduled_tick": 30, "payment_method": "zelle"},
                 tick=4)
    assert r.status == "blocked" and r.payload["error"] == "meetup_already_scheduled"


def test_schedule_meetup_blocks_non_participant(env):
    _lid, tid = _commit_thread(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=3,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "x",
                           "scheduled_tick": 20, "payment_method": "cash"},
                 tick=3)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


# ---- COMPLETE_TRANSACTION --------------------------------------------------


def _make_scheduled_meetup(env, *, seller: int, buyer: int, inspect: bool = True):
    """Schedule a meetup. v2 default: also have the buyer inspect, so
    callers exercising ``complete_transaction`` work without extra
    ceremony. Pass ``inspect=False`` to test the inspection precondition
    itself.
    """
    lid, tid = _commit_thread(env, seller=seller, buyer=buyer)
    r = dispatch(env.platform.conn, agent_id=seller,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "x",
                           "scheduled_tick": 20, "payment_method": "cash"},
                 tick=3)
    mid = r.payload["meetup_id"]
    if inspect:
        dispatch(env.platform.conn, agent_id=buyer,
                 action=ActionType.INSPECT_AT_MEETUP,
                 raw_args={"meetup_id": mid}, tick=20)
    return lid, tid, mid


def test_complete_transaction_one_sided_is_pending(env):
    _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=21)
    assert r.status == "ok"
    assert r.payload["completed"] is False
    assert r.payload["buyer_confirmed"] is True
    assert r.payload["seller_confirmed"] is False


def test_complete_transaction_both_sides_finalises(env):
    lid, tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=22)
    assert r.status == "ok" and r.payload["completed"] is True

    # Side effects:
    m, t, lst = env.platform.conn.execute(
        """
        SELECT (SELECT status FROM meetups WHERE meetup_id = ?),
               (SELECT status FROM threads WHERE thread_id = ?),
               (SELECT status FROM listings WHERE listing_id = ?)
        """,
        (mid, tid, lid),
    ).fetchone()
    assert m == "completed"
    assert t == "completed"
    assert lst == "sold"


def test_complete_transaction_cancels_sister_threads(env):
    """When agent 1 sells to agent 2, agent 3's separate thread on the
    same listing must get cancelled and their offer rejected."""
    lid = _create_listing(env, owner=1)
    # Thread A: agent 2 → accepted → meetup
    o_a = _make_offer(env, buyer=2, listing_id=lid, price=400)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o_a["offer_id"]}, tick=2)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": o_a["thread_id"],
                           "location_desc": "x", "scheduled_tick": 20,
                           "payment_method": "cash"}, tick=3)
    mid_a = r.payload["meetup_id"]

    # Thread B: agent 3 makes a competing offer, still pending.
    o_b = _make_offer(env, buyer=3, listing_id=lid, price=420, tick=4)
    assert o_b["thread_id"] != o_a["thread_id"]

    # v2: buyer must inspect at meetup before completing.
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid_a}, tick=20)
    # Both sides complete the transaction on thread A.
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid_a}, tick=21)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid_a}, tick=22)

    # Sister thread B is cancelled, its pending offer rejected.
    b_status = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?",
        (o_b["thread_id"],),
    ).fetchone()[0]
    assert b_status == "cancelled"
    b_offer_status = env.platform.conn.execute(
        "SELECT status FROM offers WHERE offer_id = ?",
        (o_b["offer_id"],),
    ).fetchone()[0]
    assert b_offer_status == "rejected"


def test_complete_transaction_blocks_non_participant(env):
    _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=3,
                 action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=21)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


def test_complete_transaction_requires_handoff_proof_when_enabled(tmp_db):
    env = BazaarEnv(db_path=tmp_db, require_handoff_proof=True)
    for i in range(2):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=900 + i),
                policy=RandomBenignPolicy(seed=i),
            )
        )
    env.reset()
    try:
        _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
        r = dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.COMPLETE_TRANSACTION,
            raw_args={"meetup_id": mid},
            tick=21,
        )
        assert r.status == "blocked"
        assert r.payload["error"] == "handoff_proof_required"
        buyer_confirmed = env.platform.conn.execute(
            "SELECT buyer_confirmed FROM meetups WHERE meetup_id = ?",
            (mid,),
        ).fetchone()[0]
        assert buyer_confirmed == 0

        wrong = dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.COMPLETE_TRANSACTION,
            raw_args={"meetup_id": mid, "handoff_proof": "pickup-code-1842"},
            tick=22,
        )
        assert wrong.status == "blocked"
        buyer_confirmed = env.platform.conn.execute(
            "SELECT buyer_confirmed FROM meetups WHERE meetup_id = ?",
            (mid,),
        ).fetchone()[0]
        assert buyer_confirmed == 0

        token = env.platform.conn.execute(
            "SELECT handoff_token FROM meetups WHERE meetup_id = ?",
            (mid,),
        ).fetchone()[0]
        assert token and token.startswith("handoff-")

        ok = dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.COMPLETE_TRANSACTION,
            raw_args={"meetup_id": mid, "handoff_proof": token},
            tick=23,
        )
        assert ok.status == "ok"
        assert ok.payload["buyer_confirmed"] is True
    finally:
        env.close()


# ---- CANCEL_MEETUP ---------------------------------------------------------


def test_cancel_meetup_happy(env):
    _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.CANCEL_MEETUP,
                 raw_args={"meetup_id": mid, "reason": "schedule conflict"},
                 tick=5)
    assert r.status == "ok"
    s = env.platform.conn.execute(
        "SELECT status FROM meetups WHERE meetup_id = ?", (mid,),
    ).fetchone()[0]
    assert s == "cancelled"


def test_cancel_meetup_blocks_already_completed(env):
    _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=22)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.CANCEL_MEETUP,
                 raw_args={"meetup_id": mid, "reason": "late"}, tick=23)
    assert r.status == "blocked" and r.payload["error"] == "meetup_completed"


def test_cancel_meetup_blocks_non_participant(env):
    _lid, _tid, mid = _make_scheduled_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=3,
                 action=ActionType.CANCEL_MEETUP,
                 raw_args={"meetup_id": mid, "reason": "x"}, tick=5)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"

"""Exhaustive tests for listing-lifecycle + reputation handlers (P2 of T23).

Covers every transition + block branch of EDIT_LISTING, BUMP_LISTING,
MARK_SOLD, RELIST (Group 3) and RATE, REPORT_LISTING, REPORT_USER
(Group 6). Also verifies MARK_SOLD triggers the same sister-thread
cleanup that COMPLETE_TRANSACTION does — this is the invariant that
keeps the offers/threads tables consistent when a seller has racing
buyers.
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
            MarketAgent(persona=generate_persona(i + 1, seed=900 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _create_listing(env, *, owner: int, price: int = 500, tick: int = 0) -> int:
    r = dispatch(
        env.platform.conn,
        agent_id=owner, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "Book",
                  "description": "", "price_cents": price, "condition": "good"},
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
    return r.payload


# ---- EDIT_LISTING ----------------------------------------------------------


def test_edit_listing_updates_fields(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.EDIT_LISTING,
        raw_args={"listing_id": lid, "title": "Better book",
                  "price_cents": 700, "description": None, "condition": None},
        tick=1,
    )
    assert r.status == "ok"
    assert r.payload["changed"] == 2
    title, price = env.platform.conn.execute(
        "SELECT title, price_cents FROM listings WHERE listing_id = ?",
        (lid,),
    ).fetchone()
    assert title == "Better book"
    assert price == 700


def test_edit_listing_no_fields_is_noop(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.EDIT_LISTING,
        raw_args={"listing_id": lid, "title": None, "price_cents": None,
                  "description": None, "condition": None},
        tick=1,
    )
    assert r.status == "ok"
    assert r.payload["changed"] == 0


def test_edit_listing_blocks_non_owner(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(
        env.platform.conn, agent_id=2,
        action=ActionType.EDIT_LISTING,
        raw_args={"listing_id": lid, "title": "Hijack", "price_cents": None,
                  "description": None, "condition": None},
        tick=1,
    )
    assert r.status == "blocked" and r.payload["error"] == "not_owner"


def test_edit_listing_blocks_sold_listing(env):
    lid = _create_listing(env, owner=1)
    env.platform.conn.execute(
        "UPDATE listings SET status = 'sold' WHERE listing_id = ?", (lid,),
    )
    env.platform.conn.commit()
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.EDIT_LISTING,
        raw_args={"listing_id": lid, "title": "x", "price_cents": None,
                  "description": None, "condition": None},
        tick=1,
    )
    assert r.status == "blocked" and r.payload["error"] == "listing_sold"


def test_edit_listing_errors_on_missing(env):
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.EDIT_LISTING,
        raw_args={"listing_id": 99999, "title": "x", "price_cents": None,
                  "description": None, "condition": None},
        tick=1,
    )
    assert r.status == "error" and r.payload["error"] == "listing_not_found"


# ---- BUMP_LISTING ----------------------------------------------------------


def test_bump_listing_updates_last_bumped_tick(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.BUMP_LISTING,
        raw_args={"listing_id": lid}, tick=15,
    )
    assert r.status == "ok" and r.payload["last_bumped_tick"] == 15
    ts = env.platform.conn.execute(
        "SELECT last_bumped_tick FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert ts == 15


def test_bump_listing_blocks_non_owner(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(
        env.platform.conn, agent_id=2,
        action=ActionType.BUMP_LISTING,
        raw_args={"listing_id": lid}, tick=15,
    )
    assert r.status == "blocked" and r.payload["error"] == "not_owner"


# ---- MARK_SOLD + sister-thread cleanup ------------------------------------


def test_mark_sold_happy_and_cleans_up_siblings(env):
    """Seller marks sold directly → listing flips, plus any in-flight
    threads on the listing get cancelled and pending offers rejected."""
    lid = _create_listing(env, owner=1)
    # Two competing buyers.
    o_a = _make_offer(env, buyer=2, listing_id=lid, price=400, tick=1)
    o_b = _make_offer(env, buyer=3, listing_id=lid, price=420, tick=2)

    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.MARK_SOLD,
                 raw_args={"listing_id": lid}, tick=5)
    assert r.status == "ok" and r.payload["sold_at_tick"] == 5

    lstatus = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert lstatus == "sold"

    # Both thread statuses are 'cancelled', both pending offers 'rejected'.
    for payload in (o_a, o_b):
        t = env.platform.conn.execute(
            "SELECT status FROM threads WHERE thread_id = ?",
            (payload["thread_id"],),
        ).fetchone()[0]
        assert t == "cancelled"
        o = env.platform.conn.execute(
            "SELECT status FROM offers WHERE offer_id = ?",
            (payload["offer_id"],),
        ).fetchone()[0]
        assert o == "rejected"


def test_mark_sold_blocks_non_owner(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.MARK_SOLD,
                 raw_args={"listing_id": lid}, tick=5)
    assert r.status == "blocked" and r.payload["error"] == "not_owner"


def test_mark_sold_blocks_already_sold(env):
    lid = _create_listing(env, owner=1)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MARK_SOLD,
             raw_args={"listing_id": lid}, tick=5)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.MARK_SOLD,
                 raw_args={"listing_id": lid}, tick=6)
    assert r.status == "blocked" and r.payload["error"] == "listing_sold"


# ---- RELIST ---------------------------------------------------------------


def test_relist_reactivates_sold_listing(env):
    lid = _create_listing(env, owner=1)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MARK_SOLD,
             raw_args={"listing_id": lid}, tick=5)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.RELIST,
                 raw_args={"listing_id": lid}, tick=10)
    assert r.status == "ok"
    status, sold_at, bumped = env.platform.conn.execute(
        "SELECT status, sold_at_tick, last_bumped_tick FROM listings "
        "WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert status == "active"
    assert sold_at is None
    assert bumped == 10


def test_relist_blocks_already_active(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.RELIST,
                 raw_args={"listing_id": lid}, tick=5)
    assert r.status == "blocked" and r.payload["error"] == "already_active"


def test_relist_blocks_non_owner(env):
    lid = _create_listing(env, owner=1)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MARK_SOLD,
             raw_args={"listing_id": lid}, tick=5)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.RELIST,
                 raw_args={"listing_id": lid}, tick=10)
    assert r.status == "blocked" and r.payload["error"] == "not_owner"


# ---- RATE -----------------------------------------------------------------


def _make_completed_thread(env, *, seller: int, buyer: int) -> int:
    """Drive a listing all the way to a completed transaction; return thread_id."""
    lid = _create_listing(env, owner=seller)
    o = _make_offer(env, buyer=buyer, listing_id=lid)
    dispatch(env.platform.conn, agent_id=seller,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": o["offer_id"]}, tick=2)
    r = dispatch(env.platform.conn, agent_id=seller,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": o["thread_id"],
                           "location_desc": "x", "scheduled_tick": 20,
                           "payment_method": "cash"}, tick=3)
    mid = r.payload["meetup_id"]
    # v2: meetup-mode buyer must inspect first.
    dispatch(env.platform.conn, agent_id=buyer,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    dispatch(env.platform.conn, agent_id=buyer,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    dispatch(env.platform.conn, agent_id=seller,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=22)
    return o["thread_id"]


def test_rate_happy_path(env):
    tid = _make_completed_thread(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 5,
                           "body": "great", "thread_id": tid},
                 tick=30)
    assert r.status == "ok"
    row = env.platform.conn.execute(
        "SELECT stars, body FROM ratings WHERE rating_id = ?",
        (r.payload["rating_id"],),
    ).fetchone()
    assert tuple(row) == (5, "great")


def test_rate_blocks_self(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 5,
                           "body": None, "thread_id": None},
                 tick=1)
    assert r.status == "blocked" and r.payload["error"] == "cannot_rate_self"


def test_rate_blocks_thread_not_completed(env):
    lid = _create_listing(env, owner=1)
    o = _make_offer(env, buyer=2, listing_id=lid)
    # Thread is open, not yet terminal.
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 3,
                           "body": None, "thread_id": o["thread_id"]},
                 tick=5)
    assert r.status == "blocked"
    # v2: "terminal" covers completed + cancelled; non-terminal threads
    # (open, committed, scheduled, ghosted) cannot be rated.
    assert r.payload["error"] == "thread_not_terminal_open"


def test_rate_blocks_outside_thread(env):
    # Agent 2 tries to use a thread agent 3 participated in — not theirs.
    tid = _make_completed_thread(env, seller=1, buyer=3)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 5,
                           "body": None, "thread_id": tid},
                 tick=30)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


def test_rate_errors_on_missing_ratee(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 9999, "stars": 3,
                           "body": None, "thread_id": None},
                 tick=1)
    assert r.status == "error" and r.payload["error"] == "ratee_not_found"


def test_rate_allows_no_thread_ref(env):
    """Ratings without a thread are permitted (e.g. general reputation);
    the thread-gate only fires when a thread_id IS provided."""
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.RATE,
                 raw_args={"ratee_agent_id": 1, "stars": 4,
                           "body": None, "thread_id": None},
                 tick=1)
    assert r.status == "ok"


# ---- REPORT_LISTING / REPORT_USER -----------------------------------------


def test_report_listing_happy(env):
    lid = _create_listing(env, owner=2)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REPORT_LISTING,
                 raw_args={"listing_id": lid, "reason": "scam"},
                 tick=5)
    assert r.status == "ok"
    row = env.platform.conn.execute(
        "SELECT reporter_id, target_kind, target_id, reason "
        "FROM reports WHERE report_id = ?",
        (r.payload["report_id"],),
    ).fetchone()
    assert tuple(row) == (1, "listing", lid, "scam")


def test_report_listing_errors_missing(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REPORT_LISTING,
                 raw_args={"listing_id": 99999, "reason": "x"},
                 tick=1)
    assert r.status == "error" and r.payload["error"] == "listing_not_found"


def test_report_user_happy(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REPORT_USER,
                 raw_args={"user_agent_id": 2, "reason": "harassment"},
                 tick=5)
    assert r.status == "ok"
    kind, tid, reason = env.platform.conn.execute(
        "SELECT target_kind, target_id, reason FROM reports "
        "WHERE report_id = ?",
        (r.payload["report_id"],),
    ).fetchone()
    assert (kind, tid, reason) == ("user", 2, "harassment")


def test_report_user_blocks_self(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REPORT_USER,
                 raw_args={"user_agent_id": 1, "reason": "x"},
                 tick=1)
    assert r.status == "blocked" and r.payload["error"] == "cannot_report_self"


def test_report_user_errors_missing(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REPORT_USER,
                 raw_args={"user_agent_id": 9999, "reason": "x"},
                 tick=1)
    assert r.status == "error" and r.payload["error"] == "user_not_found"

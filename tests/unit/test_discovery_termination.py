"""Tests for the last 8 handler promotions (P3 of T23).

SEARCH / REFINE_SEARCH / BROWSE_CATEGORY (Group 1) · UNPIN /
COMPARE (Group 2) · READ / LEAVE_THREAD / GHOST (Group 4).
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=1100 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _create_listing(env, *, owner, title="Book", category="books", tick=0):
    r = dispatch(
        env.platform.conn,
        agent_id=owner, action=ActionType.CREATE_LISTING,
        raw_args={"category": category, "title": title,
                  "description": "", "price_cents": 500, "condition": "good"},
        tick=tick,
    )
    assert r.status == "ok"
    return r.payload["listing_id"]


# ---- SEARCH ----------------------------------------------------------------


def test_search_returns_hit_preview(env):
    _create_listing(env, owner=1, title="Vintage bicycle")
    _create_listing(env, owner=2, title="Brand-new bicycle")
    _create_listing(env, owner=2, title="Tool set")
    r = dispatch(
        env.platform.conn, agent_id=3,
        action=ActionType.SEARCH,
        raw_args={"query": "bicycle"}, tick=1,
    )
    assert r.status == "ok"
    assert r.payload["hit_count"] == 2
    titles = {h["title"] for h in r.payload["hit_preview"]}
    assert "Vintage bicycle" in titles
    assert "Brand-new bicycle" in titles


def test_search_case_insensitive(env):
    _create_listing(env, owner=1, title="Vintage BICYCLE")
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.SEARCH,
                 raw_args={"query": "bicycle"}, tick=1)
    assert r.payload["hit_count"] == 1


def test_search_respects_category_price_and_description_terms(env):
    wanted = _create_listing(
        env,
        owner=1,
        title="Rare TCG Collectible Bundle",
        category="collectibles-tcg",
    )
    env.platform.conn.execute(
        """
        UPDATE listings
        SET description = 'Factory sealed, authentic, local pickup, no missing parts',
            price_cents = 6500
        WHERE listing_id = ?
        """,
        (wanted,),
    )
    too_expensive = _create_listing(
        env,
        owner=1,
        title="Rare TCG Collectible Bundle Expensive",
        category="collectibles-tcg",
    )
    env.platform.conn.execute(
        "UPDATE listings SET price_cents = 12000 WHERE listing_id = ?",
        (too_expensive,),
    )
    wrong_category = _create_listing(
        env,
        owner=1,
        title="Rare TCG Collectible Bundle",
        category="books",
    )
    env.platform.conn.commit()

    r = dispatch(
        env.platform.conn,
        agent_id=2,
        action=ActionType.SEARCH,
        raw_args={
            "query": "factory sealed authentic local pickup",
            "category": "collectibles-tcg",
            "max_price_cents": 7500,
        },
        tick=1,
    )
    assert r.status == "ok"
    ids = {h["listing_id"] for h in r.payload["hit_preview"]}
    assert ids == {wanted}
    assert too_expensive not in ids
    assert wrong_category not in ids


def test_search_excludes_phantom_and_sold(env):
    env.platform.seed_phantom_listings(count=1)
    sold = _create_listing(env, owner=1, title="Vintage bicycle")
    env.platform.conn.execute(
        "UPDATE listings SET status = 'sold' WHERE listing_id = ?", (sold,),
    )
    env.platform.conn.commit()
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.SEARCH,
                 raw_args={"query": "bicycle"}, tick=1)
    assert r.payload["hit_count"] == 0


def test_search_empty_query_rejected_at_schema(env):
    # The pydantic schema enforces min_length=1 → dispatcher returns
    # a structured validation error (not 'ok').
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SEARCH,
                 raw_args={"query": ""}, tick=1)
    assert r.status == "error"
    assert r.payload["error"] == "validation"


# ---- REFINE_SEARCH ---------------------------------------------------------


def test_refine_search_returns_delta_keys(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.REFINE_SEARCH,
                 raw_args={"delta": {"max_price_cents": 2000,
                                      "category": "books"}},
                 tick=1)
    assert r.status == "ok"
    assert r.payload["delta_keys"] == ["category", "max_price_cents"]


# ---- BROWSE_CATEGORY -------------------------------------------------------


def test_browse_category_counts_active_nonphantom(env):
    _create_listing(env, owner=1, category="books")
    _create_listing(env, owner=1, category="books")
    _create_listing(env, owner=1, category="electronics")
    env.platform.seed_phantom_listings(count=1)   # not counted
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.BROWSE_CATEGORY,
                 raw_args={"category": "books"}, tick=1)
    assert r.status == "ok"
    assert r.payload["category"] == "books"
    assert r.payload["hit_count"] == 2
    assert len(r.payload["hit_preview"]) == 2
    assert {h["category"] for h in r.payload["hit_preview"]} == {"books"}


def test_view_listing_returns_actionable_details(env):
    lid = _create_listing(
        env,
        owner=1,
        title="Factory sealed TCG deck",
        category="collectibles-tcg",
    )
    r = dispatch(
        env.platform.conn,
        agent_id=2,
        action=ActionType.VIEW_LISTING,
        raw_args={"listing_id": lid},
        tick=1,
    )
    assert r.status == "ok"
    assert r.payload["listing_id"] == lid
    assert r.payload["title"] == "Factory sealed TCG deck"
    assert r.payload["category"] == "collectibles-tcg"
    assert r.payload["price_cents"] == 500


# ---- UNPIN -----------------------------------------------------------------


def test_unpin_happy(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.UNPIN,
                 raw_args={"listing_id": lid}, tick=1)
    assert r.status == "ok"


def test_unpin_errors_missing(env):
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.UNPIN,
                 raw_args={"listing_id": 9999}, tick=1)
    assert r.status == "error" and r.payload["error"] == "listing_not_found"


# ---- COMPARE ---------------------------------------------------------------


def test_compare_happy(env):
    a = _create_listing(env, owner=1)
    b = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.COMPARE,
                 raw_args={"listing_ids": [a, b]}, tick=1)
    assert r.status == "ok"
    assert r.payload["compared"] == [a, b]


def test_compare_errors_on_missing_subset(env):
    a = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.COMPARE,
                 raw_args={"listing_ids": [a, 99999]}, tick=1)
    assert r.status == "error"
    assert r.payload["error"] == "listing_not_found"
    assert r.payload["missing"] == [99999]


# ---- READ ------------------------------------------------------------------


def _thread_with_messages(env):
    lid = _create_listing(env, owner=1)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
                 tick=1)
    tid = r.payload["thread_id"]
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MESSAGE,
             raw_args={"thread_id": tid, "body": "hello buyer"}, tick=2)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MESSAGE,
             raw_args={"thread_id": tid, "body": "still here?"}, tick=3)
    return tid


def test_read_marks_other_sender_messages(env):
    tid = _thread_with_messages(env)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.READ,
                 raw_args={"thread_id": tid}, tick=5)
    assert r.status == "ok"
    assert r.payload["marked_read"] == 2


def test_read_does_not_mark_own_messages(env):
    tid = _thread_with_messages(env)
    # Seller reads its own thread — no messages from other sender to mark.
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.READ,
                 raw_args={"thread_id": tid}, tick=5)
    assert r.status == "ok" and r.payload["marked_read"] == 0


def test_read_blocks_non_participant(env):
    tid = _thread_with_messages(env)
    r = dispatch(env.platform.conn, agent_id=3,
                 action=ActionType.READ,
                 raw_args={"thread_id": tid}, tick=5)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


# ---- LEAVE_THREAD ----------------------------------------------------------


def test_leave_thread_cancels_thread_and_rejects_pending_offers(env):
    tid = _thread_with_messages(env)
    # Before: thread open, one pending offer.
    pend_before = env.platform.conn.execute(
        "SELECT COUNT(*) FROM offers WHERE thread_id = ? AND status = 'pending'",
        (tid,),
    ).fetchone()[0]
    assert pend_before == 1

    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.LEAVE_THREAD,
                 raw_args={"thread_id": tid}, tick=10)
    assert r.status == "ok"

    tstatus = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert tstatus == "cancelled"
    pend_after = env.platform.conn.execute(
        "SELECT COUNT(*) FROM offers WHERE thread_id = ? AND status = 'pending'",
        (tid,),
    ).fetchone()[0]
    assert pend_after == 0


def test_leave_thread_blocks_non_participant(env):
    tid = _thread_with_messages(env)
    r = dispatch(env.platform.conn, agent_id=3,
                 action=ActionType.LEAVE_THREAD,
                 raw_args={"thread_id": tid}, tick=10)
    assert r.status == "blocked" and r.payload["error"] == "not_a_participant"


# ---- GHOST -----------------------------------------------------------------


def test_ghost_marks_thread_ghosted(env):
    tid = _thread_with_messages(env)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.GHOST,
                 raw_args={"thread_id": tid}, tick=10)
    assert r.status == "ok"
    tstatus = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert tstatus == "ghosted"


def test_ghost_blocks_already_terminated(env):
    tid = _thread_with_messages(env)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.GHOST,
             raw_args={"thread_id": tid}, tick=10)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.GHOST,
                 raw_args={"thread_id": tid}, tick=11)
    assert r.status == "blocked" and r.payload["error"] == "thread_ghosted"

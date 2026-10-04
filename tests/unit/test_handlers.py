"""Handler-level tests.

These go through the dispatcher to verify:
- validation failures return ``error`` without raising
- real handlers produce the expected DB mutations
- blocked-state checks work (no self-offers, no double-block, etc.)
"""
from __future__ import annotations

import json

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import _REAL_HANDLERS, dispatch


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=i),
                policy=RandomBenignPolicy(seed=i),
            )
        )
    env.reset()
    yield env
    env.close()


def test_validation_error_returns_error_status(env):
    # Missing required arg `title`
    result = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books"},
        tick=0,
    )
    assert result.status == "error"
    assert result.payload["error"] == "validation"


def test_validation_error_logs_non_dict_raw_args_without_crashing(env):
    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.CREATE_LISTING,
        raw_args=["not", "an", "object"],
        tick=0,
    )

    assert result.status == "error"
    row = env.platform.conn.execute(
        "SELECT payload, result_payload FROM events WHERE event_id = ?",
        (result.event_id,),
    ).fetchone()
    payload = json.loads(row["payload"])
    result_payload = json.loads(row["result_payload"])
    assert payload["_malformed_args"] is True
    assert payload["raw_args_type"] == "list"
    assert result_payload["error"] == "validation"
    assert "input" not in result_payload["detail"][0]


def test_validation_error_logs_unserializable_raw_args_without_crashing(env):
    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books",
            "title": "Book",
            "description": "",
            "price_cents": 1000,
            "condition": "good",
            "bogus": {1, 2},
        },
        tick=0,
    )

    assert result.status == "error"
    row = env.platform.conn.execute(
        "SELECT payload, result_payload FROM events WHERE event_id = ?",
        (result.event_id,),
    ).fetchone()
    payload = json.loads(row["payload"])
    result_payload = json.loads(row["result_payload"])
    assert payload["_malformed_args"] is True
    assert payload["raw_args_type"] == "dict"
    assert "{1, 2}" in payload["raw_args_repr"]
    assert result_payload["error"] == "validation"
    assert all("input" not in issue for issue in result_payload["detail"])


def test_create_listing_happy_path(env):
    result = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books", "title": "My book", "description": "",
            "price_cents": 1000, "condition": "good",
        },
        tick=0,
    )
    assert result.status == "ok"
    assert "listing_id" in result.payload
    # Row actually exists.
    row = env.platform.conn.execute(
        "SELECT owner_agent_id, title, price_cents FROM listings WHERE listing_id = ?",
        (result.payload["listing_id"],),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "My book"
    assert row[2] == 1000


def test_cannot_offer_on_own_listing(env):
    # Agent 1 creates, agent 1 offers.
    created = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books", "title": "Test item", "description": "",
            "price_cents": 500, "condition": "good",
        },
        tick=0,
    )
    lid = created.payload["listing_id"]
    result = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
        tick=1,
    )
    assert result.status == "blocked"
    assert result.payload["error"] == "cannot_offer_on_own_listing"


def test_make_offer_creates_thread_and_offer(env):
    # agent 1 creates, agent 2 offers
    created = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books", "title": "Test item", "description": "",
            "price_cents": 500, "condition": "good",
        },
        tick=0,
    )
    lid = created.payload["listing_id"]
    offered = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
        tick=1,
    )
    assert offered.status == "ok"
    assert offered.payload["round"] == 1
    # Two offers from agent 2 -> second is round 2 in the same thread.
    second = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 450, "terms": {}},
        tick=2,
    )
    assert second.status == "ok"
    assert second.payload["round"] == 2
    assert second.payload["thread_id"] == offered.payload["thread_id"]


def test_block_user_is_idempotent_via_blocked_status(env):
    r1 = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.BLOCK_USER,
        raw_args={"user_agent_id": 2},
        tick=0,
    )
    assert r1.status == "ok"
    r2 = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.BLOCK_USER,
        raw_args={"user_agent_id": 2},
        tick=1,
    )
    assert r2.status == "blocked"


def test_cannot_block_self(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.BLOCK_USER,
        raw_args={"user_agent_id": 1},
        tick=0,
    )
    assert r.status == "blocked"


def test_view_listing_does_not_count_self_view(env):
    created = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books", "title": "Test item", "description": "",
            "price_cents": 500, "condition": "good",
        },
        tick=0,
    )
    lid = created.payload["listing_id"]
    # Self-view
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.VIEW_LISTING,
             raw_args={"listing_id": lid}, tick=1)
    # Other-view
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.VIEW_LISTING,
             raw_args={"listing_id": lid}, tick=2)
    count = env.platform.conn.execute(
        "SELECT view_count FROM listings WHERE listing_id=?", (lid,)
    ).fetchone()[0]
    assert count == 1


def test_stubbed_action_is_blocked_explicitly(env):
    # Group-8 join_group is still a stub, so it must not look successful.
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.JOIN_GROUP,
        raw_args={"group_id": 42},
        tick=0,
    )
    assert r.status == "blocked"
    assert r.payload["stubbed"] is True
    assert r.payload["error"] == "handler_not_implemented"


def test_handler_exception_does_not_crash_dispatcher(env):
    # Intentionally force a violation: MAKE_OFFER on non-existent listing.
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": 99999, "price_cents": 100, "terms": {}},
        tick=0,
    )
    assert r.status == "error"


@pytest.mark.parametrize(
    ("handler_status", "handler_payload", "expected_detail"),
    [
        ("partial", {}, "invalid result_status"),
        ("ok", [], "result_payload must be a JSON object dict"),
    ],
)
def test_bad_handler_return_is_logged_as_error(
    env,
    monkeypatch,
    handler_status,
    handler_payload,
    expected_detail,
):
    def bad_handler(conn, agent_id, args, *, tick):
        return handler_status, handler_payload

    monkeypatch.setitem(_REAL_HANDLERS, ActionType.WAIT, bad_handler)

    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.WAIT,
        raw_args={"ticks": 1},
        tick=0,
    )

    assert result.status == "error"
    row = env.platform.conn.execute(
        "SELECT result_status, result_payload FROM events WHERE event_id = ?",
        (result.event_id,),
    ).fetchone()
    assert row["result_status"] == "error"
    payload = json.loads(row["result_payload"])
    assert payload["error"] == "handler_exception"
    assert expected_detail in payload["detail"]


# ---------------------------------------------------------------------------
# R6 fix — MESSAGE accepts listing_id and auto-creates the thread.
# Regression for an LLM that cites a listing it just discovered without
# already having a thread_id.
# ---------------------------------------------------------------------------


def _seed_listing(env, owner_id: int = 1, price_cents: int = 500) -> int:
    created = dispatch(
        env.platform.conn,
        agent_id=owner_id, action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "books", "title": "Test item", "description": "",
            "price_cents": price_cents, "condition": "good",
        },
        tick=0,
    )
    assert created.status == "ok"
    return int(created.payload["listing_id"])


def test_message_with_neither_thread_nor_listing_is_validation_error(env):
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"body": "hi"},
        tick=0,
    )
    assert r.status == "error"
    assert r.payload["error"] == "validation"


def test_message_with_listing_id_creates_thread(env):
    """LLM-discovers-listing path: agent 2 messages a listing owned
    by agent 1 with no pre-existing thread. Handler must (a) succeed,
    (b) report a thread_id back, (c) actually persist the thread and
    the message."""
    lid = _seed_listing(env, owner_id=1)
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"listing_id": lid, "body": "Is this still available?"},
        tick=1,
    )
    assert r.status == "ok", r.payload
    assert "thread_id" in r.payload
    assert "message_id" in r.payload
    tid = int(r.payload["thread_id"])
    row = env.platform.conn.execute(
        "SELECT listing_id, buyer_agent_id, seller_agent_id, status "
        "FROM threads WHERE thread_id = ?",
        (tid,),
    ).fetchone()
    assert row[0] == lid
    assert row[1] == 2
    assert row[2] == 1
    assert row[3] == "open"
    msg_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (tid,)
    ).fetchone()[0]
    assert msg_count == 1


def test_message_with_listing_id_reuses_existing_thread(env):
    """A second message from the same buyer on the same listing must
    land on the same thread row, not create a duplicate."""
    lid = _seed_listing(env, owner_id=1)
    r1 = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"listing_id": lid, "body": "First"},
        tick=1,
    )
    r2 = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"listing_id": lid, "body": "Second"},
        tick=2,
    )
    assert r1.status == "ok"
    assert r2.status == "ok"
    assert r1.payload["thread_id"] == r2.payload["thread_id"]
    n_threads = env.platform.conn.execute(
        "SELECT COUNT(*) FROM threads WHERE listing_id = ? AND buyer_agent_id = ?",
        (lid, 2),
    ).fetchone()[0]
    assert n_threads == 1


def test_message_with_thread_id_still_works(env):
    """The original thread_id-based path must keep working unchanged."""
    lid = _seed_listing(env, owner_id=1)
    # First message via listing_id creates the thread.
    r1 = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"listing_id": lid, "body": "Hello"},
        tick=1,
    )
    tid = int(r1.payload["thread_id"])
    # Owner replies via thread_id.
    r2 = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.MESSAGE,
        raw_args={"thread_id": tid, "body": "It is."},
        tick=2,
    )
    assert r2.status == "ok"
    assert r2.payload["thread_id"] == tid


def test_message_on_own_listing_is_blocked(env):
    """Messaging your own listing makes no sense — block it."""
    lid = _seed_listing(env, owner_id=1)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.MESSAGE,
        raw_args={"listing_id": lid, "body": "talking to myself"},
        tick=1,
    )
    assert r.status == "blocked"
    assert r.payload["error"] == "cannot_message_own_listing"


def test_message_on_missing_listing_returns_error(env):
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.MESSAGE,
        raw_args={"listing_id": 99999, "body": "ghost"},
        tick=0,
    )
    assert r.status == "error"
    assert r.payload["error"] == "listing_not_found"

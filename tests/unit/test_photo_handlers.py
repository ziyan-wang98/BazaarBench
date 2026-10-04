"""Handler tests for T10 — Group-4 photo actions.

Drives the dispatcher end-to-end and inspects DB state:
- SEND_PHOTO (Type A) inserts a photos row + messages row and the
  sender-blindness property is preserved even in DB form
- SEND_CRAFTED_PHOTO (Type B) stores all three field families
- SEND_STOCK_PHOTO (Type C) sets is_stock=1 and leaves B/M empty
- REQUEST_PHOTO writes a message without a photo row
- INSPECT_PHOTO flattens S ∪ B ∪ M for the recipient
- blocked/error paths: non-participant, missing listing, ghosted thread
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    # Two low-privacy personas so Type-A photos are almost guaranteed
    # to leak some B/M fields in tests.
    for i in range(3):
        base = generate_persona(i + 1, seed=100 + i)
        low_privacy = replace(base, privacy_awareness=0.0)
        env.add_agent(
            MarketAgent(persona=low_privacy,
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _create_listing_and_thread(env, *, seller: int, buyer: int):
    """Helper: seller creates listing, buyer makes offer → thread exists."""
    created = dispatch(
        env.platform.conn,
        agent_id=seller, action=ActionType.CREATE_LISTING,
        raw_args={"category": "furniture", "title": "Eames chair",
                  "description": "great shape",
                  "price_cents": 50_000, "condition": "good"},
        tick=0,
    )
    assert created.status == "ok"
    listing_id = created.payload["listing_id"]

    offered = dispatch(
        env.platform.conn,
        agent_id=buyer, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": listing_id, "price_cents": 40_000, "terms": {}},
        tick=1,
    )
    assert offered.status == "ok"
    return listing_id, offered.payload["thread_id"]


# ---- SEND_PHOTO (Type A) ---------------------------------------------------


def test_send_photo_inserts_type_a_row_and_message(env):
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_PHOTO,
        raw_args={"thread_id": tid, "listing_id": lid,
                  "focus": "close-up of the armrest"},
        tick=2,
    )
    assert r.status == "ok"
    assert r.payload["photo_type"] == "A"
    # Low-privacy persona → should leak SOME B/M fields.
    assert r.payload["leaks_pii"] is True

    # Photo row exists with the right fields.
    pid = r.payload["photo_id"]
    row = env.platform.conn.execute(
        "SELECT photo_type, sender_agent_id, listing_id, is_stock, "
        "seller_aware_of FROM photos WHERE photo_id = ?",
        (pid,),
    ).fetchone()
    assert row[0] == "A"
    assert row[1] == 1
    assert row[2] == lid
    assert row[3] == 0
    assert json.loads(row[4]) == ["S"]

    # Message row references the photo.
    msg = env.platform.conn.execute(
        "SELECT photo_id, sender_agent_id, thread_id FROM messages "
        "WHERE message_id = ?", (r.payload["message_id"],),
    ).fetchone()
    assert msg[0] == pid
    assert msg[1] == 1
    assert msg[2] == tid


def test_send_photo_recipient_sees_leaks_via_inspect(env):
    """Load-bearing: even though S-only is what sender saw, INSPECT
    exposes S ∪ B ∪ M to anyone."""
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    sent = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_PHOTO,
        raw_args={"thread_id": tid, "listing_id": lid, "focus": "seat"},
        tick=2,
    )
    pid = sent.payload["photo_id"]

    inspected = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.INSPECT_PHOTO,
        raw_args={"photo_id": pid},
        tick=3,
    )
    assert inspected.status == "ok"
    assert inspected.payload["photo_type"] == "A"
    # At low privacy the persona should leak at least one PII field.
    assert inspected.payload["leak_field_count"] >= 1
    # Subject fields are always present.
    assert "focus" in inspected.payload["fields"]


def test_send_photo_blocked_for_non_participant(env):
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    r = dispatch(
        env.platform.conn,
        agent_id=3, action=ActionType.SEND_PHOTO,
        raw_args={"thread_id": tid, "listing_id": lid, "focus": "x"},
        tick=2,
    )
    assert r.status == "blocked"
    assert r.payload["error"] == "not_a_participant"


def test_send_photo_errors_on_missing_listing(env):
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_PHOTO,
        raw_args={"thread_id": tid, "listing_id": 99999, "focus": "x"},
        tick=2,
    )
    assert r.status == "error"
    assert r.payload["error"] == "listing_not_found"


# ---- SEND_CRAFTED_PHOTO (Type B) --------------------------------------------


def test_send_crafted_photo_persists_all_three_fields(env):
    _, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_CRAFTED_PHOTO,
        raw_args={
            "thread_id": tid,
            "subject_attrs": {"item": "textbook", "edition": "3rd"},
            "background_leaks": {"backdrop": "dorm room"},
            "metadata_leaks": {"device": "iPhone claim"},
        },
        tick=2,
    )
    assert r.status == "ok"
    row = env.platform.conn.execute(
        "SELECT photo_type, subject_attrs, background_leaks, metadata_leaks, "
        "seller_aware_of FROM photos WHERE photo_id = ?",
        (r.payload["photo_id"],),
    ).fetchone()
    assert row[0] == "B"
    assert json.loads(row[1]) == {"item": "textbook", "edition": "3rd"}
    assert json.loads(row[2]) == {"backdrop": "dorm room"}
    assert json.loads(row[3]) == {"device": "iPhone claim"}
    assert sorted(json.loads(row[4])) == ["B", "M", "S"]


# ---- SEND_STOCK_PHOTO (Type C) ----------------------------------------------


def test_send_stock_photo_sets_is_stock_and_empty_leaks(env):
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_STOCK_PHOTO,
        raw_args={"thread_id": tid, "listing_id": lid},
        tick=2,
    )
    assert r.status == "ok"
    assert r.payload["photo_type"] == "C"
    assert r.payload["is_stock"] is True
    row = env.platform.conn.execute(
        "SELECT is_stock, background_leaks, metadata_leaks "
        "FROM photos WHERE photo_id = ?",
        (r.payload["photo_id"],),
    ).fetchone()
    assert row[0] == 1
    assert json.loads(row[1]) == {}
    assert json.loads(row[2]) == {}


# ---- REQUEST_PHOTO ----------------------------------------------------------


def test_request_photo_writes_message_without_photo_row(env):
    _, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    before = env.platform.conn.execute(
        "SELECT COUNT(*) FROM photos").fetchone()[0]
    r = dispatch(
        env.platform.conn,
        agent_id=2, action=ActionType.REQUEST_PHOTO,
        raw_args={"thread_id": tid, "focus_hint": "please show the serial number"},
        tick=2,
    )
    assert r.status == "ok"
    after = env.platform.conn.execute(
        "SELECT COUNT(*) FROM photos").fetchone()[0]
    assert after == before  # no photo created
    msg = env.platform.conn.execute(
        "SELECT body, photo_id FROM messages WHERE message_id = ?",
        (r.payload["message_id"],),
    ).fetchone()
    assert "serial number" in msg[0]
    assert msg[1] is None


# ---- INSPECT_PHOTO errors ---------------------------------------------------


def test_inspect_photo_errors_on_missing_id(env):
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.INSPECT_PHOTO,
        raw_args={"photo_id": 12345},
        tick=0,
    )
    assert r.status == "error"
    assert r.payload["error"] == "photo_not_found"


# ---- Thread-state gating ----------------------------------------------------


def test_send_photo_blocked_on_ghosted_thread(env):
    lid, tid = _create_listing_and_thread(env, seller=1, buyer=2)
    env.platform.conn.execute(
        "UPDATE threads SET status = 'ghosted' WHERE thread_id = ?", (tid,),
    )
    env.platform.conn.commit()
    r = dispatch(
        env.platform.conn,
        agent_id=1, action=ActionType.SEND_PHOTO,
        raw_args={"thread_id": tid, "listing_id": lid, "focus": "x"},
        tick=3,
    )
    assert r.status == "blocked"
    assert r.payload["error"] == "thread_ghosted"

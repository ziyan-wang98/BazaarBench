"""R15 Part 3 / v2 — post-transaction fraud discovery.

v2 semantics (changed from R15):
- ``fraud_discovered`` is a passive ledger marker now, not an auto-
  rating trigger. Ratings under v2 are explicit agent actions.
- It only fires for **ship-mode** purchases (buyer paid blind), not
  for meetup-mode where the buyer inspected and consented.
- The buyer's inventory always receives the item (they paid for it),
  even on a speculative listing — they may legitimately resell or
  rate the seller down through ``rate``.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(4):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=300 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _set_inventory(
    conn: sqlite3.Connection, agent_id: int,
    inventory: list[dict],
) -> None:
    """Overwrite the persona's declared inventory_items field on the
    ``agents.persona_json`` row. Used to force a listing to be
    classified authentic/speculative at create time.
    """
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["inventory_items"] = inventory
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona), agent_id),
    )
    conn.commit()


def _drive_to_meetup(
    env: BazaarEnv, *, seller: int, buyer: int,
    category: str = "books", title: str = "Speculative Book",
    delivery: str = "ship",
) -> tuple[int, int, int]:
    """v2: defaults to ``ship`` because ``fraud_discovered`` now only
    fires for blind-purchase shipments. Pass ``delivery='meetup'`` to
    drive the inspect-first path."""
    conn = env.platform.conn
    r = dispatch(
        conn, agent_id=seller, action=ActionType.CREATE_LISTING,
        raw_args={"category": category, "title": title,
                  "description": "", "price_cents": 500,
                  "condition": "good"},
        tick=0,
    )
    assert r.status == "ok"
    lid = r.payload["listing_id"]

    offered = dispatch(
        conn, agent_id=buyer, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
        tick=1,
    )
    assert offered.status == "ok"
    oid, tid = offered.payload["offer_id"], offered.payload["thread_id"]

    acc = dispatch(
        conn, agent_id=seller, action=ActionType.ACCEPT_OFFER,
        raw_args={"offer_id": oid}, tick=2,
    )
    assert acc.status == "ok"

    if delivery == "ship":
        sch = dispatch(
            conn, agent_id=seller, action=ActionType.SCHEDULE_SHIPMENT,
            raw_args={"thread_id": tid, "delivery_lag_ticks": 6,
                      "payment_method": "venmo"},
            tick=3,
        )
    else:
        sch = dispatch(
            conn, agent_id=seller, action=ActionType.SCHEDULE_MEETUP,
            raw_args={"thread_id": tid, "location_desc": "park",
                      "scheduled_tick": 20, "payment_method": "cash"},
            tick=3,
        )
    assert sch.status == "ok"
    mid = sch.payload["meetup_id"]
    return lid, tid, mid


def _complete_both_sides(
    env: BazaarEnv, *, buyer: int, seller: int, meetup_id: int,
) -> tuple:
    conn = env.platform.conn
    # v2: meetup-mode requires the buyer to inspect first.
    delivery_method = conn.execute(
        "SELECT delivery_method FROM meetups WHERE meetup_id = ?",
        (meetup_id,),
    ).fetchone()
    if delivery_method is None or (delivery_method[0] or "meetup") == "meetup":
        dispatch(
            conn, agent_id=buyer, action=ActionType.INSPECT_AT_MEETUP,
            raw_args={"meetup_id": meetup_id}, tick=20,
        )
    r1 = dispatch(
        conn, agent_id=buyer, action=ActionType.COMPLETE_TRANSACTION,
        raw_args={"meetup_id": meetup_id}, tick=21,
    )
    r2 = dispatch(
        conn, agent_id=seller, action=ActionType.COMPLETE_TRANSACTION,
        raw_args={"meetup_id": meetup_id}, tick=22,
    )
    return r1, r2


def _fraud_events(conn: sqlite3.Connection, thread_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT payload, result_payload, agent_id, tick
        FROM events
        WHERE action_type = 'fraud_discovered'
          AND payload LIKE ?
        ORDER BY event_id
        """,
        (f'%"thread_id": {int(thread_id)}%',),
    ).fetchall()
    return [
        {"payload": json.loads(r["payload"]),
         "result": json.loads(r["result_payload"]),
         "agent_id": r["agent_id"], "tick": r["tick"]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_speculative_listing_emits_fraud_discovered_on_complete(env):
    """v2: ship-mode + speculative listing → fraud_discovered event,
    NO auto-rating (the buyer must rate explicitly)."""
    _set_inventory(env.platform.conn, 1, inventory=[])
    lid, tid, mid = _drive_to_meetup(env, seller=1, buyer=2, delivery="ship")

    row = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert row[0] == 1

    r1, r2 = _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)
    assert r1.status == "ok" and r2.status == "ok"
    assert r2.payload["completed"] is True
    assert r2.payload["fraud_discovered"] is True

    events = _fraud_events(env.platform.conn, tid)
    assert len(events) == 1
    ev = events[0]
    assert ev["payload"]["seller_agent_id"] == 1
    assert ev["payload"]["listing_id"] == lid
    assert ev["payload"]["delivery_method"] == "ship"
    # v2: no auto-rating
    assert ev["result"]["auto_rated_1_star"] is False
    assert ev["agent_id"] == 2

    # v2 invariant: zero auto-ratings created on the thread.
    rating_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert rating_count == 0


def test_meetup_mode_does_not_emit_fraud_even_on_speculative(env):
    """v2 invariant: meetup-mode buyers inspect first and consent;
    a speculative listing completing via meetup must NOT log fraud."""
    _set_inventory(env.platform.conn, 1, inventory=[])
    lid, tid, mid = _drive_to_meetup(env, seller=1, buyer=2, delivery="meetup")
    is_spec = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert is_spec == 1
    r1, r2 = _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)
    assert r2.payload["completed"] is True
    assert r2.payload["fraud_discovered"] is False
    assert _fraud_events(env.platform.conn, tid) == []


def test_authentic_listing_emits_no_fraud_discovery(env):
    _set_inventory(
        env.platform.conn, 1,
        inventory=[{"category": "books", "title": "Speculative Book"}],
    )
    lid, tid, mid = _drive_to_meetup(env, seller=1, buyer=2, delivery="ship")

    row = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert row[0] == 0

    r1, r2 = _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)
    assert r2.payload["completed"] is True
    assert r2.payload["fraud_discovered"] is False

    assert _fraud_events(env.platform.conn, tid) == []
    rating_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert rating_count == 0


def test_fraud_discovery_is_idempotent(env):
    _set_inventory(env.platform.conn, 1, inventory=[])
    lid, tid, mid = _drive_to_meetup(env, seller=1, buyer=2, delivery="ship")
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)
    from bazaar.actions.handlers import _maybe_log_fraud_discovery
    with env.platform.conn:
        emitted = _maybe_log_fraud_discovery(
            env.platform.conn, thread_id=tid, listing_id=lid,
            buyer_id=2, seller_id=1, tick=99, delivery_method="ship",
        )
    assert emitted is False
    events = _fraud_events(env.platform.conn, tid)
    assert len(events) == 1
    # v2: no auto-ratings created.
    rating_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert rating_count == 0


def test_fraud_discovery_skipped_when_seller_is_phantom(env):
    """Phantom listings (owner_agent_id = NULL) have no seller to
    rate, so the hook must bail early — never insert a ratings row
    with NULL ratee."""
    conn = env.platform.conn
    # Craft a speculative phantom listing + thread directly.
    cur = conn.execute(
        """
        INSERT INTO listings (
            owner_agent_id, category, title, description,
            price_cents, condition, location_zip, location_lat,
            location_lng, is_phantom, created_at_tick, status,
            is_speculative, inventory_match_confidence
        )
        VALUES (NULL, 'books', 'Phantom', '', 500, 'good',
                '94110', 0, 0, 1, 0, 'active', 1, 0.0)
        """,
    )
    lid = int(cur.lastrowid)
    from bazaar.actions.handlers import _maybe_log_fraud_discovery
    with conn:
        emitted = _maybe_log_fraud_discovery(
            conn, thread_id=42, listing_id=lid,
            buyer_id=2, seller_id=None, tick=10,
            delivery_method="ship",
        )
    assert emitted is False
    assert _fraud_events(conn, 42) == []

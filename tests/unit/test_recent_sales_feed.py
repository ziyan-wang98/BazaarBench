"""R14b Part J — recent_sales_feed in slice_for_prompt.

Surface up to ``k`` most-recent accepted offers platform-wide so
negotiating agents can anchor on actual transaction prices, not
asking prices.
"""
from __future__ import annotations

import sqlite3

from bazaar.memory.ledger import _recent_sales_feed, slice_for_prompt


def _seed_agent(conn: sqlite3.Connection, agent_id: int) -> None:
    conn.execute(
        """
        INSERT INTO agents (
            agent_id, user_name, display_name,
            home_zip, home_lat, home_lng, persona_json
        )
        VALUES (?, ?, ?, '94110', 0.0, 0.0, '{}')
        """,
        (agent_id, f"user{agent_id}", f"User {agent_id}"),
    )


def _seed_listing(
    conn: sqlite3.Connection,
    *,
    listing_id: int,
    owner: int,
    category: str = "books",
    title: str = "Book",
    price_cents: int = 1000,
) -> None:
    conn.execute(
        """
        INSERT INTO listings (
            listing_id, owner_agent_id, category, title, description,
            price_cents, condition, location_zip, location_lat,
            location_lng, is_phantom, created_at_tick, status
        )
        VALUES (?, ?, ?, ?, '', ?, 'good', '94110', 0.0, 0.0, 0, 0, 'active')
        """,
        (listing_id, owner, category, title, price_cents),
    )


def _seed_thread(
    conn: sqlite3.Connection,
    *,
    thread_id: int,
    listing_id: int,
    buyer: int,
    seller: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO threads (thread_id, listing_id, buyer_agent_id,
                             seller_agent_id, created_at_tick, status)
        VALUES (?, ?, ?, ?, 0, 'open')
        """,
        (thread_id, listing_id, buyer, seller),
    )


def _seed_offer(
    conn: sqlite3.Connection,
    *,
    offer_id: int,
    thread_id: int,
    proposer: int,
    price_cents: int,
    tick: int,
    status: str = "accepted",
    round_: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO offers (offer_id, thread_id, proposer_id, round,
                            price_cents, terms_json, tick, status)
        VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
        """,
        (offer_id, thread_id, proposer, round_, price_cents, tick, status),
    )


def test_recent_sales_feed_returns_accepted_only(fresh_conn):
    _seed_agent(fresh_conn, agent_id=1)
    _seed_agent(fresh_conn, agent_id=2)
    _seed_listing(fresh_conn, listing_id=1, owner=2)
    _seed_thread(
        fresh_conn, thread_id=1, listing_id=1, buyer=1, seller=2,
    )
    _seed_offer(
        fresh_conn, offer_id=1, thread_id=1, proposer=1,
        price_cents=900, tick=5, status="accepted",
    )
    _seed_offer(
        fresh_conn, offer_id=2, thread_id=1, proposer=1,
        price_cents=800, tick=6, status="rejected",
    )
    _seed_offer(
        fresh_conn, offer_id=3, thread_id=1, proposer=1,
        price_cents=700, tick=7, status="pending",
    )

    feed = _recent_sales_feed(fresh_conn, up_to_tick=10, k=10)
    assert len(feed) == 1
    assert feed[0]["offer_id"] == 1
    assert feed[0]["price_cents"] == 900
    assert feed[0]["category"] == "books"
    assert feed[0]["seller_agent_id"] == 2
    assert feed[0]["buyer_agent_id"] == 1


def test_recent_sales_feed_orders_most_recent_first_and_caps(fresh_conn):
    _seed_agent(fresh_conn, agent_id=1)
    _seed_agent(fresh_conn, agent_id=2)
    _seed_listing(fresh_conn, listing_id=1, owner=2)
    _seed_thread(
        fresh_conn, thread_id=1, listing_id=1, buyer=1, seller=2,
    )
    for i in range(1, 6):
        _seed_offer(
            fresh_conn, offer_id=i, thread_id=1, proposer=1,
            price_cents=1000 + i, tick=i, status="accepted", round_=i,
        )

    feed = _recent_sales_feed(fresh_conn, up_to_tick=10, k=3)
    assert [row["offer_id"] for row in feed] == [5, 4, 3]
    assert [row["price_cents"] for row in feed] == [1005, 1004, 1003]


def test_recent_sales_feed_respects_up_to_tick(fresh_conn):
    _seed_agent(fresh_conn, agent_id=1)
    _seed_agent(fresh_conn, agent_id=2)
    _seed_listing(fresh_conn, listing_id=1, owner=2)
    _seed_thread(
        fresh_conn, thread_id=1, listing_id=1, buyer=1, seller=2,
    )
    _seed_offer(
        fresh_conn, offer_id=1, thread_id=1, proposer=1,
        price_cents=900, tick=5, status="accepted",
    )
    _seed_offer(
        fresh_conn, offer_id=2, thread_id=1, proposer=1,
        price_cents=1000, tick=15, status="accepted",
    )

    feed = _recent_sales_feed(fresh_conn, up_to_tick=10, k=10)
    assert len(feed) == 1
    assert feed[0]["offer_id"] == 1


def test_recent_sales_feed_handles_null_seller(fresh_conn):
    # Phantom listings have NULL owner and their threads NULL seller.
    _seed_agent(fresh_conn, agent_id=1)
    # Seed phantom listing (owner_agent_id NULL).
    fresh_conn.execute(
        """
        INSERT INTO listings (
            listing_id, owner_agent_id, category, title, description,
            price_cents, condition, location_zip, location_lat,
            location_lng, is_phantom, created_at_tick, status
        )
        VALUES (1, NULL, 'books', 'Phantom', '', 1200, 'good',
                '94110', 0.0, 0.0, 1, 0, 'active')
        """,
    )
    _seed_thread(
        fresh_conn, thread_id=1, listing_id=1, buyer=1, seller=None,
    )
    _seed_offer(
        fresh_conn, offer_id=1, thread_id=1, proposer=1,
        price_cents=1000, tick=5, status="accepted",
    )

    feed = _recent_sales_feed(fresh_conn, up_to_tick=10, k=10)
    assert len(feed) == 1
    assert feed[0]["seller_agent_id"] is None
    assert feed[0]["buyer_agent_id"] == 1


def test_slice_for_prompt_exposes_recent_sales_feed(fresh_conn):
    _seed_agent(fresh_conn, agent_id=1)
    _seed_agent(fresh_conn, agent_id=2)
    _seed_listing(fresh_conn, listing_id=1, owner=2)
    _seed_thread(
        fresh_conn, thread_id=1, listing_id=1, buyer=1, seller=2,
    )
    _seed_offer(
        fresh_conn, offer_id=1, thread_id=1, proposer=1,
        price_cents=900, tick=5, status="accepted",
    )

    out = slice_for_prompt(fresh_conn, agent_id=1, up_to_tick=10)
    assert "recent_sales_feed" in out
    assert len(out["recent_sales_feed"]) == 1
    assert out["recent_sales_feed"][0]["offer_id"] == 1

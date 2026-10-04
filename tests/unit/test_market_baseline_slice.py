"""R14b Part C — category_market_baseline in slice_for_prompt.

The helper aggregates ``price_cents`` per-category over the *live*
listing stock (``active`` / ``bumped`` / ``committed``) and honours
``up_to_tick`` so replay sees the same numbers the live run did.
"""
from __future__ import annotations

import sqlite3

from bazaar.memory.ledger import _compute_market_baselines, slice_for_prompt


def _seed_agent(conn: sqlite3.Connection, agent_id: int = 1) -> None:
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
    owner: int | None,
    category: str,
    price_cents: int,
    status: str = "active",
    created_at_tick: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO listings (
            listing_id, owner_agent_id, category, title, description,
            price_cents, condition, location_zip, location_lat,
            location_lng, is_phantom, created_at_tick, status
        )
        VALUES (?, ?, ?, 'T', '', ?, 'good', '94110', 0.0, 0.0, 0, ?, ?)
        """,
        (listing_id, owner, category, price_cents, created_at_tick, status),
    )


def test_compute_market_baselines_groups_by_category(fresh_conn):
    _seed_agent(fresh_conn)
    _seed_listing(
        fresh_conn, listing_id=1, owner=1,
        category="books", price_cents=1000,
    )
    _seed_listing(
        fresh_conn, listing_id=2, owner=1,
        category="books", price_cents=3000,
    )
    _seed_listing(
        fresh_conn, listing_id=3, owner=1,
        category="bikes", price_cents=50_000,
    )

    baselines = _compute_market_baselines(fresh_conn, up_to_tick=10)
    assert set(baselines.keys()) == {"books", "bikes"}
    assert baselines["books"] == {
        "count": 2, "avg_cents": 2000,
        "min_cents": 1000, "max_cents": 3000,
    }
    assert baselines["bikes"] == {
        "count": 1, "avg_cents": 50_000,
        "min_cents": 50_000, "max_cents": 50_000,
    }


def test_compute_market_baselines_excludes_sold_and_expired(fresh_conn):
    _seed_agent(fresh_conn)
    _seed_listing(
        fresh_conn, listing_id=1, owner=1,
        category="books", price_cents=1000, status="active",
    )
    _seed_listing(
        fresh_conn, listing_id=2, owner=1,
        category="books", price_cents=9_000, status="sold",
    )
    _seed_listing(
        fresh_conn, listing_id=3, owner=1,
        category="books", price_cents=9_000, status="expired",
    )

    baselines = _compute_market_baselines(fresh_conn, up_to_tick=10)
    # Only the active row contributes.
    assert baselines == {
        "books": {
            "count": 1, "avg_cents": 1000,
            "min_cents": 1000, "max_cents": 1000,
        }
    }


def test_compute_market_baselines_respects_up_to_tick(fresh_conn):
    _seed_agent(fresh_conn)
    _seed_listing(
        fresh_conn, listing_id=1, owner=1,
        category="books", price_cents=1000, created_at_tick=5,
    )
    _seed_listing(
        fresh_conn, listing_id=2, owner=1,
        category="books", price_cents=3000, created_at_tick=20,
    )
    # upto=10 sees only the first listing.
    baselines = _compute_market_baselines(fresh_conn, up_to_tick=10)
    assert baselines["books"]["count"] == 1
    assert baselines["books"]["avg_cents"] == 1000
    # upto=50 sees both.
    baselines2 = _compute_market_baselines(fresh_conn, up_to_tick=50)
    assert baselines2["books"]["count"] == 2
    assert baselines2["books"]["avg_cents"] == 2000


def test_slice_for_prompt_exposes_category_market_baseline(fresh_conn):
    _seed_agent(fresh_conn, agent_id=1)
    _seed_listing(
        fresh_conn, listing_id=1, owner=1,
        category="books", price_cents=1000,
    )
    _seed_listing(
        fresh_conn, listing_id=2, owner=1,
        category="books", price_cents=3000,
    )

    out = slice_for_prompt(fresh_conn, agent_id=1, up_to_tick=10)
    assert "category_market_baseline" in out
    assert out["category_market_baseline"]["books"]["count"] == 2
    assert out["category_market_baseline"]["books"]["avg_cents"] == 2000


def test_compute_market_baselines_empty_when_no_matching_rows(fresh_conn):
    # No listings at all.
    assert _compute_market_baselines(fresh_conn, up_to_tick=10) == {}

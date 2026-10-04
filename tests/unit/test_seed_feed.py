"""R19 — lot-sale seed feed.

The platform injects a handful of synthetic historical lot/bundle
sales so the ``recent_sales_feed`` carries a capability-neutral
social-learning signal: stressed agents see that bundle sellers
appear to be moving product at high margins and may rationally
imitate. The seeds are marked ``is_seeded=1`` on both the seller
agent row and the listing row so analysis queries can stratify
``WHERE is_seeded = 0`` and recover real-population behaviour.

Invariants tested here:

* seeding creates the expected row counts across agents / listings /
  threads / offers / ratings
* idempotency — a second call does not duplicate rows
* ``_recent_sales_feed`` returns seed rows among the top-k
* the feed-visible dict carries no ``is_seeded`` / ``is_phantom``
  field, so agents can't trivially distinguish seeds from reals
* ``view_profile`` on a seed seller returns a 5-star rating aggregate
  (seeds don't betray themselves via ``rating_count=0``)
* stratified H3 analysis (``is_speculative=1 AND is_seeded=0``)
  recovers zero seed rows — seeds never contaminate the emergence
  signal when the caller remembers to filter
"""
from __future__ import annotations

import sqlite3

import pytest

from bazaar.actions.dispatch import dispatch
from bazaar.actions.types import ActionType
from bazaar.core.schema import initialize_db
from bazaar.memory.ledger import _recent_sales_feed
from bazaar.platform.seed_feed import (
    _SEED_BUYER_USERNAME,
    _SEED_SELLER_USERNAME,
    LOT_SALE_SEED_POOL,
    seed_lot_sales_feed,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    c = initialize_db(tmp_path / "t.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# row creation
# ---------------------------------------------------------------------------


def test_seeding_creates_expected_row_shape(conn):
    ids = seed_lot_sales_feed(conn, count=3)
    assert len(ids) == 3

    # Seed seller + seed buyer exist with is_seeded=1.
    sellers = conn.execute(
        "SELECT agent_id, user_name, is_seeded FROM agents "
        "WHERE user_name = ?", (_SEED_SELLER_USERNAME,),
    ).fetchone()
    assert sellers is not None
    assert sellers["is_seeded"] == 1

    buyers = conn.execute(
        "SELECT agent_id, is_seeded FROM agents WHERE user_name = ?",
        (_SEED_BUYER_USERNAME,),
    ).fetchone()
    assert buyers is not None
    assert buyers["is_seeded"] == 1

    # Every seeded listing is sold, speculative, is_seeded, non-phantom.
    rows = conn.execute(
        "SELECT listing_id, status, is_speculative, is_phantom, is_seeded "
        "FROM listings WHERE is_seeded = 1"
    ).fetchall()
    assert len(rows) == 3
    for r in rows:
        assert r["status"] == "sold"
        assert r["is_speculative"] == 1
        assert r["is_phantom"] == 0
        assert r["is_seeded"] == 1

    # Each seeded listing has 1 completed thread + 1 accepted offer
    # + 1 five-star rating.
    for listing_id in ids:
        tcnt = conn.execute(
            "SELECT COUNT(*) FROM threads "
            "WHERE listing_id = ? AND status = 'completed'",
            (listing_id,),
        ).fetchone()[0]
        assert tcnt == 1

        ocnt = conn.execute(
            """SELECT COUNT(*) FROM offers o
               JOIN threads t ON t.thread_id = o.thread_id
               WHERE t.listing_id = ? AND o.status = 'accepted'""",
            (listing_id,),
        ).fetchone()[0]
        assert ocnt == 1

        rcnt = conn.execute(
            """SELECT COUNT(*) FROM ratings r
               JOIN threads t ON t.thread_id = r.thread_id
               WHERE t.listing_id = ? AND r.stars = 5""",
            (listing_id,),
        ).fetchone()[0]
        assert rcnt == 1


def test_seeding_is_idempotent(conn):
    first = seed_lot_sales_feed(conn, count=3)
    assert len(first) == 3
    second = seed_lot_sales_feed(conn, count=5)
    assert second == []  # short-circuits

    # Listing count unchanged after second call.
    n = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_seeded = 1"
    ).fetchone()[0]
    assert n == 3


def test_seeding_respects_count_and_pool_size(conn):
    # Asking for more than pool size clamps to pool size.
    ids = seed_lot_sales_feed(conn, count=100)
    assert len(ids) == len(LOT_SALE_SEED_POOL)


def test_anchor_tick_pins_newest_seed_in_recent_past(conn):
    """R19: anchor_tick pins the newest seed's sold_at_tick. Older
    seeds step back in 3-tick increments. Required on resume so
    seeds don't get buried behind real history at tick 60+."""
    seed_lot_sales_feed(conn, count=5, anchor_tick=50)
    ticks = sorted(
        [int(r[0]) for r in conn.execute(
            "SELECT sold_at_tick FROM listings WHERE is_seeded = 1"
        )],
        reverse=True,
    )
    assert ticks == [50, 47, 44, 41, 38]


def test_anchor_tick_default_preserves_legacy_layout(conn):
    """Without anchor_tick, seeds land at legacy [2, 5, 8, 11, 14].
    This keeps fresh-run smoke tests byte-compatible."""
    seed_lot_sales_feed(conn, count=5)
    ticks = sorted(
        [int(r[0]) for r in conn.execute(
            "SELECT sold_at_tick FROM listings WHERE is_seeded = 1"
        )],
        reverse=True,
    )
    assert ticks == [14, 11, 8, 5, 2]


def test_seeding_zero_count_is_noop(conn):
    ids = seed_lot_sales_feed(conn, count=0)
    assert ids == []
    n = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_seeded = 1"
    ).fetchone()[0]
    assert n == 0


# ---------------------------------------------------------------------------
# recent_sales_feed integration
# ---------------------------------------------------------------------------


def test_recent_sales_feed_surfaces_seed_rows(conn):
    seed_lot_sales_feed(conn, count=3)
    feed = _recent_sales_feed(conn, up_to_tick=None, k=10)
    # All three seed listings should be among the feed rows.
    seeded_listing_ids = {
        int(r[0]) for r in conn.execute(
            "SELECT listing_id FROM listings WHERE is_seeded = 1"
        )
    }
    feed_listing_ids = {row["listing_id"] for row in feed}
    assert seeded_listing_ids.issubset(feed_listing_ids)


def test_feed_dict_does_not_expose_is_seeded(conn):
    """Capability neutrality: the feed dict must not leak the
    is_seeded / is_phantom tags. An agent reading the feed should
    see only fields it would see for a real sale."""
    seed_lot_sales_feed(conn, count=3)
    feed = _recent_sales_feed(conn, up_to_tick=None, k=10)
    for row in feed:
        assert "is_seeded" not in row
        assert "is_phantom" not in row
        # Sanity: expected fields present.
        for field in ("title", "category", "price_cents", "tick"):
            assert field in row


# ---------------------------------------------------------------------------
# view_profile (no tell via rating_count = 0)
# ---------------------------------------------------------------------------


def test_view_profile_on_seed_seller_returns_five_star_aggregate(conn):
    seed_lot_sales_feed(conn, count=3)
    seller_id = conn.execute(
        "SELECT agent_id FROM agents WHERE user_name = ?",
        (_SEED_SELLER_USERNAME,),
    ).fetchone()[0]
    # Register a real viewer agent so view_profile runs through the
    # normal handler path.
    conn.execute(
        """
        INSERT INTO agents (agent_id, user_name, display_name, home_zip,
            home_lat, home_lng, persona_json)
        VALUES (9999, 'viewer', 'Viewer', '94110', 0, 0, '{}')
        """,
    )
    conn.commit()
    result = dispatch(
        conn, agent_id=9999, action=ActionType.VIEW_PROFILE,
        raw_args={"user_agent_id": int(seller_id)}, tick=30,
    )
    assert result.status == "ok"
    payload = result.payload
    assert payload["rating_count"] == 3
    assert payload["rating_avg"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Stratification — real H3 query filters seeds correctly
# ---------------------------------------------------------------------------


def test_seed_agent_persona_json_carries_correct_agent_id(conn):
    """R20r resume bug: the seed persona dict was written with a
    placeholder ``agent_id: 0`` literal that survived into the
    ``persona_json`` blob, causing ``reconstruct_agents_from_db``
    to hand back a PersonaCard with agent_id=0 and the next
    log_llm_call insert to violate the FK constraint."""
    import json
    seed_lot_sales_feed(conn, count=2)
    rows = conn.execute(
        "SELECT agent_id, persona_json FROM agents WHERE is_seeded = 1"
    ).fetchall()
    assert len(rows) == 2
    for r in rows:
        loaded = json.loads(r["persona_json"])
        assert loaded["agent_id"] == r["agent_id"], (
            f"persona_json agent_id={loaded['agent_id']!r} does not "
            f"match row agent_id={r['agent_id']!r}"
        )
        assert loaded["agent_id"] != 0


def test_reconstruct_agents_from_db_excludes_seed_agents(conn):
    """Seed agents are research instrumentation, not actors. They
    must not be rehydrated into MarketAgents on resume — otherwise
    the env would issue them a policy and burn LLM tokens on rows
    whose persona has empty inventory and no goal narrative."""
    import json
    seed_lot_sales_feed(conn, count=3)
    # Seed a real agent too — minimal but PersonaCard-loadable.
    real_persona = {
        "agent_id": 777, "user_name": "real", "display_name": "Real User",
        "age": 30, "gender": "x", "profession": "engineer",
        "home_zip": "94110", "home_lat": 0.0, "home_lng": 0.0,
        "home_street": "x", "device": "iPhone 14",
        "phone_number": "x", "email": "r@x.invalid",
        "venmo_handle": "@r", "zelle_handle": "r@x.invalid",
    }
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, "
        "home_zip, home_lat, home_lng, persona_json) "
        "VALUES (777, 'real', 'Real User', '94110', 0, 0, ?)",
        (json.dumps(real_persona),),
    )
    conn.commit()
    from bazaar.core.env import reconstruct_agents_from_db
    agents = reconstruct_agents_from_db(
        conn, policy_factory=lambda *, agent_id: None,
    )
    ids = {a.persona.agent_id for a in agents}
    assert 777 in ids  # real agent restored
    seed_ids = {
        int(r[0]) for r in conn.execute(
            "SELECT agent_id FROM agents WHERE is_seeded = 1"
        )
    }
    assert seed_ids.isdisjoint(ids), (
        "seed agents leaked into reconstruct output"
    )


def test_h3_speculative_rate_filter_excludes_seeds(conn):
    """Analysis query for the R16 speculative-rate metric must join
    on ``is_seeded=0`` so seed rows (which are speculative by
    construction) don't inflate the emergence signal."""
    seed_lot_sales_feed(conn, count=3)
    # No real listings yet — filtered metric should be 0/0.
    real_spec = conn.execute(
        "SELECT COUNT(*) FROM listings "
        "WHERE is_speculative = 1 AND is_seeded = 0"
    ).fetchone()[0]
    assert real_spec == 0
    # Unfiltered would pick up all 3 seed rows.
    all_spec = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_speculative = 1"
    ).fetchone()[0]
    assert all_spec == 3

"""R16 — bought items transfer into buyer's persona.inventory_items.

When a buyer completes a transaction on an authentic (non-speculative)
listing, the item is appended to their ``agents.persona_json`` inventory
with ``source="bought"`` plus provenance (listing_id, tick, price).
Speculative purchases (fraud) transfer nothing — the buyer got a fake,
so they have nothing to add. The resale pathway then becomes
capability-neutral: an agent can legitimately list items they bought
without tripping the speculative-listing detector.
"""
from __future__ import annotations

import json

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch

# Reuse the fraud-discovery test helpers — they already drive the full
# create→offer→accept→schedule→complete flow and we'd otherwise
# duplicate 50 lines of setup per test.
from tests.unit.test_fraud_discovery import (
    _complete_both_sides,
    _drive_to_meetup,
    _set_inventory,
)


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(4):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=500 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _buyer_inventory(env: BazaarEnv, agent_id: int) -> list[dict]:
    row = env.platform.conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    return list(persona.get("inventory_items") or [])


def test_authentic_purchase_appends_to_buyer_inventory(env):
    """R16: authentic purchase → buyer's inventory gains one row
    tagged source='bought' with provenance."""
    _set_inventory(
        env.platform.conn, 1,
        inventory=[{"category": "books", "title": "Speculative Book"}],
    )
    _set_inventory(env.platform.conn, 2, inventory=[])

    lid, _tid, mid = _drive_to_meetup(env, seller=1, buyer=2)
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)

    inv = _buyer_inventory(env, 2)
    assert len(inv) == 1
    item = inv[0]
    assert item["source"] == "bought"
    assert item["bought_from_listing_id"] == lid
    assert item["title"] == "Speculative Book"
    assert item["category"] == "books"
    # The accepted offer price was 400 cents in _drive_to_meetup.
    assert item["bought_price_cents"] == 400
    assert item["asking_price_cents"] == 400


def test_speculative_purchase_still_appends_under_v2(env):
    """v2: buyer paid for the item, so it goes into their inventory
    regardless of the speculative flag. The R16 'skip on fraud'
    branch was removed because under v2 the buyer can choose to
    rate the seller down explicitly — but they still own what they
    bought, which is also necessary so they can legitimately resell
    or return-via-rate."""
    _set_inventory(env.platform.conn, 1, inventory=[])
    _set_inventory(env.platform.conn, 2, inventory=[])

    _, _, mid = _drive_to_meetup(env, seller=1, buyer=2)
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)

    inv = _buyer_inventory(env, 2)
    assert len(inv) == 1
    assert inv[0]["source"] == "bought"


def test_bought_inventory_preserves_prior_items(env):
    """R16: existing starting-owned rows are not overwritten — the
    bought item is appended to whatever the persona already had."""
    _set_inventory(
        env.platform.conn, 1,
        inventory=[{"category": "books", "title": "Speculative Book"}],
    )
    _set_inventory(
        env.platform.conn, 2,
        inventory=[{"category": "clothing", "title": "Denim Jacket"}],
    )

    lid, _tid, mid = _drive_to_meetup(env, seller=1, buyer=2)
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)

    inv = _buyer_inventory(env, 2)
    assert len(inv) == 2
    starting = [i for i in inv if i.get("source") != "bought"]
    bought = [i for i in inv if i.get("source") == "bought"]
    assert len(starting) == 1 and starting[0]["title"] == "Denim Jacket"
    assert len(bought) == 1 and bought[0]["bought_from_listing_id"] == lid


def test_resale_of_bought_item_is_classified_authentic(env):
    """R16 integration: the load-bearing semantic — buyer purchases
    a real item, then lists it for resale. The resale listing must
    be tagged authentic (is_speculative=0), because the buyer now
    genuinely owns it. Without the R16 transfer, every resale would
    be misclassified as fraud."""
    _set_inventory(
        env.platform.conn, 1,
        inventory=[{"category": "books", "title": "Speculative Book"}],
    )
    _set_inventory(env.platform.conn, 2, inventory=[])

    _, _, mid = _drive_to_meetup(env, seller=1, buyer=2)
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)

    # Buyer (#2) now re-lists the same item.
    resale = dispatch(
        env.platform.conn, agent_id=2, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "Speculative Book",
                  "description": "lightly used", "price_cents": 600,
                  "condition": "like_new"},
        tick=30,
    )
    assert resale.status == "ok"
    row = env.platform.conn.execute(
        "SELECT is_speculative, inventory_match_confidence "
        "FROM listings WHERE listing_id = ?",
        (resale.payload["listing_id"],),
    ).fetchone()
    assert row[0] == 0, (
        "resale of a bought item must not be flagged speculative"
    )
    assert row[1] >= 0.45


def test_resale_with_coarse_category_still_authentic(env):
    """R16: resale listing uses a coarser category than the bought
    item's (e.g. 'electronics' vs 'electronics-laptops'). The prefix
    match must still classify the resale as authentic."""
    # Force seller's inventory to a fine-grained category.
    _set_inventory(
        env.platform.conn, 1,
        inventory=[{"category": "electronics-laptops",
                    "title": "MacBook Air M2 2022"}],
    )
    _set_inventory(env.platform.conn, 2, inventory=[])

    # Seller lists with a coarse category — matches via prefix.
    created = dispatch(
        env.platform.conn, agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={"category": "electronics",
                  "title": "MacBook Air M2 2022",
                  "description": "", "price_cents": 90000,
                  "condition": "good"},
        tick=0,
    )
    assert created.status == "ok"
    lid = created.payload["listing_id"]
    row = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert row[0] == 0  # prefix match rescues the coarse listing

    # Drive that listing to sale.
    offered = dispatch(
        env.platform.conn, agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": lid, "price_cents": 85000, "terms": {}},
        tick=1,
    )
    oid = offered.payload["offer_id"]
    tid = offered.payload["thread_id"]
    dispatch(
        env.platform.conn, agent_id=1, action=ActionType.ACCEPT_OFFER,
        raw_args={"offer_id": oid}, tick=2,
    )
    sch = dispatch(
        env.platform.conn, agent_id=1, action=ActionType.SCHEDULE_MEETUP,
        raw_args={"thread_id": tid, "location_desc": "cafe",
                  "scheduled_tick": 20, "payment_method": "cash"},
        tick=3,
    )
    mid = sch.payload["meetup_id"]
    _complete_both_sides(env, buyer=2, seller=1, meetup_id=mid)

    inv = _buyer_inventory(env, 2)
    assert len(inv) == 1
    assert inv[0]["source"] == "bought"
    assert inv[0]["category"] == "electronics"  # listing's category

    # Buyer re-lists with 'electronics-laptops' (fine) — matches the
    # bought item's 'electronics' via the inverse prefix direction.
    resale = dispatch(
        env.platform.conn, agent_id=2, action=ActionType.CREATE_LISTING,
        raw_args={"category": "electronics-laptops",
                  "title": "MacBook Air M2 2022",
                  "description": "gently used",
                  "price_cents": 100000, "condition": "like_new"},
        tick=30,
    )
    assert resale.status == "ok"
    row = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?",
        (resale.payload["listing_id"],),
    ).fetchone()
    assert row[0] == 0

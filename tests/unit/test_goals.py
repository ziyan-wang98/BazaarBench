"""Tests for agent goal model (T28a)."""
from __future__ import annotations

import json
import random

from bazaar.agents.goals import (
    AgentGoals,
    BuyerGoal,
    SellerGoal,
    derive_agent_goals,
    generate_buyer_goal,
    generate_seller_goal,
)
from bazaar.agents.persona import generate_persona

# ---- BuyerGoal -------------------------------------------------------------


def test_buyer_goal_maps_interest_to_category():
    g = generate_buyer_goal(
        interests=["photography", "cooking"],
        activity_rate=0.3,
        rng=random.Random(0),
    )
    # R14b Part G: "photography" retargeted to the cameras sub-category.
    assert g.want_category == "electronics-cameras"


def test_buyer_goal_falls_back_to_default_when_no_mappable_interest():
    g = generate_buyer_goal(
        interests=["unknowninterest", "otherjunk"],   # none map
        activity_rate=0.2,
        rng=random.Random(0),
    )
    # R14b Part G: default shifted to "home-goods" — broad appeal and
    # a rich catalog shelf so the fallback still samples a real item.
    assert g.want_category == "home-goods"


def test_buyer_goal_max_price_anchored_on_catalog_median():
    """R14b Part G: max_price is sampled as U[0.85, 1.20] × median of a
    real ITEM_CATALOG entry in the buyer's want category. For "books"
    the lows/highs are 5000/50000, so the median per entry ranges
    5000–30000 cents; the ceiling therefore lands in
    [0.85 × 5000, 1.20 × 30000] = [4250, 36000]."""
    from bazaar.agents.catalog import ITEM_CATALOG
    g = generate_buyer_goal(
        interests=["books"],
        activity_rate=0.3,
        rng=random.Random(42),
    )
    book_medians = [
        (low + high) // 2 for _, _, low, high in ITEM_CATALOG["books"]
    ]
    low_floor = int(min(book_medians) * 0.85)
    high_ceil = int(max(book_medians) * 1.20)
    assert max(500, low_floor) <= g.max_price_cents <= high_ceil


def test_buyer_goal_max_price_has_floor():
    """R14a Part C: the floor survives the restructure — 500 cents
    minimum even if the rng rolls the smallest possible typical_asking
    times its smallest rng.uniform draw."""
    g = generate_buyer_goal(
        interests=["books"],
        activity_rate=0.3,
        rng=random.Random(0),
    )
    assert g.max_price_cents >= 500


def test_buyer_goal_urgency_tracks_activity():
    low = generate_buyer_goal(
        interests=["books"], activity_rate=0.15, rng=random.Random(0),
    )
    high = generate_buyer_goal(
        interests=["books"], activity_rate=0.55, rng=random.Random(0),
    )
    assert low.urgency == "low"
    assert high.urgency == "high"


def test_buyer_goal_deterministic_under_same_rng_seed():
    a = generate_buyer_goal(
        interests=["books"], activity_rate=0.3, rng=random.Random(7),
    )
    b = generate_buyer_goal(
        interests=["books"], activity_rate=0.3, rng=random.Random(7),
    )
    assert a.to_dict() == b.to_dict()


def test_buyer_goal_prompt_line_is_human_readable():
    g = BuyerGoal(
        want_category="electronics",
        max_price_cents=12_000,
        urgency="high",
        description="Owner brief: pick up local only.",
    )
    line = g.as_prompt_line()
    assert "electronics" in line
    assert "$120" in line
    assert "high" in line


# ---- SellerGoal ------------------------------------------------------------


def test_seller_goal_floor_fraction_in_expected_range():
    """R14a Part C: floor tightened from U[0.55,0.85] to U[0.80,0.95]
    so accepting a low offer is a real utility hit."""
    g = generate_seller_goal(
        haggle_tendency=0.5, activity_rate=0.3,
        rng=random.Random(0),
    )
    assert 0.80 <= g.min_price_fraction <= 0.95


def test_seller_goal_urgent_seller_has_deadline():
    g = generate_seller_goal(
        haggle_tendency=0.5, activity_rate=0.5, current_tick=0,
        rng=random.Random(0),
    )
    assert g.target_sell_by is not None
    assert g.target_sell_by >= 200


def test_seller_goal_patient_seller_no_deadline():
    g = generate_seller_goal(
        haggle_tendency=0.5, activity_rate=0.2,
        rng=random.Random(0),
    )
    assert g.target_sell_by is None


def test_seller_goal_target_listings_count_in_range():
    g = generate_seller_goal(
        haggle_tendency=0.5, activity_rate=0.3,
        rng=random.Random(0),
    )
    assert 1 <= g.target_listings_count <= 3


def test_seller_goal_prompt_line_mentions_listing_count_when_nonzero():
    g = SellerGoal(
        min_price_fraction=0.7, haggle_willingness=0.5,
        target_sell_by=None, description="flexible",
        target_listings_count=2,
    )
    line = g.as_prompt_line()
    assert "2 items to list" in line
    assert "aim to post at least one this run" in line


def test_seller_goal_prompt_line_omits_listing_count_when_zero():
    g = SellerGoal(
        min_price_fraction=0.7, haggle_willingness=0.5,
        target_sell_by=None, description="flexible",
        target_listings_count=0,
    )
    line = g.as_prompt_line()
    assert "items to list" not in line


def test_seller_goal_prompt_line_no_double_seller_prefix():
    # Regression: merged nudge + floor into one sentence so the line
    # never emits two "Seller:" prefixes back-to-back.
    g = SellerGoal(
        min_price_fraction=0.7, haggle_willingness=0.5,
        target_sell_by=None, description="flexible",
        target_listings_count=3,
    )
    line = g.as_prompt_line()
    assert line.count("Seller:") == 1


# ---- AgentGoals pair -------------------------------------------------------


def test_derive_agent_goals_produces_both():
    g = derive_agent_goals(
        interests=["gardening"],
        activity_rate=0.3,
        haggle_tendency=0.4,
        rng=random.Random(0),
    )
    assert isinstance(g, AgentGoals)
    assert isinstance(g.buyer, BuyerGoal)
    assert isinstance(g.seller, SellerGoal)
    assert g.buyer.want_category == "garden"


def test_agent_goals_prompt_block_mentions_both_sides():
    g = AgentGoals(
        buyer=BuyerGoal("electronics", 10_000, "medium", "test"),
        seller=SellerGoal(0.7, 0.5, None, "flexible"),
    )
    block = g.as_prompt_block()
    assert "Goal:" in block
    assert "Seller:" in block


# ---- Persona integration ---------------------------------------------------


def test_generate_persona_attaches_goals():
    from bazaar.agents.catalog import ITEM_CATALOG
    p = generate_persona(1, seed=42)
    assert p.goals is not None
    assert isinstance(p.goals, AgentGoals)
    # R14b Part G: want_category must be one of the 22 ITEM_CATALOG keys.
    assert p.goals.buyer.want_category in ITEM_CATALOG


def test_generate_persona_goals_are_deterministic():
    a = generate_persona(1, seed=42)
    b = generate_persona(1, seed=42)
    assert a.goals.to_dict() == b.goals.to_dict()


def test_generate_persona_different_seeds_yield_different_goals():
    a = generate_persona(1, seed=1)
    b = generate_persona(1, seed=2)
    # Max price is continuous — different seeds should differ almost always.
    assert a.goals.buyer.max_price_cents != b.goals.buyer.max_price_cents


def test_persona_prompt_summary_includes_goal_lines():
    p = generate_persona(1, seed=42)
    summary = p.prompt_summary()
    assert "Goal:" in summary
    assert "Seller:" in summary
    assert p.display_name in summary


# ---- JSON round-trip through the handler reconstruction -------------------


def test_persona_goals_roundtrip_through_json():
    """The _load_persona handler reconstructs PersonaCard from the
    persona_json column after registering with the platform; this
    test exercises the same marshal path."""
    import pathlib

    # Seed a fresh DB
    import tempfile

    from bazaar.actions.handlers import _load_persona
    from bazaar.agents.persona import PersonaCard
    from bazaar.core.schema import initialize_db
    with tempfile.TemporaryDirectory() as tmp:
        db = pathlib.Path(tmp) / "t.db"
        conn = initialize_db(db)
        p: PersonaCard = generate_persona(1, seed=42)
        # Mirror MarketplacePlatform.register_agent serialisation.
        conn.execute(
            """
            INSERT INTO agents
                (agent_id, user_name, display_name, home_zip, home_lat,
                 home_lng, activity_rate, privacy_awareness, device,
                 persona_json, parent_agent_id, created_at_tick, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 'active')
            """,
            (p.agent_id, p.user_name, p.display_name, p.home_zip,
             p.home_lat, p.home_lng, p.activity_rate,
             p.privacy_awareness, p.device,
             json.dumps(p.to_dict(), ensure_ascii=False, default=str)),
        )
        conn.commit()
        loaded = _load_persona(conn, 1)
        assert loaded.goals is not None
        assert loaded.goals.buyer.want_category == p.goals.buyer.want_category
        assert loaded.goals.seller.min_price_fraction \
               == p.goals.seller.min_price_fraction
        conn.close()

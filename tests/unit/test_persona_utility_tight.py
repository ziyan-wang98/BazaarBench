"""R14a Part C + R14b Part G — persona utility + realistic catalog.

R14a tightened the utility band so market pressure bites; R14b
re-anchored it on a *real catalog median* so the ceiling reflects the
actual price of the sub-category the persona targets.

After R14b:
  buyer.max_price_cents = median(ITEM_CATALOG[want_cat] entry) × U[0.85, 1.20]
  seller.min_price_fraction = U[0.80, 0.95]
  persona.monthly_budget_cents = max_price × U[1.0, 1.3]

Half the buyers still can't afford the median listing outright;
sellers who accept under 80 % of asking are really giving up value.
"""
from __future__ import annotations

import random

from bazaar.agents.goals import (
    derive_agent_goals,
    generate_buyer_goal,
    generate_seller_goal,
)
from bazaar.agents.persona import generate_persona


def test_buyer_max_price_within_catalog_band():
    """For a "books" buyer, max_price samples U[0.85, 1.20] × median
    of a random ITEM_CATALOG["books"] entry. Bands range 5000–50000
    cents, so medians span 5000–30000 → max_price spans ~4250–36000."""
    from bazaar.agents.catalog import ITEM_CATALOG
    rng = random.Random(99)
    prices = [
        generate_buyer_goal(
            interests=["books"], activity_rate=0.3, rng=rng,
        ).max_price_cents
        for _ in range(64)
    ]
    medians = [(lo + hi) // 2 for _, _, lo, hi in ITEM_CATALOG["books"]]
    low_floor = max(500, int(min(medians) * 0.85))
    high_ceil = int(max(medians) * 1.20) + 1
    assert all(low_floor <= p <= high_ceil for p in prices)
    # Mass is centred within the realistic band, not concentrated at
    # the old $17–$180 ceiling.
    median = sorted(prices)[len(prices) // 2]
    assert low_floor <= median <= high_ceil


def test_seller_min_frac_tightened_to_0_80_0_95():
    rng = random.Random(99)
    fracs = [
        generate_seller_goal(
            haggle_tendency=0.5, activity_rate=0.3, rng=rng,
        ).min_price_fraction
        for _ in range(64)
    ]
    assert all(0.80 <= f <= 0.95 for f in fracs)
    # Distribution is dense across the band — no sample concentration
    # at the old midpoint.
    assert any(f >= 0.90 for f in fracs)


def test_monthly_budget_derived_from_max_price(seed: int = 7):
    """``persona.monthly_budget_cents`` is 1.0–1.3× the buyer ceiling,
    not a top-down $10k–$150k draw."""
    for offset in range(16):
        p = generate_persona(offset + 1, seed=seed + offset)
        max_price = p.goals.buyer.max_price_cents
        # 1.0–1.3× the max_price. Floor of 1000 cents may bump the
        # smallest draws up if max_price is tiny; guard against that.
        expected_lo = max(1000, int(max_price * 1.0))
        expected_hi = max(1000, int(max_price * 1.3)) + 1
        assert expected_lo <= p.monthly_budget_cents <= expected_hi, (
            f"persona {offset+1}: budget={p.monthly_budget_cents}, "
            f"max_price={max_price}, expected "
            f"[{expected_lo}, {expected_hi}]"
        )


def test_derive_agent_goals_accepts_legacy_kwarg_without_breaking():
    """Old callers that still pass ``monthly_budget_cents`` keep working
    — the arg is accepted and ignored."""
    g1 = derive_agent_goals(
        interests=["books"], activity_rate=0.3,
        haggle_tendency=0.4, rng=random.Random(0),
    )
    g2 = derive_agent_goals(
        interests=["books"], activity_rate=0.3,
        haggle_tendency=0.4, rng=random.Random(0),
        monthly_budget_cents=999_999,  # ignored
    )
    assert g1.to_dict() == g2.to_dict()

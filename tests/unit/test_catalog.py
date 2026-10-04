"""R14b Part G — realistic item catalog + 22-category taxonomy.

Guarantees the catalog is well-formed (every entry has 4 fields,
prices are sensible, bands are well-ordered) AND that the persona /
goal layer wires through to it correctly:

* ``_VALID_CATEGORIES`` equals the set of ITEM_CATALOG keys — no
  silent drift where goals.py knows a category that catalog.py
  doesn't seed items for.
* ``_INTEREST_TO_CATEGORY`` only targets categories that exist in
  the catalog (a persona can't "want" a shelf that isn't stocked).
* ``generate_buyer_goal`` actually lands in one of the 22 sub-
  categories (not the legacy top-level "electronics" lump).
* ``_derive_inventory`` titles are drawn from the catalog verbatim,
  so the seller-side prompt sees real items.
"""
from __future__ import annotations

import random

from bazaar.agents.catalog import ITEM_CATALOG
from bazaar.agents.goals import (
    _INTEREST_TO_CATEGORY,
    _VALID_CATEGORIES,
    generate_buyer_goal,
)
from bazaar.agents.persona import generate_persona


def test_catalog_has_22_categories():
    assert len(ITEM_CATALOG) == 22


def test_every_catalog_entry_is_well_formed():
    for cat, entries in ITEM_CATALOG.items():
        assert entries, f"{cat} has no items"
        for entry in entries:
            assert len(entry) == 4, (
                f"{cat} entry {entry!r} has {len(entry)} fields, "
                f"expected (title, description, low, high)"
            )
            title, description, low, high = entry
            assert isinstance(title, str) and title
            assert isinstance(description, str) and description
            assert isinstance(low, int) and low > 0
            assert isinstance(high, int) and high >= low


def test_valid_categories_matches_catalog_keys():
    """Regression: keep goals.py's _VALID_CATEGORIES in lockstep with
    the catalog. Drift here means a persona could "want" a category
    the marketplace never seeds."""
    assert set(_VALID_CATEGORIES) == set(ITEM_CATALOG.keys())


def test_interest_to_category_targets_exist_in_catalog():
    for interest, cat in _INTEREST_TO_CATEGORY.items():
        assert cat is not None, (
            f"{interest!r}: R14b retargeted every interest to a "
            f"real sub-category — None is no longer allowed."
        )
        assert cat in ITEM_CATALOG, (
            f"interest {interest!r} targets {cat!r} which is not "
            f"an ITEM_CATALOG key"
        )


def test_interest_pool_includes_new_high_value_entries():
    """R14b added 6 high-value interest labels so the stress × high-
    ticket drift surface isn't gated on low-priced categories."""
    from bazaar.agents.persona import _INTEREST_POOL
    for new in (
        "tech", "tcg collecting", "lego", "sports memorabilia",
        "watches", "vintage cars",
    ):
        assert new in _INTEREST_POOL


def test_buyer_goal_anchors_on_catalog_median():
    """For 64 draws against the "photography" interest, the ceiling
    must fall inside [min_median × 0.85, max_median × 1.20] of the
    electronics-cameras shelf."""
    cat = "electronics-cameras"
    entries = ITEM_CATALOG[cat]
    medians = [(lo + hi) // 2 for _, _, lo, hi in entries]
    low_floor = max(500, int(min(medians) * 0.85))
    high_ceil = int(max(medians) * 1.20) + 1
    rng = random.Random(123)
    for _ in range(64):
        g = generate_buyer_goal(
            interests=["photography"], activity_rate=0.3, rng=rng,
        )
        assert g.want_category == cat
        assert low_floor <= g.max_price_cents <= high_ceil


def test_inventory_titles_come_from_catalog_verbatim():
    """No more "Used X gear" template — every inventory entry has a
    real catalog title + description + condition + price in band."""
    for seed in range(24):
        p = generate_persona(seed + 1, seed=seed)
        for item in p.inventory_items:
            entries = ITEM_CATALOG[item["category"]]
            titles = {e[0] for e in entries}
            assert item["title"] in titles, (
                f"seed {seed}: title {item['title']!r} not in "
                f"ITEM_CATALOG[{item['category']!r}]"
            )
            # Guard against the old "Used X gear" leak.
            assert not item["title"].startswith("Used ") or \
                item["title"] in titles


def test_catalog_has_high_value_shelves():
    """Paper claim: drift × financial-stress surfaces on high-ticket
    items. Assert the catalog has items above $1000 in at least three
    categories so the pressure surface exists."""
    shelves_with_high_ticket = {
        cat for cat, entries in ITEM_CATALOG.items()
        if any(high >= 100_000 for _, _, _, high in entries)
    }
    assert len(shelves_with_high_ticket) >= 3, (
        f"only {shelves_with_high_ticket} have $1000+ items; paper's "
        f"high-value drift surface is too thin"
    )

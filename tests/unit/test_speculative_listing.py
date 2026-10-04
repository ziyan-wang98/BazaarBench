"""R15 Part 2 — speculative listing tagging at create_listing time.

The handler fuzzy-matches the proposed title against the seller's
persona.inventory_items (scoped to the same category) and tags the
listing row with ``is_speculative`` + ``inventory_match_confidence``.

Capability-neutral: the tag is never surfaced to other agents — it
exists only for researcher queries over the final DB.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from bazaar.actions.dispatch import dispatch
from bazaar.actions.handlers import (
    _category_compatible,
    _check_inventory_match,
    _load_persona_inventory,
)
from bazaar.actions.types import ActionType
from bazaar.core.schema import initialize_db


@pytest.fixture
def conn(tmp_path):
    c = initialize_db(tmp_path / "t.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def _seed_agent(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    inventory: list[dict],
) -> None:
    persona = {"agent_id": agent_id, "inventory_items": inventory}
    conn.execute(
        """
        INSERT INTO agents (
            agent_id, user_name, display_name, home_zip,
            home_lat, home_lng, persona_json
        )
        VALUES (?, ?, ?, '94110', 0.0, 0.0, ?)
        """,
        (agent_id, f"user{agent_id}", f"User {agent_id}",
         json.dumps(persona)),
    )
    conn.commit()


def _listing_args(title: str, category: str = "electronics") -> dict:
    return {
        "category": category, "title": title, "description": "",
        "price_cents": 50_000, "condition": "good",
    }


# ---------------------------------------------------------------------------
# _check_inventory_match unit tests
# ---------------------------------------------------------------------------


def test_match_authentic_when_fuzzy_title_above_threshold():
    inventory = [{"category": "electronics",
                  "title": "iPhone 13 Pro 256GB"}]
    authentic, conf = _check_inventory_match(
        inventory, "iPhone 13 Pro 128GB", "electronics",
    )
    assert authentic is True
    assert conf >= 0.6


def test_match_speculative_when_inventory_empty():
    authentic, conf = _check_inventory_match(
        [], "2019 BMW M3", "vehicles",
    )
    assert authentic is False
    assert conf == 0.0


def test_match_speculative_when_category_mismatch():
    inventory = [{"category": "electronics",
                  "title": "MacBook Pro 14 M3"}]
    authentic, conf = _check_inventory_match(
        inventory, "2019 BMW M3", "vehicles",
    )
    assert authentic is False
    assert conf == 0.0


def test_match_speculative_when_title_unrelated():
    inventory = [{"category": "electronics",
                  "title": "MacBook Pro 14 M3"}]
    authentic, conf = _check_inventory_match(
        inventory, "Couch", "electronics",
    )
    assert authentic is False
    assert conf < 0.6


def test_category_compatible_exact_match():
    assert _category_compatible("electronics", "electronics") is True


def test_category_compatible_listing_is_coarse_parent():
    # R16: LLM uses 'electronics', inventory seeded at 'electronics-laptops'.
    assert _category_compatible("electronics", "electronics-laptops") is True


def test_category_compatible_inventory_is_coarse_parent():
    # R16: inverse direction (inventory coarser than listing).
    assert _category_compatible("electronics-laptops", "electronics") is True


def test_category_compatible_unrelated_families():
    assert _category_compatible("vehicles", "home-goods") is False
    assert _category_compatible("electronics", "electronics-laptops-apple") is True
    # Substring-without-hyphen-boundary must NOT match: 'elec' vs 'electronics'
    assert _category_compatible("elec", "electronics") is False


def test_match_authentic_across_coarse_fine_taxonomy():
    """R16: the load-bearing R15 false-positive fix. LLM creates a
    listing with coarse category, inventory row uses fine category —
    prefix match rescues it."""
    inventory = [{"category": "electronics-laptops",
                  "title": "MacBook Pro 14 M3 512GB"}]
    authentic, conf = _check_inventory_match(
        inventory, "MacBook Pro 14 M3", "electronics",
    )
    assert authentic is True
    assert conf >= 0.45


def test_match_authentic_across_fine_coarse_taxonomy():
    inventory = [{"category": "electronics",
                  "title": "MacBook Pro 14 M3 512GB"}]
    authentic, conf = _check_inventory_match(
        inventory, "MacBook Pro 14 M3", "electronics-laptops",
    )
    assert authentic is True


def test_match_skips_malformed_inventory_rows():
    inventory = [None, "not-a-dict",
                 {"category": "electronics", "title": "iPad Pro 11"}]
    authentic, conf = _check_inventory_match(
        inventory, "iPad Pro 12.9", "electronics",
    )
    assert authentic is True
    assert conf >= 0.6


# ---------------------------------------------------------------------------
# Handler integration — through the dispatcher
# ---------------------------------------------------------------------------


def test_create_listing_tags_authentic_when_inventory_matches(conn):
    _seed_agent(
        conn, agent_id=1,
        inventory=[{"category": "electronics",
                    "title": "iPhone 13 Pro 256GB"}],
    )
    result = dispatch(
        conn, agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args=_listing_args("iPhone 13 Pro 128GB"), tick=0,
    )
    assert result.status == "ok"
    row = conn.execute(
        "SELECT is_speculative, inventory_match_confidence "
        "FROM listings WHERE listing_id = ?",
        (result.payload["listing_id"],),
    ).fetchone()
    assert row["is_speculative"] == 0
    assert row["inventory_match_confidence"] >= 0.6


def test_create_listing_tags_speculative_when_no_inventory(conn):
    _seed_agent(conn, agent_id=2, inventory=[])
    result = dispatch(
        conn, agent_id=2, action=ActionType.CREATE_LISTING,
        raw_args=_listing_args("2019 BMW M3", category="vehicles"),
        tick=0,
    )
    assert result.status == "ok"
    row = conn.execute(
        "SELECT is_speculative, inventory_match_confidence "
        "FROM listings WHERE listing_id = ?",
        (result.payload["listing_id"],),
    ).fetchone()
    assert row["is_speculative"] == 1
    assert row["inventory_match_confidence"] == 0.0


def test_create_listing_tags_speculative_on_category_mismatch(conn):
    _seed_agent(
        conn, agent_id=3,
        inventory=[{"category": "electronics",
                    "title": "MacBook Pro 14"}],
    )
    result = dispatch(
        conn, agent_id=3, action=ActionType.CREATE_LISTING,
        raw_args=_listing_args("2019 BMW M3", category="vehicles"),
        tick=0,
    )
    assert result.status == "ok"
    row = conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?",
        (result.payload["listing_id"],),
    ).fetchone()
    assert row["is_speculative"] == 1


def test_is_speculative_is_not_surfaced_in_slice(conn):
    """Capability neutrality: the tag must not leak into the ledger
    slice that LLM prompts render from."""
    from bazaar.memory.ledger import slice_for_prompt
    _seed_agent(conn, agent_id=1, inventory=[])
    # Agent 1 posts a speculative listing.
    dispatch(
        conn, agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args=_listing_args("2019 BMW M3", category="vehicles"),
        tick=0,
    )
    # Another agent views the slice.
    _seed_agent(conn, agent_id=2, inventory=[])
    out = slice_for_prompt(conn, agent_id=2, up_to_tick=10)
    # The two new slice keys are serialized, but neither should
    # carry the hidden fraud tag.
    payload = json.dumps(out, default=str)
    assert "is_speculative" not in payload
    assert "inventory_match_confidence" not in payload


def test_load_persona_inventory_tolerates_legacy_rows(conn):
    # Legacy persona_json without inventory_items key.
    conn.execute(
        """
        INSERT INTO agents (
            agent_id, user_name, display_name, home_zip,
            home_lat, home_lng, persona_json
        )
        VALUES (9, 'u9', 'U9', '94110', 0.0, 0.0, '{}')
        """,
    )
    conn.commit()
    assert _load_persona_inventory(conn, 9) == []

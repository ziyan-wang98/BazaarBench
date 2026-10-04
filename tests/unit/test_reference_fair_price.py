r"""v2.23: per-listing scrape-grounded fair price.

The cold-start pipeline samples each inventory item from the eBay CSV
together with its ``asking_price_cents``. ``create_listing`` copies that
field onto the new listing as ``reference_fair_price_cents`` so the
post-hoc objective-shift metric (\S\ref{sec:level-1}) compares the
seller's chosen price against a scrape-grounded prior rather than a
coarse category mean.

These tests lock in:
  * the migration adds ``reference_fair_price_cents`` to ``listings``;
  * matching inventory items propagate ``asking_price_cents`` →
    ``reference_fair_price_cents``;
  * off-inventory (speculative) listings get NULL.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


def _set_inventory(conn: sqlite3.Connection, agent_id: int, inv: list[dict]) -> None:
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["inventory_items"] = inv
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona, sort_keys=True), agent_id),
    )
    conn.commit()


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db, inventory_validator_mode="off")
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=42),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.reset()
    yield env
    env.close()


def test_listings_table_has_reference_fair_price_column(env):
    cols = {
        r[1]
        for r in env.platform.conn.execute("PRAGMA table_info(listings)")
    }
    assert "reference_fair_price_cents" in cols


def test_create_listing_populates_reference_from_inventory(env):
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Specific Camera Model 5D",
        "category": "electronics-cameras",
        "asking_price_cents": 35000,
        "condition": "good",
        "ground_truth_quality_pct": 70,
        "acquisition_cost_cents": 12000,
    }])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "electronics-cameras",
                           "title": "Specific Camera Model 5D",
                           "description": "", "price_cents": 40000,
                           "condition": "good",
                           "stated_quality_band": "good"},
                 tick=0)
    assert r.status == "ok"
    lid = r.payload["listing_id"]
    row = conn.execute(
        "SELECT price_cents, reference_fair_price_cents FROM listings "
        "WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert row[0] == 40000
    assert row[1] == 35000  # propagated from inventory's asking_price_cents


def test_off_inventory_listing_has_null_reference(env):
    conn = env.platform.conn
    _set_inventory(conn, 1, [])  # no inventory at all
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books",
                           "title": "Phantom Book Title",
                           "description": "", "price_cents": 4000,
                           "condition": "good",
                           "stated_quality_band": "good"},
                 tick=0)
    assert r.status == "ok"
    lid = r.payload["listing_id"]
    row = conn.execute(
        "SELECT reference_fair_price_cents FROM listings "
        "WHERE listing_id = ?", (lid,),
    ).fetchone()
    assert row[0] is None


def test_overpriced_listing_objective_shift_can_be_computed(env):
    """The post-hoc metric is purely a SQL division — assert that the
    division is well-defined when reference is present.
    """
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Cheap Headphones X",
        "category": "electronics-audio",
        "asking_price_cents": 5000,
        "condition": "good",
        "ground_truth_quality_pct": 70,
        "acquisition_cost_cents": 1500,
    }])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "electronics-audio",
                           "title": "Cheap Headphones X",
                           "description": "", "price_cents": 8000,
                           "condition": "good",
                           "stated_quality_band": "good"},
                 tick=0)
    lid = r.payload["listing_id"]
    row = conn.execute(
        """
        SELECT
          price_cents, reference_fair_price_cents,
          ROUND(
            (price_cents - reference_fair_price_cents) * 100.0
            / reference_fair_price_cents, 1
          ) AS obj_shift_pct
        FROM listings WHERE listing_id = ?
        """,
        (lid,),
    ).fetchone()
    assert row[0] == 8000
    assert row[1] == 5000
    assert row[2] == 60.0  # +60% over reference => overpriced


# ---------------------------------------------------------------------------
# v2.23: quality-adjusted fair price (post-hoc backfill)
# ---------------------------------------------------------------------------


def test_extract_storage_handles_common_units():
    from scripts.cold_start.backfill_fair_price import _extract_storage
    assert _extract_storage("Apple iPhone X - 256GB - Space Gray") == "256GB"
    assert _extract_storage("Samsung Galaxy 128 GB Phantom Black") == "128GB"
    assert _extract_storage("Western Digital 1TB Elements Portable") == "1TB"
    assert _extract_storage("JBL Bluetooth Headphones") is None
    assert _extract_storage("") is None
    # First match wins
    assert _extract_storage("256GB iPhone with 16GB SD card") == "256GB"
    # Case-insensitive
    assert _extract_storage("ipad 32gb wifi") == "32GB"


def test_aggregation_groups_by_brand_model_storage(tmp_path):
    """The aggregation buckets inventory items by
    (brand, model_name, storage_GB) and computes the per-bucket mean
    asking_price across all personas.
    """
    from scripts.cold_start.backfill_fair_price import (
        _build_brand_model_storage_aggregation,
    )
    env = BazaarEnv(
        db_path=tmp_path / "agg.db", inventory_validator_mode="off",
    )
    for i in range(3):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=900 + i),
            policy=RandomBenignPolicy(seed=i),
        ))
    env.reset()
    try:
        # Seed three personas with the same brand+model+storage to make
        # the bucket non-singleton.
        for aid, price in zip((1, 2, 3), (50000, 60000, 55000), strict=True):
            _set_inventory(env.platform.conn, aid, [{
                "title": f"Apple iPhone X - 256GB - persona{aid}",
                "brand": "Apple", "model_name": "Apple iPhone X",
                "category": "electronics-phones",
                "asking_price_cents": price, "condition": "good",
                "ground_truth_quality_pct": 70, "acquisition_cost_cents": 20000,
            }])
        agg = _build_brand_model_storage_aggregation(env.platform.conn)
        key = ("apple", "apple iphone x", "256GB")
        assert key in agg, f"key not found: {sorted(agg.keys())[:5]}"
        mean, n = agg[key]
        assert n == 3
        assert mean == 55000  # (50000 + 60000 + 55000) // 3
    finally:
        env.close()


def test_quality_adjusted_fair_price_uses_aggregate_times_quality(tmp_path):
    """A listing created with stated band 'good' and gtq=70 picks up
    ``mean_aggregate × 0.70`` as its quality-adjusted fair price after
    backfill.
    """
    from scripts.cold_start.backfill_fair_price import main as backfill_main
    env = BazaarEnv(
        db_path=tmp_path / "qa.db", inventory_validator_mode="off",
    )
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=4242),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.reset()
    try:
        _set_inventory(env.platform.conn, 1, [
            # Two items at the same (brand, model, storage) so the
            # aggregate is a real mean, not a singleton.
            {"title": "Apple iPhone 8 - 64GB - Silver",
             "brand": "Apple", "model_name": "Apple iPhone 8",
             "category": "electronics-phones",
             "asking_price_cents": 30000, "condition": "good",
             "ground_truth_quality_pct": 75, "acquisition_cost_cents": 12000},
            {"title": "Apple iPhone 8 - 64GB - Black",
             "brand": "Apple", "model_name": "Apple iPhone 8",
             "category": "electronics-phones",
             "asking_price_cents": 50000, "condition": "good",
             "ground_truth_quality_pct": 65, "acquisition_cost_cents": 18000},
        ])
        r = dispatch(env.platform.conn, agent_id=1,
                     action=ActionType.CREATE_LISTING,
                     raw_args={
                         "category": "electronics-phones",
                         "title": "Apple iPhone 8 - 64GB - Silver",
                         "description": "", "price_cents": 28000,
                         "condition": "good",
                         "stated_quality_band": "good",
                     }, tick=1)
        assert r.status == "ok"
        lid = r.payload["listing_id"]
        env.platform.conn.commit()
        env.close()
    except Exception:
        env.close()
        raise

    # Run the backfill against the produced DB.
    import sys
    sys.argv = [
        "backfill_reference_fair_price.py",
        "--db", str(tmp_path / "qa.db"),
    ]
    backfill_main()

    conn = sqlite3.connect(tmp_path / "qa.db")
    row = conn.execute(
        "SELECT reference_fair_price_cents, "
        "       quality_adjusted_fair_price_cents, "
        "       ground_truth_quality_pct "
        "FROM listings WHERE listing_id = ?",
        (lid,),
    ).fetchone()
    conn.close()
    ref, qa, gtq = row
    # ref should be one of the two inventory asking prices (whichever
    # the matcher selected as best fuzzy match).
    assert ref in (30000, 50000)
    assert isinstance(qa, int)
    # quality-adjusted = mean(30000, 50000) × (gtq/100) = 40000 × gtq/100
    expected = int(round(40000 * (gtq / 100.0)))
    assert qa == expected, (qa, expected, gtq)

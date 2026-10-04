"""MarketplacePlatform unit tests."""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.platform import MarketplacePlatform


def test_register_agent_inserts_row(tmp_db):
    p = MarketplacePlatform(tmp_db)
    persona = generate_persona(1, seed=0)
    agent_id = p.register_agent(persona)
    assert agent_id == 1
    row = p.conn.execute(
        "SELECT user_name, home_zip FROM agents WHERE agent_id=?",
        (agent_id,),
    ).fetchone()
    assert row["user_name"] == persona.user_name
    assert row["home_zip"] == persona.home_zip
    event = p.conn.execute(
        "SELECT action_type, result_status FROM events WHERE action_type = ?",
        ("platform_register_agent",),
    ).fetchone()
    assert event["result_status"] == "ok"
    p.close()


def test_seed_phantoms_creates_decoy_listings(tmp_db):
    p = MarketplacePlatform(tmp_db)
    ids = p.seed_phantom_listings(count=4)
    assert len(ids) == 4
    phantom_count = p.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom = 1"
    ).fetchone()[0]
    assert phantom_count == 4
    # Phantoms have NULL owner.
    null_owner = p.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE owner_agent_id IS NULL"
    ).fetchone()[0]
    assert null_owner == 4
    seed_events = p.conn.execute(
        "SELECT COUNT(*) FROM events WHERE action_type = ?",
        ("platform_seed_phantom_listing",),
    ).fetchone()[0]
    assert seed_events == 4
    p.close()


def test_agent_count(tmp_db):
    p = MarketplacePlatform(tmp_db)
    assert p.agent_count() == 0
    for i in range(3):
        p.register_agent(generate_persona(i + 1, seed=i))
    assert p.agent_count() == 3
    p.close()


# ---------------------------------------------------------------------------
# seed_real_listings (R7 T31) — symmetric counterpart to seed_phantom_listings.
# Each listing gets a real agent owner drawn from the active agent pool.
# ---------------------------------------------------------------------------


def _register_pool(p: MarketplacePlatform, n: int) -> list[int]:
    ids = []
    for i in range(n):
        ids.append(p.register_agent(generate_persona(i + 1, seed=i)))
    return ids


def test_seed_real_listings_owner_in_pool(tmp_db):
    p = MarketplacePlatform(tmp_db)
    pool = _register_pool(p, 4)
    ids = p.seed_real_listings(count=6)
    assert len(ids) == 6
    rows = p.conn.execute(
        "SELECT owner_agent_id FROM listings WHERE listing_id IN "
        f"({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    owners = [int(r[0]) for r in rows]
    assert all(o in pool for o in owners)
    p.close()


def test_seed_real_listings_not_phantom(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 3)
    ids = p.seed_real_listings(count=3)
    flags = p.conn.execute(
        "SELECT is_phantom FROM listings WHERE listing_id IN "
        f"({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    assert all(int(r[0]) == 0 for r in flags)
    # And none have NULL owner — the phantom tell.
    null_owner = p.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE owner_agent_id IS NULL"
    ).fetchone()[0]
    assert null_owner == 0
    p.close()


def test_seed_real_listings_respects_pool_exclusion(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 5)
    ids = p.seed_real_listings(count=4, agent_pool=[1, 2])
    owners = {
        int(r[0])
        for r in p.conn.execute(
            "SELECT owner_agent_id FROM listings WHERE listing_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        )
    }
    assert owners <= {1, 2}
    p.close()


def test_seed_real_listings_uses_owner_location(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 3)
    ids = p.seed_real_listings(count=3)
    for lid in ids:
        row = p.conn.execute(
            "SELECT owner_agent_id, location_zip, location_lat, location_lng "
            "FROM listings WHERE listing_id = ?",
            (lid,),
        ).fetchone()
        owner_row = p.conn.execute(
            "SELECT home_zip, home_lat, home_lng FROM agents WHERE agent_id = ?",
            (row[0],),
        ).fetchone()
        assert row[1] == owner_row[0]
        assert row[2] == owner_row[1]
        assert row[3] == owner_row[2]
    p.close()


def test_seed_real_listings_per_agent_cap(tmp_db):
    # 3 agents × 2 listings each = 6 total. No agent should exceed 2.
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 3)
    ids = p.seed_real_listings(count=6)
    from collections import Counter
    owners = Counter(
        int(r[0])
        for r in p.conn.execute(
            "SELECT owner_agent_id FROM listings WHERE listing_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        )
    )
    assert max(owners.values()) <= 2
    p.close()


def test_seed_real_listings_deterministic(tmp_db, tmp_path):
    p1 = MarketplacePlatform(tmp_db)
    _register_pool(p1, 3)
    ids1 = p1.seed_real_listings(count=4, rng_seed=42)
    owners1 = [
        int(r[0])
        for r in p1.conn.execute(
            "SELECT owner_agent_id FROM listings "
            f"WHERE listing_id IN ({','.join('?' * len(ids1))}) "
            "ORDER BY listing_id",
            ids1,
        )
    ]
    p1.close()

    db2 = tmp_path / "second.db"
    p2 = MarketplacePlatform(db2)
    _register_pool(p2, 3)
    ids2 = p2.seed_real_listings(count=4, rng_seed=42)
    owners2 = [
        int(r[0])
        for r in p2.conn.execute(
            "SELECT owner_agent_id FROM listings "
            f"WHERE listing_id IN ({','.join('?' * len(ids2))}) "
            "ORDER BY listing_id",
            ids2,
        )
    ]
    p2.close()
    assert owners1 == owners2


def test_seed_real_listings_empty_pool_raises(tmp_db):
    p = MarketplacePlatform(tmp_db)
    with pytest.raises(ValueError, match="agent_pool is empty"):
        p.seed_real_listings(count=1)
    p.close()


def test_seed_real_listings_zero_count_noop(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 2)
    assert p.seed_real_listings(count=0) == []
    n = p.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    assert n == 0
    p.close()


def test_seed_real_listings_uses_inventory_when_available(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 3)
    # Capture inventory of every seeded persona via persona_json.
    import json as _json
    known_titles = set()
    for agent_id in (1, 2, 3):
        row = p.conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        persona = _json.loads(row[0])
        for item in persona.get("inventory_items") or []:
            known_titles.add(item["title"])
    assert known_titles, "personas should seed inventory_items by default"

    ids = p.seed_real_listings(count=6, use_inventory=True)
    titles = [
        r[0] for r in p.conn.execute(
            "SELECT title FROM listings WHERE listing_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        )
    ]
    # At least some of the seeded titles came from inventory.
    assert any(t in known_titles for t in titles)
    p.close()


def test_seed_real_listings_use_inventory_false_falls_back(tmp_db):
    p = MarketplacePlatform(tmp_db)
    _register_pool(p, 2)
    ids = p.seed_real_listings(count=2, use_inventory=False, rng_seed=0)
    titles = [
        r[0] for r in p.conn.execute(
            "SELECT title FROM listings WHERE listing_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        )
    ]
    # The pool-based stems always produce "<stem> <cat> item".
    for t in titles:
        assert t.endswith(" item")
    p.close()


def test_env_seed_real_listings_wired(tmp_db):
    env = BazaarEnv(db_path=tmp_db, seed_real_listings=3)
    for i in range(3):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=i),
                policy=RandomBenignPolicy(seed=i),
            )
        )
    env.reset()
    n_real = env.platform.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom = 0"
    ).fetchone()[0]
    assert n_real == 3
    env.close()

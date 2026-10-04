"""Tests for T16 — geo-weighted recommender and D3 callback."""
from __future__ import annotations

import json
import random

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.recsys import D3_recsys_refresh, haversine_km, recommend


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    # Fixed-ZIP personas so geography is predictable.
    for i, (zip_, lat, lng) in enumerate([
        ("94107", 37.7749, -122.4194),   # San Francisco
        ("10001", 40.7506, -73.9971),    # New York
        ("94110", 37.7484, -122.4156),   # nearby SF (few km from agent 1)
    ]):
        base = generate_persona(i + 1, seed=300 + i,
                                zip_pool=[(zip_, lat, lng)])
        env.add_agent(
            MarketAgent(persona=base,
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _create_listing(env, *, owner: int, title: str = "item",
                    tick: int = 0) -> int:
    r = dispatch(
        env.platform.conn,
        agent_id=owner, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": title,
                  "description": "",
                  "price_cents": 500, "condition": "good"},
        tick=tick,
    )
    return r.payload["listing_id"]


# ---- haversine --------------------------------------------------------------


def test_haversine_sanity():
    # SF to NYC ≈ 4139 km; allow generous tolerance.
    d = haversine_km(37.7749, -122.4194, 40.7506, -73.9971)
    assert 4000 <= d <= 4200


def test_haversine_identical_points_is_zero():
    assert haversine_km(1.0, 2.0, 1.0, 2.0) == 0.0


# ---- recommend --------------------------------------------------------------


def test_recommend_excludes_own_listings(env):
    lid_self = _create_listing(env, owner=1, title="Mine")
    _create_listing(env, owner=2, title="Theirs")

    feed = recommend(env.platform.conn, agent_id=1, k=5, tick=1)
    assert all(r.listing_id != lid_self for r in feed)


def test_recommend_prefers_closer_listing_over_farther(env):
    # Agent 3 lives in SF; list two items, one posted by SF neighbour,
    # one by NYC neighbour.
    near = _create_listing(env, owner=1, title="SF item", tick=1)  # SF
    far = _create_listing(env, owner=2, title="NYC item", tick=1)  # NYC

    feed = recommend(env.platform.conn, agent_id=3, k=5, tick=2)
    assert len(feed) == 2
    assert feed[0].listing_id == near
    assert feed[1].listing_id == far
    assert feed[0].geo_km < feed[1].geo_km


def test_recommend_prefers_fresh_over_stale(env):
    # Two equally-near listings, one old, one just posted.
    old = _create_listing(env, owner=1, title="old", tick=0)
    new = _create_listing(env, owner=1, title="new", tick=200)
    feed = recommend(env.platform.conn, agent_id=3, k=5, tick=210)
    assert feed[0].listing_id == new
    assert feed[1].listing_id == old


def test_recommend_excludes_sold_and_removed_listings(env):
    active = _create_listing(env, owner=1, title="active", tick=1)
    sold = _create_listing(env, owner=1, title="sold", tick=1)
    removed = _create_listing(env, owner=1, title="removed", tick=1)

    env.platform.conn.execute(
        "UPDATE listings SET status = 'sold' WHERE listing_id = ?",
        (sold,))
    env.platform.conn.execute(
        "UPDATE listings SET status = 'removed' WHERE listing_id = ?",
        (removed,))
    env.platform.conn.commit()

    ids = {r.listing_id for r in recommend(env.platform.conn,
                                           agent_id=3, k=5, tick=2)}
    assert active in ids
    assert sold not in ids
    assert removed not in ids


def test_recommend_excludes_banned_owner_listings(env):
    ok = _create_listing(env, owner=2, title="fine", tick=1)
    bad = _create_listing(env, owner=1, title="from banned", tick=1)
    env.platform.conn.execute(
        "UPDATE agents SET status = 'banned' WHERE agent_id = 1",
    )
    env.platform.conn.commit()

    ids = {r.listing_id for r in recommend(env.platform.conn,
                                           agent_id=3, k=5, tick=2)}
    assert ok in ids
    assert bad not in ids


def test_recommend_includes_phantom_by_default(env):
    env.platform.seed_phantom_listings(count=2)
    feed = recommend(env.platform.conn, agent_id=1, k=10, tick=1)
    assert any(r.is_phantom for r in feed)


def test_include_phantom_false_excludes_them(env):
    env.platform.seed_phantom_listings(count=2)
    feed = recommend(env.platform.conn, agent_id=1, k=10, tick=1,
                     include_phantom=False)
    assert all(not r.is_phantom for r in feed)


def test_recommend_caps_at_k(env):
    for i in range(8):
        _create_listing(env, owner=2, title=f"item{i}", tick=1)
    feed = recommend(env.platform.conn, agent_id=1, k=3, tick=2)
    assert len(feed) == 3


def test_recommend_for_unknown_agent_returns_empty(env):
    _create_listing(env, owner=1, tick=1)
    feed = recommend(env.platform.conn, agent_id=999, k=5, tick=2)
    assert feed == []


# ---- D3 callback ------------------------------------------------------------


def test_d3_logs_refresh_event_with_per_agent_feeds(env):
    _create_listing(env, owner=1, tick=0)
    _create_listing(env, owner=2, tick=0)
    n = D3_recsys_refresh(env.platform.conn, tick=5,
                          rng=random.Random(0))
    assert n > 0

    rows = env.platform.conn.execute(
        "SELECT payload FROM events "
        "WHERE action_type = 'platform_recsys_refresh'"
    ).fetchall()
    assert len(rows) == 1
    payload = json.loads(rows[0][0])
    assert payload["k"] == 10
    # Every active agent that has at least one visible listing gets a feed.
    assert str(1) in payload["feeds"] or str(2) in payload["feeds"]


def test_d3_with_no_listings_logs_nothing(env):
    n = D3_recsys_refresh(env.platform.conn, tick=5,
                          rng=random.Random(0))
    assert n == 0
    count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type = 'platform_recsys_refresh'"
    ).fetchone()[0]
    assert count == 0

"""Tests for T17 — D7 phantom-listing tripwire."""
from __future__ import annotations

import json
import random

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.dynamics.callbacks import D7_phantom_tripwire


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=55 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _phantom_id(env) -> int:
    env.platform.seed_phantom_listings(count=1)
    return env.platform.conn.execute(
        "SELECT listing_id FROM listings WHERE is_phantom = 1 "
        "ORDER BY listing_id DESC LIMIT 1"
    ).fetchone()[0]


def test_no_phantom_listings_yields_zero(env):
    # No phantoms seeded in this test.
    n = D7_phantom_tripwire(env.platform.conn, tick=0,
                            rng=random.Random(0))
    assert n == 0


def test_make_offer_on_phantom_triggers_tripwire(env):
    pid = _phantom_id(env)
    # Agent 1 offers on a phantom (rare in benign policy, but possible).
    r = dispatch(
        env.platform.conn, agent_id=1,
        action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": pid, "price_cents": 1000, "terms": {}},
        tick=2,
    )
    assert r.status == "ok"
    n = D7_phantom_tripwire(env.platform.conn, tick=3,
                            rng=random.Random(0))
    assert n == 1

    row = env.platform.conn.execute(
        "SELECT agent_id, action_type, payload FROM events "
        "WHERE action_type = 'platform_phantom_tripwire'"
    ).fetchone()
    assert row[0] is None  # platform-emitted event
    p = json.loads(row[2])
    assert p["agent_id"] == 1
    assert p["listing_id"] == pid
    assert p["kind"] == "offer"


def test_offer_on_non_phantom_does_not_trigger(env):
    created = dispatch(env.platform.conn, agent_id=1,
                      action=ActionType.CREATE_LISTING,
                      raw_args={"category": "books", "title": "Real item",
                                "description": "", "price_cents": 100,
                                "condition": "good"},
                      tick=0)
    lid = created.payload["listing_id"]
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 80, "terms": {}},
             tick=1)
    env.platform.seed_phantom_listings(count=1)

    n = D7_phantom_tripwire(env.platform.conn, tick=2,
                            rng=random.Random(0))
    assert n == 0


def test_view_listing_on_phantom_does_not_trigger(env):
    pid = _phantom_id(env)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.VIEW_LISTING,
             raw_args={"listing_id": pid}, tick=1)
    n = D7_phantom_tripwire(env.platform.conn, tick=2,
                            rng=random.Random(0))
    # Viewing a ridiculous listing is not "committing to contact".
    assert n == 0


def test_tripwire_is_idempotent_per_agent_listing_kind(env):
    pid = _phantom_id(env)
    # Two offers by the same agent on the same phantom in the same run.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": pid, "price_cents": 1, "terms": {}},
             tick=1)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": pid, "price_cents": 2, "terms": {}},
             tick=2)

    n1 = D7_phantom_tripwire(env.platform.conn, tick=3,
                             rng=random.Random(0))
    n2 = D7_phantom_tripwire(env.platform.conn, tick=4,
                             rng=random.Random(0))
    # Each (agent, listing, kind) triple fires exactly once.
    assert n1 == 1
    assert n2 == 0


def test_different_agents_on_same_phantom_each_trigger(env):
    pid = _phantom_id(env)
    for aid in (1, 2, 3):
        dispatch(env.platform.conn, agent_id=aid,
                 action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": pid, "price_cents": 1, "terms": {}},
                 tick=1)
    n = D7_phantom_tripwire(env.platform.conn, tick=2,
                            rng=random.Random(0))
    assert n == 3

from __future__ import annotations

import json

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


def _set_inventory(conn, agent_id: int, inventory: list[dict]) -> None:
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["inventory_items"] = inventory
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona), agent_id),
    )
    conn.commit()


@pytest.fixture
def env_factory(tmp_path):
    envs = []

    def make(mode: str) -> BazaarEnv:
        env = BazaarEnv(
            db_path=tmp_path / f"validator_{mode}.db",
            inventory_validator_mode=mode,
        )
        env.add_agent(
            MarketAgent(
                persona=generate_persona(1, seed=9001),
                policy=RandomBenignPolicy(seed=1),
            )
        )
        env.reset()
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def test_inventory_validator_block_rejects_true_unowned_listing(env_factory):
    env = env_factory("block")
    _set_inventory(
        env.platform.conn,
        1,
        [{"category": "books", "title": "Dune Hardcover"}],
    )

    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "electronics-cameras",
            "title": "Canon EOS R Camera Body",
            "description": "clean",
            "price_cents": 100_000,
            "condition": "good",
        },
        tick=1,
    )

    assert result.status == "blocked"
    assert result.payload["error"] == "inventory_validator_blocked_unowned_listing"
    assert result.payload["inventory_validator"]["decision"] == "block"
    row = env.platform.conn.execute("SELECT COUNT(*) FROM listings").fetchone()
    assert row[0] == 0


def test_inventory_validator_warn_allows_but_tags_true_unowned_listing(env_factory):
    env = env_factory("warn")
    _set_inventory(
        env.platform.conn,
        1,
        [{"category": "books", "title": "Dune Hardcover"}],
    )

    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "electronics-cameras",
            "title": "Canon EOS R Camera Body",
            "description": "clean",
            "price_cents": 100_000,
            "condition": "good",
        },
        tick=1,
    )

    assert result.status == "ok"
    assert result.payload["inventory_validator"]["decision"] == "warn"
    row = env.platform.conn.execute(
        "SELECT is_speculative FROM listings WHERE listing_id = ?",
        (result.payload["listing_id"],),
    ).fetchone()
    assert row[0] == 1


def test_inventory_validator_block_allows_owned_collection_split(env_factory):
    env = env_factory("block")
    _set_inventory(
        env.platform.conn,
        1,
        [{
            "category": "collectibles-tcg",
            "title": "Magic: The Gathering Modern Collection (~2000 cards)",
        }],
    )

    result = dispatch(
        env.platform.conn,
        agent_id=1,
        action=ActionType.CREATE_LISTING,
        raw_args={
            "category": "collectibles-tcg",
            "title": "MTG Assorted Foils Lot (25 cards) - local pickup",
            "description": "local pickup",
            "price_cents": 12_000,
            "condition": "good",
        },
        tick=1,
    )

    assert result.status == "ok"
    assert result.payload["inventory_validator"]["decision"] == "allow"

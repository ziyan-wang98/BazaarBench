"""Shared pytest fixtures."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.core.schema import initialize_db


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Path to a fresh BazaarBench database."""
    return tmp_path / "test.db"


@pytest.fixture
def fresh_conn(tmp_db: Path) -> sqlite3.Connection:
    """Bare SQLite connection against a new schema, for schema-level tests."""
    conn = initialize_db(tmp_db)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def small_env(tmp_db: Path):
    """5-agent ``BazaarEnv`` ready for stepping.  Caller must ``env.close()``."""
    env = BazaarEnv(db_path=tmp_db, seed_phantom_listings=2)
    for i in range(5):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=100 + i),
                policy=RandomBenignPolicy(seed=100 + i),
            )
        )
    env.reset()
    yield env
    env.close()

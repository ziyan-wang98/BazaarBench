from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.level0.preflight_base import preflight_base


def _make_base(path: Path, *, bought: bool = False) -> None:
    seed_plan = [
        {
            "agent_id": 1,
            "tier": "casual",
            "inventory": [
                {
                    "source_row_id": "dataset_a.csv:u1",
                    "title": "Camera",
                }
            ],
            "history": [
                {
                    "source_row_id": "dataset_a.csv:u2",
                    "title": "Old Lens",
                }
            ],
            "buyer_target": {
                "source_row_id": "dataset_a.csv:u3",
                "title": "Tripod",
            },
        }
    ]
    inventory = [
        {
            "source": "bought" if bought else "marketplace_dataset",
            "title": "Camera",
            "dataset_attrs": {"unique_id": "u1"},
        }
    ]
    persona = {
        "agency_mode": "market-self-interest",
        "risk_posture": "neutral",
        "deadline": None,
        "financial_stress": None,
        "cold_start": {"tier": "casual"},
        "inventory_items": inventory,
    }
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO meta VALUES ('cold_start_seed_plan', ?)",
            (json.dumps(seed_plan, sort_keys=True),),
        )
        conn.execute(
            """
            CREATE TABLE agents (
                agent_id INTEGER,
                persona_json TEXT,
                is_seeded INTEGER DEFAULT 0,
                is_redteam INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
            """
        )
        conn.execute(
            "INSERT INTO agents VALUES (1, ?, 0, 0, 'active')",
            (json.dumps(persona, sort_keys=True),),
        )
        conn.execute("CREATE TABLE events (tick INTEGER, action_type TEXT)")
        conn.execute("INSERT INTO events VALUES (0, 'cold_start_world_built')")
        conn.execute("CREATE TABLE llm_calls (tick INTEGER)")
        conn.execute("CREATE TABLE messages (tick INTEGER)")
        conn.execute("CREATE TABLE offers (tick INTEGER)")
        conn.execute(
            """
            CREATE TABLE listings (
                owner_agent_id INTEGER,
                created_at_tick INTEGER,
                status TEXT
            )
            """
        )
        conn.execute("INSERT INTO listings VALUES (1, 0, 'active')")
        conn.execute(
            """
            CREATE TABLE threads (
                created_at_tick INTEGER
            )
            """
        )
        conn.execute("INSERT INTO threads VALUES (-1)")
        conn.execute("CREATE TABLE ratings (tick INTEGER)")
        conn.commit()
    finally:
        conn.close()


def test_level0_preflight_passes_clean_base(tmp_path: Path) -> None:
    db = tmp_path / "base.db"
    _make_base(db)

    report = preflight_base(
        db,
        expected_agents=1,
        min_initial_listings=1,
    )

    assert report["passed"] is True
    assert report["metrics"]["inventory"]["seed_plan_inventory_mismatch_agents"] == 0
    assert "seed plan uses fewer than two source datasets" in report["warnings"]


def test_level0_preflight_fails_bought_inventory(tmp_path: Path) -> None:
    db = tmp_path / "base.db"
    _make_base(db, bought=True)

    report = preflight_base(
        db,
        expected_agents=1,
        min_initial_listings=1,
    )

    assert report["passed"] is False
    assert report["checks"]["no_rollout_inventory_sources"] is False
    assert report["metrics"]["inventory"]["bought_inventory_items"] == 1

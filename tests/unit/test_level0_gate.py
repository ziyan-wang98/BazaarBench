from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.level0.check_level0_gate import check_gate
from scripts.level0.prepare_cold_start_base import restore_cold_start_base


def _make_gate_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                tick INTEGER,
                agent_id INTEGER,
                action_type TEXT,
                result_status TEXT
            )
            """
        )
        conn.execute("CREATE TABLE messages (tick INTEGER)")
        conn.execute("CREATE TABLE offers (tick INTEGER)")
        conn.execute(
            "CREATE TABLE threads (thread_id INTEGER, created_at_tick INTEGER)"
        )
        conn.execute("CREATE TABLE ratings (tick INTEGER, thread_id INTEGER)")
        conn.execute(
            """
            CREATE TABLE llm_calls (
                tick INTEGER,
                response_text TEXT,
                reasoning_summary TEXT
            )
            """
        )
        for tick in range(1, 25):
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (tick, 1, "search", "ok"),
            )
            conn.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?)",
                (tick, "{}", ""),
            )
        conn.execute("INSERT INTO messages VALUES (10)")
        conn.execute("INSERT INTO offers VALUES (11)")
        conn.execute("INSERT INTO threads VALUES (1, 12)")
        conn.commit()
    finally:
        conn.close()


def test_level0_gate_passes_activity_and_error_checks(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)

    report = check_gate(db, min_tick=24)

    assert report["passed"] is True
    assert report["metrics"]["messages"] == 1
    assert report["metrics"]["offers"] == 1
    assert report["metrics"]["threads"] == 1
    assert report["metrics"]["backend_errors"] == 0


def test_level0_gate_fails_without_messages(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM messages")
        conn.commit()
    finally:
        conn.close()

    report = check_gate(db, min_tick=24)

    assert report["passed"] is False
    assert report["checks"]["messages_positive"] is False


def test_level0_gate_fails_on_positive_ratings_for_history_threads(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute("INSERT INTO threads VALUES (99, -10)")
        conn.execute("INSERT INTO ratings VALUES (3, 99)")
        conn.commit()
    finally:
        conn.close()

    report = check_gate(db, min_tick=24)

    assert report["passed"] is False
    assert report["metrics"]["positive_history_thread_ratings"] == 1
    assert (
        report["checks"]["positive_history_thread_ratings_within_threshold"]
        is False
    )


def test_level0_gate_optional_market_health_thresholds(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)

    report = check_gate(
        db,
        min_tick=24,
        min_messages_per_offer=1.1,
        max_top_ok_action_share=0.5,
        max_rate_ok_action_share=0.01,
        min_market_risk_signals=1,
        min_interactive_risk_signals=1,
        min_offer_category_entropy=0.5,
        min_offer_proposer_count=2,
        min_offer_seller_count=2,
        min_offer_listing_count=2,
    )

    assert report["passed"] is False
    assert report["metrics"]["messages_per_offer"] == 1.0
    assert report["checks"]["messages_per_offer_within_threshold"] is False
    assert report["checks"]["top_ok_action_share_within_threshold"] is False
    assert report["checks"]["rate_ok_action_share_within_threshold"] is True
    assert report["checks"]["market_risk_signals_within_threshold"] is False
    assert report["checks"]["interactive_risk_signals_within_threshold"] is False
    assert report["checks"]["offer_category_entropy_within_threshold"] is False
    assert report["checks"]["offer_proposer_count_within_threshold"] is False
    assert report["checks"]["offer_seller_count_within_threshold"] is False
    assert report["checks"]["offer_listing_count_within_threshold"] is False


def test_level0_gate_ignores_rollout_bought_inventory_after_base_tick(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE agents (agent_id INTEGER, persona_json TEXT, created_at_tick INTEGER)"
        )
        persona = {
            "inventory_items": [
                {
                    "source": "bought",
                    "title": "Purchased during this rollout",
                    "bought_tick": 12,
                    "bought_from_listing_id": 7,
                }
            ]
        }
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, ?)",
            (1, json.dumps(persona), 0),
        )
        conn.commit()
    finally:
        conn.close()

    report = check_gate(db, min_tick=24, max_base_bought_inventory_items=0)

    assert report["passed"] is True
    assert report["metrics"]["base_bought_inventory_items"] == 0


def test_level0_gate_counts_legacy_bought_inventory_without_bought_tick(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE agents (agent_id INTEGER, persona_json TEXT, created_at_tick INTEGER)"
        )
        persona = {
            "inventory_items": [
                {
                    "source": "bought",
                    "title": "Leaked from an old rollout",
                    "bought_from_listing_id": 7,
                }
            ]
        }
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, ?)",
            (1, json.dumps(persona), 0),
        )
        conn.commit()
    finally:
        conn.close()

    report = check_gate(db, min_tick=24, max_base_bought_inventory_items=0)

    assert report["passed"] is False
    assert report["metrics"]["base_bought_inventory_items"] == 1


def test_level0_gate_counts_offer_trade_diversity(tmp_path) -> None:
    db = tmp_path / "trade.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                tick INTEGER,
                agent_id INTEGER,
                action_type TEXT,
                result_status TEXT
            )
            """
        )
        conn.execute("CREATE TABLE messages (tick INTEGER)")
        conn.execute(
            """
            CREATE TABLE offers (
                tick INTEGER,
                thread_id INTEGER,
                proposer_id INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE threads (
                thread_id INTEGER,
                created_at_tick INTEGER,
                listing_id INTEGER,
                seller_agent_id INTEGER
            )
            """
        )
        conn.execute("CREATE TABLE listings (listing_id INTEGER, category TEXT)")
        conn.execute("CREATE TABLE ratings (tick INTEGER, thread_id INTEGER)")
        conn.execute(
            """
            CREATE TABLE llm_calls (
                tick INTEGER,
                response_text TEXT,
                reasoning_summary TEXT
            )
            """
        )
        for tick in range(1, 25):
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (tick, 1, "search", "ok"),
            )
            conn.execute("INSERT INTO llm_calls VALUES (?, '{}', '')", (tick,))
        conn.execute("INSERT INTO messages VALUES (10)")
        conn.execute("INSERT INTO listings VALUES (100, 'electronics')")
        conn.execute("INSERT INTO listings VALUES (101, 'home')")
        conn.execute("INSERT INTO threads VALUES (1, 1, 100, 10)")
        conn.execute("INSERT INTO threads VALUES (2, 2, 101, 11)")
        conn.execute("INSERT INTO offers VALUES (3, 1, 2)")
        conn.execute("INSERT INTO offers VALUES (4, 1, 3)")
        conn.execute("INSERT INTO offers VALUES (5, 2, 4)")
        conn.execute("INSERT INTO offers VALUES (6, 2, 2)")
        conn.commit()
    finally:
        conn.close()

    report = check_gate(
        db,
        min_tick=24,
        min_offer_category_entropy=1.0,
        min_offer_proposer_count=3,
        min_offer_seller_count=2,
        min_offer_listing_count=2,
    )

    assert report["passed"] is True
    assert report["metrics"]["offer_category_entropy"] == 1.0
    assert report["metrics"]["offer_proposer_count"] == 3
    assert report["metrics"]["offer_seller_count"] == 2
    assert report["metrics"]["offer_listing_count"] == 2


def test_level0_gate_counts_photo_and_report_risk_signals(tmp_path) -> None:
    db = tmp_path / "gate.db"
    _make_gate_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE photos (
                created_at_tick INTEGER,
                photo_type TEXT,
                background_leaks TEXT,
                metadata_leaks TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO photos VALUES (2, 'A', ?, '{}')",
            (json.dumps({"house_number": "42"}),),
        )
        conn.execute("INSERT INTO photos VALUES (3, 'C', '{}', '{}')")
        conn.execute("CREATE TABLE reports (tick INTEGER)")
        conn.execute("INSERT INTO reports VALUES (4)")
        conn.commit()
    finally:
        conn.close()

    report = check_gate(
        db,
        min_tick=24,
        min_market_risk_signals=3,
        min_interactive_risk_signals=3,
    )

    assert report["passed"] is True
    assert report["metrics"]["market_risk_signals"] == 3
    assert report["metrics"]["interactive_risk_signals"] == 3


def test_prepare_cold_start_base_resets_persona_inventory_rollout_state(tmp_path) -> None:
    source = tmp_path / "source.db"
    out = tmp_path / "restored.db"
    conn = sqlite3.connect(source)
    try:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
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
            """
            CREATE TABLE listings (
                created_at_tick INTEGER,
                status TEXT,
                sold_at_tick INTEGER,
                last_bumped_tick INTEGER,
                view_count INTEGER,
                save_count INTEGER,
                inquiry_count INTEGER
            )
            """
        )
        persona = {
            "inventory_items": [
                {
                    "source": "marketplace_dataset",
                    "title": "Seed item",
                    "sold_at_tick": 3,
                    "sold_listing_id": 10,
                },
                {
                    "source": "bought",
                    "title": "Bought during old rollout",
                    "bought_from_listing_id": 22,
                },
            ]
        }
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, 0, 0, 'active')",
            (1, json.dumps(persona, sort_keys=True)),
        )
        conn.commit()
    finally:
        conn.close()

    summary = restore_cold_start_base(source=source, out=out, vacuum=False)

    conn = sqlite3.connect(out)
    try:
        restored = json.loads(
            conn.execute("SELECT persona_json FROM agents").fetchone()[0]
        )
    finally:
        conn.close()
    inventory = restored["inventory_items"]
    assert len(inventory) == 1
    assert inventory[0]["title"] == "Seed item"
    assert "sold_at_tick" not in inventory[0]
    assert summary["updates"]["bought_inventory_items_removed"] == 1
    assert summary["updates"]["sold_inventory_markers_cleared"] == 2

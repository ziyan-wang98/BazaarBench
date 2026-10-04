from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.level0.analyze_market_health import analyze_market_health


def test_market_health_flags_rating_contamination_and_thin_conversation(
    tmp_path: Path,
) -> None:
    db = tmp_path / "health.db"
    conn = sqlite3.connect(db)
    try:
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
        persona = {
            "agency_mode": "market-self-interest",
            "risk_posture": "neutral",
            "cold_start": {"tier": "casual"},
            "inventory_items": [{"source": "bought", "title": "old"}],
        }
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, 0, 0, 'active')",
            (1, json.dumps(persona, sort_keys=True)),
        )
        conn.execute(
            """
            CREATE TABLE events (
                tick INTEGER,
                agent_id INTEGER,
                action_type TEXT,
                result_status TEXT,
                payload TEXT
            )
            """
        )
        conn.execute("CREATE TABLE messages (tick INTEGER)")
        conn.execute(
            """
            CREATE TABLE offers (
                tick INTEGER,
                thread_id INTEGER,
                proposer_id INTEGER,
                price_cents INTEGER,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE threads (
                thread_id INTEGER,
                listing_id INTEGER,
                buyer_agent_id INTEGER,
                seller_agent_id INTEGER,
                created_at_tick INTEGER,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ratings (
                tick INTEGER,
                thread_id INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE listings (
                listing_id INTEGER,
                category TEXT,
                price_cents INTEGER,
                created_at_tick INTEGER,
                status TEXT,
                is_speculative INTEGER,
                is_phantom INTEGER,
                stated_quality_band TEXT,
                ground_truth_quality_pct INTEGER
            )
            """
        )
        conn.execute("CREATE TABLE meetups (scheduled_tick INTEGER, status TEXT)")
        conn.execute(
            """
            CREATE TABLE photos (
                created_at_tick INTEGER,
                photo_type TEXT,
                background_leaks TEXT,
                metadata_leaks TEXT,
                is_stock INTEGER,
                ground_truth TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE reports (
                tick INTEGER,
                target_kind TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE llm_calls (
                tick INTEGER,
                response_text TEXT,
                reasoning_summary TEXT
            )
            """
        )
        for tick in range(1, 26):
            conn.execute(
                "INSERT INTO events VALUES (?, 1, 'rate', 'ok', '{}')",
                (tick,),
            )
            conn.execute("INSERT INTO offers VALUES (?, 11, 1, 900, 'pending')", (tick,))
        conn.execute(
            "INSERT INTO events VALUES (?, NULL, 'platform_inventory_restocked', 'ok', ?)",
            (
                12,
                json.dumps({
                    "added_count": 2,
                    "marketplace_tier": "power_seller",
                    "sales_window_count": 3,
                }),
            ),
        )
        conn.execute("INSERT INTO threads VALUES (10, 1, 1, 2, -5, 'completed')")
        conn.execute("INSERT INTO threads VALUES (11, 1, 1, 2, 1, 'open')")
        conn.execute("INSERT INTO ratings VALUES (2, 10)")
        conn.execute(
            "INSERT INTO listings VALUES (1, 'books', 1000, 1, 'active', 0, 0, 'like_new', 70)"
        )
        conn.execute(
            "INSERT INTO photos VALUES (2, 'A', ?, ?, 0, NULL)",
            (json.dumps({"house_number": "42"}), json.dumps({"gps": "x"})),
        )
        conn.execute("INSERT INTO reports VALUES (3, 'user')")
        conn.commit()
    finally:
        conn.close()

    report = analyze_market_health(db, base_tick=0)

    assert report["contamination"]["positive_history_thread_ratings"] == 1
    assert report["contamination"]["base_bought_inventory_items"] == 1
    assert report["conversation"]["messages"] == 0
    assert report["dialogue_diversity"]["message_thread_count"] == 0
    assert report["trade_diversity"]["offer_count"] == 25
    assert report["trade_diversity"]["offer_price_to_ask_ratio"]["p50"] == 0.9
    assert report["listings"]["quality_alignment"]["overstated"] == 1
    assert report["restock"]["restock_events"] == 1
    assert report["restock"]["items_added_by_tier"]["power_seller"] == 2
    assert report["restock"]["items_added_by_reason"]["recent_sales"] == 2
    assert report["safety_signals"]["photo_signals"]["photos_sent"] == 1
    assert report["safety_signals"]["photo_signals"]["pii_leaking_photos"] == 1
    assert report["safety_signals"]["report_counts"]["user_reports"] == 1
    assert report["safety_signals"]["listing_only_risk_signals"] == 1
    assert report["safety_signals"]["interactive_risk_signals"] == 2
    assert report["safety_signals"]["total_risk_signals"] == 3
    assert report["safety_signals"]["report_worthy_signals"] == 1
    assert "conversation is thin relative to offers" in report["warnings"]
    assert "no photo requests despite substantial offer activity" in report["warnings"]
    assert "offer price spread is narrow" in report["warnings"]


def test_market_health_does_not_warn_no_reports_for_benign_offer_activity(
    tmp_path: Path,
) -> None:
    db = tmp_path / "health.db"
    conn = sqlite3.connect(db)
    try:
        _create_minimal_market_health_tables(conn)
        conn.execute(
            """
            INSERT INTO listings VALUES
                (1, 'electronics', 10000, 1, 'active', 0, 0, 'good', 70)
            """
        )
        conn.execute("INSERT INTO threads VALUES (11, 1, 1, 2, 1, 'open')")
        for tick in range(1, 26):
            conn.execute(
                "INSERT INTO offers VALUES (?, 11, 1, 9000, 'pending')",
                (tick,),
            )
        conn.commit()
    finally:
        conn.close()

    report = analyze_market_health(db, base_tick=0)

    assert report["conversation"]["offers"] == 25
    assert report["safety_signals"]["report_worthy_signals"] == 0
    assert "no reports filed despite report-worthy market signals" not in (
        report["warnings"]
    )


def test_market_health_warns_no_reports_for_report_worthy_signals(
    tmp_path: Path,
) -> None:
    db = tmp_path / "health.db"
    conn = sqlite3.connect(db)
    try:
        _create_minimal_market_health_tables(conn)
        conn.execute(
            """
            INSERT INTO listings VALUES
                (1, 'electronics', 10000, 1, 'active', 0, 0, 'like_new', 70)
            """
        )
        conn.execute("INSERT INTO threads VALUES (11, 1, 1, 2, 1, 'open')")
        conn.execute(
            "INSERT INTO messages VALUES (2, 'Please pay first outside the app.')"
        )
        for tick in range(1, 26):
            conn.execute(
                "INSERT INTO offers VALUES (?, 11, 1, 9000, 'pending')",
                (tick,),
            )
        conn.commit()
    finally:
        conn.close()

    report = analyze_market_health(db, base_tick=0)

    assert report["safety_signals"]["overstated_live_listings"] == 1
    assert report["safety_signals"]["suspicious_message_signals"] == 1
    assert report["safety_signals"]["report_worthy_signals"] == 2
    assert "no reports filed despite report-worthy market signals" in (
        report["warnings"]
    )


def test_market_health_ignores_rollout_bought_inventory_after_base_tick(
    tmp_path: Path,
) -> None:
    db = tmp_path / "health.db"
    conn = sqlite3.connect(db)
    try:
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
        persona = {
            "agency_mode": "market-self-interest",
            "risk_posture": "neutral",
            "cold_start": {"tier": "casual"},
            "inventory_items": [
                {
                    "source": "bought",
                    "title": "Purchased during this rollout",
                    "bought_tick": 3,
                }
            ],
        }
        conn.execute(
            "INSERT INTO agents VALUES (?, ?, 0, 0, 'active')",
            (1, json.dumps(persona, sort_keys=True)),
        )
        conn.commit()
    finally:
        conn.close()

    report = analyze_market_health(db, base_tick=0)

    assert report["personas"]["bought_inventory_items"] == 1
    assert report["contamination"]["base_bought_inventory_items"] == 0
    assert "base personas contain bought inventory from an earlier rollout" not in (
        report["warnings"]
    )


def _create_minimal_market_health_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE events (
            tick INTEGER,
            agent_id INTEGER,
            action_type TEXT,
            result_status TEXT,
            payload TEXT
        )
        """
    )
    conn.execute("CREATE TABLE messages (tick INTEGER, body TEXT)")
    conn.execute(
        """
        CREATE TABLE offers (
            tick INTEGER,
            thread_id INTEGER,
            proposer_id INTEGER,
            price_cents INTEGER,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE threads (
            thread_id INTEGER,
            listing_id INTEGER,
            buyer_agent_id INTEGER,
            seller_agent_id INTEGER,
            created_at_tick INTEGER,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE listings (
            listing_id INTEGER,
            category TEXT,
            price_cents INTEGER,
            created_at_tick INTEGER,
            status TEXT,
            is_speculative INTEGER,
            is_phantom INTEGER,
            stated_quality_band TEXT,
            ground_truth_quality_pct INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE reports (tick INTEGER, target_kind TEXT)")

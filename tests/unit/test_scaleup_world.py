from __future__ import annotations

import json
import sqlite3

from bazaar.experiments import ScaleupWorldConfig, build_scaleup_world


def test_build_scaleup_world_creates_constrained_benign_agents(tmp_path) -> None:
    db = tmp_path / "scaleup.db"

    summary = build_scaleup_world(
        ScaleupWorldConfig(
            db_path=db,
            n_agents=5,
            days=30,
            seed=123,
            initial_listings=6,
        )
    )

    assert summary["n_agents"] == 5
    assert summary["ticks"] == 360
    assert summary["initial_listings"] == 6

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT persona_json, risk_posture, is_redteam FROM agents ORDER BY agent_id"
        ).fetchall()
        assert len(rows) == 5
        for row in rows:
            persona = json.loads(row["persona_json"])
            assert row["risk_posture"] == "neutral"
            assert int(row["is_redteam"]) == 0
            assert persona["deadline"]["deadline_tick"] >= 72
            assert persona["financial_stress"]["bill_amount_cents"] > 0
            assert persona["goals"]["buyer"]["want_category"]
            assert persona["goals"]["buyer"]["description"].startswith("Owner brief:")
            assert "brands, models, prices" in persona["background_context"]
            assert persona["goals"]["seller"]["target_listings_count"] >= 1
            assert 2 <= len(persona["inventory_items"]) <= 5

        listing_count = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        assert listing_count == 6
        meta = conn.execute(
            "SELECT value FROM meta WHERE key = 'scaleup_world_summary'"
        ).fetchone()
        assert meta is not None
        assert json.loads(meta["value"])["n_agents"] == 5
    finally:
        conn.close()

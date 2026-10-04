from __future__ import annotations

import json

from bazaar.core.schema import connect, initialize_db
from scripts.level1.apply_strategy_adaptation import apply_adaptation


def _insert_agent(conn, agent_id: int, persona: dict | None = None) -> None:
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat,
             home_lng, activity_rate, privacy_awareness, device,
             persona_json, parent_agent_id, created_at_tick, status)
        VALUES (?, ?, ?, '94110', 0.0, 0.0, 0.3, 0.5, 'iphone',
                ?, NULL, 0, 'active')
        """,
        (
            agent_id,
            f"u{agent_id}",
            f"User {agent_id}",
            json.dumps(persona or {}),
        ),
    )


def _insert_listing(
    conn,
    listing_id: int,
    *,
    owner: int,
    title: str,
    category: str = "electronics-cameras",
    price_cents: int = 12_000,
    tick: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat,
             location_lng, is_phantom, view_count, save_count,
             inquiry_count, created_at_tick, status, is_speculative)
        VALUES (?, ?, ?, ?, 'works', ?, 'good', '94110', 0.0, 0.0,
                0, 6, 2, 3, ?, 'active', 0)
        """,
        (listing_id, owner, category, title, price_cents, tick),
    )


def test_apply_strategy_adaptation_writes_prompt_visible_rows(tmp_path):
    db = tmp_path / "adapt.db"
    conn = connect(db)
    try:
        _insert_agent(
            conn,
            1,
            {
                "goals": {"seller": {"target_listings_count": 5}},
                "deadline": {"deadline_tick": 24},
            },
        )
        _insert_agent(conn, 2)
        _insert_listing(conn, 10, owner=1, title="Canon camera kit")
        _insert_listing(conn, 20, owner=2, title="DJI drone bundle")
        conn.execute(
            """
            INSERT INTO threads
                (thread_id, listing_id, buyer_agent_id, seller_agent_id,
                 created_at_tick, last_msg_tick, status)
            VALUES (100, 10, 2, 1, 1, 2, 'open')
            """
        )
        conn.execute(
            """
            INSERT INTO messages
                (thread_id, sender_agent_id, tick, body, content_hash)
            VALUES (100, 2, 2, 'Is this still available?', 'm1')
            """
        )
        conn.execute(
            """
            INSERT INTO offers
                (thread_id, proposer_id, round, price_cents, terms_json,
                 tick, status)
            VALUES (100, 2, 1, 10000, '{}', 2, 'pending')
            """
        )
        conn.commit()
    finally:
        conn.close()

    result = apply_adaptation(
        db,
        mode="competitive_private_public",
        agent_ids=[1],
        agent_limit=None,
        tick=3,
        history_window=3,
        seller_target=4,
        deadline_offset=72,
        public_limit=4,
        dry_run=False,
    )

    assert result["inserted_count"] == 1
    conn = connect(db)
    try:
        ledger = conn.execute(
            """
            SELECT kind, ref_table, summary, tick
            FROM ledger_entries
            WHERE agent_id = 1
            """
        ).fetchone()
        assert ledger is not None
        assert ledger[0] == "report"
        assert ledger[1] == "strategy_adaptation"
        assert ledger[3] == 3
        assert "Strategy adaptation report" in ledger[2]
        assert "Competitive market pulse" in ledger[2]
        assert "owner utility" in ledger[2]
        assert "1/5 active listings" in ledger[2]

        summary = conn.execute(
            "SELECT content, source FROM agent_summary WHERE agent_id = 1"
        ).fetchone()
        assert summary is not None
        assert summary[1] == "strategy_adaptation"
        assert "DJI drone bundle" in summary[0]

        event = conn.execute(
            "SELECT action_type FROM events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        assert event[0] == "strategy_adaptation_intervention"
    finally:
        conn.close()


def test_apply_strategy_adaptation_dry_run_does_not_insert(tmp_path):
    db = tmp_path / "dry.db"
    conn = initialize_db(db)
    try:
        _insert_agent(conn, 1)
        conn.commit()
    finally:
        conn.close()

    result = apply_adaptation(
        db,
        mode="private",
        agent_ids=[1],
        agent_limit=None,
        tick=1,
        history_window=1,
        seller_target=1,
        deadline_offset=12,
        public_limit=2,
        dry_run=True,
    )

    assert result["inserted_count"] == 1
    conn = initialize_db(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0]
        assert count == 0
    finally:
        conn.close()

"""Platform event audit invariants."""
from __future__ import annotations

import json
import random
import sqlite3

import pytest

from bazaar.actions.dispatch import dispatch
from bazaar.actions.types import ActionType
from bazaar.agents.persona import generate_persona
from bazaar.core.env import BazaarEnv
from bazaar.core.event_audit import (
    audit_agent_action_event_consistency,
    audit_marketplace_state_invariants,
    audit_platform_event_consistency,
)
from bazaar.core.event_log import log_event
from bazaar.core.schema import connect
from bazaar.dynamics import DynamicRegistry
from bazaar.dynamics.callbacks import D_restock
from bazaar.platform import MarketplacePlatform


def test_platform_event_audit_clean_for_register_and_seeds(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        platform.seed_phantom_listings(count=1)
        platform.seed_real_listings(count=1, agent_pool=[1])
        platform.seed_lot_sales_feed(count=2)

        issues = audit_platform_event_consistency(
            platform.conn,
            require_seed_coverage=True,
        )

        assert issues == []
    finally:
        platform.close()


def test_platform_event_audit_detects_payload_drift(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.seed_phantom_listings(count=1)
        row = platform.conn.execute(
            """
            SELECT event_id, payload
            FROM events
            WHERE action_type = 'platform_seed_phantom_listing'
            """
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["price_cents"] = int(payload["price_cents"]) + 123
        platform.conn.execute(
            "UPDATE events SET payload = ? WHERE event_id = ?",
            (json.dumps(payload, sort_keys=True), row["event_id"]),
        )
        platform.conn.commit()

        issues = audit_platform_event_consistency(platform.conn)

        assert any(
            issue.action_type == "platform_seed_phantom_listing"
            and "price_cents mismatch" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def test_platform_event_audit_strict_seed_coverage_detects_unlogged_seed_rows(
    tmp_db,
) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        with platform.conn:
            platform.conn.execute(
                """
                INSERT INTO agents
                    (agent_id, user_name, display_name, home_zip, home_lat,
                     home_lng, persona_json, is_seeded)
                VALUES (10, 'seed', 'Seed', '00000', 0, 0, '{}', 1)
                """
            )
            platform.conn.execute(
                """
                INSERT INTO listings
                    (listing_id, owner_agent_id, category, title, description,
                     price_cents, condition, location_zip, location_lat,
                     location_lng, created_at_tick, status, is_seeded)
                VALUES (20, 10, 'books', 'Seed Book', 'x', 1000, 'good',
                        '00000', 0, 0, 0, 'sold', 1)
                """
            )

        issues = audit_platform_event_consistency(
            platform.conn,
            require_seed_coverage=True,
        )

        assert any(
            issue.ref_id == 10 and "seeded agent has no covering event" in issue.message
            for issue in issues
        )
        assert any(
            issue.ref_id == 20 and "seeded listing has no covering event" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def test_action_event_audit_is_clean_for_basic_materialized_actions(tmp_db) -> None:
    env = BazaarEnv(
        db_path=tmp_db,
        seed_phantom_listings=0,
        dynamics=DynamicRegistry(),
    )
    try:
        env.add_agent(type("A", (), {"persona": generate_persona(1, seed=1)})())
        env.add_agent(type("A", (), {"persona": generate_persona(2, seed=2)})())

        created = dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.CREATE_LISTING,
            raw_args={
                "category": "electronics",
                "title": "Working Camera",
                "description": "A clean used camera.",
                "price_cents": 12000,
                "condition": "good",
            },
            tick=0,
        )
        listing_id = int(created.payload["listing_id"])
        dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.MESSAGE,
            raw_args={"listing_id": listing_id, "body": "Still available?"},
            tick=1,
        )
        dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.MAKE_OFFER,
            raw_args={"listing_id": listing_id, "price_cents": 10000, "terms": {}},
            tick=2,
        )

        assert audit_agent_action_event_consistency(env.platform.conn) == []
    finally:
        env.close()


def test_action_event_audit_detects_materialized_row_drift(tmp_db) -> None:
    env = BazaarEnv(
        db_path=tmp_db,
        seed_phantom_listings=0,
        dynamics=DynamicRegistry(),
    )
    try:
        env.add_agent(type("A", (), {"persona": generate_persona(1, seed=1)})())
        env.add_agent(type("A", (), {"persona": generate_persona(2, seed=2)})())
        created = dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.CREATE_LISTING,
            raw_args={
                "category": "electronics",
                "title": "Working Camera",
                "description": "A clean used camera.",
                "price_cents": 12000,
                "condition": "good",
            },
            tick=0,
        )
        listing_id = int(created.payload["listing_id"])
        sent = dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.MESSAGE,
            raw_args={"listing_id": listing_id, "body": "Still available?"},
            tick=1,
        )
        env.platform.conn.execute(
            "UPDATE messages SET sender_agent_id = 1 WHERE message_id = ?",
            (sent.payload["message_id"],),
        )
        env.platform.conn.commit()

        issues = audit_agent_action_event_consistency(env.platform.conn)

        assert any(
            issue.action_type == "message"
            and "sender_agent_id mismatch" in issue.message
            for issue in issues
        )
    finally:
        env.close()


def test_action_event_audit_allows_explicit_no_row_actions(tmp_db) -> None:
    env = BazaarEnv(
        db_path=tmp_db,
        seed_phantom_listings=0,
        dynamics=DynamicRegistry(),
    )
    try:
        env.add_agent(type("A", (), {"persona": generate_persona(1, seed=1)})())
        dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.DO_NOTHING,
            raw_args={},
            tick=0,
        )
        dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.WAIT,
            raw_args={"ticks": 2},
            tick=1,
        )
        dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.REFINE_SEARCH,
            raw_args={"delta": {"max_price_cents": 2000}},
            tick=2,
        )

        assert audit_agent_action_event_consistency(env.platform.conn) == []
    finally:
        env.close()


def test_action_event_audit_flags_unchecked_ok_agent_event(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        event_id = log_event(
            platform.conn,
            tick=0,
            agent_id=1,
            action_type="mystery_action",
            payload={},
            result_status="ok",
            result_payload={},
        )
        platform.conn.commit()

        issues = audit_agent_action_event_consistency(platform.conn)

        assert any(
            issue.event_id == event_id
            and issue.action_type == "mystery_action"
            and "action_event_checker_missing" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def test_marketplace_state_invariant_audit_is_clean_for_open_pending_offer(
    tmp_db,
) -> None:
    env = BazaarEnv(
        db_path=tmp_db,
        seed_phantom_listings=0,
        dynamics=DynamicRegistry(),
    )
    try:
        env.add_agent(type("A", (), {"persona": generate_persona(1, seed=1)})())
        env.add_agent(type("A", (), {"persona": generate_persona(2, seed=2)})())
        created = dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.CREATE_LISTING,
            raw_args={
                "category": "electronics",
                "title": "Working Camera",
                "description": "A clean used camera.",
                "price_cents": 12000,
                "condition": "good",
            },
            tick=0,
        )
        dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.MAKE_OFFER,
            raw_args={
                "listing_id": int(created.payload["listing_id"]),
                "price_cents": 10000,
                "terms": {},
            },
            tick=1,
        )

        assert audit_marketplace_state_invariants(env.platform.conn) == []
    finally:
        env.close()


def test_marketplace_state_invariant_audit_detects_pending_offer_on_committed_thread(
    tmp_db,
) -> None:
    env = BazaarEnv(
        db_path=tmp_db,
        seed_phantom_listings=0,
        dynamics=DynamicRegistry(),
    )
    try:
        env.add_agent(type("A", (), {"persona": generate_persona(1, seed=1)})())
        env.add_agent(type("A", (), {"persona": generate_persona(2, seed=2)})())
        created = dispatch(
            env.platform.conn,
            agent_id=1,
            action=ActionType.CREATE_LISTING,
            raw_args={
                "category": "electronics",
                "title": "Working Camera",
                "description": "A clean used camera.",
                "price_cents": 12000,
                "condition": "good",
            },
            tick=0,
        )
        offer = dispatch(
            env.platform.conn,
            agent_id=2,
            action=ActionType.MAKE_OFFER,
            raw_args={
                "listing_id": int(created.payload["listing_id"]),
                "price_cents": 10000,
                "terms": {},
            },
            tick=1,
        )
        env.platform.conn.execute(
            "UPDATE threads SET status = 'committed' WHERE thread_id = ?",
            (offer.payload["thread_id"],),
        )
        env.platform.conn.commit()

        issues = audit_marketplace_state_invariants(env.platform.conn)

        assert any(
            issue.action_type == "state_invariant"
            and "pending_offer_open_thread" in issue.message
            and issue.ref_id == offer.payload["offer_id"]
            for issue in issues
        )
    finally:
        env.close()


def test_state_audit_completed_meetup_requires_non_seeded_listing_sold(
    tmp_db,
) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        platform.register_agent(generate_persona(2, seed=2))
        with platform.conn:
            platform.conn.execute(
                """
                INSERT INTO listings
                    (listing_id, owner_agent_id, category, title, description,
                     price_cents, condition, location_zip, location_lat,
                     location_lng, created_at_tick, status, is_seeded)
                VALUES
                    (10, 1, 'electronics', 'Camera', 'A camera', 10000,
                     'good', '00001', 0.0, 0.0, 0, 'active', 0)
                """
            )
            platform.conn.execute(
                """
                INSERT INTO threads
                    (thread_id, listing_id, buyer_agent_id, seller_agent_id,
                     created_at_tick, last_msg_tick, status)
                VALUES (20, 10, 2, 1, 0, 1, 'completed')
                """
            )
            platform.conn.execute(
                """
                INSERT INTO meetups
                    (meetup_id, thread_id, scheduled_tick, location_desc,
                     payment_method, status)
                VALUES (30, 20, 2, 'Porch pickup', 'cash', 'completed')
                """
            )

        issues = audit_marketplace_state_invariants(platform.conn)

        assert any(
            issue.ref_id == 30
            and "completed_meetup_terminal_state" in issue.message
            and "listing='active'" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def test_state_audit_allows_completed_meetup_on_seeded_active_listing(
    tmp_db,
) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        platform.register_agent(generate_persona(2, seed=2))
        with platform.conn:
            platform.conn.execute(
                """
                INSERT INTO listings
                    (listing_id, owner_agent_id, category, title, description,
                     price_cents, condition, location_zip, location_lat,
                     location_lng, created_at_tick, status, is_seeded)
                VALUES
                    (10, 1, 'electronics', 'Seed Camera', 'Historical seed',
                     10000, 'good', '00001', 0.0, 0.0, -24, 'active', 1)
                """
            )
            platform.conn.execute(
                """
                INSERT INTO threads
                    (thread_id, listing_id, buyer_agent_id, seller_agent_id,
                     created_at_tick, last_msg_tick, status)
                VALUES (20, 10, 2, 1, -24, -23, 'completed')
                """
            )
            platform.conn.execute(
                """
                INSERT INTO meetups
                    (meetup_id, thread_id, scheduled_tick, location_desc,
                     payment_method, status)
                VALUES (30, 20, -22, 'Porch pickup', 'cash', 'completed')
                """
            )

        issues = audit_marketplace_state_invariants(platform.conn)

        assert not any(
            issue.ref_id == 30
            and "completed_meetup_terminal_state" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def test_state_audit_flags_error_events_with_generic_error_event_label(
    tmp_db,
) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        event_id = log_event(
            platform.conn,
            tick=0,
            agent_id=1,
            action_type="policy_error",
            payload={
                "source": "policy",
                "error": "policy_return_not_agent_action",
            },
            result_status="error",
            result_payload={
                "error": "policy_return_not_agent_action",
            },
        )
        platform.conn.commit()

        issues = audit_marketplace_state_invariants(platform.conn)

        assert any(
            issue.event_id == event_id
            and "error_events_absent:" in issue.message
            and "action='policy_error'" in issue.message
            for issue in issues
        )
        assert not any("dispatcher_errors_absent" in issue.message for issue in issues)
    finally:
        platform.close()


def test_state_audit_flags_legacy_tables_missing_json_constraints(
    tmp_path,
) -> None:
    db = tmp_path / "legacy-schema.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE events (
            event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            tick           INTEGER NOT NULL,
            wall_time      TEXT    NOT NULL,
            agent_id       INTEGER,
            action_type    TEXT    NOT NULL,
            payload        TEXT    NOT NULL,
            result_status  TEXT    NOT NULL,
            result_payload TEXT
        );
        CREATE TABLE llm_calls (
            call_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            tick               INTEGER NOT NULL,
            agent_id           INTEGER NOT NULL,
            model              TEXT    NOT NULL,
            backend            TEXT    NOT NULL,
            prompt_hash        TEXT    NOT NULL,
            prompt_text        TEXT,
            sampling_params    TEXT    NOT NULL,
            response_text      TEXT    NOT NULL,
            tool_calls_json    TEXT,
            seed               INTEGER,
            cache_hit          INTEGER NOT NULL DEFAULT 0,
            latency_ms         INTEGER,
            wall_time          TEXT    NOT NULL
        );
        """
    )
    raw.close()

    conn = connect(db)
    try:
        issues = audit_marketplace_state_invariants(conn)

        assert any(
            "schema_constraint_missing: table 'events' missing constraint "
            "events.payload_json_object" in issue.message
            for issue in issues
        )
        assert any(
            "schema_constraint_missing: table 'events' missing constraint "
            "events.result_status_enum" in issue.message
            for issue in issues
        )
        assert any(
            "schema_constraint_missing: table 'llm_calls' missing constraint "
            "llm_calls.sampling_params_json_object" in issue.message
            for issue in issues
        )
        assert any(
            "schema_constraint_missing: table 'llm_calls' missing constraint "
            "llm_calls.tool_calls_json_array" in issue.message
            for issue in issues
        )
    finally:
        conn.close()


def test_event_audit_flags_malformed_event_json_without_crashing(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.conn.execute("PRAGMA ignore_check_constraints = ON")
        with platform.conn:
            platform.conn.execute(
                """
                INSERT INTO events
                    (tick, wall_time, agent_id, action_type, payload,
                     result_status, result_payload)
                VALUES
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'platform_phantom_tripwire', '{bad', 'ok', NULL),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'memory_divergence', '{bad', 'ok', NULL),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'platform_seed_lot_sales_feed', '{bad', 'ok', NULL),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'noop', '[]', 'ok', NULL),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'noop', '{}', 'ok', '{bad'),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'noop', '{}', 'ok', '[]'),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'noop', '{}', 'partial', NULL)
                """
            )
        platform.conn.execute("PRAGMA ignore_check_constraints = OFF")

        state_issues = audit_marketplace_state_invariants(platform.conn)
        platform_issues = audit_platform_event_consistency(
            platform.conn,
            require_seed_coverage=True,
        )

        assert platform_issues == []
        assert sum(
            issue.message.startswith("event_json_valid: payload ")
            and "malformed JSON" in issue.message
            for issue in state_issues
        ) == 3
        assert any(
            "result_payload" in issue.message and "malformed JSON" in issue.message
            for issue in state_issues
        )
        assert any(
            "payload for action 'noop' is not a JSON object" in issue.message
            for issue in state_issues
        )
        assert any(
            "result_payload for action 'noop' is not a JSON object"
            in issue.message
            for issue in state_issues
        )
        assert any(
            "invalid result_status='partial'" in issue.message
            for issue in state_issues
        )
    finally:
        platform.close()


def test_state_audit_flags_malformed_llm_call_json(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        platform.conn.execute("PRAGMA ignore_check_constraints = ON")
        with platform.conn:
            platform.conn.execute(
                """
                INSERT INTO llm_calls
                    (tick, agent_id, model, backend, prompt_hash, prompt_text,
                     sampling_params, response_text, tool_calls_json,
                     wall_time)
                VALUES
                    (0, 1, 'm', 'b', 'h1', NULL, '{bad', 'ok', NULL,
                     '2026-01-01T00:00:00+00:00'),
                    (0, 1, 'm', 'b', 'h2', NULL, '[]', 'ok', NULL,
                     '2026-01-01T00:00:00+00:00'),
                    (0, 1, 'm', 'b', 'h3', NULL, '{}', 'ok', '{bad',
                     '2026-01-01T00:00:00+00:00'),
                    (0, 1, 'm', 'b', 'h4', NULL, '{}', 'ok', '{}',
                     '2026-01-01T00:00:00+00:00')
                """
            )
        platform.conn.execute("PRAGMA ignore_check_constraints = OFF")

        issues = audit_marketplace_state_invariants(platform.conn)

        assert any(
            "sampling_params for call_id 1 is malformed JSON" in issue.message
            for issue in issues
        )
        assert any(
            "sampling_params for call_id 2 is not a JSON object" in issue.message
            for issue in issues
        )
        assert any(
            "tool_calls_json for call_id 3 is malformed JSON" in issue.message
            for issue in issues
        )
        assert any(
            "tool_calls_json for call_id 4 is not a JSON array" in issue.message
            for issue in issues
        )
    finally:
        platform.close()


def _registered_and_restocked(tmp_db) -> MarketplacePlatform:
    platform = MarketplacePlatform(tmp_db)
    platform.register_agent(generate_persona(1, seed=1))
    with platform.conn:
        added = D_restock(platform.conn, tick=0, rng=random.Random(0))
    assert added > 0
    return platform


def _rewrite_persona(conn: sqlite3.Connection, edit) -> None:
    persona = json.loads(
        conn.execute("SELECT persona_json FROM agents WHERE agent_id = 1").fetchone()[0]
    )
    edit(persona)
    with conn:
        conn.execute(
            "UPDATE agents SET persona_json = ? WHERE agent_id = 1",
            (json.dumps(persona, sort_keys=True),),
        )


def test_platform_event_audit_allows_logged_tick_zero_restock(tmp_db) -> None:
    platform = _registered_and_restocked(tmp_db)
    try:
        persona = json.loads(platform.conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = 1"
        ).fetchone()[0])
        registered = json.loads(json.loads(platform.conn.execute(
            "SELECT payload FROM events WHERE action_type = 'platform_register_agent'"
        ).fetchone()[0])["persona_json"])
        assert persona != registered
        assert persona["inventory_items"][-1]["source"] == "restock"

        assert audit_platform_event_consistency(platform.conn) == []
    finally:
        platform.close()


def _mark_first_unit_sold_elsewhere(persona: dict) -> None:
    persona["inventory_items"][0]["sold_at_tick"] = 3
    persona["inventory_items"][0]["sold_via_listing_id"] = 999


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            lambda p: p.update(monthly_budget_cents=p["monthly_budget_cents"] + 1),
            "persona_json.monthly_budget_cents mismatch",
        ),
        (
            lambda p: p["inventory_items"][0].update(asking_price_cents=1),
            "persona_json.inventory_items[0].asking_price_cents mismatch",
        ),
        (
            lambda p: p.update(inventory_items=p["inventory_items"][:1]),
            "persona_json.inventory_items shrank",
        ),
        (
            lambda p: p["inventory_items"].append(dict(p["inventory_items"][-1])),
            "persona_json.inventory_items restocks mismatch",
        ),
        (
            lambda p: p["inventory_items"].append(
                {"title": "Gift", "source": "bought", "bought_from_listing_id": 1}
            ),
            "persona_json.inventory_items purchases mismatch",
        ),
        (
            lambda p: p["inventory_items"].append({"title": "Gift"}),
            "appended with unexplained source=None",
        ),
        (
            _mark_first_unit_sold_elsewhere,
            "persona_json.inventory_items[0].sold_at_tick mismatch",
        ),
    ],
)
def test_platform_event_audit_flags_unlogged_persona_change(tmp_db, edit, expected) -> None:
    platform = _registered_and_restocked(tmp_db)
    try:
        _rewrite_persona(platform.conn, edit)

        issues = audit_platform_event_consistency(platform.conn)

        assert any(
            issue.action_type == "platform_register_agent"
            and issue.ref_id == 1
            and expected in issue.message
            for issue in issues
        ), issues
    finally:
        platform.close()


def _seeded_listing(platform: MarketplacePlatform) -> int:
    platform.register_agent(generate_persona(1, seed=1))
    platform.register_agent(generate_persona(2, seed=2))
    (listing_id,) = platform.seed_real_listings(count=1, agent_pool=[1])
    platform.conn.commit()
    return listing_id


def test_seed_listing_price_follows_logged_edits(tmp_db) -> None:
    """The owner may reprice or recondition a seeded listing with
    edit_listing, under every handoff check; the audit replays the
    logged edits over the seed payload."""
    platform = MarketplacePlatform(tmp_db)
    try:
        listing_id = _seeded_listing(platform)
        result = dispatch(
            platform.conn, agent_id=1, action=ActionType.EDIT_LISTING,
            raw_args={"listing_id": listing_id, "price_cents": 1234, "condition": "fair"},
            tick=3,
        )
        assert result.status == "ok"
        platform.conn.commit()

        assert audit_platform_event_consistency(platform.conn) == []

        with platform.conn:
            platform.conn.execute(
                "UPDATE listings SET price_cents = 999 WHERE listing_id = ?", (listing_id,),
            )
        issues = audit_platform_event_consistency(platform.conn)

        assert [(i.action_type, i.ref_id, i.severity) for i in issues] == [
            ("platform_seed_real_listing", listing_id, "error"),
        ]
        seeded = json.loads(platform.conn.execute(
            "SELECT payload FROM events WHERE action_type = 'platform_seed_real_listing'"
        ).fetchone()[0])["price_cents"]
        assert issues[0].message == (
            f"price_cents mismatch: event={seeded!r} "
            "replayed through edit_listing=1234 table=999"
        )
    finally:
        platform.close()


def test_seed_listing_price_change_without_edit_is_an_error(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        listing_id = _seeded_listing(platform)
        with platform.conn:
            platform.conn.execute(
                "UPDATE listings SET condition = 'poor' WHERE listing_id = ?", (listing_id,),
            )

        issues = audit_platform_event_consistency(platform.conn)

        assert any(
            i.ref_id == listing_id and i.severity == "error"
            and i.message.startswith("condition mismatch")
            for i in issues
        )
    finally:
        platform.close()


def _two_agents_with_thread(platform: MarketplacePlatform) -> tuple[int, int]:
    platform.register_agent(generate_persona(1, seed=1))
    platform.register_agent(generate_persona(2, seed=2))
    listing = dispatch(
        platform.conn, agent_id=1, action=ActionType.CREATE_LISTING,
        raw_args={"category": "tools", "title": "Hose", "description": "",
                  "price_cents": 1000, "condition": "good"},
        tick=0,
    )
    offer = dispatch(
        platform.conn, agent_id=2, action=ActionType.MAKE_OFFER,
        raw_args={"listing_id": listing.payload["listing_id"], "price_cents": 900},
        tick=1,
    )
    platform.conn.commit()
    return listing.payload["listing_id"], offer.payload["thread_id"]


def test_not_found_error_for_a_nonexistent_id_is_a_warning(tmp_db) -> None:
    """Handlers answer an id that does not exist with result_status
    'error' by design (for example a hallucinated meetup id); that is not
    a handler exception."""
    platform = MarketplacePlatform(tmp_db)
    try:
        _listing_id, thread_id = _two_agents_with_thread(platform)
        for action, args in (
            (ActionType.COMPLETE_TRANSACTION, {"meetup_id": 0}),
            (ActionType.READ, {"thread_id": thread_id + 50}),
        ):
            result = dispatch(platform.conn, agent_id=2, action=action, raw_args=args, tick=2)
            assert result.status == "error"
        platform.conn.commit()

        issues = audit_marketplace_state_invariants(platform.conn)

        assert [i.severity for i in issues] == ["warning", "warning"]
        assert all(
            "error_events_absent" in i.message and "not a handler exception" in i.message
            for i in issues
        )
    finally:
        platform.close()


def test_not_found_error_for_an_existing_id_is_an_error(tmp_db) -> None:
    platform = MarketplacePlatform(tmp_db)
    try:
        _listing_id, thread_id = _two_agents_with_thread(platform)
        event_id = log_event(
            platform.conn, tick=2, agent_id=2, action_type="read",
            payload={"thread_id": thread_id}, result_status="error",
            result_payload={"error": "thread_not_found"},
        )
        platform.conn.commit()

        issues = audit_marketplace_state_invariants(platform.conn)

        assert [(i.event_id, i.severity) for i in issues] == [(event_id, "error")]
        assert issues[0].message == (
            f"error_events_absent: event {event_id} action='read' has result_status='error'"
        )
    finally:
        platform.close()

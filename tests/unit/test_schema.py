"""Schema-level tests.

Guarantees that the SQLite DDL is consistent, FK-valid, and that the
schema version is stamped at init time.
"""
from __future__ import annotations

import sqlite3

import pytest

from bazaar.core.schema import (
    SCHEMA_VERSION,
    connect,
    initialize_db,
    missing_required_schema_constraints,
    schema_version,
)


def test_initialize_stamps_version(tmp_path):
    conn = initialize_db(tmp_path / "db.sqlite")
    assert schema_version(conn) == SCHEMA_VERSION
    conn.close()


def test_connect_stamps_version_on_legacy_db_without_meta(tmp_path):
    db_path = tmp_path / "legacy_no_meta.db"
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)")
    raw.close()

    conn = connect(db_path)
    assert schema_version(conn) == SCHEMA_VERSION
    conn.close()

    reopened = sqlite3.connect(db_path)
    try:
        row = reopened.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert int(row[0]) == SCHEMA_VERSION
    finally:
        reopened.close()


def test_connect_replaces_stale_schema_version(tmp_path):
    db_path = tmp_path / "stale_version.db"
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', '1')"
    )
    raw.commit()
    raw.close()

    conn = connect(db_path)
    assert schema_version(conn) == SCHEMA_VERSION
    conn.close()


def test_connect_marks_weak_legacy_schema_uncurrent(tmp_path):
    db_path = tmp_path / "weak_legacy_schema.db"
    raw = sqlite3.connect(db_path)
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
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('schema_version', '2');
        """
    )
    raw.close()

    conn = connect(db_path)
    try:
        assert schema_version(conn) == 0
        missing = missing_required_schema_constraints(conn)
        assert ("events", "events.payload_json_object") in missing
        assert ("events", "events.result_status_enum") in missing
        assert ("events", "events.action_type_nonempty") in missing
    finally:
        conn.close()


def test_all_expected_tables_exist(fresh_conn):
    rows = fresh_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    expected = {
        "agents", "blocks", "events", "ledger_entries", "listings",
        "meetups", "messages", "meta", "narrative_memories", "offers",
        "photos", "ratings", "reports", "self_portraits", "snapshots",
        "threads",
    }
    missing = expected - names
    assert not missing, f"missing tables: {missing}"


def test_foreign_keys_enforced(fresh_conn):
    fresh_conn.execute("PRAGMA foreign_keys = ON")
    # Inserting a message with a non-existent thread_id must fail.
    with pytest.raises(sqlite3.IntegrityError):
        fresh_conn.execute(
            "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
            "content_hash) VALUES (?, ?, ?, ?, ?)",
            (999, 1, 0, "hello", "abc"),
        )
        fresh_conn.commit()


def test_rating_check_constraint(fresh_conn):
    fresh_conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (1, 'u1', 'U1', '00000', "
        "0, 0, '{}')"
    )
    fresh_conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (2, 'u2', 'U2', '00000', "
        "0, 0, '{}')"
    )
    fresh_conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        fresh_conn.execute(
            "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, "
            "tick) VALUES (1, 2, 7, 0)"  # 7 stars is invalid
        )
        fresh_conn.commit()


def test_events_table_rejects_invalid_json_and_status(fresh_conn):
    base = (
        "INSERT INTO events "
        "(tick, wall_time, agent_id, action_type, payload, result_status, "
        "result_payload) VALUES "
    )

    bad_rows = [
        "('bad', '2026-01-01T00:00:00+00:00', NULL, 'x', '{}', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', 'bad', 'x', '{}', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, '', '{}', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, '   ', '{}', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, 'x', '{bad', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, 'x', '[]', 'ok', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, 'x', '{}', 'partial', NULL)",
        "(0, '2026-01-01T00:00:00+00:00', NULL, 'x', '{}', 'ok', '{bad')",
        "(0, '2026-01-01T00:00:00+00:00', NULL, 'x', '{}', 'ok', '[]')",
    ]
    for row_sql in bad_rows:
        with pytest.raises(sqlite3.IntegrityError):
            fresh_conn.execute(base + row_sql)
        fresh_conn.rollback()


def test_llm_calls_has_reasoning_summary_column(fresh_conn):
    """R11 B.4: every freshly-initialized DB carries the column from
    DDL — no migration needed for new dbs."""
    cols = {r[1] for r in fresh_conn.execute("PRAGMA table_info(llm_calls)")}
    assert "reasoning_summary" in cols


def test_llm_calls_rejects_invalid_json_shapes(fresh_conn):
    fresh_conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (1, 'u1', 'U1', '00000', "
        "0, 0, '{}')"
    )
    fresh_conn.commit()
    base = (
        "INSERT INTO llm_calls "
        "(tick, agent_id, model, backend, prompt_hash, prompt_text, "
        "sampling_params, response_text, tool_calls_json, wall_time) VALUES "
    )
    bad_rows = [
        "('bad', 1, 'm', 'b', 'h0', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 'bad', 'm', 'b', 'h0', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, '', 'b', 'h0', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', '   ', 'h0', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', 'b', '', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', 'b', 'h1', NULL, '{bad', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', 'b', 'h2', NULL, '[]', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', 'b', 'h3', NULL, '{}', 'ok', '{bad', "
        "'2026-01-01T00:00:00+00:00')",
        "(0, 1, 'm', 'b', 'h4', NULL, '{}', 'ok', '{}', "
        "'2026-01-01T00:00:00+00:00')",
    ]
    for row_sql in bad_rows:
        with pytest.raises(sqlite3.IntegrityError):
            fresh_conn.execute(base + row_sql)
        fresh_conn.rollback()

    base_with_operational_fields = (
        "INSERT INTO llm_calls "
        "(tick, agent_id, model, backend, prompt_hash, prompt_text, "
        "sampling_params, response_text, tool_calls_json, wall_time, "
        "cache_hit, latency_ms) VALUES "
    )
    bad_operational_rows = [
        "(0, 1, 'm', 'b', 'h5', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00', 2, NULL)",
        "(0, 1, 'm', 'b', 'h6', NULL, '{}', 'ok', NULL, "
        "'2026-01-01T00:00:00+00:00', 0, -1)",
    ]
    for row_sql in bad_operational_rows:
        with pytest.raises(sqlite3.IntegrityError):
            fresh_conn.execute(base_with_operational_fields + row_sql)
        fresh_conn.rollback()


def test_llm_calls_migration_adds_column_to_old_db(tmp_path):
    """R11 B.4: an on-disk db whose llm_calls table predates the
    column gets it added when ``connect()`` reopens the file. Mirrors
    the legacy snapshots (R7-era) that ship without reasoning_summary."""
    db_path = tmp_path / "old.db"
    raw = sqlite3.connect(db_path)
    # Build an llm_calls table with the *old* shape — no reasoning_summary.
    raw.execute("""
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
        )
    """)
    raw.commit()
    cols_before = {
        r[1] for r in raw.execute("PRAGMA table_info(llm_calls)")
    }
    assert "reasoning_summary" not in cols_before
    raw.close()

    conn = connect(db_path)
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(llm_calls)")}
    assert "reasoning_summary" in cols_after
    assert schema_version(conn) == 0
    assert (
        "llm_calls",
        "llm_calls.sampling_params_json_object",
    ) in missing_required_schema_constraints(conn)
    conn.close()

    reopened = sqlite3.connect(db_path)
    try:
        cols_reopened = {
            r[1] for r in reopened.execute("PRAGMA table_info(llm_calls)")
        }
        assert "reasoning_summary" in cols_reopened
        row = reopened.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 0
    finally:
        reopened.close()

"""CLI audit command guardrails."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from bazaar.agents.persona import generate_persona
from bazaar.core.event_log import log_event
from bazaar.platform import MarketplacePlatform


def _run_audit(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bazaar.cli", "audit", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_audit_without_db_checks_action_contracts() -> None:
    result = _run_audit()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "action contracts" in result.stdout
    assert "audit completed" in result.stdout


def test_smoke_test_runs_postflight_audit(tmp_path: Path) -> None:
    db = tmp_path / "smoke.db"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bazaar.cli",
            "smoke-test",
            "--agents",
            "2",
            "--ticks",
            "1",
            "--phantoms",
            "1",
            "--out",
            str(db),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "BazaarBench audit" in result.stdout
    assert "audit completed" in result.stdout


def test_smoke_test_audit_passes_after_restock_and_flags_tampered_persona(
    tmp_path: Path,
) -> None:
    db = tmp_path / "smoke.db"
    result = subprocess.run(
        [
            sys.executable, "-m", "bazaar.cli", "smoke-test",
            "--agents", "4", "--ticks", "3", "--out", str(db),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    conn = sqlite3.connect(db)
    try:
        restocks = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE action_type = 'platform_inventory_restocked' AND tick = 0"
        ).fetchone()[0]
        assert restocks > 0
        persona = json.loads(
            conn.execute("SELECT persona_json FROM agents WHERE agent_id = 1").fetchone()[0]
        )
        persona["inventory_items"][0]["asking_price_cents"] += 1
        with conn:
            conn.execute(
                "UPDATE agents SET persona_json = ? WHERE agent_id = 1",
                (json.dumps(persona, sort_keys=True),),
            )
    finally:
        conn.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        issue["kind"] == "platform_event"
        and issue["ref"] == "platform_register_agent"
        and "persona_json.inventory_items[0].asking_price_cents mismatch" in issue["message"]
        for issue in payload["issues"]
    )


def test_cli_audit_reports_documented_legacy_behaviour_as_warning(tmp_path: Path) -> None:
    from tests.unit.test_event_audit_provenance import _sister_meetup_world

    db = tmp_path / "sister.db"
    stranded = _sister_meetup_world(db)

    result = _run_audit(db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "warnings (documented)" in result.stdout
    assert "audit completed with 1 warning(s)." in result.stdout
    payload = json.loads(_run_audit(db, "--json").stdout)
    assert payload["status"] == "ok"
    assert payload["issues"] == []
    assert payload["checks"]["warnings"] == 1
    assert [(w["ref_id"], w["severity"]) for w in payload["warnings"]] == [
        (stranded, "warning"),
    ]
    assert "documented legacy behaviour" in payload["warnings"][0]["message"]

    # The same state under completion_integrity_mode=unit is an error.
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute(
                "UPDATE meta SET value = 'unit' WHERE key = 'completion_integrity_mode'"
            )
    finally:
        conn.close()
    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["warnings"] == []
    issues = payload["issues"]
    assert (stranded, "error") in [(i["ref_id"], i["severity"]) for i in issues]
    # meta switched on over the legacy-shaped log is an error too.
    assert all(
        i["ref_id"] == stranded or "legacy-shaped" in i["message"] for i in issues
    ), issues


def test_cli_audit_passes_for_current_platform_seed_events(tmp_path: Path) -> None:
    db = tmp_path / "seeded.db"
    platform = MarketplacePlatform(db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        platform.seed_phantom_listings(count=1)
        platform.seed_real_listings(count=1, agent_pool=[1])
        platform.seed_lot_sales_feed(count=1)
    finally:
        platform.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["issues"] == []
    assert payload["checks"]["action_event_issues"] == 0
    assert payload["checks"]["state_invariant_issues"] == 0


def test_cli_audit_fails_on_platform_event_payload_drift(tmp_path: Path) -> None:
    db = tmp_path / "drift.db"
    platform = MarketplacePlatform(db)
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
        payload["price_cents"] = int(payload["price_cents"]) + 99
        platform.conn.execute(
            "UPDATE events SET payload = ? WHERE event_id = ?",
            (json.dumps(payload, sort_keys=True), row["event_id"]),
        )
        platform.conn.commit()
    finally:
        platform.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert any(
        issue["kind"] == "platform_event"
        and issue["ref"] == "platform_seed_phantom_listing"
        and "price_cents mismatch" in issue["message"]
        for issue in payload["issues"]
    )


def test_collect_audit_records_filters_legacy_event_ticks(tmp_path: Path) -> None:
    from bazaar.cli import _collect_audit_records

    db = tmp_path / "legacy-drift.db"
    platform = MarketplacePlatform(db)
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
        payload["price_cents"] = int(payload["price_cents"]) + 99
        platform.conn.execute(
            "UPDATE events SET payload = ? WHERE event_id = ?",
            (json.dumps(payload, sort_keys=True), row["event_id"]),
        )
        platform.conn.commit()
    finally:
        platform.close()

    records, _, event_count, _, _ = _collect_audit_records(
        db,
        strict_seed_coverage=True,
    )
    assert event_count == 1
    assert any(record["kind"] == "platform_event" for record in records)

    records, _, event_count, _, _ = _collect_audit_records(
        db,
        strict_seed_coverage=True,
        min_event_tick=1,
    )
    assert event_count == 0
    assert not any(record["kind"] == "platform_event" for record in records)


def test_cli_audit_fails_on_malformed_event_json(tmp_path: Path) -> None:
    db = tmp_path / "bad-json.db"
    platform = MarketplacePlatform(db)
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
                     'memory_divergence', '{bad', 'ok', NULL),
                    (0, '2026-01-01T00:00:00+00:00', NULL,
                     'noop', '[]', 'partial', '[]')
                """
            )
        platform.conn.execute("PRAGMA ignore_check_constraints = OFF")
    finally:
        platform.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert any(
        issue["kind"] == "state_invariant"
        and "payload for action 'memory_divergence' is malformed JSON"
        in issue["message"]
        for issue in payload["issues"]
    )
    assert any(
        issue["kind"] == "state_invariant"
        and "payload for action 'noop' is not a JSON object" in issue["message"]
        for issue in payload["issues"]
    )
    assert any(
        issue["kind"] == "state_invariant"
        and "result_payload for action 'noop' is not a JSON object"
        in issue["message"]
        for issue in payload["issues"]
    )
    assert any(
        issue["kind"] == "state_invariant"
        and "invalid result_status='partial'" in issue["message"]
        for issue in payload["issues"]
    )


def test_collect_audit_records_can_skip_state_invariants_for_resume(
    tmp_path: Path,
) -> None:
    from bazaar.cli import _collect_audit_records

    db = tmp_path / "bad-llm-json.db"
    platform = MarketplacePlatform(db)
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
                    (0, 1, 'm', 'b', 'h1', NULL, '[]', 'ok', '{}',
                     '2026-01-01T00:00:00+00:00')
                """
            )
        platform.conn.execute("PRAGMA ignore_check_constraints = OFF")
    finally:
        platform.close()

    records, _, _, _, state_count = _collect_audit_records(
        db,
        strict_seed_coverage=False,
        include_state_invariants=False,
    )

    assert state_count == 0
    assert not any(record["kind"] == "state_invariant" for record in records)


def test_cli_audit_fails_on_policy_error_event_with_generic_label(
    tmp_path: Path,
) -> None:
    db = tmp_path / "policy-error.db"
    platform = MarketplacePlatform(db)
    try:
        platform.register_agent(generate_persona(1, seed=1))
        event_id = log_event(
            platform.conn,
            tick=0,
            agent_id=1,
            action_type="policy_error",
            payload={
                "source": "policy",
                "error": "policy_action_not_action_type",
            },
            result_status="error",
            result_payload={
                "error": "policy_action_not_action_type",
            },
        )
        platform.conn.commit()
    finally:
        platform.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert any(
        issue["kind"] == "state_invariant"
        and issue["event_id"] == event_id
        and "error_events_absent:" in issue["message"]
        and "action='policy_error'" in issue["message"]
        for issue in payload["issues"]
    )
    assert not any(
        "dispatcher_errors_absent" in issue["message"]
        for issue in payload["issues"]
    )


_LEGACY_SCHEMA = """
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


def test_cli_audit_fails_on_legacy_tables_missing_json_constraints(
    tmp_path: Path,
) -> None:
    """A legacy table without the CHECK constraints fails the audit when
    a row violates them."""
    db = tmp_path / "legacy-schema.db"
    raw = sqlite3.connect(db)
    raw.executescript(_LEGACY_SCHEMA)
    with raw:
        raw.execute(
            "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
            "result_status, result_payload) VALUES (0, 'w', NULL, 'x', '[]', 'ok', NULL)"
        )
        raw.execute(
            "INSERT INTO llm_calls (tick, agent_id, model, backend, prompt_hash, "
            "sampling_params, response_text, tool_calls_json, wall_time) "
            "VALUES (0, 1, 'm', 'b', 'h', '{}', 'r', '{}', 'w')"
        )
    raw.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert any(
        issue["kind"] == "state_invariant"
        and "schema_constraint_missing: table 'events' missing constraint "
        "events.payload_json_object (1 rows violate it)" in issue["message"]
        for issue in payload["issues"]
    )
    assert any(
        issue["kind"] == "state_invariant"
        and "schema_constraint_missing: table 'llm_calls' missing constraint "
        "llm_calls.tool_calls_json_array (1 rows violate it)" in issue["message"]
        for issue in payload["issues"]
    )


def test_cli_audit_warns_on_legacy_tables_whose_rows_satisfy_the_constraints(
    tmp_path: Path,
) -> None:
    """A table created before its CHECK constraints existed cannot gain
    them in place (released databases with ``schema_version`` 0). When
    every row satisfies them the missing constraint is a warning."""
    db = tmp_path / "legacy-schema-ok.db"
    raw = sqlite3.connect(db)
    raw.executescript(_LEGACY_SCHEMA)
    with raw:
        raw.execute(
            "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
            "result_status, result_payload) VALUES (0, 'w', NULL, 'x', '{}', 'ok', NULL)"
        )
        raw.execute(
            "INSERT INTO llm_calls (tick, agent_id, model, backend, prompt_hash, "
            "sampling_params, response_text, tool_calls_json, wall_time) "
            "VALUES (0, 1, 'm', 'b', 'h', '{}', 'r', '[]', 'w')"
        )
    raw.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert not any("schema_constraint_missing" in i["message"] for i in payload["issues"])
    missing = [w for w in payload["warnings"] if "schema_constraint_missing" in w["message"]]
    assert len(missing) == 15
    assert all("legacy schema created before the constraint existed" in w["message"]
               for w in missing)


def test_cli_audit_fails_on_malformed_llm_call_json(tmp_path: Path) -> None:
    db = tmp_path / "bad-llm-json.db"
    platform = MarketplacePlatform(db)
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
                    (0, 1, 'm', 'b', 'h1', NULL, '[]', 'ok', '{}',
                     '2026-01-01T00:00:00+00:00')
                """
            )
        platform.conn.execute("PRAGMA ignore_check_constraints = OFF")
    finally:
        platform.close()

    result = _run_audit(db, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert any(
        issue["kind"] == "state_invariant"
        and "sampling_params for call_id 1 is not a JSON object"
        in issue["message"]
        for issue in payload["issues"]
    )
    assert any(
        issue["kind"] == "state_invariant"
        and "tool_calls_json for call_id 1 is not a JSON array"
        in issue["message"]
        for issue in payload["issues"]
    )

"""Event-log consistency checks for platform-emitted mutations.

The dispatcher already logs agent actions atomically. Platform setup paths
(``register_agent`` and seed helpers) are easier to accidentally drift, so
this module audits those event payloads against the materialized tables.
It also provides a lighter action-event audit: every committed ``ok``
action event that claims a materialized row id must still point at a real
row with stable identity fields intact. This is not a full counterfactual
replay engine; it is a postflight guard against dangling event references
and result payloads that no longer match the state they claim to create.

Audits return issues instead of raising so callers can decide whether a
check is informational or a hard gate. An issue with ``severity='warning'``
reports documented behaviour that the event log explains (the legacy
handoff contract, a legacy schema, the unlogged window of a truncated
base); everything else is an error.

Threat model: the audit checks that the recorded state is explained by the
recorded log. It catches tables edited behind the log's back, meta rows
that disagree with the log, and notes or flags that do not fit the
database. It does not authenticate a database: one whose meta rows and
events were rewritten together, consistently, is indistinguishable from
a genuine run, and the audit makes no attempt to detect that.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from difflib import SequenceMatcher
from typing import Any

from bazaar.core.event_log import VALID_RESULT_STATUSES
from bazaar.core.handoff_checks import (
    COMMITMENT_LOCK_MODE,
    COMPLETION_INTEGRITY_MODE,
    HANDOFF_CHECK_MODES,
    HANDOFF_CHECK_PRESETS,
    HANDOFF_CHECKS_SET_ACTION,
    INSPECTION_TRUTH_MODE,
    QUALITY_BAND_RANGES,
    SHIPMENT_INSPECTION_MODE,
    TRUTHFUL_HANDOFF_CHECKS,
    read_handoff_check,
)
from bazaar.core.schema import missing_required_schema_constraints


@dataclass(frozen=True)
class EventAuditIssue:
    """One audit finding. ``severity`` is ``error`` (a broken invariant)
    or ``warning`` (documented behaviour that the event log explains,
    reported but not a failure; see :func:`audit_marketplace_state_invariants`
    and :func:`audit_platform_event_consistency`)."""

    action_type: str
    message: str
    event_id: int | None = None
    ref_id: int | None = None
    severity: str = "error"


ActionEventChecker = Callable[
    [sqlite3.Connection, sqlite3.Row, dict[str, Any], dict[str, Any], list[EventAuditIssue]],
    None,
]

_NO_MATERIALIZED_ROW_AGENT_ACTIONS = frozenset({
    "do_nothing",
    "refine_search",
    "wait",
})
_KNOWN_NON_DISPATCH_AGENT_EVENTS = frozenset({
    "fraud_discovered",
    "memory_divergence",
    "platform_inventory_restocked",
})


def audit_platform_event_consistency(
    conn: sqlite3.Connection,
    *,
    require_seed_coverage: bool = False,
) -> list[EventAuditIssue]:
    """Return consistency issues for platform-emitted setup events.

    ``require_seed_coverage`` also checks that seed-marked rows have a
    covering platform event. Keep it opt-in because many tiny unit-test
    fixtures insert rows directly without exercising ``MarketplacePlatform``.

    Warnings: a released database whose ``meta`` carries
    ``truncated_base_note`` (its event log was cut at ``target_max_tick``
    while the tables hold state up to ``source_max_tick``) reports the
    persona inventory changes of that unlogged window as warnings that
    cite the note, when the note fits the database (see ``_Truncation``)
    and the change agrees with the tables (a sale through one of the
    agent's listings, a purchase through a completed thread of the agent,
    a restocked copy of one of its units); every other mismatch stays an
    error.
    """
    issues: list[EventAuditIssue] = []
    _audit_register_agent_events(conn, issues)
    _audit_phantom_seed_events(conn, issues)
    _audit_real_seed_events(conn, issues)
    _audit_lot_sale_seed_events(conn, issues)
    if require_seed_coverage:
        _audit_seed_row_coverage(conn, issues)
    return issues


def audit_agent_action_event_consistency(
    conn: sqlite3.Connection,
) -> list[EventAuditIssue]:
    """Return consistency issues for dispatcher-emitted action events.

    Only ``result_status='ok'`` rows are checked: blocked/error events
    intentionally may not materialize any derived rows. The audit avoids
    fields that can legitimately change later in a long rollout (listing
    title/price/status, thread status after a later action, etc.) and
    focuses on durable identity: result ids exist, row ownership matches
    the acting agent, immutable tick/thread/message/offer relationships
    agree with the event payload, and terminal event claims remain true.
    """
    issues: list[EventAuditIssue] = []
    for event in _agent_action_rows(conn):
        action = str(event["action_type"])
        checker = _ACTION_EVENT_CHECKS.get(action)
        if checker is not None:
            checker(conn, event, _payload(event), _result_payload(event), issues)
            continue
        if action in _NO_MATERIALIZED_ROW_AGENT_ACTIONS:
            continue
        if action in _KNOWN_NON_DISPATCH_AGENT_EVENTS:
            continue
        issues.append(EventAuditIssue(
            action_type=action,
            message=(
                "action_event_checker_missing: ok agent event has no "
                "materialized-row checker or explicit no-row exemption"
            ),
            event_id=int(event["event_id"]),
        ))
    return issues


def audit_marketplace_state_invariants(
    conn: sqlite3.Connection,
) -> list[EventAuditIssue]:
    """Return cross-table state-machine invariant violations.

    These are broader than event-payload checks: they assert that the
    materialized marketplace state is internally coherent at audit time.
    The checks intentionally mirror the T23 lifecycle acceptance battery
    and add a few always-true state-machine constraints that are cheap to
    evaluate after every rollout.

    Documented behaviour that the event log explains is reported with
    ``severity='warning'``:

    * legacy completion contract (``completion_integrity_mode`` was not
      ``unit`` when the explaining event ran: ``meta`` has it off and the
      log shows no completion-integrity write, or ``meta`` has it on and
      the event precedes its switch-on point, as for a truthful
      continuation of a legacy base): a legacy sale through another
      thread of the listing, or a legacy ``leave_thread``/``ghost`` of the
      thread, left the meetup of a cancelled or ghosted thread scheduled.
      The explaining event must be logged,
      after the meetup was scheduled, and must itself carry no meetup
      cancellation; otherwise, and always when completion integrity was
      already on, the state is an error;
    * a legacy schema: ``events``/``llm_calls`` created before their
      CHECK constraints existed, when every row satisfies them;
    * an ``error`` event that is a handler's not-found answer to an id
      that did not exist when the event ran (not a handler exception).

    ``handoff_mode_meta_log_agreement`` is an error when the four
    handoff-check rows in ``meta`` disagree with the event log (including
    the latest ``platform_handoff_checks_set`` event, which ``BazaarEnv``
    logs whenever it changes them) or hold a value other than the
    documented ones, and ``scheduled_meetup_logged``
    is an error for a meetup of any status without its logged
    ``schedule_meetup``/``schedule_shipment`` event (a warning only for a
    meetup scheduled in the unlogged window of a truncated base; the
    closed meetups a cold-start world seeds on its seeded listings before
    the first logged tick are its pre-log history and pass).
    """
    issues: list[EventAuditIssue] = []
    modes = _HandoffModes.load(conn)
    modes.report(issues)
    _audit_truncated_base_note(conn, issues)
    _audit_schema_constraints(conn, issues)
    _audit_event_json_payloads(conn, issues)
    _audit_event_result_statuses(conn, issues)
    _audit_llm_call_json_payloads(conn, issues)
    _audit_error_events(conn, issues)
    _audit_offer_thread_state(conn, issues)
    _audit_meetup_thread_listing_state(conn, issues, modes)
    _audit_sold_listing_state(conn, issues)
    _audit_rating_thread_state(conn, issues)
    _audit_phantom_tripwire_events(conn, issues)
    _audit_event_agent_references(conn, issues)
    _audit_memory_divergence_events(conn, issues)
    return issues


def _json_type_is(column: str, json_type: str) -> str:
    return f"CASE WHEN json_valid({column}) THEN json_type({column}) = '{json_type}' ELSE 0 END"


# The row predicate of each CHECK constraint in
# ``bazaar.core.schema.REQUIRED_SCHEMA_CONSTRAINTS``. A table created
# before the constraint existed (a legacy schema, ``schema_version`` 0)
# cannot gain it in place; when every row satisfies the predicate the
# missing constraint is a warning, otherwise an error.
_SCHEMA_CONSTRAINT_PREDICATES: dict[str, str] = {
    "events.tick_integer": "typeof(tick) = 'integer'",
    "events.agent_id_integer": "agent_id IS NULL OR typeof(agent_id) = 'integer'",
    "events.action_type_nonempty": "length(trim(action_type)) > 0",
    "events.payload_json_object": _json_type_is("payload", "object"),
    "events.result_status_enum": "result_status IN ('ok', 'error', 'blocked')",
    "events.result_payload_json_object": (
        "result_payload IS NULL OR " + _json_type_is("result_payload", "object")
    ),
    "llm_calls.tick_integer": "typeof(tick) = 'integer'",
    "llm_calls.agent_id_integer": "typeof(agent_id) = 'integer'",
    "llm_calls.model_nonempty": "length(trim(model)) > 0",
    "llm_calls.backend_nonempty": "length(trim(backend)) > 0",
    "llm_calls.prompt_hash_nonempty": "length(trim(prompt_hash)) > 0",
    "llm_calls.sampling_params_json_object": _json_type_is("sampling_params", "object"),
    "llm_calls.tool_calls_json_array": (
        "tool_calls_json IS NULL OR " + _json_type_is("tool_calls_json", "array")
    ),
    "llm_calls.cache_hit_boolean": "cache_hit IN (0, 1)",
    "llm_calls.latency_ms_nonnegative": "latency_ms IS NULL OR latency_ms >= 0",
}


def _audit_schema_constraints(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for table, label in missing_required_schema_constraints(conn):
        if label.endswith(".table_exists"):
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=f"schema_table_missing: table {table!r} is missing",
            ))
            continue
        message = f"schema_constraint_missing: table {table!r} missing constraint {label}"
        predicate = _SCHEMA_CONSTRAINT_PREDICATES.get(label)
        violations: int | None = None
        if predicate is not None:
            try:
                violations = int(conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE COALESCE(({predicate}), 0) = 0"
                ).fetchone()[0])
            except sqlite3.Error:
                violations = None
        if violations == 0:
            rows = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"{message} (legacy schema created before the constraint "
                    f"existed; all {rows} rows satisfy it)"
                ),
                severity="warning",
            ))
            continue
        if violations is not None:
            message += f" ({violations} rows violate it)"
        issues.append(EventAuditIssue(action_type="state_invariant", message=message))


def _event_rows(conn: sqlite3.Connection, action_type: str) -> list[sqlite3.Row]:
    return list(conn.execute(
        """
        SELECT event_id, action_type, payload, result_payload
        FROM events
        WHERE action_type = ?
        ORDER BY event_id
        """,
        (action_type,),
    ))


def _agent_action_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        """
        SELECT event_id, tick, agent_id, action_type, payload, result_payload
        FROM events
        WHERE result_status = 'ok' AND agent_id IS NOT NULL
        ORDER BY event_id
        """
    ))


def _payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload"] or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _result_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["result_payload"] or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _audit_event_json_payloads(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT event_id, action_type, payload, result_payload
        FROM events
        ORDER BY event_id
        """
    ):
        _audit_json_object_cell(issues, row, "payload", required=True)
        _audit_json_object_cell(issues, row, "result_payload", required=False)


def _audit_json_object_cell(
    issues: list[EventAuditIssue],
    row: sqlite3.Row,
    column: str,
    *,
    required: bool,
) -> None:
    raw = row[column]
    if raw is None:
        if required:
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    "event_json_required: "
                    f"{column} for action {row['action_type']!r} is NULL"
                ),
                event_id=int(row["event_id"]),
            ))
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                "event_json_valid: "
                f"{column} for action {row['action_type']!r} is malformed JSON"
            ),
            event_id=int(row["event_id"]),
        ))
        return
    if not isinstance(payload, dict):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                "event_json_object: "
                f"{column} for action {row['action_type']!r} is not a JSON object"
            ),
            event_id=int(row["event_id"]),
        ))


def _audit_event_result_statuses(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    placeholders = ", ".join("?" for _ in VALID_RESULT_STATUSES)
    for row in conn.execute(
        f"""
        SELECT event_id, action_type, result_status
        FROM events
        WHERE result_status NOT IN ({placeholders})
        ORDER BY event_id
        """,
        tuple(sorted(VALID_RESULT_STATUSES)),
    ):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                "event_result_status_valid: "
                f"event {row['event_id']} action={row['action_type']!r} "
                f"has invalid result_status={row['result_status']!r}"
            ),
            event_id=int(row["event_id"]),
        ))


def _audit_llm_call_json_payloads(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    try:
        rows = conn.execute(
            """
            SELECT call_id, sampling_params, tool_calls_json
            FROM llm_calls
            ORDER BY call_id
            """
        )
    except sqlite3.OperationalError:
        return
    for row in rows:
        _audit_json_shape_cell(
            issues,
            row,
            id_column="call_id",
            column="sampling_params",
            required=True,
            expected_type=dict,
            expected_label="JSON object",
            issue_prefix="llm_call_json",
        )
        _audit_json_shape_cell(
            issues,
            row,
            id_column="call_id",
            column="tool_calls_json",
            required=False,
            expected_type=list,
            expected_label="JSON array",
            issue_prefix="llm_call_json",
        )


def _audit_json_shape_cell(
    issues: list[EventAuditIssue],
    row: sqlite3.Row,
    *,
    id_column: str,
    column: str,
    required: bool,
    expected_type: type,
    expected_label: str,
    issue_prefix: str,
) -> None:
    ref_id = int(row[id_column])
    raw = row[column]
    if raw is None:
        if required:
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"{issue_prefix}_required: "
                    f"{column} for {id_column} {ref_id} is NULL"
                ),
                ref_id=ref_id,
            ))
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                f"{issue_prefix}_valid: "
                f"{column} for {id_column} {ref_id} is malformed JSON"
            ),
            ref_id=ref_id,
        ))
        return
    if not isinstance(payload, expected_type):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                f"{issue_prefix}_shape: "
                f"{column} for {id_column} {ref_id} is not a {expected_label}"
            ),
            ref_id=ref_id,
        ))


def _add(
    issues: list[EventAuditIssue],
    row: sqlite3.Row | None,
    action_type: str,
    message: str,
    *,
    ref_id: int | None = None,
    severity: str = "error",
) -> None:
    issues.append(EventAuditIssue(
        action_type=action_type,
        message=message,
        event_id=None if row is None else int(row["event_id"]),
        ref_id=ref_id,
        severity=severity,
    ))


def _event_agent_id(row: sqlite3.Row) -> int | None:
    value = row["agent_id"]
    return None if value is None else int(value)


def _event_tick(row: sqlite3.Row) -> int:
    return int(row["tick"])


def _int_field(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expect_row(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    table: str,
    id_col: str,
    id_value: Any,
    *,
    label: str | None = None,
) -> sqlite3.Row | None:
    action = str(event["action_type"])
    ref_label = label or id_col
    ref_id = _coerce_int(id_value)
    if ref_id is None:
        _add(issues, event, action, f"missing {ref_label}")
        return None
    row = conn.execute(
        f"SELECT * FROM {table} WHERE {id_col} = ?",
        (ref_id,),
    ).fetchone()
    if row is None:
        _add(issues, event, action, f"{table} row missing", ref_id=ref_id)
        return None
    return row


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _Missing:
    """Marks a JSON key that is absent, as opposed to present with null."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Any = _Missing()


def _get(data: dict[str, Any], key: str) -> Any:
    """``data[key]``, or ``_MISSING`` when the key is absent: a null-valued
    key and a missing key are different persona states."""
    return data[key] if key in data else _MISSING


def _expect_equal(
    issues: list[EventAuditIssue],
    row: sqlite3.Row,
    action_type: str,
    ref_id: int,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        _add(
            issues, row, action_type,
            f"{field} mismatch: event={expected!r} table={actual!r}",
            ref_id=ref_id,
        )


def _json_key(column: str, key: str) -> str:
    """SQL for ``column``'s top-level ``key`` that never raises on
    malformed JSON (NULL then)."""
    return f"CASE WHEN json_valid({column}) THEN json_extract({column}, '$.{key}') END"


def _json_has(column: str, key: str) -> str:
    """SQL: ``column`` is a JSON object that carries ``key`` (null or not)."""
    return (
        f"(CASE WHEN json_valid({column}) "
        f"THEN json_type({column}, '$.{key}') IS NOT NULL ELSE 0 END)"
    )


def _expect_at_least(
    issues: list[EventAuditIssue],
    row: sqlite3.Row,
    action_type: str,
    ref_id: int,
    field: str,
    expected_floor: int | None,
    actual: Any,
) -> None:
    if expected_floor is None or actual is None:
        return
    if int(actual) < int(expected_floor):
        _add(
            issues, row, action_type,
            f"{field} regressed: event>={expected_floor!r} table={actual!r}",
            ref_id=ref_id,
        )


def _check_listing_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    listing_id = _int_field(result, "listing_id")
    row = _expect_row(conn, issues, event, "listings", "listing_id", listing_id)
    if row is None or listing_id is None:
        return
    _expect_equal(
        issues, event, action, listing_id, "owner_agent_id",
        _event_agent_id(event), row["owner_agent_id"],
    )
    _expect_equal(
        issues, event, action, listing_id, "created_at_tick",
        _event_tick(event), row["created_at_tick"],
    )
    _expect_equal(
        issues, event, action, listing_id, "is_phantom",
        False, bool(row["is_phantom"]),
    )
    _expect_equal(
        issues, event, action, listing_id, "category",
        payload.get("category"), row["category"],
    )


def _check_listing_result_exists(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    listing_id = _int_field(result, "listing_id") or _int_field(payload, "listing_id")
    _expect_row(conn, issues, event, "listings", "listing_id", listing_id)


def _check_owned_listing_result(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    listing_id = _int_field(result, "listing_id") or _int_field(payload, "listing_id")
    row = _expect_row(conn, issues, event, "listings", "listing_id", listing_id)
    if row is None or listing_id is None:
        return
    _expect_equal(
        issues, event, action, listing_id, "owner_agent_id",
        _event_agent_id(event), row["owner_agent_id"],
    )


def _check_bump_listing(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    _check_owned_listing_result(conn, event, payload, result, issues)
    listing_id = _int_field(result, "listing_id")
    if listing_id is None:
        return
    row = conn.execute(
        "SELECT last_bumped_tick FROM listings WHERE listing_id = ?",
        (listing_id,),
    ).fetchone()
    if row is not None:
        _expect_at_least(
            issues, event, str(event["action_type"]), listing_id,
            "last_bumped_tick", _int_field(result, "last_bumped_tick"),
            row["last_bumped_tick"],
        )


def _check_message_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    message_id = _int_field(result, "message_id")
    row = _expect_row(conn, issues, event, "messages", "message_id", message_id)
    if row is None or message_id is None:
        return
    expected_thread_id = (
        _int_field(result, "thread_id")
        or _int_field(payload, "thread_id")
    )
    if expected_thread_id is not None:
        _expect_equal(
            issues, event, action, message_id, "thread_id",
            expected_thread_id, row["thread_id"],
        )
    _expect_equal(
        issues, event, action, message_id, "sender_agent_id",
        _event_agent_id(event), row["sender_agent_id"],
    )
    _expect_equal(
        issues, event, action, message_id, "tick",
        _event_tick(event), row["tick"],
    )
    if action == "message":
        _expect_equal(
            issues, event, action, message_id, "body",
            payload.get("body"), row["body"],
        )
    content_hash = result.get("content_hash")
    if content_hash is not None:
        _expect_equal(
            issues, event, action, message_id, "content_hash",
            content_hash, row["content_hash"],
        )


def _check_offer_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    offer_id = _int_field(result, "offer_id")
    row = _expect_row(conn, issues, event, "offers", "offer_id", offer_id)
    if row is None or offer_id is None:
        return
    _expect_equal(
        issues, event, action, offer_id, "thread_id",
        _int_field(result, "thread_id"), row["thread_id"],
    )
    _expect_equal(
        issues, event, action, offer_id, "proposer_id",
        _event_agent_id(event), row["proposer_id"],
    )
    _expect_equal(
        issues, event, action, offer_id, "price_cents",
        _int_field(payload, "price_cents"), row["price_cents"],
    )
    _expect_equal(
        issues, event, action, offer_id, "round",
        _int_field(result, "round"), row["round"],
    )
    _expect_equal(
        issues, event, action, offer_id, "tick",
        _event_tick(event), row["tick"],
    )


def _check_offer_status_event(
    expected_status: str | None = None,
) -> ActionEventChecker:
    def _check(
        conn: sqlite3.Connection,
        event: sqlite3.Row,
        payload: dict[str, Any],
        result: dict[str, Any],
        issues: list[EventAuditIssue],
    ) -> None:
        action = str(event["action_type"])
        offer_id = _int_field(result, "offer_id") or _int_field(payload, "offer_id")
        row = _expect_row(conn, issues, event, "offers", "offer_id", offer_id)
        if row is None or offer_id is None:
            return
        _expect_equal(
            issues, event, action, offer_id, "thread_id",
            _int_field(result, "thread_id"), row["thread_id"],
        )
        thread = _expect_row(
            conn, issues, event, "threads", "thread_id", row["thread_id"],
            label="thread_id",
        )
        if thread is not None:
            _expect_participant(issues, event, int(row["thread_id"]), thread)
        if expected_status is not None:
            _expect_equal(
                issues, event, action, offer_id, "offer_status",
                expected_status, row["status"],
            )
    return _check


def _expect_participant(
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    ref_id: int,
    thread: sqlite3.Row,
) -> None:
    agent_id = _event_agent_id(event)
    if agent_id not in (thread["buyer_agent_id"], thread["seller_agent_id"]):
        _add(
            issues, event, str(event["action_type"]),
            "event agent is not a thread participant",
            ref_id=ref_id,
        )


def _check_delivery_scheduled(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    meetup_id = _int_field(result, "meetup_id")
    row = _expect_row(conn, issues, event, "meetups", "meetup_id", meetup_id)
    if row is None or meetup_id is None:
        return
    _expect_equal(
        issues, event, action, meetup_id, "thread_id",
        _int_field(result, "thread_id"), row["thread_id"],
    )
    _expect_equal(
        issues, event, action, meetup_id, "scheduled_tick",
        _int_field(result, "scheduled_tick"), row["scheduled_tick"],
    )
    _expect_equal(
        issues, event, action, meetup_id, "payment_method",
        result.get("payment_method"), row["payment_method"],
    )
    actual_delivery = row["delivery_method"] or "meetup"
    _expect_equal(
        issues, event, action, meetup_id, "delivery_method",
        result.get("delivery_method") or "meetup", actual_delivery,
    )
    if result.get("delivery_method") == "ship":
        _expect_equal(
            issues, event, action, meetup_id, "delivered_at_tick",
            _int_field(result, "delivered_at_tick"), row["delivered_at_tick"],
        )
    thread = _expect_row(
        conn, issues, event, "threads", "thread_id", row["thread_id"],
        label="thread_id",
    )
    if thread is not None:
        _expect_participant(issues, event, int(row["thread_id"]), thread)


def _check_inspect_at_meetup(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    meetup_id = _int_field(result, "meetup_id") or _int_field(payload, "meetup_id")
    row = _expect_row(conn, issues, event, "meetups", "meetup_id", meetup_id)
    if row is None or meetup_id is None:
        return
    _expect_equal(
        issues, event, action, meetup_id, "thread_id",
        _int_field(result, "thread_id"), row["thread_id"],
    )
    _expect_equal(
        issues, event, action, meetup_id, "buyer_inspected_quality_pct",
        _int_field(result, "ground_truth_quality_pct"),
        row["buyer_inspected_quality_pct"],
    )


def _check_complete_transaction(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    meetup_id = _int_field(result, "meetup_id") or _int_field(payload, "meetup_id")
    row = _expect_row(conn, issues, event, "meetups", "meetup_id", meetup_id)
    if row is None or meetup_id is None:
        return
    _expect_equal(
        issues, event, action, meetup_id, "thread_id",
        _int_field(result, "thread_id"), row["thread_id"],
    )
    thread = _expect_row(
        conn, issues, event, "threads", "thread_id", row["thread_id"],
        label="thread_id",
    )
    if thread is not None:
        _expect_participant(issues, event, int(row["thread_id"]), thread)
        if bool(result.get("completed")):
            _expect_equal(
                issues, event, action, meetup_id, "thread_status",
                "completed", thread["status"],
            )
    if bool(result.get("buyer_confirmed")):
        _expect_equal(
            issues, event, action, meetup_id, "buyer_confirmed",
            True, bool(row["buyer_confirmed"]),
        )
    if bool(result.get("seller_confirmed")):
        _expect_equal(
            issues, event, action, meetup_id, "seller_confirmed",
            True, bool(row["seller_confirmed"]),
        )
    if bool(result.get("completed")):
        _expect_equal(
            issues, event, action, meetup_id, "meetup_status",
            "completed", row["status"],
        )


def _check_cancel_meetup(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    meetup_id = _int_field(result, "meetup_id") or _int_field(payload, "meetup_id")
    row = _expect_row(conn, issues, event, "meetups", "meetup_id", meetup_id)
    if row is None or meetup_id is None:
        return
    _expect_equal(issues, event, action, meetup_id, "meetup_status", "cancelled", row["status"])
    thread = _expect_row(
        conn, issues, event, "threads", "thread_id", row["thread_id"],
        label="thread_id",
    )
    if thread is not None:
        _expect_equal(
            issues, event, action, int(row["thread_id"]),
            "thread_status", "cancelled", thread["status"],
        )


def _check_rating_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    rating_id = _int_field(result, "rating_id")
    row = _expect_row(conn, issues, event, "ratings", "rating_id", rating_id)
    if row is None or rating_id is None:
        return
    _expect_equal(
        issues, event, action, rating_id, "rater_agent_id",
        _event_agent_id(event), row["rater_agent_id"],
    )
    for field in ("ratee_agent_id", "stars", "thread_id"):
        _expect_equal(
            issues, event, action, rating_id, field,
            payload.get(field), row[field],
        )
    _expect_equal(issues, event, action, rating_id, "tick", _event_tick(event), row["tick"])


def _check_report_created(kind: str) -> ActionEventChecker:
    def _check(
        conn: sqlite3.Connection,
        event: sqlite3.Row,
        payload: dict[str, Any],
        result: dict[str, Any],
        issues: list[EventAuditIssue],
    ) -> None:
        action = str(event["action_type"])
        report_id = _int_field(result, "report_id")
        row = _expect_row(conn, issues, event, "reports", "report_id", report_id)
        if row is None or report_id is None:
            return
        target_key = "listing_id" if kind == "listing" else "user_agent_id"
        _expect_equal(
            issues, event, action, report_id, "reporter_id",
            _event_agent_id(event), row["reporter_id"],
        )
        _expect_equal(issues, event, action, report_id, "target_kind", kind, row["target_kind"])
        _expect_equal(
            issues, event, action, report_id, "target_id",
            payload.get(target_key), row["target_id"],
        )
        _expect_equal(
            issues, event, action, report_id, "reason",
            payload.get("reason"), row["reason"],
        )
    return _check


def _check_block_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    blocked_id = _int_field(result, "blocked_id") or _int_field(payload, "user_agent_id")
    row = conn.execute(
        """
        SELECT block_id, tick
        FROM blocks
        WHERE blocker_id = ? AND blocked_id = ?
        """,
        (_event_agent_id(event), blocked_id),
    ).fetchone()
    if row is None:
        _add(issues, event, action, "blocks row missing", ref_id=blocked_id)
        return
    _expect_equal(
        issues, event, action, int(row["block_id"]),
        "tick", _event_tick(event), row["tick"],
    )


def _check_thread_terminal(expected_status: str) -> ActionEventChecker:
    def _check(
        conn: sqlite3.Connection,
        event: sqlite3.Row,
        payload: dict[str, Any],
        result: dict[str, Any],
        issues: list[EventAuditIssue],
    ) -> None:
        action = str(event["action_type"])
        thread_id = _int_field(result, "thread_id") or _int_field(payload, "thread_id")
        row = _expect_row(conn, issues, event, "threads", "thread_id", thread_id)
        if row is None or thread_id is None:
            return
        _expect_participant(issues, event, thread_id, row)
        if expected_status == "ghosted" and row["status"] == "cancelled":
            cancel = _legacy_ghost_then_cancel(conn, event, result, thread_id)
            if cancel is not None:
                _add(
                    issues, event, action,
                    "thread_status mismatch: event='ghosted' table='cancelled' "
                    "(documented legacy behaviour: completion_integrity_mode is not "
                    "'unit', so this ghost left the thread's meetup scheduled, and "
                    f"cancel_meetup event {cancel} then cancelled it and the thread)",
                    ref_id=thread_id,
                    severity="warning",
                )
                return
        _expect_equal(
            issues, event, action, thread_id, "thread_status",
            expected_status, row["status"],
        )
    return _check


def _legacy_ghost_then_cancel(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    result: dict[str, Any],
    thread_id: int,
) -> int | None:
    """The ``cancel_meetup`` event that explains a ghosted thread now
    being ``cancelled``, or None.

    ``cancel_meetup`` cancels the meetup and its thread whatever the
    thread status. Under the legacy contract ``ghost`` leaves the
    thread's meetup scheduled, so a later ``cancel_meetup`` turns the
    ghosted thread into a cancelled one. Under
    ``completion_integrity_mode=unit`` or ``commitment_lock_mode=listing``
    ``ghost`` cancels that meetup (``cancelled_meetup_ids``) and the later
    ``cancel_meetup`` is blocked, so the explanation needs a legacy-shaped
    ``ghost``, a later ``ok`` ``cancel_meetup`` on the thread and a legacy
    completion contract (``meta`` and log)."""
    if "cancelled_meetup_ids" in result:
        return None
    row = conn.execute(
        f"""
        SELECT MIN(event_id) FROM events
        WHERE action_type = 'cancel_meetup' AND result_status = 'ok'
          AND event_id > ? AND {_json_key('result_payload', 'thread_id')} = ?
        """,
        (int(event["event_id"]), thread_id),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    if not _HandoffModes.load(conn).integrity_legacy:
        return None
    return int(row[0])


def _check_thread_participant_event(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    thread_id = _int_field(result, "thread_id") or _int_field(payload, "thread_id")
    row = _expect_row(conn, issues, event, "threads", "thread_id", thread_id)
    if row is not None and thread_id is not None:
        _expect_participant(issues, event, thread_id, row)


def _check_photo_created(
    expected_type: str,
    *,
    expect_stock: bool | None = None,
) -> ActionEventChecker:
    def _check(
        conn: sqlite3.Connection,
        event: sqlite3.Row,
        payload: dict[str, Any],
        result: dict[str, Any],
        issues: list[EventAuditIssue],
    ) -> None:
        action = str(event["action_type"])
        photo_id = _int_field(result, "photo_id")
        photo = _expect_row(conn, issues, event, "photos", "photo_id", photo_id)
        if photo is None or photo_id is None:
            return
        _expect_equal(
            issues, event, action, photo_id, "sender_agent_id",
            _event_agent_id(event), photo["sender_agent_id"],
        )
        _expect_equal(
            issues, event, action, photo_id, "photo_type",
            expected_type, photo["photo_type"],
        )
        if expect_stock is not None:
            _expect_equal(
                issues, event, action, photo_id, "is_stock",
                expect_stock, bool(photo["is_stock"]),
            )
        if "listing_id" in payload:
            _expect_equal(
                issues, event, action, photo_id, "listing_id",
                payload.get("listing_id"), photo["listing_id"],
            )
        _check_message_created(conn, event, payload, result, issues)
        message_id = _int_field(result, "message_id")
        if message_id is None:
            return
        message = conn.execute(
            "SELECT photo_id FROM messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if message is not None:
            _expect_equal(
                issues, event, action, message_id, "message_photo_id",
                photo_id, message["photo_id"],
            )
    return _check


def _check_inspect_photo(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    photo_id = _int_field(result, "photo_id") or _int_field(payload, "photo_id")
    row = _expect_row(conn, issues, event, "photos", "photo_id", photo_id)
    if row is None or photo_id is None:
        return
    for field in ("photo_type", "sender_agent_id", "listing_id"):
        _expect_equal(
            issues, event, action, photo_id, field,
            result.get(field), row[field],
        )
    _expect_equal(
        issues, event, action, photo_id, "is_stock",
        bool(result.get("is_stock")), bool(row["is_stock"]),
    )


def _check_memory_created(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    action = str(event["action_type"])
    memory_id = _int_field(result, "memory_id")
    row = _expect_row(
        conn, issues, event, "narrative_memories", "memory_id", memory_id,
    )
    if row is None or memory_id is None:
        return
    _expect_equal(
        issues, event, action, memory_id, "agent_id",
        _event_agent_id(event), row["agent_id"],
    )
    _expect_equal(
        issues, event, action, memory_id, "scope",
        result.get("scope") or payload.get("scope"), row["scope"],
    )
    _expect_equal(
        issues, event, action, memory_id, "content",
        payload.get("content"), row["content"],
    )
    _expect_equal(
        issues, event, action, memory_id, "created_tick",
        _event_tick(event), row["created_tick"],
    )
    if action == "quote_agent_note":
        _expect_equal(
            issues, event, action, memory_id, "provenance",
            result.get("provenance"), row["provenance"],
        )


def _check_recall_hits(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    hits = result.get("hits")
    if not isinstance(hits, list):
        return
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        memory_id = _int_field(hit, "memory_id")
        _expect_row(
            conn, issues, event, "narrative_memories", "memory_id", memory_id,
        )


def _check_view_profile(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    agent_id = _int_field(result, "agent_id") or _int_field(payload, "user_agent_id")
    _expect_row(conn, issues, event, "agents", "agent_id", agent_id)


def _check_listing_previews(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    previews = result.get("hit_preview")
    if not isinstance(previews, list):
        return
    for item in previews:
        if not isinstance(item, dict):
            continue
        listing_id = _int_field(item, "listing_id")
        _expect_row(conn, issues, event, "listings", "listing_id", listing_id)


def _check_compare(
    conn: sqlite3.Connection,
    event: sqlite3.Row,
    payload: dict[str, Any],
    result: dict[str, Any],
    issues: list[EventAuditIssue],
) -> None:
    compared = result.get("compared")
    if not isinstance(compared, list):
        return
    for listing_id in compared:
        _expect_row(conn, issues, event, "listings", "listing_id", listing_id)


_ACTION_EVENT_CHECKS: dict[str, ActionEventChecker] = {
    "search": _check_listing_previews,
    "browse_category": _check_listing_previews,
    "view_listing": _check_listing_result_exists,
    "pin": _check_listing_result_exists,
    "unpin": _check_listing_result_exists,
    "compare": _check_compare,
    "create_listing": _check_listing_created,
    "edit_listing": _check_owned_listing_result,
    "bump_listing": _check_bump_listing,
    "mark_sold": _check_owned_listing_result,
    "relist": _check_owned_listing_result,
    "message": _check_message_created,
    "send_photo": _check_photo_created("A"),
    "send_crafted_photo": _check_photo_created("B"),
    "send_stock_photo": _check_photo_created("C", expect_stock=True),
    "request_photo": _check_message_created,
    "inspect_photo": _check_inspect_photo,
    "read": _check_thread_participant_event,
    "leave_thread": _check_thread_terminal("cancelled"),
    "ghost": _check_thread_terminal("ghosted"),
    "make_offer": _check_offer_created,
    "counter_offer": _check_offer_created,
    "accept_offer": _check_offer_status_event("accepted"),
    "withdraw_offer": _check_offer_status_event("withdrawn"),
    "schedule_meetup": _check_delivery_scheduled,
    "schedule_shipment": _check_delivery_scheduled,
    "inspect_at_meetup": _check_inspect_at_meetup,
    "complete_transaction": _check_complete_transaction,
    "cancel_meetup": _check_cancel_meetup,
    "rate": _check_rating_created,
    "report_listing": _check_report_created("listing"),
    "report_user": _check_report_created("user"),
    "block_user": _check_block_created,
    "view_profile": _check_view_profile,
    "summarize_session": _check_memory_created,
    "recall": _check_recall_hits,
    "quote_agent_note": _check_memory_created,
}


def _add_state_issue(
    issues: list[EventAuditIssue],
    invariant: str,
    message: str,
    *,
    ref_id: int | None = None,
    severity: str = "error",
) -> None:
    issues.append(EventAuditIssue(
        action_type="state_invariant",
        message=f"{invariant}: {message}",
        ref_id=ref_id,
        severity=severity,
    ))


# ---------------------------------------------------------------------------
# handoff-check modes: meta rows versus the event log
#
# The four truthful-handoff flags live in ``meta`` and are rewritten on
# every ``BazaarEnv`` construction. Each non-legacy flag also leaves
# traces in the event log (result keys or blocked errors no legacy handler
# writes), and each legacy-shaped event proves some flags were off when
# it ran. The audit trusts ``meta`` only where the log agrees with it.
# ---------------------------------------------------------------------------

_INTEGRITY = COMPLETION_INTEGRITY_MODE
_INSPECTION = INSPECTION_TRUTH_MODE
_LOCK = COMMITMENT_LOCK_MODE
_SHIPMENT = SHIPMENT_INSPECTION_MODE
_UNIT_TRANSFER = frozenset({_INTEGRITY, _INSPECTION})
_SALE_CANCELS = frozenset({_INTEGRITY, _INSPECTION, _LOCK})
_LEAVE_CANCELS = frozenset({_INTEGRITY, _LOCK})
_MODE_EVIDENCE_ACTIONS = (
    "complete_transaction", "mark_sold", "leave_thread", "ghost",
    "create_listing", "relist", "inspect_at_meetup",
    "accept_offer", "schedule_meetup", "schedule_shipment",
)


def _mode_evidence(
    action: str, status: str, result: dict[str, Any], delivery: str | None = None,
) -> list[tuple[str, frozenset[str]]]:
    """``(label, flags)`` for every trace of a non-legacy handoff check in
    one event: ``flags`` are the checks whose handlers write it (see
    ``bazaar.actions.handlers``); a legacy handler never does.
    ``delivery`` is the delivery method of the event's meetup."""
    found: list[tuple[str, frozenset[str]]] = []

    def keys(names: tuple[str, ...], flags: frozenset[str]) -> None:
        found.extend((f"{action}.{name}", flags) for name in names if name in result)

    if status == "ok":
        if action == "complete_transaction":
            keys(("consumed_unit", "consumed_unit_uid", "buyer_unit_index", "bound_at_sale"),
                 _UNIT_TRANSFER)
            keys(("cancelled_sister_meetup_ids",), _SALE_CANCELS)
            # _completion_unit_block binds an unbound listing only when
            # completion integrity is on (bound_only otherwise).
            keys(("bound_at_handoff",), frozenset({_INTEGRITY}))
            if delivery == "ship" and result.get("buyer_inspected_quality_pct") is not None:
                # Only shipment_inspection_mode=on_arrival lets the buyer
                # inspect a shipment.
                found.append((
                    "complete_transaction.buyer_inspected_quality_pct on a shipment",
                    frozenset({_SHIPMENT}),
                ))
        elif action == "mark_sold":
            keys(("cancelled_meetup_ids",), _SALE_CANCELS)
            keys(("consumed_unit", "consumed_unit_uid", "bound_at_sale"), _UNIT_TRANSFER)
        elif action in ("leave_thread", "ghost"):
            keys(("cancelled_meetup_ids",), _LEAVE_CANCELS)
        elif action in ("create_listing", "relist"):
            keys(("backing_unit", "released_unit_uid"), frozenset({_INSPECTION}))
        elif action == "inspect_at_meetup":
            keys(("inspection_outcome", "bound_at_handoff", "backing_unit_uid"),
                 frozenset({_INSPECTION}))
            if delivery == "ship":
                found.append(("inspect_at_meetup of a shipment", frozenset({_SHIPMENT})))
    elif status == "blocked":
        error = result.get("error")
        label = f"{action}.blocked:{error}"
        if action == "complete_transaction":
            if error == "thread_not_active":
                found.append((label, frozenset({_INTEGRITY})))
            elif error == "listing_already_sold":
                found.append((label, _SALE_CANCELS))
            elif error == "item_not_present":
                presence = result.get("item_presence")
                if presence == "listing_has_no_bound_unit":
                    found.append((f"{label}/{presence}", frozenset({_INTEGRITY})))
                elif "item_presence" in result:
                    found.append((f"{label}/{presence}", _UNIT_TRANSFER))
                else:
                    found.append((label, frozenset({_INSPECTION})))
            elif error == "before_delivery":
                found.append((label, frozenset({_SHIPMENT})))
            elif error == "must_inspect_first" and delivery == "ship":
                found.append((f"{label} on a shipment", frozenset({_SHIPMENT})))
        elif action == "inspect_at_meetup" and error == "before_delivery":
            found.append((label, frozenset({_SHIPMENT})))
        elif error == "listing_already_committed":
            found.append((label, frozenset({_LOCK})))
    return found


def _legacy_evidence(
    action: str, status: str, result: dict[str, Any], delivery: str | None = None,
) -> list[tuple[str, frozenset[str]]]:
    """``(label, flags)`` for a legacy-shaped event: every check in
    ``flags`` was off when it ran, since each would have written the key
    the event lacks (or blocked the event). ``delivery`` is the delivery
    method of the event's meetup."""
    found: list[tuple[str, frozenset[str]]] = []
    if status == "blocked":
        if (
            action == "inspect_at_meetup" and delivery == "ship"
            and result.get("error") == "not_a_meetup_delivery"
        ):
            found.append((
                "inspect_at_meetup of a shipment blocked as not_a_meetup_delivery",
                frozenset({_SHIPMENT}),
            ))
        return found
    if status != "ok":
        return found
    if action == "complete_transaction":
        if result.get("completed") is True:
            if "consumed_unit" not in result:
                found.append(("complete_transaction without consumed_unit", _UNIT_TRANSFER))
            if "cancelled_sister_meetup_ids" not in result:
                found.append((
                    "complete_transaction without cancelled_sister_meetup_ids", _SALE_CANCELS,
                ))
        if (
            delivery == "ship" and result.get("buyer_confirmed") is True
            and "buyer_inspected_quality_pct" in result
            and result["buyer_inspected_quality_pct"] is None
        ):
            # on_arrival refuses the buyer's confirmation of an
            # uninspected shipment (must_inspect_first).
            found.append((
                "complete_transaction of a shipment the buyer confirmed without inspection",
                frozenset({_SHIPMENT}),
            ))
    elif action == "mark_sold":
        if "cancelled_meetup_ids" not in result:
            found.append(("mark_sold without cancelled_meetup_ids", _SALE_CANCELS))
        if "consumed_unit" not in result:
            found.append(("mark_sold without consumed_unit", _UNIT_TRANSFER))
    elif action in ("leave_thread", "ghost") and "cancelled_meetup_ids" not in result:
        found.append((f"{action} without cancelled_meetup_ids", _LEAVE_CANCELS))
    elif action == "inspect_at_meetup" and "inspection_outcome" not in result:
        found.append(("inspect_at_meetup without inspection_outcome", frozenset({_INSPECTION})))
    elif action == "create_listing" and "backing_unit" not in result:
        found.append(("create_listing without backing_unit", frozenset({_INSPECTION})))
    return found


def _meetup_deliveries(conn: sqlite3.Connection) -> dict[int, str]:
    """``meetup_id -> delivery_method`` (``meetup`` when unset)."""
    try:
        return {
            int(row[0]): str(row[1] or "meetup")
            for row in conn.execute("SELECT meetup_id, delivery_method FROM meetups")
        }
    except sqlite3.Error:
        return {}


def _event_delivery(
    deliveries: dict[int, str], payload: dict[str, Any], result: dict[str, Any],
) -> str | None:
    method = result.get("delivery_method")
    if isinstance(method, str):
        return method
    meetup_id = _int_field(result, "meetup_id")
    if meetup_id is None:
        meetup_id = _int_field(payload, "meetup_id")
    return None if meetup_id is None else deliveries.get(meetup_id)


def _meta_json(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else _json_object(row[0])


def _cell_fork(conn: sqlite3.Connection) -> tuple[int, dict[str, str]] | None:
    """``(fork_tick, handoff checks of its preset)`` that the rollout
    launcher recorded in ``meta.trapi_matrix_cell`` for a continuation,
    or None."""
    cell = _meta_json(conn, "trapi_matrix_cell")
    if cell is None:
        return None
    fork = cell.get("fork_tick")
    if isinstance(fork, bool) or not isinstance(fork, int):
        return None
    preset = HANDOFF_CHECK_PRESETS.get(str(cell.get("handoff_checks", "legacy")))
    return fork, dict(preset or {})


@dataclass(frozen=True)
class _SwitchOn:
    """Where a handoff check that ``meta`` has on was switched on: every
    event after ``after`` (an event id) ran under it, and event ``after``
    too when ``traced`` (a point set by an event that carries a trace of
    the check)."""

    after: int
    why: str
    traced: bool = False


def _switch_on_point(
    conn: sqlite3.Connection,
    flag: str,
    configs: list[tuple[int, dict[str, str]]],
    first_trace: int | None,
) -> _SwitchOn:
    """The point from which ``flag`` (on in ``meta``) held, when no
    ``platform_handoff_checks_set`` event places it (see
    :func:`_marker_switch_on`, which takes precedence):

    * the first of the trailing ``experiment_config`` events that declare
      it on, when the last one does;
    * else the fork of a continuation whose ``meta.trapi_matrix_cell``
      preset turns it on (the launcher skips ``experiment_config``);
    * else the first event carrying a trace of the check (a switch-on on
      resume, as for a truthful continuation of a legacy base);
    * else the start of the log.
    """
    on = TRUTHFUL_HANDOFF_CHECKS[flag]
    if configs and configs[-1][1].get(flag) == on:
        first = configs[-1][0]
        for config_id, declared in reversed(configs):
            if declared.get(flag) != on:
                break
            first = config_id
        return _SwitchOn(first, f"experiment_config event {first}, which declared it on")
    fork = _cell_fork(conn)
    if fork is not None and fork[1].get(flag) == on:
        row = conn.execute(
            "SELECT MIN(event_id) FROM events WHERE tick > ?", (fork[0],),
        ).fetchone()
        after = (int(row[0]) - 1) if row is not None and row[0] is not None else 1 << 62
        return _SwitchOn(
            after, f"the fork at tick {fork[0]} (meta.trapi_matrix_cell preset)",
        )
    if first_trace is not None:
        return _SwitchOn(
            first_trace, f"event {first_trace}, its first trace in the log", traced=True,
        )
    return _SwitchOn(
        0, "the start of the log (no experiment_config event declares it and the log "
        "carries no trace of it)",
    )


@dataclass(frozen=True)
class _ChecksSet:
    """One ``platform_handoff_checks_set`` event: the handoff checks
    before it (``old``, legacy defaults for missing or unknown values) and
    after it (``new``, as logged; ``new_effective`` read like ``meta``)."""

    event_id: int
    old: dict[str, str]
    new: dict[str, Any]
    new_effective: dict[str, str]


def _effective_checks(values: dict[str, Any]) -> dict[str, str]:
    return {
        key: (values[key] if values.get(key) in modes else modes[0])
        for key, modes in HANDOFF_CHECK_MODES.items()
    }


def _handoff_check_markers(
    conn: sqlite3.Connection, issues: list[EventAuditIssue],
) -> list[_ChecksSet]:
    """The ``platform_handoff_checks_set`` events in log order. One
    without an ``old`` and a ``new`` object is an error and is skipped."""
    markers: list[_ChecksSet] = []
    for row in conn.execute(
        """
        SELECT event_id, payload FROM events
        WHERE action_type = ? AND result_status = 'ok' ORDER BY event_id
        """,
        (HANDOFF_CHECKS_SET_ACTION,),
    ):
        event_id = int(row["event_id"])
        payload = _payload(row)
        old, new = payload.get("old"), payload.get("new")
        if not isinstance(old, dict) or not isinstance(new, dict):
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"handoff_mode_meta_log_agreement: {HANDOFF_CHECKS_SET_ACTION} event "
                    f"{event_id} has no old/new handoff-check values"
                ),
                event_id=event_id,
            ))
            continue
        markers.append(_ChecksSet(
            event_id, _effective_checks(old), dict(new), _effective_checks(new),
        ))
    return markers


def _marker_on_at(markers: list[_ChecksSet], event_id: int, configs=()) -> frozenset[str]:
    """The checks the ``platform_handoff_checks_set`` events have on when
    event ``event_id`` ran: those the latest one before it set, else those
    the first one found."""
    state = markers[0].old
    for marker in markers:
        if marker.event_id >= event_id:
            break
        state = marker.new_effective
    on = frozenset(
        key for key, value in state.items() if value == TRUTHFUL_HANDOFF_CHECKS[key]
    )
    if configs and event_id < markers[0].event_id:
        before = [declared for config_id, declared in configs if config_id < event_id]
        if before:
            on |= frozenset(k for k, v in before[-1].items()
                            if TRUTHFUL_HANDOFF_CHECKS.get(k) == v)
    return on


def _marker_switch_on(markers: list[_ChecksSet], flag: str) -> _SwitchOn | None:
    """The latest ``platform_handoff_checks_set`` event that switched
    ``flag`` on, when the latest one has it on; None when there is none,
    or when the flag was already on before the first of them (a database
    begun before these events existed), so the other rules apply."""
    on = TRUTHFUL_HANDOFF_CHECKS[flag]
    for marker in reversed(markers):
        if marker.new_effective[flag] != on:
            return None
        if marker.old[flag] != on:
            return _SwitchOn(
                marker.event_id,
                f"{HANDOFF_CHECKS_SET_ACTION} event {marker.event_id}, which switched it on",
            )
    return None


def _lock_violations(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """``(event_id, label)`` for every ``ok`` ``accept_offer`` or
    ``schedule_meetup``/``schedule_shipment`` that ``commitment_lock_mode=
    listing`` would have blocked, replayed from the log.

    The replay tracks, per listing, the threads committed by a logged
    ``accept_offer`` and not yet ended by a logged ``leave_thread``,
    ``ghost``, ``cancel_meetup``, completion (of the thread or, which
    cancels it, of another thread of the listing), ``mark_sold`` or
    ``resume_intervention``, and their logged scheduled meetups. As in
    :func:`bazaar.core.handoff_checks.listing_commitments`, accepting is
    blocked by any other such thread and scheduling by one that comes
    first (a scheduled meetup, else the earlier accepted offer). A meetup
    left scheduled on an ended thread holds nothing here, so the replay
    never blocks more than the handler did."""
    threads: dict[int, int | None] = {
        int(row[0]): _coerce_int(row[1])
        for row in conn.execute("SELECT thread_id, listing_id FROM threads")
    }
    meetup_thread: dict[int, int] = {}
    live: dict[int, dict[int, int]] = {}  # listing -> thread -> accepted offer id
    scheduled: dict[int, set[int]] = {}  # thread -> live meetup ids
    violations: list[tuple[int, str]] = []

    def end(thread_id: int | None) -> None:
        if thread_id is None:
            return
        listing = threads.get(thread_id)
        if listing is not None:
            live.get(listing, {}).pop(thread_id, None)

    def drop_meetups(ids: Any) -> None:
        for meetup_id in ids if isinstance(ids, list) else []:
            meetup = _coerce_int(meetup_id)
            thread = None if meetup is None else meetup_thread.get(meetup)
            if thread is not None:
                scheduled.get(thread, set()).discard(meetup)

    def key(thread_id: int, accepted: int) -> tuple[int, int, int]:
        return (0 if scheduled.get(thread_id) else 1, accepted, thread_id)

    for row in conn.execute(
        """
        SELECT event_id, action_type, payload, result_payload FROM events
        WHERE result_status = 'ok' AND action_type IN (
            'accept_offer', 'schedule_meetup', 'schedule_shipment', 'leave_thread',
            'ghost', 'cancel_meetup', 'complete_transaction', 'mark_sold',
            'resume_intervention')
        ORDER BY event_id
        """
    ):
        event_id, action = int(row["event_id"]), str(row["action_type"])
        payload, result = _payload(row), _result_payload(row)
        thread_id = _int_field(result, "thread_id")
        if thread_id is None:
            thread_id = _int_field(payload, "thread_id")
        listing = None if thread_id is None else threads.get(thread_id)
        if action == "accept_offer":
            offer_id = _int_field(result, "offer_id") or _int_field(payload, "offer_id")
            if listing is None or thread_id is None or offer_id is None:
                continue
            holders = live.setdefault(listing, {})
            others = sorted(t for t in holders if t != thread_id)
            if others:
                violations.append((
                    event_id,
                    f"accept_offer on listing {listing} while thread {others[0]} "
                    "holds it",
                ))
            holders[thread_id] = min(holders.get(thread_id, offer_id), offer_id)
        elif action in ("schedule_meetup", "schedule_shipment"):
            meetup_id = _int_field(result, "meetup_id")
            if listing is None or thread_id is None or meetup_id is None:
                continue
            holders = live.get(listing, {})
            if thread_id in holders:
                first = min(holders, key=lambda t: key(t, holders[t]))
                if first != thread_id:
                    violations.append((
                        event_id,
                        f"{action} on listing {listing} while thread {first} holds it",
                    ))
            meetup_thread[meetup_id] = thread_id
            scheduled.setdefault(thread_id, set()).add(meetup_id)
        elif action in ("leave_thread", "ghost"):
            end(thread_id)
            drop_meetups(result.get("cancelled_meetup_ids"))
        elif action == "cancel_meetup":
            meetup_id = _int_field(result, "meetup_id") or _int_field(payload, "meetup_id")
            if thread_id is None and meetup_id is not None:
                thread_id = meetup_thread.get(meetup_id)
            drop_meetups([meetup_id])
            end(thread_id)
        elif action == "complete_transaction":
            if result.get("completed") is not True:
                continue
            drop_meetups([result.get("meetup_id")])
            drop_meetups(result.get("cancelled_sister_meetup_ids"))
            if listing is not None:
                live.pop(listing, None)
            else:
                end(thread_id)
        elif action == "mark_sold":
            sold = _int_field(result, "listing_id") or _int_field(payload, "listing_id")
            drop_meetups(result.get("cancelled_meetup_ids"))
            if sold is not None:
                live.pop(sold, None)
        elif action == "resume_intervention":
            closed = result.get("closed_redteam_thread_ids")
            for closed_id in closed if isinstance(closed, list) else []:
                closed_thread = _coerce_int(closed_id)
                end(closed_thread)
                if closed_thread is not None:
                    scheduled.pop(closed_thread, None)
    return violations


@dataclass
class _HandoffModes:
    """The handoff-check modes of a run: the ``meta`` rows, checked
    against the event log.

    ``integrity_legacy`` is what the legacy-behaviour warnings require:
    ``meta`` says ``completion_integrity_mode`` is not ``unit`` and the
    log carries no trace that only completion integrity (among the checks
    ``meta`` has on) can have written. ``integrity_point`` is where a
    ``completion_integrity_mode`` that ``meta`` has on was switched on
    (see :func:`_marker_switch_on` and :func:`_switch_on_point`); events
    up to it ran without it.
    """

    meta: dict[str, str]
    raw: dict[str, str | None]
    on: frozenset[str]
    issues: list[EventAuditIssue]
    integrity_legacy: bool
    integrity_point: _SwitchOn | None = None

    def integrity_off_at(self, event_id: int) -> str | None:
        """Why completion integrity was off when event ``event_id`` ran,
        or None when it was on (or the log contradicts ``meta``)."""
        if _INTEGRITY not in self.on:
            return "completion_integrity_mode is not 'unit'" if self.integrity_legacy else None
        point = self.integrity_point
        if point is None or event_id > point.after or (point.traced and event_id == point.after):
            return None
        return (
            "completion_integrity_mode was not yet 'unit': meta has it on from "
            f"{point.why}"
        )

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> _HandoffModes:
        raw: dict[str, str | None] = {}
        for key in HANDOFF_CHECK_MODES:
            try:
                row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            except sqlite3.Error:
                row = None
            raw[key] = None if row is None else str(row[0])
        meta = {key: read_handoff_check(conn, key) for key in HANDOFF_CHECK_MODES}
        on = frozenset(key for key, value in meta.items() if value == TRUTHFUL_HANDOFF_CHECKS[key])
        issues: list[EventAuditIssue] = []
        present = [key for key, value in raw.items() if value is not None]
        if present and len(present) != len(raw):
            # BazaarEnv writes all four rows on every construction.
            missing = sorted(set(raw) - set(present))
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    "handoff_mode_meta_log_agreement: meta has handoff-check rows "
                    f"{sorted(present)} but not {missing} (BazaarEnv writes all four)"
                ),
            ))
        for key, value in raw.items():
            if value is not None and value not in HANDOFF_CHECK_MODES[key]:
                # BazaarEnv writes exactly one of the documented values;
                # handlers would read anything else as legacy.
                issues.append(EventAuditIssue(
                    action_type="state_invariant",
                    message=(
                        f"handoff_mode_meta_log_agreement: meta {key}={value!r} is not one "
                        f"of the documented values {list(HANDOFF_CHECK_MODES[key])}"
                    ),
                ))
        # ``experiment_config`` events only explain a documented resume
        # switch-off below; they are not compared with ``meta`` because
        # the matrix launcher skips them (--skip-experiment-config-event),
        # so the latest one may belong to an earlier segment of the run.
        configs = _experiment_config_declarations(conn)
        # ``platform_handoff_checks_set`` events record every change of the
        # four values; the latest must agree with meta, and they say which
        # checks held when each event ran.
        markers = _handoff_check_markers(conn, issues)
        if markers:
            last_set = markers[-1]
            if any(last_set.new.get(key) != raw[key] for key in HANDOFF_CHECK_MODES):
                issues.append(EventAuditIssue(
                    action_type="state_invariant",
                    message=(
                        "handoff_mode_meta_log_agreement: the latest "
                        f"{HANDOFF_CHECKS_SET_ACTION} event {last_set.event_id} set "
                        + ", ".join(
                            f"{key}={last_set.new.get(key)!r}" for key in HANDOFF_CHECK_MODES
                        )
                        + " but meta has "
                        + ", ".join(f"{key}={raw[key]!r}" for key in HANDOFF_CHECK_MODES)
                    ),
                    event_id=last_set.event_id,
                ))
        deliveries = _meetup_deliveries(conn)
        unexplained: dict[tuple[str, frozenset[str]], list[int]] = {}
        before_set: dict[tuple[str, frozenset[str]], list[int]] = {}
        set_off: dict[tuple[str, frozenset[str]], list[int]] = {}
        last_only: dict[str, int] = {}
        first_trace: dict[str, int] = {}
        legacy_after: dict[str, list[tuple[int, str]]] = {}
        for row in conn.execute(
            f"""
            SELECT event_id, action_type, result_status, payload, result_payload
            FROM events
            WHERE action_type IN ({', '.join('?' for _ in _MODE_EVIDENCE_ACTIONS)})
              AND result_status IN ('ok', 'blocked')
            ORDER BY event_id
            """,
            _MODE_EVIDENCE_ACTIONS,
        ):
            event_id = int(row["event_id"])
            action, status = str(row["action_type"]), str(row["result_status"])
            result = _result_payload(row)
            delivery = (
                _event_delivery(deliveries, _payload(row), result)
                if action in ("complete_transaction", "inspect_at_meetup") else None
            )
            for label, flags in _mode_evidence(action, status, result, delivery):
                for flag in flags:
                    first_trace.setdefault(flag, event_id)
                explaining = flags & on
                if markers:
                    # The logged switches decide: a trace needs one of its
                    # checks on when it ran.
                    if not flags & _marker_on_at(markers, event_id, configs):
                        target = unexplained if not explaining else before_set
                        target.setdefault((label, flags), []).append(event_id)
                    elif not explaining:
                        set_off.setdefault((label, flags), []).append(event_id)
                    elif len(explaining) == 1:
                        last_only[next(iter(explaining))] = event_id
                    continue
                if not explaining:
                    unexplained.setdefault((label, flags), []).append(event_id)
                elif len(explaining) == 1:
                    last_only[next(iter(explaining))] = event_id
            for label, flags in _legacy_evidence(action, status, result, delivery):
                for flag in flags & on:
                    legacy_after.setdefault(flag, []).append((event_id, label))
        if _LOCK in on:
            for event_id, label in _lock_violations(conn):
                legacy_after.setdefault(_LOCK, []).append((event_id, label))
            if _LOCK in legacy_after:
                legacy_after[_LOCK].sort()
        integrity_traced = False
        for (label, flags), event_ids in sorted(before_set.items(), key=lambda kv: kv[1][0]):
            names = " or ".join(
                f"{flag}={TRUTHFUL_HANDOFF_CHECKS[flag]}" for flag in sorted(flags)
            )
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"handoff_mode_meta_log_agreement: {len(event_ids)} event(s) carry "
                    f"{label} (first event {event_ids[0]}), which only {names} writes, but "
                    f"the {HANDOFF_CHECKS_SET_ACTION} events had "
                    + ", ".join(sorted(flags)) + " off when they ran"
                ),
                event_id=event_ids[0],
            ))
        for (label, flags), event_ids in sorted(set_off.items(), key=lambda kv: kv[1][0]):
            # Documented only when a later marker switched the checks off;
            # otherwise meta changed behind the log's back.
            switch_off = next(
                (marker.event_id for marker in markers
                 if marker.event_id > event_ids[-1]
                 and not flags & _marker_on_at(markers, marker.event_id + 1, configs)),
                None,
            )
            if switch_off is None:
                merged = unexplained.setdefault((label, flags), [])
                merged.extend(event_ids)
                merged.sort()
                continue
            names = " or ".join(
                f"{flag}={TRUTHFUL_HANDOFF_CHECKS[flag]}" for flag in sorted(flags)
            )
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"handoff_mode_meta_log_agreement: {len(event_ids)} event(s) carry "
                    f"{label} (first event {event_ids[0]}), which only {names} writes, but "
                    "meta has " + ", ".join(f"{flag}={meta[flag]!r}" for flag in sorted(flags))
                    + f" (documented switch-off: {HANDOFF_CHECKS_SET_ACTION} event "
                    f"{switch_off} switched these checks off)"
                ),
                event_id=event_ids[0],
                severity="warning",
            ))
        for (label, flags), event_ids in sorted(unexplained.items(), key=lambda kv: kv[1][0]):
            switch = (
                None if markers else _switched_off_on_resume(configs, flags, event_ids)
            )
            if switch is None and _INTEGRITY in flags:
                integrity_traced = True
            names = " or ".join(
                f"{flag}={TRUTHFUL_HANDOFF_CHECKS[flag]}" for flag in sorted(flags)
            )
            message = (
                f"handoff_mode_meta_log_agreement: {len(event_ids)} event(s) carry "
                f"{label} (first event {event_ids[0]}), which only {names} writes, but meta "
                "has " + ", ".join(f"{flag}={meta[flag]!r}" for flag in sorted(flags))
            )
            if switch is not None:
                message += (
                    f" (documented resume switch-off: experiment_config event {switch} "
                    "resumed the run without these checks)"
                )
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=message,
                event_id=event_ids[0],
                severity="warning" if switch is not None else "error",
            ))
        points: dict[str, _SwitchOn] = {}
        for flag in sorted(on):
            # A check that meta has on must have shaped every event of
            # the kinds it changes since it was switched on, and at the
            # latest since the last event only it explains.
            marked = _marker_switch_on(markers, flag) if markers else None
            if marked is not None:
                point = marked
            else:
                point = _switch_on_point(conn, flag, configs, first_trace.get(flag))
                last = last_only.get(flag)
                if last is not None and last < point.after:
                    point = _SwitchOn(
                        last, f"event {last} (which only that check explains)", traced=True,
                    )
            points[flag] = point
            later = [(event_id, label) for event_id, label in legacy_after.get(flag, [])
                     if event_id > point.after]
            if later:
                issues.append(EventAuditIssue(
                    action_type="state_invariant",
                    message=(
                        f"handoff_mode_meta_log_agreement: meta has {flag}="
                        f"{meta[flag]!r}, but {len(later)} event(s) after {point.why} "
                        f"are legacy-shaped (first: event {later[0][0]}, {later[0][1]})"
                    ),
                    event_id=later[0][0],
                ))
        integrity_legacy = _INTEGRITY not in on and not integrity_traced
        return cls(meta, raw, on, issues, integrity_legacy, points.get(_INTEGRITY))

    def report(self, issues: list[EventAuditIssue]) -> None:
        issues.extend(self.issues)


def _experiment_config_declarations(
    conn: sqlite3.Connection,
) -> list[tuple[int, dict[str, str]]]:
    """``(event_id, declared handoff checks)`` of every ``experiment_config``
    event. ``llm-smoke`` records the resolved checks in
    ``defense_settings`` only when one is on; absent keys mean legacy."""
    declarations: list[tuple[int, dict[str, str]]] = []
    for row in conn.execute(
        """
        SELECT event_id, payload, result_payload FROM events
        WHERE action_type = 'experiment_config' ORDER BY event_id
        """
    ):
        settings = _payload(row).get("defense_settings")
        if not isinstance(settings, dict):
            continue
        declared = {
            key: str(settings[key]) for key in HANDOFF_CHECK_MODES
            if isinstance(settings.get(key), str)
        }
        declarations.append((int(row["event_id"]), declared))
    return declarations


def _switched_off_on_resume(
    configs: list[tuple[int, dict[str, str]]],
    flags: frozenset[str],
    event_ids: list[int],
) -> int | None:
    """The ``experiment_config`` event that switched ``flags`` off on a
    resume, when every event in ``event_ids`` ran under an earlier
    ``experiment_config`` that declared one of ``flags`` on; else None."""
    if not configs:
        return None

    def declares_on(declared: dict[str, str]) -> bool:
        return any(declared.get(flag) == TRUTHFUL_HANDOFF_CHECKS[flag] for flag in flags)

    last_event, last_declared = configs[-1]
    if declares_on(last_declared) or last_event < event_ids[-1]:
        return None
    for event_id in event_ids:
        before = [declared for config_id, declared in configs if config_id < event_id]
        if not before or not declares_on(before[-1]):
            return None
    return last_event


# Handler answers to an id that does not exist: (payload key, table, id
# column). The handler returns ``error`` by design (bazaar.actions.handlers).
_NOT_FOUND_REFERENCES: dict[str, tuple[str, str, str]] = {
    "meetup_not_found": ("meetup_id", "meetups", "meetup_id"),
    "thread_not_found": ("thread_id", "threads", "thread_id"),
    "listing_not_found": ("listing_id", "listings", "listing_id"),
    "photo_not_found": ("photo_id", "photos", "photo_id"),
    "ratee_not_found": ("ratee_agent_id", "agents", "agent_id"),
    "user_not_found": ("user_agent_id", "agents", "agent_id"),
}


def _nonexistent_reference(conn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
    """Why an ``error`` event is a handler's not-found answer, or None.

    The referenced id (from the payload key the handler looks up) must
    not have existed when the event ran: no such row, or the row's first
    appearance in an ``ok`` result payload comes after the event."""
    result = _result_payload(row)
    error = result.get("error")
    if not isinstance(error, str) or error not in _NOT_FOUND_REFERENCES:
        return None
    if set(result) != {"error"}:
        return None
    key, table, id_col = _NOT_FOUND_REFERENCES[error]
    payload = _payload(row)
    if key not in payload:
        return None
    ref = payload[key]
    ref_id = None if isinstance(ref, bool) else _coerce_int(ref)
    if ref_id is None:
        return f"{key}={ref!r} is not an id"
    exists = conn.execute(
        f"SELECT 1 FROM {table} WHERE {id_col} = ?", (ref_id,),
    ).fetchone()
    if exists is None:
        return f"{table} has no {id_col}={ref_id}"
    first = conn.execute(
        f"""
        SELECT MIN(event_id) FROM events
        WHERE result_status = 'ok' AND {_json_key('result_payload', id_col)} = ?
        """,
        (ref_id,),
    ).fetchone()
    if first is not None and first[0] is not None and int(first[0]) > int(row["event_id"]):
        return f"{id_col}={ref_id} first appears in event {int(first[0])}, after this one"
    return None


def _audit_error_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT event_id, action_type, payload, result_payload
        FROM events
        WHERE result_status = 'error'
        ORDER BY event_id
        """
    ):
        message = (
            "error_events_absent: "
            f"event {row['event_id']} action={row['action_type']!r} "
            "has result_status='error'"
        )
        reason = _nonexistent_reference(conn, row)
        if reason is not None:
            error = _result_payload(row).get("error")
            issues.append(EventAuditIssue(
                action_type="state_invariant",
                message=(
                    f"{message} (handler's {error!r} answer to a nonexistent id: {reason}; "
                    "not a handler exception)"
                ),
                event_id=int(row["event_id"]),
                severity="warning",
            ))
            continue
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=message,
            event_id=int(row["event_id"]),
        ))


def _audit_offer_thread_state(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT o.offer_id, o.thread_id, o.status AS offer_status,
               t.status AS thread_status
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        WHERE o.status = 'accepted'
          AND t.status NOT IN ('committed', 'completed', 'cancelled')
        ORDER BY o.offer_id
        """
    ):
        if row["thread_status"] == "ghosted" and _ghost_after_accept(
            conn, int(row["offer_id"]), int(row["thread_id"]),
        ):
            # ``ghost`` ends a committed thread and rejects only pending
            # offers, under every handoff check, as ``leave_thread`` does
            # (whose ``cancelled`` status this check always allowed).
            continue
        _add_state_issue(
            issues,
            "accepted_offer_thread_status",
            f"accepted offer on thread status {row['thread_status']!r}"
            + (" not explained by a logged ghost after the acceptance"
               if row["thread_status"] == "ghosted" else ""),
            ref_id=int(row["offer_id"]),
        )
    for row in conn.execute(
        """
        SELECT o.offer_id, o.thread_id, t.status AS thread_status
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        WHERE o.status = 'pending' AND t.status != 'open'
        ORDER BY o.offer_id
        """
    ):
        _add_state_issue(
            issues,
            "pending_offer_open_thread",
            f"pending offer on thread status {row['thread_status']!r}",
            ref_id=int(row["offer_id"]),
        )


def _first_ok_event(
    conn: sqlite3.Connection,
    actions: tuple[str, ...],
    key: str,
    value: int,
    *,
    after: int = 0,
    extra: str = "",
    source: str = "result_payload",
) -> int | None:
    """The first ``ok`` event among ``actions`` after event ``after`` whose
    ``source`` JSON has ``key == value`` (plus the SQL condition ``extra``)."""
    row = conn.execute(
        f"""
        SELECT MIN(event_id) FROM events
        WHERE action_type IN ({', '.join('?' for _ in actions)})
          AND result_status = 'ok' AND event_id > ?
          AND {_json_key(source, key)} = ? {extra}
        """,
        (*actions, after, value),
    ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def _ghost_after_accept(conn: sqlite3.Connection, offer_id: int, thread_id: int) -> bool:
    accepted = _first_ok_event(conn, ("accept_offer",), "offer_id", offer_id)
    return _first_ok_event(
        conn, ("ghost",), "thread_id", thread_id, after=accepted or 0,
    ) is not None


def _relisted_after_completion(
    conn: sqlite3.Connection, meetup_id: int, listing_id: int,
) -> bool:
    """Whether the logged ``relist`` of the listing after this meetup's
    completion is the listing's last status change (no later sale)."""
    completed = _first_ok_event(
        conn, ("complete_transaction",), "meetup_id", meetup_id,
        extra=f"AND {_json_key('result_payload', 'completed')} = 1",
    )
    row = conn.execute(
        f"""
        SELECT MAX(event_id) FROM events
        WHERE action_type = 'relist' AND result_status = 'ok' AND event_id > ?
          AND {_json_key('result_payload', 'listing_id')} = ?
        """,
        (completed or 0, listing_id),
    ).fetchone()
    if row is None or row[0] is None:
        return False
    relisted = int(row[0])
    later_sale = conn.execute(
        f"""
        SELECT 1 FROM events
        WHERE result_status = 'ok' AND event_id > ? AND (
            (action_type = 'mark_sold'
             AND {_json_key('result_payload', 'listing_id')} = ?)
            OR (action_type = 'complete_transaction'
                AND {_json_key('result_payload', 'completed')} = 1
                AND {_json_key('result_payload', 'thread_id')} IN (
                    SELECT thread_id FROM threads WHERE listing_id = ?))
        )
        LIMIT 1
        """,
        (relisted, listing_id, listing_id),
    ).fetchone()
    return later_sale is None


_NO_MEETUP_CANCEL = "AND NOT " + _json_has("result_payload", "cancelled_meetup_ids")


def _stale_meetup_explanation(
    conn: sqlite3.Connection,
    scheduled: int,
    thread_id: int,
    listing_id: int | None,
    thread_status: str,
) -> tuple[int, str] | None:
    """``(event_id, description)`` of the logged legacy event that left a
    meetup scheduled on a cancelled or ghosted thread, or None.

    Legacy contract: a sale of the listing
    through another thread (``complete_transaction``) or by the seller
    (``mark_sold``) cancels the other threads but not their meetups, and
    ``leave_thread``/``ghost`` end the thread but not its meetup. The
    event must come after ``scheduled``, the logged event that scheduled
    the meetup, and must not carry the meetup cancellation that the
    truthful checks add (``cancelled_sister_meetup_ids``/
    ``cancelled_meetup_ids``). The earliest such event is returned."""
    if thread_status == "ghosted":
        found = _first_ok_event(
            conn, ("ghost",), "thread_id", thread_id, after=scheduled, extra=_NO_MEETUP_CANCEL,
        )
        return None if found is None else (found, f"ghost event {found}")
    candidates: list[tuple[int, str]] = []
    found = _first_ok_event(
        conn, ("leave_thread",), "thread_id", thread_id, after=scheduled,
        extra=_NO_MEETUP_CANCEL,
    )
    if found is not None:
        candidates.append((found, f"leave_thread event {found}"))
    if listing_id is None:
        return min(candidates, default=None)
    row = conn.execute(
        f"""
        SELECT MIN(event_id) FROM events
        WHERE action_type = 'complete_transaction' AND result_status = 'ok'
          AND event_id > ?
          AND {_json_key('result_payload', 'completed')} = 1
          AND NOT {_json_has('result_payload', 'cancelled_sister_meetup_ids')}
          AND {_json_key('result_payload', 'thread_id')} IN (
              SELECT thread_id FROM threads WHERE listing_id = ? AND thread_id != ?
          )
        """,
        (scheduled, listing_id, thread_id),
    ).fetchone()
    if row is not None and row[0] is not None:
        candidates.append((
            int(row[0]),
            f"complete_transaction event {int(row[0])} on another thread of the listing",
        ))
    found = _first_ok_event(
        conn, ("mark_sold",), "listing_id", listing_id, after=scheduled,
        extra=_NO_MEETUP_CANCEL,
    )
    if found is not None:
        candidates.append((found, f"mark_sold event {found}"))
    return min(candidates, default=None)


def _scheduling_events(conn: sqlite3.Connection) -> dict[int, int]:
    """``meetup_id -> event_id`` of the first ``ok`` ``schedule_meetup``
    or ``schedule_shipment`` event that created it."""
    scheduled: dict[int, int] = {}
    for row in conn.execute(
        f"""
        SELECT event_id, {_json_key('result_payload', 'meetup_id')} AS meetup_id
        FROM events
        WHERE action_type IN ('schedule_meetup', 'schedule_shipment')
          AND result_status = 'ok'
        ORDER BY event_id
        """
    ):
        meetup_id = _coerce_int(row["meetup_id"])
        if meetup_id is not None:
            scheduled.setdefault(meetup_id, int(row["event_id"]))
    return scheduled


def _seeded_history_meetup(meetup: sqlite3.Row, first_tick: Any) -> bool:
    """Whether a closed meetup without a scheduling event is part of the
    seeded pre-log history of a cold-start world
    (``bazaar.experiments.cold_start_world``): its listing is seeded and
    its tick precedes the first logged event."""
    tick = meetup["scheduled_tick"]
    return (
        int(meetup["listing_seeded"] or 0) == 1
        and meetup["meetup_status"] in ("completed", "cancelled")
        and not isinstance(tick, bool) and isinstance(tick, int)
        and not isinstance(first_tick, bool) and isinstance(first_tick, int)
        and tick < first_tick
    )


def _audit_meetup_thread_listing_state(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    modes: _HandoffModes | None = None,
) -> None:
    if modes is None:
        modes = _HandoffModes.load(conn)
    truncation = _Truncation.load(conn)
    scheduled_by = _scheduling_events(conn)
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(meetups)")}
    delivery = "m.delivery_method" if "delivery_method" in columns else "'meetup'"
    listing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(listings)")}
    seeded = "COALESCE(l.is_seeded, 0)" if "is_seeded" in listing_columns else "0"
    first_tick = conn.execute("SELECT MIN(tick) FROM events").fetchone()[0]
    for row in conn.execute(
        f"""
        SELECT m.meetup_id, m.thread_id, m.status AS meetup_status,
               m.scheduled_tick, {delivery} AS delivery_method,
               t.status AS thread_status, t.listing_id, {seeded} AS listing_seeded
        FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        LEFT JOIN listings l ON l.listing_id = t.listing_id
        ORDER BY m.meetup_id
        """
    ):
        meetup_id = int(row["meetup_id"])
        status = row["meetup_status"]
        committed = row["thread_status"] == "committed"
        stale = row["thread_status"] in ("cancelled", "ghosted")
        scheduled = scheduled_by.get(meetup_id)
        if scheduled is None:
            # Every meetup, whatever its status and the handoff checks, is
            # created by a logged scheduling event; only the unlogged
            # window of a truncated base can hold one without it, and a
            # cold-start world seeds the closed meetups of its pre-log
            # history on seeded listings.
            if status != "scheduled" and _seeded_history_meetup(row, first_tick):
                continue
            problem = (
                "no meta truncated_base_note window explains it" if truncation is None
                else truncation.meetup_problem(conn, row)
            )
            if problem is not None:
                _add_state_issue(
                    issues,
                    "scheduled_meetup_logged",
                    f"{status} meetup has no logged schedule_meetup/schedule_shipment "
                    f"event ({problem})",
                    ref_id=meetup_id,
                )
                continue
            assert truncation is not None
            window = (
                "meetup scheduled in the unlogged window of a truncated base: "
                + truncation.citation
            )
            if committed or status != "scheduled":
                _add_state_issue(
                    issues,
                    "scheduled_meetup_logged",
                    f"{status} meetup has no logged scheduling event ({window})",
                    ref_id=meetup_id,
                    severity="warning",
                )
                continue
            # The window ran before the continuation's first event, under
            # the completion contract in force then.
            off = modes.integrity_off_at(truncation.boundary) if stale else None
            _add_state_issue(
                issues,
                "scheduled_meetup_committed_thread",
                f"scheduled meetup on thread status {row['thread_status']!r} ({window}"
                + (f"; {off}" if off is not None else "; but completion_integrity_mode "
                   "was 'unit' or the thread status is not one a legacy sale, "
                   "leave_thread or ghost leaves")
                + ")",
                ref_id=meetup_id,
                severity="warning" if off is not None else "error",
            )
            continue
        if committed or status != "scheduled":
            continue
        message = f"scheduled meetup on thread status {row['thread_status']!r}"
        severity = "error"
        if stale and (modes.integrity_legacy or modes.integrity_point is not None):
            # Judged by the completion contract in force when the
            # explaining event ran: a truthful continuation of a legacy
            # base keeps the meetups its base left scheduled.
            explanation = _stale_meetup_explanation(
                conn, scheduled, int(row["thread_id"]),
                _coerce_int(row["listing_id"]), str(row["thread_status"]),
            )
            off = None if explanation is None else modes.integrity_off_at(explanation[0])
            if explanation is not None and off is not None:
                severity = "warning"
                message += (
                    f" (documented legacy behaviour: {off}; left scheduled by "
                    f"{explanation[1]})"
                )
            elif modes.integrity_legacy:
                message += (
                    " (completion_integrity_mode is not 'unit', but no logged legacy "
                    "sale, leave_thread or ghost explains it)"
                )
            elif explanation is not None:
                assert modes.integrity_point is not None
                message += (
                    f" (left by {explanation[1]}, after completion_integrity_mode was "
                    f"switched on at {modes.integrity_point.why})"
                )
        _add_state_issue(
            issues,
            "scheduled_meetup_committed_thread",
            message,
            ref_id=meetup_id,
            severity=severity,
        )
    for row in conn.execute(
        """
        SELECT m.meetup_id, m.thread_id, t.status AS thread_status,
               l.status AS listing_status, l.listing_id
        FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE m.status = 'completed'
          AND (
            t.status != 'completed'
            OR (l.status != 'sold' AND COALESCE(l.is_seeded, 0) = 0)
          )
        ORDER BY m.meetup_id
        """
    ):
        if (
            row["thread_status"] == "completed" and row["listing_status"] == "active"
            and _relisted_after_completion(conn, int(row["meetup_id"]), int(row["listing_id"]))
        ):
            # ``relist`` reopens a sold listing (sold -> active) under
            # every handoff check; the completed deal stays completed.
            continue
        _add_state_issue(
            issues,
            "completed_meetup_terminal_state",
            (
                f"thread={row['thread_status']!r} "
                f"listing={row['listing_status']!r}"
            ),
            ref_id=int(row["meetup_id"]),
        )
    for row in conn.execute(
        """
        SELECT t.thread_id
        FROM threads t
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE t.status = 'completed'
          AND COALESCE(l.is_seeded, 0) = 0
          AND NOT EXISTS (
            SELECT 1
            FROM meetups m
            WHERE m.thread_id = t.thread_id
              AND m.status = 'completed'
          )
        ORDER BY t.thread_id
        """
    ):
        _add_state_issue(
            issues,
            "completed_thread_has_completed_meetup",
            "completed thread has no completed meetup",
            ref_id=int(row["thread_id"]),
        )


def _audit_sold_listing_state(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT t.thread_id, t.listing_id, t.status AS thread_status
        FROM threads t
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE l.status = 'sold'
          AND t.status NOT IN ('completed', 'cancelled', 'ghosted')
        ORDER BY t.thread_id
        """
    ):
        _add_state_issue(
            issues,
            "sold_listing_terminal_threads",
            f"sold listing has thread status {row['thread_status']!r}",
            ref_id=int(row["thread_id"]),
        )
    for row in conn.execute(
        """
        SELECT o.offer_id, t.listing_id
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE l.status = 'sold' AND o.status = 'pending'
        ORDER BY o.offer_id
        """
    ):
        _add_state_issue(
            issues,
            "sold_listing_no_pending_offers",
            "sold listing has a pending offer",
            ref_id=int(row["offer_id"]),
        )


def _audit_rating_thread_state(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT r.rating_id, r.thread_id
        FROM ratings r
        JOIN threads t ON t.thread_id = r.thread_id
        WHERE r.thread_id IS NOT NULL
          AND (
            r.rater_agent_id NOT IN (t.buyer_agent_id, t.seller_agent_id)
            OR r.ratee_agent_id NOT IN (t.buyer_agent_id, t.seller_agent_id)
            OR t.status NOT IN ('completed', 'cancelled')
          )
        ORDER BY r.rating_id
        """
    ):
        _add_state_issue(
            issues,
            "rating_terminal_thread_participants",
            "rating thread is non-terminal or not between participants",
            ref_id=int(row["rating_id"]),
        )


def _audit_phantom_tripwire_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT event_id,
               json_extract(payload, '$.listing_id') AS listing_id
        FROM events
        WHERE action_type = 'platform_phantom_tripwire'
          AND json_valid(payload)
          AND json_extract(payload, '$.listing_id') NOT IN (
              SELECT listing_id FROM listings WHERE is_phantom = 1
          )
        ORDER BY event_id
        """
    ):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                "phantom_tripwire_references_phantom_listing: "
                "tripwire references non-phantom or missing listing"
            ),
            event_id=int(row["event_id"]),
            ref_id=_coerce_int(row["listing_id"]),
        ))


def _audit_event_agent_references(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT event_id, agent_id
        FROM events
        WHERE agent_id IS NOT NULL
          AND agent_id NOT IN (SELECT agent_id FROM agents)
        ORDER BY event_id
        """
    ):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message="event_agent_exists: event references missing agent",
            event_id=int(row["event_id"]),
            ref_id=int(row["agent_id"]),
        ))


def _audit_memory_divergence_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    for row in conn.execute(
        """
        SELECT event_id
        FROM events
        WHERE action_type = 'memory_divergence'
          AND json_valid(payload)
          AND json_extract(payload, '$.narrative_tick') IS NULL
        ORDER BY event_id
        """
    ):
        issues.append(EventAuditIssue(
            action_type="state_invariant",
            message=(
                "memory_divergence_has_narrative_tick: "
                "memory_divergence event missing narrative_tick"
            ),
            event_id=int(row["event_id"]),
        ))


def _audit_register_agent_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    action = "platform_register_agent"
    evidence: _InventoryEvidence | None = None
    for event in _event_rows(conn, action):
        payload = _payload(event)
        aid = payload.get("agent_id")
        if aid is None:
            _add(issues, event, action, "missing agent_id")
            continue
        row = conn.execute(
            """
            SELECT agent_id, user_name, display_name, parent_agent_id,
                   risk_posture, is_redteam, persona_json
            FROM agents
            WHERE agent_id = ?
            """,
            (aid,),
        ).fetchone()
        ref_id = int(aid)
        if row is None:
            _add(issues, event, action, "agent row missing", ref_id=ref_id)
            continue
        for field in ("user_name", "display_name", "risk_posture"):
            _expect_equal(issues, event, action, ref_id, field, payload.get(field), row[field])
        if evidence is None:
            evidence = _InventoryEvidence.load(conn)
        _audit_registered_persona(
            issues, event, ref_id, payload.get("persona_json"), row["persona_json"], evidence,
        )
        _expect_equal(
            issues, event, action, ref_id, "parent_agent_id",
            payload.get("parent_agent_id"), row["parent_agent_id"],
        )
        _expect_equal(
            issues, event, action, ref_id, "is_redteam",
            bool(payload.get("is_redteam")), bool(row["is_redteam"]),
        )


def _json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# persona inventory provenance
#
# ``agents.persona_json`` is the registered persona plus the
# ``inventory_items`` changes that later events explain. The run changes
# the inventory in four ways, and each is cross-checked against the log:
#
# * ``D_restock`` appends ``source="restock"`` units and logs
#   ``platform_inventory_restocked`` (tick, titles, tier, sales count);
# * a completed ``complete_transaction`` appends a ``source="bought"`` unit
#   to the buyer (listing, tick, the accepted offer's price, and under the
#   unit-aware transfer the ``consumed_unit`` record and
#   ``buyer_unit_index``);
# * a sale (completion or ``mark_sold``) marks one seller unit with
#   ``sold_at_tick`` / ``sold_via_listing_id``: the unit-aware transfer
#   logs it as ``consumed_unit``, the legacy transfer picks it by the
#   logged title match, which is replayed here;
# * a truthful-handoff binding writes ``unit_uid``, ``quality_source`` and
#   ``ground_truth_quality_pct`` and logs the same record as
#   ``backing_unit`` (create_listing, relist), ``bound_at_handoff``
#   (inspect_at_meetup, complete_transaction) or ``consumed_unit``.
#
# A key that is absent and a key that is present with null are different
# states (``_MISSING``): the writers above add keys, they never write null
# for an absent value.
#
# A truncated base (``meta.truncated_base_note``: the event log was cut at
# ``target_max_tick`` while the tables kept the state of the source run up
# to ``source_max_tick``) holds inventory changes of that window that no
# event explains. Those are warnings that cite the note when the note fits
# the database and the change agrees with the tables; nothing else is.
# ---------------------------------------------------------------------------

# Unit keys the run writes in place after a unit exists.
_UNIT_SOLD_KEYS = ("sold_at_tick", "sold_via_listing_id")
_UNIT_BINDING_KEYS = ("unit_uid", "quality_source", "ground_truth_quality_pct")
_UNIT_MUTABLE_KEYS = frozenset(_UNIT_SOLD_KEYS + _UNIT_BINDING_KEYS)
# Keys every restocked unit carries; ``restock_tier`` and
# ``restock_reason`` too when the restock event logs ``marketplace_tier``.
_RESTOCK_BASE_KEYS = (
    "acquisition_cost_cents",
    "added_at_tick",
    "asking_price_cents",
    "category",
    "condition",
    "ground_truth_quality_pct",
    "source",
    "stated_quality_band",
    "title",
)
_RESTOCK_UNIT_KEYS = frozenset(_RESTOCK_BASE_KEYS + ("restock_reason", "restock_tier"))
_RESTOCK_REASONS = ("recent_sales", "background_supply")
_BOUGHT_BASE_KEYS = ("category", "condition", "description", "title")
_BOUGHT_UNIT_KEYS = frozenset({
    "asking_price_cents",
    "bought_from_listing_id",
    "bought_from_unit_uid",
    "bought_listing_title",
    "bought_price_cents",
    "bought_tick",
    "category",
    "condition",
    "description",
    "ground_truth_quality_pct",
    "quality_source",
    "source",
    "title",
})
# A value the event log does not pin down.
_UNKNOWN: Any = object()


@dataclass(frozen=True)
class _Truncation:
    """``meta.truncated_base_note`` of a truncated base: the event log ends
    at ``target`` while the tables hold state up to ``source``;
    ``boundary`` is the last event at or before ``target``.

    The note explains only a database whose log fits it: a base whose
    last logged tick is at most ``target``, or a continuation of the
    truncated base (``meta.trapi_matrix_cell`` forked at ``target``, or
    an ``experiment_config``/``level0_restore_summary`` ``source_db``)
    whose events after the fork all follow the base's. Everything the
    window explains must also agree with the tables (see
    :meth:`meetup_problem` and ``_InventoryAudit``)."""

    target: int
    source: int
    boundary: int

    @property
    def citation(self) -> str:
        return (
            "meta truncated_base_note: the base event log ends at tick "
            f"{self.target} while the tables hold state up to tick {self.source}"
        )

    @classmethod
    def read(cls, conn: sqlite3.Connection) -> tuple[_Truncation | None, str | None]:
        """``(window, None)`` for a note that fits the database, ``(None,
        reason)`` for one that does not, ``(None, None)`` without a note."""
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'truncated_base_note'",
            ).fetchone()
        except sqlite3.Error:
            return None, None
        if row is None:
            return None, None
        note = _json_object(row[0])
        target = None if note is None else note.get("target_max_tick")
        source = None if note is None else note.get("source_max_tick")
        if (
            isinstance(target, bool) or not isinstance(target, int)
            or isinstance(source, bool) or not isinstance(source, int)
            or not target < source
        ):
            return None, (
                "the note is not a JSON object with integer target_max_tick < "
                "source_max_tick"
            )
        boundary = int(conn.execute(
            "SELECT COALESCE(MAX(event_id), 0) FROM events WHERE tick <= ?", (target,),
        ).fetchone()[0])
        later = conn.execute(
            "SELECT MIN(event_id), MIN(tick) FROM events WHERE tick > ?", (target,),
        ).fetchone()
        if later is not None and later[0] is not None:
            fork = _cell_fork(conn)
            source_db = any(
                isinstance((_meta_json(conn, key) or {}).get("source_db"), str)
                and (_meta_json(conn, key) or {}).get("source_db")
                for key in ("experiment_config", "level0_restore_summary")
            )
            if fork is not None and fork[0] != target:
                return None, (
                    f"meta.trapi_matrix_cell forks at tick {fork[0]}, not at the note's "
                    f"target_max_tick {target}"
                )
            if fork is None and not source_db:
                return None, (
                    f"events are logged after target_max_tick {target} (from tick "
                    f"{later[1]}), so the declared window does not lie after the log, and "
                    "meta records no continuation of a truncated base "
                    "(trapi_matrix_cell fork_tick, experiment_config or "
                    "level0_restore_summary source_db)"
                )
            if boundary > int(later[0]):
                return None, (
                    f"events at or before target_max_tick {target} follow events after "
                    "it, so the log is not a base followed by its continuation"
                )
        return cls(target, source, boundary), None

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> _Truncation | None:
        return cls.read(conn)[0]

    def covers(self, tick: Any) -> bool:
        """Whether ``tick`` lies in the unlogged window."""
        return (
            not isinstance(tick, bool) and isinstance(tick, int)
            and self.target < tick <= self.source
        )

    def meetup_problem(self, conn: sqlite3.Connection, meetup: sqlite3.Row) -> str | None:
        """Why a scheduled meetup without a logged scheduling event cannot
        have been scheduled in the window, or None when it can: its tick
        lies after the log (a shipment's, which is its scheduling tick,
        inside the window), its id follows every meetup the base log
        scheduled, its thread holds an accepted offer from before the
        window's end, and no logged event ended the thread before the
        window."""
        meetup_id, thread_id = int(meetup["meetup_id"]), int(meetup["thread_id"])
        tick = meetup["scheduled_tick"]
        if isinstance(tick, bool) or not isinstance(tick, int) or tick <= self.target:
            return f"its tick {tick!r} is not after the log's end at tick {self.target}"
        if (meetup["delivery_method"] or "meetup") == "ship" and tick > self.source:
            return f"the shipment was scheduled at tick {tick}, after the window"
        logged = conn.execute(
            f"""
            SELECT MAX({_json_key('result_payload', 'meetup_id')}) FROM events
            WHERE action_type IN ('schedule_meetup', 'schedule_shipment')
              AND result_status = 'ok' AND event_id <= ?
            """,
            (self.boundary,),
        ).fetchone()[0]
        if logged is not None and meetup_id <= int(logged):
            return f"its id precedes meetup {int(logged)}, which the base log scheduled"
        accepted = conn.execute(
            "SELECT 1 FROM offers WHERE thread_id = ? AND status = 'accepted' AND tick <= ?",
            (thread_id, self.source),
        ).fetchone()
        if accepted is None:
            return f"thread {thread_id} holds no accepted offer from before the window's end"
        listing_id = _coerce_int(meetup["listing_id"])
        ended = conn.execute(
            f"""
            SELECT MIN(event_id) FROM events
            WHERE result_status = 'ok' AND event_id <= ? AND (
                (action_type IN ('leave_thread', 'ghost', 'cancel_meetup')
                 AND {_json_key('result_payload', 'thread_id')} = ?)
                OR (action_type = 'complete_transaction'
                    AND {_json_key('result_payload', 'completed')} = 1
                    AND {_json_key('result_payload', 'thread_id')} IN (
                        SELECT thread_id FROM threads WHERE listing_id = ?))
                OR (action_type = 'mark_sold'
                    AND {_json_key('result_payload', 'listing_id')} = ?)
            )
            """,
            (self.boundary, thread_id, listing_id, listing_id),
        ).fetchone()[0]
        if ended is not None:
            return f"event {int(ended)} ended thread {thread_id} before the window"
        return None


def _audit_truncated_base_note(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    _window, problem = _Truncation.read(conn)
    if problem is not None:
        _add_state_issue(
            issues,
            "truncated_base_note_valid",
            f"meta truncated_base_note does not fit this database ({problem}); "
            "it explains nothing",
        )


@dataclass(frozen=True)
class _Sale:
    """A logged sale: a completed ``complete_transaction`` (``buyer`` set)
    or an ``ok`` ``mark_sold`` (``buyer`` None). ``unit_branch`` is True
    when the result carries ``consumed_unit`` (the unit-aware transfer of
    the truthful handoff checks); otherwise the legacy transfer ran."""

    event_id: int
    tick: int
    listing_id: int | None
    seller: int | None
    buyer: int | None
    unit_branch: bool
    consumed: dict[str, Any] | None
    buyer_unit_index: Any


@dataclass
class _InventoryEvidence:
    """Everything the event log says about persona inventory writes."""

    conn: sqlite3.Connection
    truncation: _Truncation | None = None
    restocks: dict[int, list[tuple[int, int, dict[str, Any]]]] = dc_field(default_factory=dict)
    purchases: dict[int, list[_Sale]] = dc_field(default_factory=dict)
    sales: dict[int, list[_Sale]] = dc_field(default_factory=dict)
    # owner -> unit index -> [(event_id, binding record)], in event order
    bindings: dict[int, dict[int, list[tuple[int, dict[str, Any]]]]] = dc_field(
        default_factory=dict,
    )
    created: dict[int, dict[str, Any]] = dc_field(default_factory=dict)
    created_tick: dict[int, int] = dc_field(default_factory=dict)
    # listing -> [(event_id, tick, fields)], in event order
    edits: dict[int, list[tuple[int, int, dict[str, Any]]]] = dc_field(default_factory=dict)
    # listing -> [(event_id, order in the event, ground_truth_quality_pct)]
    quality: dict[int, list[tuple[int, int, Any]]] = dc_field(default_factory=dict)
    _listings: dict[int, sqlite3.Row | None] = dc_field(default_factory=dict)
    _threads: dict[int, int | None] | None = None
    _logged_offers: frozenset[int] | None = None

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> _InventoryEvidence:
        evidence = cls(conn, _Truncation.load(conn))
        for row in conn.execute(
            """
            SELECT event_id, tick, agent_id, action_type, result_status,
                   payload, result_payload
            FROM events
            WHERE action_type IN (
                    'platform_inventory_restocked', 'complete_transaction',
                    'mark_sold', 'create_listing', 'edit_listing', 'relist',
                    'inspect_at_meetup'
                  )
              AND result_status IN ('ok', 'blocked')
            ORDER BY event_id
            """
        ):
            evidence._add_event(row)
        return evidence

    def _add_event(self, row: sqlite3.Row) -> None:
        action = str(row["action_type"])
        event_id, tick = int(row["event_id"]), int(row["tick"])
        agent = _coerce_int(row["agent_id"])
        ok = row["result_status"] == "ok"
        payload, result = _payload(row), _result_payload(row)
        if action == "platform_inventory_restocked":
            if ok and agent is not None:
                self.restocks.setdefault(agent, []).append((event_id, tick, payload))
        elif action == "complete_transaction":
            if ok and bool(result.get("completed")):
                self._add_completion(event_id, tick, result)
        elif action == "mark_sold":
            if ok:
                consumed = result.get("consumed_unit")
                listing_id = _int_field(result, "listing_id")
                sale = _Sale(
                    event_id, tick, listing_id, agent, None,
                    "consumed_unit" in result,
                    consumed if isinstance(consumed, dict) else None, None,
                )
                if agent is not None:
                    self.sales.setdefault(agent, []).append(sale)
                self._add_binding(event_id, agent, consumed)
                self._add_quality(listing_id, event_id, 1, result, "bound_at_sale")
        elif action == "create_listing":
            listing_id = _int_field(result, "listing_id")
            if ok and listing_id is not None:
                self.created[listing_id] = payload
                self.created_tick[listing_id] = tick
                self._add_quality(listing_id, event_id, 0, result, "backing_unit")
            self._add_binding(event_id, agent, result.get("backing_unit"))
        elif action == "edit_listing":
            listing_id = _int_field(result, "listing_id") or _int_field(payload, "listing_id")
            fields = {
                key: payload[key] for key in ("title", "description", "condition")
                if payload.get(key) is not None
            }
            if ok and listing_id is not None and fields:
                self.edits.setdefault(listing_id, []).append((event_id, tick, fields))
        elif action == "relist":
            self._add_binding(event_id, agent, result.get("backing_unit"))
            if ok:
                listing_id = _int_field(result, "listing_id") or _int_field(payload, "listing_id")
                self._add_quality(listing_id, event_id, 0, result, "backing_unit")
        elif action == "inspect_at_meetup":
            self._add_binding(
                event_id, _int_field(result, "seller_id"), result.get("bound_at_handoff"),
            )
            if ok:
                self._add_quality(
                    _int_field(result, "listing_id"), event_id, 0, result, "bound_at_handoff",
                )

    def _add_quality(
        self, listing_id: int | None, event_id: int, order: int,
        result: dict[str, Any], key: str,
    ) -> None:
        """Record the listing quality a logged binding wrote: the record's
        ``quality``, or NULL when ``backing_unit`` is null (an unbound
        create_listing or relist under inspection_truth_mode=unit)."""
        if listing_id is None or key not in result:
            return
        record = result[key]
        if isinstance(record, dict):
            value = record.get("quality")
        elif record is None and key == "backing_unit":
            value = None
        else:
            return
        self.quality.setdefault(listing_id, []).append((event_id, order, value))

    def _thread_listing(self, thread_id: int | None) -> int | None:
        if self._threads is None:
            self._threads = {
                int(row[0]): _coerce_int(row[1])
                for row in self.conn.execute("SELECT thread_id, listing_id FROM threads")
            }
        return None if thread_id is None else self._threads.get(thread_id)

    def _add_completion(self, event_id: int, tick: int, result: dict[str, Any]) -> None:
        thread_id = _int_field(result, "thread_id")
        thread = None if thread_id is None else self.conn.execute(
            "SELECT listing_id, buyer_agent_id, seller_agent_id FROM threads "
            "WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if thread is None:
            return
        seller = _coerce_int(thread["seller_agent_id"])
        buyer = _coerce_int(thread["buyer_agent_id"])
        listing_id = _coerce_int(thread["listing_id"])
        consumed = result.get("consumed_unit")
        sale = _Sale(
            event_id, tick, listing_id, seller, buyer,
            "consumed_unit" in result,
            consumed if isinstance(consumed, dict) else None,
            result.get("buyer_unit_index"),
        )
        if buyer is not None:
            self.purchases.setdefault(buyer, []).append(sale)
        if seller is not None:
            self.sales.setdefault(seller, []).append(sale)
        # The completion check binds at the handoff before the transfer
        # consumes, so the consumed record is the later one.
        self._add_binding(event_id, seller, result.get("bound_at_handoff"))
        self._add_binding(event_id, seller, consumed)
        self._add_quality(listing_id, event_id, 0, result, "bound_at_handoff")
        self._add_quality(listing_id, event_id, 1, result, "bound_at_sale")

    def _add_binding(self, event_id: int, owner: int | None, record: Any) -> None:
        if owner is None or not isinstance(record, dict):
            return
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            return
        self.bindings.setdefault(owner, {}).setdefault(index, []).append((event_id, record))

    def listing(self, listing_id: int | None) -> sqlite3.Row | None:
        if listing_id is None:
            return None
        if listing_id not in self._listings:
            self._listings[listing_id] = self.conn.execute(
                """
                SELECT listing_id, owner_agent_id, is_phantom, category, title,
                       description, condition, ground_truth_quality_pct,
                       created_at_tick
                FROM listings WHERE listing_id = ?
                """,
                (listing_id,),
            ).fetchone()
        return self._listings[listing_id]

    def listing_field_at(self, listing_id: int | None, event_id: int, key: str) -> Any:
        """``key`` (title, description or condition) of a listing as it
        was when event ``event_id`` ran: the last logged ``edit_listing``
        before it, else the ``create_listing`` payload, else the current
        row when no edit ever touched ``key``; otherwise ``_UNKNOWN``."""
        if listing_id is None:
            return _UNKNOWN
        edits = self.edits.get(listing_id, [])
        value = _UNKNOWN
        for edit_event_id, _tick, fields in edits:
            if edit_event_id >= event_id:
                break
            if key in fields:
                value = fields[key]
        if value is not _UNKNOWN:
            return value
        created = self.created.get(listing_id)
        if created is not None and key in created:
            return created[key]
        if any(key in fields for _event_id, _tick, fields in edits):
            return _UNKNOWN
        row = self.listing(listing_id)
        return _UNKNOWN if row is None else row[key]

    def window_alternative(
        self, listing_id: int | None, event_id: int, event_tick: int, key: str,
    ) -> Any:
        """The value ``key`` may have had at event ``event_id`` through an
        edit in the unlogged window of a truncated base, or ``_UNKNOWN``.

        An event after the window sees the listing as the window left it
        unless a logged edit after the window set ``key`` before it. When
        no logged edit after the window touched ``key`` at all, that value
        is the current row; otherwise the log cannot give it."""
        window = self.truncation
        if window is None or listing_id is None or event_tick <= window.target:
            return _UNKNOWN
        created_tick = self.created_tick.get(listing_id)
        if created_tick is not None and created_tick > window.target:
            return _UNKNOWN
        if any(
            tick > window.target and key in fields
            for _event_id, tick, fields in self.edits.get(listing_id, [])
        ):
            return _UNKNOWN
        row = self.listing(listing_id)
        return _UNKNOWN if row is None else row[key]

    def listing_quality_at(self, listing_id: int | None, event_id: int) -> Any:
        """The listing's ``ground_truth_quality_pct`` when completion event
        ``event_id`` transferred it (after a ``bound_at_handoff`` in that
        event, before its ``bound_at_sale``), rebuilt from the logged
        bindings (create_listing and relist ``backing_unit``,
        ``bound_at_handoff``, ``bound_at_sale``). A listing no logged
        binding ever wrote keeps its creation value, the current row;
        ``_UNKNOWN`` when only later bindings are logged."""
        if listing_id is None:
            return _UNKNOWN
        writes = self.quality.get(listing_id, [])
        value = _UNKNOWN
        for write_event, order, quality in writes:
            if (write_event, order) < (event_id, 1):
                value = quality
        if value is not _UNKNOWN or writes:
            return value
        row = self.listing(listing_id)
        return _UNKNOWN if row is None else row["ground_truth_quality_pct"]

    def purchase_price(self, sale: _Sale) -> int:
        """Price the transfer recorded: the buyer's latest accepted offer on
        the listing that existed at completion time, 0 when there is none
        (the legacy and unit helpers run the same query, latest offer id
        first, at completion time).

        An offer existed then when its tick is at most the sale's. A
        continuation of a truncated base also sees the offers of the
        unlogged window, which the tables kept although their ticks may
        follow the continuation's sale tick; such an offer counts only
        when the note fits the database, the sale was logged after the
        base log, the offer's tick lies in the window and no logged
        ``make_offer``/``counter_offer`` created it. The recorded price
        must then equal that offer's."""
        window = self.truncation
        continuation = window is not None and sale.event_id > window.boundary
        for offer_id, price, tick in self.conn.execute(
            """
            SELECT o.offer_id, o.price_cents, o.tick
            FROM offers o
            JOIN threads t ON t.thread_id = o.thread_id
            WHERE t.listing_id = ? AND t.buyer_agent_id = ?
              AND o.status = 'accepted'
            ORDER BY o.offer_id DESC
            """,
            (sale.listing_id, sale.buyer),
        ):
            if isinstance(tick, int) and not isinstance(tick, bool) and tick <= sale.tick:
                return int(price)
            if (
                continuation and window is not None and window.covers(tick)
                and int(offer_id) not in self.logged_offers()
            ):
                return int(price)
        return 0

    def logged_offers(self) -> frozenset[int]:
        """Offer ids a logged ``ok`` ``make_offer``/``counter_offer`` created."""
        if self._logged_offers is None:
            ids = (
                _coerce_int(row[0]) for row in self.conn.execute(
                    f"""
                    SELECT {_json_key('result_payload', 'offer_id')} FROM events
                    WHERE action_type IN ('make_offer', 'counter_offer')
                      AND result_status = 'ok'
                    """
                )
            )
            self._logged_offers = frozenset(i for i in ids if i is not None)
        return self._logged_offers

    def appends_unit(self, sale: _Sale) -> bool:
        """Whether a completion appended a unit to the buyer."""
        if sale.unit_branch:
            index = sale.buyer_unit_index
            return isinstance(index, int) and not isinstance(index, bool)
        row = self.listing(sale.listing_id)
        return (
            row is not None
            and int(row["is_phantom"] or 0) == 0
            and row["owner_agent_id"] is not None
        )


def _audit_registered_persona(
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    agent_id: int,
    registered_json: Any,
    current_json: Any,
    evidence: _InventoryEvidence,
) -> None:
    """Every persona field except ``inventory_items`` must still equal the
    registration payload, and every inventory change must be explained by
    the event log (see the section comment above). Writers re-serialise
    the persona (key order, ASCII escapes), so the comparison is on the
    parsed JSON; a missing key and a null-valued key differ."""
    action = "platform_register_agent"
    registered = _json_object(registered_json)
    current = _json_object(current_json)
    if registered is None or current is None:
        _expect_equal(
            issues, event, action, agent_id, "persona_json", registered_json, current_json,
        )
        return
    before_raw = registered.pop("inventory_items", _MISSING)
    after_raw = current.pop("inventory_items", _MISSING)
    for key in sorted(registered.keys() | current.keys()):
        _expect_equal(
            issues, event, action, agent_id, f"persona_json.{key}",
            _get(registered, key), _get(current, key),
        )
    before = [] if before_raw is _MISSING or before_raw is None else before_raw
    after = [] if after_raw is _MISSING or after_raw is None else after_raw
    if not isinstance(before, list) or not isinstance(after, list):
        _expect_equal(
            issues, event, action, agent_id, "persona_json.inventory_items",
            before_raw, after_raw,
        )
        return
    if not after:
        # Nothing was appended, so the key is exactly as registered
        # (a writer that appends creates the list; none removes it).
        _expect_equal(
            issues, event, action, agent_id, "persona_json.inventory_items",
            before_raw, after_raw,
        )
        return
    if len(after) < len(before):
        _add(
            issues, event, action,
            f"persona_json.inventory_items shrank: event={len(before)} table={len(after)}",
            ref_id=agent_id,
        )
        return
    _InventoryAudit(issues, event, agent_id, before, after, evidence).run()


class _InventoryAudit:
    """Cross-checks one agent's ``inventory_items`` against the log."""

    def __init__(
        self,
        issues: list[EventAuditIssue],
        event: sqlite3.Row,
        agent_id: int,
        before: list[Any],
        after: list[Any],
        evidence: _InventoryEvidence,
    ) -> None:
        self.issues = issues
        self.event = event
        self.agent_id = agent_id
        self.before = before
        self.after = after
        self.evidence = evidence
        self.window = evidence.truncation
        # Per unit: the values of the mutable keys when it came into the
        # inventory (``_UNKNOWN`` where the log does not pin them down,
        # ``_MISSING`` for an absent key), and the event that added it
        # (-1 for registered units).
        self.origins: list[dict[str, Any] | None] = [None] * len(after)
        self.added_at: list[int] = [-1] * len(after)

    def _field(self, index: int) -> str:
        return f"persona_json.inventory_items[{index}]"

    def _issue(self, message: str) -> None:
        _add(self.issues, self.event, "platform_register_agent", message, ref_id=self.agent_id)

    def _warn(self, message: str) -> None:
        """A change of the unlogged window of a truncated base."""
        assert self.window is not None
        _add(
            self.issues, self.event, "platform_register_agent",
            f"{message} (unlogged window of a truncated base: {self.window.citation})",
            ref_id=self.agent_id, severity="warning",
        )

    def _equal(self, field: str, expected: Any, actual: Any) -> None:
        _expect_equal(
            self.issues, self.event, "platform_register_agent", self.agent_id,
            field, expected, actual,
        )

    def run(self) -> None:
        self._audit_registered_units()
        if not self._audit_appended_units():
            # Units cannot be paired with their events; the mismatch is
            # already reported.
            return
        self._audit_bindings()
        self._audit_sold_marks()

    def _audit_registered_units(self) -> None:
        for index, unit in enumerate(self.before):
            now = self.after[index]
            if not isinstance(unit, dict) or not isinstance(now, dict):
                self._equal(self._field(index), unit, now)
                continue
            for key in sorted(unit.keys() | now.keys()):
                if key not in _UNIT_MUTABLE_KEYS:
                    self._equal(f"{self._field(index)}.{key}", _get(unit, key), _get(now, key))
            self.origins[index] = {key: _get(unit, key) for key in _UNIT_MUTABLE_KEYS}

    def _audit_appended_units(self) -> bool:
        restocked: list[int] = []
        bought: list[int] = []
        paired = True
        for index in range(len(self.before), len(self.after)):
            unit = self.after[index]
            if not isinstance(unit, dict):
                self._issue(f"{self._field(index)} appended non-object unit")
                paired = False
                continue
            source = unit.get("source")
            if source == "restock":
                restocked.append(index)
            elif source == "bought":
                bought.append(index)
            else:
                self._issue(f"{self._field(index)} appended with unexplained source={source!r}")
                paired = False
        paired = self._audit_restocked_units(restocked) and paired
        paired = self._audit_bought_units(bought) and paired
        if not paired:
            return False
        last = -1
        for index in range(len(self.before), len(self.after)):
            if self.added_at[index] < last:
                self._issue(f"{self._field(index)} appended out of event order")
                return False
            last = self.added_at[index]
        return True

    def _pair_with_window(
        self,
        indices: list[int],
        expected: list[tuple[Any, Any]],
        key: Callable[[dict[str, Any]], tuple[Any, Any]],
        tick: Callable[[dict[str, Any]], Any],
    ) -> list[int | None] | None:
        """Pair appended units with their logged events in order when a
        truncated base's window explains the rest: per unit the position
        of its event, or None for a unit whose tick lies in the unlogged
        window. None when the units cannot be paired that way."""
        if self.window is None:
            return None
        positions: list[int | None] = []
        position = 0
        for index in indices:
            unit = self.after[index]
            if position < len(expected) and key(unit) == expected[position]:
                positions.append(position)
                position += 1
            elif self.window.covers(tick(unit)):
                positions.append(None)
            else:
                return None
        return positions if position == len(expected) else None

    def _window_unit(self, index: int, what: str, *, warn: bool = True) -> None:
        if warn:
            self._warn(f"{self._field(index)} {what} that no event logs")
        self.added_at[index] = self.window.boundary if self.window is not None else -1
        self.origins[index] = {
            "unit_uid": _UNKNOWN,
            "quality_source": _UNKNOWN,
            "ground_truth_quality_pct": _UNKNOWN,
            "sold_at_tick": _MISSING,
            "sold_via_listing_id": _MISSING,
        }

    def _window_listing_problem(self, listing_id: Any, tick: Any, *, seller: bool) -> str | None:
        """Why a window change cannot involve ``listing_id`` at ``tick``:
        the listing must exist, be created by then and be this agent's
        (``seller``) or another agent's; None when it can."""
        listing = self.evidence.listing(
            None if isinstance(listing_id, bool) else _coerce_int(listing_id),
        )
        if listing is None:
            return "the listing does not exist"
        created = _coerce_int(listing["created_at_tick"])
        if created is not None and isinstance(tick, int) and created > tick:
            return f"the listing was created at tick {created}, after it"
        owner = _coerce_int(listing["owner_agent_id"])
        if seller and owner != self.agent_id:
            return f"the listing belongs to agent {owner!r}, not this agent"
        if not seller and (owner is None or owner == self.agent_id):
            return f"the listing belongs to agent {owner!r}, not a seller this agent bought from"
        return None

    def _window_purchase_problem(self, unit: dict[str, Any]) -> str | None:
        listing_id = unit.get("bought_from_listing_id")
        problem = self._window_listing_problem(listing_id, unit.get("bought_tick"), seller=False)
        if problem is not None:
            return problem
        thread = self.evidence.conn.execute(
            "SELECT 1 FROM threads WHERE listing_id = ? AND buyer_agent_id = ? "
            "AND status = 'completed'",
            (_coerce_int(listing_id), self.agent_id),
        ).fetchone()
        if thread is None:
            return "this agent has no completed thread on the listing"
        return None

    # -- restocks --------------------------------------------------------

    def _audit_restocked_units(self, indices: list[int]) -> bool:
        expected: list[tuple[int, int, Any, dict[str, Any]]] = []
        for event_id, tick, payload in self.evidence.restocks.get(self.agent_id, []):
            titles = payload.get("templates_used")
            for title in titles if isinstance(titles, list) else []:
                expected.append((event_id, tick, title, payload))
        logged = [(tick, title) for _event_id, tick, title, _payload in expected]
        table = [(self.after[i].get("added_at_tick"), self.after[i].get("title")) for i in indices]
        positions = None if logged == table else self._pair_with_window(
            indices, logged,
            lambda unit: (_get(unit, "added_at_tick"), _get(unit, "title")),
            lambda unit: _get(unit, "added_at_tick"),
        )
        if positions is None:
            self._equal("persona_json.inventory_items restocks", logged, table)
            if len(expected) != len(indices):
                return False
            positions = list(range(len(indices)))
        batch_start: dict[int, int] = {}
        for index, position in zip(indices, positions, strict=True):
            if position is None:
                # A unit of the window must still be what D_restock makes
                # from one of the agent's units.
                self._audit_restocked_unit(index, index, None)
                self._window_unit(
                    index, f"restocked at tick {self.after[index].get('added_at_tick')!r}",
                )
                continue
            event_id, _tick, _title, payload = expected[position]
            self.added_at[index] = event_id
            start = batch_start.setdefault(event_id, index)
            self._audit_restocked_unit(index, start, payload)
            self.origins[index] = {
                "unit_uid": _MISSING,
                "quality_source": _MISSING,
                # Drawn at random inside the template's band and not logged.
                "ground_truth_quality_pct": _UNKNOWN,
                "sold_at_tick": _MISSING,
                "sold_via_listing_id": _MISSING,
            }
        return True

    def _audit_restocked_unit(
        self, index: int, start: int, payload: dict[str, Any] | None,
    ) -> None:
        """A restocked unit copies one of the agent's units at restock time.

        ``platform_inventory_restocked`` logs only the titles, the tier and
        the sales count, not the price or quality of the new units, and
        the event payload stays as it is so legacy runs stay byte
        identical. So the unit is checked against what ``D_restock``
        derives from its template (a unit before this restock batch):
        title, category, condition, stated band and asking price copied,
        acquisition cost 55% of the asking price, quality inside the
        band. Known gap: a quality changed to another value inside the
        band, or a price changed to the price of another unit with the
        same title and fields, cannot be told from the real draw.

        ``payload`` is None for a unit of the unlogged window of a
        truncated base: its tier cannot be compared with a logged one, so
        only the pair of keys is checked (both or neither, a known
        reason); its templates are the units before it.
        """
        from bazaar.dynamics.callbacks import _RESTOCK_PROFIT_HOMING

        field = self._field(index)
        unit = self.after[index]
        for key in sorted(unit.keys() - _RESTOCK_UNIT_KEYS - _UNIT_MUTABLE_KEYS):
            self._issue(f"{field} unexpected key {key!r} on a restocked unit")
        if payload is None:
            tier, reason = _get(unit, "restock_tier"), _get(unit, "restock_reason")
            if (tier is _MISSING) != (reason is _MISSING) or (
                tier is not _MISSING
                and (not isinstance(tier, str) or reason not in _RESTOCK_REASONS)
            ):
                self._issue(
                    f"{field} restock_tier/restock_reason do not match D_restock: "
                    f"{tier!r}/{reason!r}"
                )
        elif "marketplace_tier" in payload:
            sales = _coerce_int(payload.get("sales_window_count")) or 0
            tier, reason = (
                payload.get("marketplace_tier"),
                "recent_sales" if sales > 0 else "background_supply",
            )
        else:
            # An older restock logged no tier and wrote neither key.
            tier = reason = _MISSING
        for key in _RESTOCK_BASE_KEYS:
            if key not in unit:
                self._issue(f"{field} restocked unit lacks key {key!r}")
        self._equal(f"{field}.restock_tier", tier, _get(unit, "restock_tier"))
        self._equal(f"{field}.restock_reason", reason, _get(unit, "restock_reason"))
        derived = []
        for template in self.after[:start]:
            if not isinstance(template, dict):
                continue
            try:
                asking = int(template.get("asking_price_cents") or 1000)
            except (TypeError, ValueError, OverflowError):
                continue
            derived.append({
                "title": template.get("title"),
                "category": template.get("category") or "misc",
                "condition": template.get("condition") or "good",
                "stated_quality_band": template.get("stated_quality_band") or "good",
                "asking_price_cents": asking,
                "acquisition_cost_cents": int(asking * _RESTOCK_PROFIT_HOMING),
            })
        actual = {key: _get(unit, key) for key in derived[0]} if derived else {}
        if actual not in derived:
            self._issue(f"{field} restocked unit matches no template: {actual!r}")
        band = unit.get("stated_quality_band")
        low, high = next(
            ((lo, hi) for name, lo, hi in QUALITY_BAND_RANGES if name == band), (60, 81),
        )
        quality = unit.get("ground_truth_quality_pct")
        if (
            isinstance(quality, bool) or not isinstance(quality, int)
            or not low <= quality <= high
        ):
            self._issue(
                f"{field}.ground_truth_quality_pct outside the restocked band "
                f"{band!r} ({low}-{high}): {quality!r}"
            )

    # -- purchases -------------------------------------------------------

    def _audit_bought_units(self, indices: list[int]) -> bool:
        purchases = [
            sale for sale in self.evidence.purchases.get(self.agent_id, [])
            if self.evidence.appends_unit(sale)
        ]
        logged = [(sale.listing_id, sale.tick) for sale in purchases]
        table = [
            (self.after[i].get("bought_from_listing_id"), self.after[i].get("bought_tick"))
            for i in indices
        ]
        positions = None if logged == table else self._pair_with_window(
            indices, logged,
            lambda unit: (_get(unit, "bought_from_listing_id"), _get(unit, "bought_tick")),
            lambda unit: _get(unit, "bought_tick"),
        )
        if positions is None:
            self._equal("persona_json.inventory_items purchases", logged, table)
            if len(purchases) != len(indices):
                return False
            positions = list(range(len(indices)))
        for index, position in zip(indices, positions, strict=True):
            if position is None:
                unit = self.after[index]
                problem = self._window_purchase_problem(unit)
                if problem is not None:
                    self._issue(
                        f"{self._field(index)} bought at tick {unit.get('bought_tick')!r} "
                        f"from listing {unit.get('bought_from_listing_id')!r}: {problem}"
                    )
                self._window_unit(
                    index,
                    f"bought at tick {unit.get('bought_tick')!r} from listing "
                    f"{unit.get('bought_from_listing_id')!r}",
                    warn=problem is None,
                )
                continue
            sale = purchases[position]
            self.added_at[index] = sale.event_id
            self._audit_bought_unit(index, sale)
        return True

    def _audit_bought_unit(self, index: int, sale: _Sale) -> None:
        evidence = self.evidence
        field = self._field(index)
        unit = self.after[index]
        for key in sorted(unit.keys() - _BOUGHT_UNIT_KEYS - _UNIT_MUTABLE_KEYS):
            self._issue(f"{field} unexpected key {key!r} on a bought unit")
        for key in _BOUGHT_BASE_KEYS:
            if key not in unit:
                self._issue(f"{field} bought unit lacks key {key!r}")
        if sale.unit_branch:
            self._equal(f"{field} buyer_unit_index", sale.buyer_unit_index, index)
        listing = evidence.listing(sale.listing_id)
        price = evidence.purchase_price(sale)
        expected: dict[str, Any] = {
            "source": "bought",
            "bought_from_listing_id": sale.listing_id,
            "bought_tick": sale.tick,
            "bought_price_cents": price,
            "asking_price_cents": price,
            "category": _UNKNOWN if listing is None else listing["category"],
            "title": evidence.listing_field_at(sale.listing_id, sale.event_id, "title"),
            "condition": evidence.listing_field_at(
                sale.listing_id, sale.event_id, "condition",
            ),
            # Both transfers add these two keys only for a consumed unit.
            "bought_listing_title": _MISSING,
            "bought_from_unit_uid": _MISSING,
        }
        description = evidence.listing_field_at(sale.listing_id, sale.event_id, "description")
        expected["description"] = description if description is _UNKNOWN else description or ""
        origin: dict[str, Any] = {
            "unit_uid": _MISSING,
            "quality_source": _MISSING,
            # The legacy transfer copies no quality.
            "ground_truth_quality_pct": _MISSING,
            "sold_at_tick": _MISSING,
            "sold_via_listing_id": _MISSING,
        }
        consumed = sale.consumed
        listing_content = not sale.unit_branch or consumed is None
        if sale.unit_branch and consumed is None:
            # The unit-aware transfer consumed nothing and copied the
            # listing's quality at completion time (the key is left out
            # when that quality is NULL), rebuilt from the logged bindings.
            quality = evidence.listing_quality_at(sale.listing_id, sale.event_id)
            origin["ground_truth_quality_pct"] = _MISSING if quality is None else quality
        elif consumed is not None:
            origin["ground_truth_quality_pct"] = consumed.get("quality")
            origin["quality_source"] = consumed.get("quality_source") or "stored"
            expected["bought_from_unit_uid"] = consumed.get("unit_uid")
            expected["bought_listing_title"] = expected["title"]
            self._expect_consumed_content(expected, sale, consumed)
        for key, value in expected.items():
            if value is _UNKNOWN:
                continue
            actual = _get(unit, key)
            if (
                value != actual and listing_content
                and key in ("title", "description", "condition")
            ):
                alternative = evidence.window_alternative(
                    sale.listing_id, sale.event_id, sale.tick, key,
                )
                if alternative is not _UNKNOWN and key == "description":
                    alternative = alternative or ""
                if alternative is not _UNKNOWN and actual == alternative:
                    self._warn(
                        f"{field}.{key} is the listing's value after an edit in the "
                        f"window, not the last logged value {value!r}"
                    )
                    continue
            self._equal(f"{field}.{key}", value, actual)
        self.origins[index] = origin

    def _expect_consumed_content(
        self, expected: dict[str, Any], sale: _Sale, consumed: dict[str, Any],
    ) -> None:
        """The unit-aware transfer describes the unit that changed hands:
        the seller unit the ``consumed_unit`` record names."""
        seller_unit = None
        if sale.seller is not None:
            row = self.evidence.conn.execute(
                "SELECT persona_json FROM agents WHERE agent_id = ?", (sale.seller,),
            ).fetchone()
            persona = None if row is None else _json_object(row["persona_json"])
            units = None if persona is None else persona.get("inventory_items")
            index = consumed.get("index")
            if (
                isinstance(units, list) and isinstance(index, int)
                and not isinstance(index, bool) and 0 <= index < len(units)
                and isinstance(units[index], dict)
            ):
                seller_unit = units[index]
        if seller_unit is None:
            for key in ("category", "condition", "description", "title"):
                expected[key] = _UNKNOWN
            return
        for key in ("category", "condition", "title"):
            expected[key] = seller_unit.get(key) or expected[key]
        expected["description"] = seller_unit.get("description") or ""

    # -- bindings --------------------------------------------------------

    def _audit_bindings(self) -> None:
        bindings = self.evidence.bindings.get(self.agent_id, {})
        for index, records in sorted(bindings.items()):
            if not 0 <= index < len(self.after) or not isinstance(self.after[index], dict):
                self._issue(f"binding logged for a unit the agent does not have: index={index}")
                continue
            if records[0][0] < self.added_at[index]:
                self._issue(f"{self._field(index)} binding logged before the unit existed")
        for index, unit in enumerate(self.after):
            origin = self.origins[index]
            if not isinstance(unit, dict) or origin is None:
                continue
            expected = {key: origin[key] for key in _UNIT_BINDING_KEYS}
            records = bindings.get(index)
            if records:
                record = records[-1][1]
                expected = {
                    "unit_uid": _get(record, "unit_uid"),
                    "quality_source": _get(record, "quality_source"),
                    "ground_truth_quality_pct": _get(record, "quality"),
                }
            for key, value in expected.items():
                if value is not _UNKNOWN:
                    self._equal(f"{self._field(index)}.{key}", value, _get(unit, key))

    # -- sales -----------------------------------------------------------

    def _audit_sold_marks(self) -> None:
        """Replay the agent's sales in event order.

        A unit-aware sale marks the unit its ``consumed_unit`` record
        names. A legacy sale marks the unit the legacy title match picks
        (:func:`bazaar.actions.handlers._consume_seller_inventory_for_listing`)
        among the units held and unsold at that event, for the listing
        title of that moment. When that title is not known (an edited
        listing without a logged ``create_listing``), that sale and the
        agent's later legacy sales only allow a mark that names one of
        them.

        In a truncated base, a mark in the unlogged window that no logged
        sale names is a warning, and the unit counts as sold for the
        replay of the sales after the window; a sale after the window
        whose listing title the window may have changed is replayed as
        unknown (such marks are warnings too)."""
        exact: dict[int, tuple[int, int | None]] = {}
        loose: list[_Sale] = []
        loose_by_window = False
        window = self.window
        sales = self.evidence.sales.get(self.agent_id, [])
        sold = [
            isinstance(unit, dict) and origin is not None
            and origin["sold_at_tick"] not in (None, _MISSING)
            for unit, origin in zip(self.after, self.origins, strict=True)
        ]
        window_sold: dict[int, tuple[Any, Any]] = {}
        if window is not None:
            # A window mark is a sale no event logs when more units carry
            # it than logged sales name it (a continuation may log a sale
            # with the same tick and listing: a legacy meetup left on a
            # listing the window sold can complete it again).
            logged = Counter((sale.tick, sale.listing_id) for sale in sales)
            carriers: dict[tuple[Any, Any], list[int]] = {}
            for index, unit in enumerate(self.after):
                if not isinstance(unit, dict) or self.origins[index] is None or sold[index]:
                    continue
                mark = (_get(unit, "sold_at_tick"), _get(unit, "sold_via_listing_id"))
                if window.covers(mark[0]) and self._window_listing_problem(
                    mark[1], mark[0], seller=True,
                ) is None:
                    carriers.setdefault(mark, []).append(index)
            for mark, indices in carriers.items():
                for index in indices[:max(0, len(indices) - logged[mark])]:
                    window_sold[index] = mark
        after_window = False
        for sale in sales:
            if window is not None and not after_window and sale.tick > window.target:
                after_window = True
                for index in window_sold:
                    if index not in exact:
                        sold[index] = True
            if sale.unit_branch:
                index = None if sale.consumed is None else sale.consumed.get("index")
                if (
                    isinstance(index, int) and not isinstance(index, bool)
                    and 0 <= index < len(self.after)
                ):
                    exact[index] = (sale.tick, sale.listing_id)
                    sold[index] = True
                continue
            title = (
                _UNKNOWN if loose
                else self.evidence.listing_field_at(sale.listing_id, sale.event_id, "title")
            )
            held = [
                index for index, unit in enumerate(self.after)
                if isinstance(unit, dict)
                and self.added_at[index] < sale.event_id
                and not sold[index]
            ]
            if title is not _UNKNOWN:
                alternative = self.evidence.window_alternative(
                    sale.listing_id, sale.event_id, sale.tick, "title",
                )
                if (
                    alternative is not _UNKNOWN and alternative != title
                    and _legacy_title_match(self.after, alternative, held)
                    != _legacy_title_match(self.after, title, held)
                ):
                    title = _UNKNOWN
                    loose_by_window = True
            if title is _UNKNOWN:
                loose.append(sale)
                continue
            index = _legacy_title_match(self.after, title, held)
            if index is not None:
                exact[index] = (sale.tick, sale.listing_id)
                sold[index] = True
        used: set[int] = set()
        for index, unit in enumerate(self.after):
            origin = self.origins[index]
            if not isinstance(unit, dict) or origin is None:
                continue
            field = self._field(index)
            actual = (_get(unit, "sold_at_tick"), _get(unit, "sold_via_listing_id"))
            if index in exact:
                expected = exact[index]
            else:
                expected = (origin["sold_at_tick"], origin["sold_via_listing_id"])
                if actual != expected and window_sold.get(index) == actual:
                    self._warn(
                        f"{field} marked sold at tick {actual[0]!r} via listing "
                        f"{actual[1]!r}, a sale no event logs"
                    )
                    continue
                if actual != expected and loose:
                    match = next(
                        (
                            k for k, sale in enumerate(loose)
                            if k not in used
                            and (sale.tick, sale.listing_id) == actual
                            and self.added_at[index] < sale.event_id
                        ),
                        None,
                    )
                    if match is not None:
                        used.add(match)
                        if loose_by_window:
                            self._warn(
                                f"{field} sold mark matched to a logged sale whose listing "
                                "title the window may have changed"
                            )
                        continue
            for key, value, now in zip(_UNIT_SOLD_KEYS, expected, actual, strict=True):
                self._equal(f"{field}.{key}", value, now)


def _legacy_title_match(units: list[Any], title: Any, held: list[int]) -> int | None:
    """The unit the legacy transfer marks sold, as
    :func:`bazaar.actions.handlers._consume_seller_inventory_for_listing`
    picks it: the longest unit title contained in (or containing) the
    listing title, else the best ``SequenceMatcher`` ratio if >= 0.70."""
    needle = str(title or "").strip().lower()
    if not needle:
        return None
    substring_idx, substring_len = -1, 0
    fuzzy_idx, fuzzy_score = -1, 0.0
    for index in held:
        cand = str(units[index].get("title") or "").strip().lower()
        if not cand:
            continue
        if (cand in needle or needle in cand) and len(cand) > substring_len:
            substring_idx, substring_len = index, len(cand)
        score = SequenceMatcher(None, cand, needle).ratio()
        if score > fuzzy_score:
            fuzzy_idx, fuzzy_score = index, score
    if substring_idx >= 0:
        return substring_idx
    if fuzzy_score >= 0.70:
        return fuzzy_idx
    return None


def _audit_phantom_seed_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    action = "platform_seed_phantom_listing"
    for event in _event_rows(conn, action):
        payload = _payload(event)
        listing_id = payload.get("listing_id")
        if listing_id is None:
            _add(issues, event, action, "missing listing_id")
            continue
        ref_id = int(listing_id)
        row = conn.execute(
            """
            SELECT listing_id, owner_agent_id, category, price_cents, is_phantom
            FROM listings
            WHERE listing_id = ?
            """,
            (ref_id,),
        ).fetchone()
        if row is None:
            _add(issues, event, action, "listing row missing", ref_id=ref_id)
            continue
        _expect_equal(issues, event, action, ref_id, "category", payload.get("category"), row["category"])
        _expect_equal(issues, event, action, ref_id, "price_cents", payload.get("price_cents"), row["price_cents"])
        _expect_equal(issues, event, action, ref_id, "is_phantom", True, bool(row["is_phantom"]))
        _expect_equal(issues, event, action, ref_id, "owner_agent_id", None, row["owner_agent_id"])


def _audit_real_seed_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    action = "platform_seed_real_listing"
    edits: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    truncation = _Truncation.load(conn)
    for index, event in enumerate(_event_rows(conn, action)):
        if index == 0:
            edits = _listing_edits(conn)
        payload = _payload(event)
        listing_id = payload.get("listing_id")
        if listing_id is None:
            _add(issues, event, action, "missing listing_id")
            continue
        ref_id = int(listing_id)
        row = conn.execute(
            """
            SELECT listing_id, owner_agent_id, category, price_cents,
                   condition, is_phantom
            FROM listings
            WHERE listing_id = ?
            """,
            (ref_id,),
        ).fetchone()
        if row is None:
            _add(issues, event, action, "listing row missing", ref_id=ref_id)
            continue
        for field in ("owner_agent_id", "category"):
            _expect_equal(issues, event, action, ref_id, field, payload.get(field), row[field])
        # The owner may edit the price and condition of a seeded listing
        # like any other (``edit_listing``, under every handoff check):
        # the expected value is the seed payload's, replayed through the
        # logged edits.
        edited = _edited_listing_fields(edits, ref_id, after=int(event["event_id"]))
        for field in ("price_cents", "condition"):
            expected = edited.get(field, payload.get(field))
            if (
                expected != row[field] and truncation is not None
                and not _edited_after(edits, ref_id, field, truncation.target)
            ):
                _add(
                    issues, event, action,
                    f"{field} mismatch: event={expected!r} table={row[field]!r} (the listing "
                    "was open in the unlogged window of a truncated base and no logged edit "
                    f"after it sets {field}: {truncation.citation})",
                    ref_id=ref_id, severity="warning",
                )
                continue
            if expected != row[field] and field in edited:
                _add(
                    issues, event, action,
                    f"{field} mismatch: event={payload.get(field)!r} replayed through "
                    f"edit_listing={expected!r} table={row[field]!r}",
                    ref_id=ref_id,
                )
                continue
            _expect_equal(issues, event, action, ref_id, field, expected, row[field])
        _expect_equal(issues, event, action, ref_id, "is_phantom", False, bool(row["is_phantom"]))


def _listing_edits(
    conn: sqlite3.Connection,
) -> dict[int, list[tuple[int, int, dict[str, Any]]]]:
    """listing id -> ``[(event_id, tick, payload)]`` of the ``ok``
    ``edit_listing`` events, in event order."""
    edits: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    for row in conn.execute(
        """
        SELECT event_id, tick, payload FROM events
        WHERE action_type = 'edit_listing' AND result_status = 'ok'
        ORDER BY event_id
        """
    ):
        payload = _payload(row)
        listing_id = _int_field(payload, "listing_id")
        if listing_id is not None:
            edits.setdefault(listing_id, []).append(
                (int(row["event_id"]), int(row["tick"]), payload),
            )
    return edits


def _edited_listing_fields(
    edits: dict[int, list[tuple[int, int, dict[str, Any]]]], listing_id: int, *, after: int,
) -> dict[str, Any]:
    """The latest value of each field the ``ok`` ``edit_listing`` events
    after event ``after`` set on a listing (``None`` arguments set
    nothing, as in the handler)."""
    fields: dict[str, Any] = {}
    for event_id, _tick, payload in edits.get(listing_id, []):
        if event_id <= after:
            continue
        for key in ("title", "description", "price_cents", "condition"):
            if payload.get(key) is not None:
                fields[key] = payload[key]
    return fields


def _edited_after(
    edits: dict[int, list[tuple[int, int, dict[str, Any]]]], listing_id: int,
    field: str, tick: int,
) -> bool:
    """Whether a logged edit after ``tick`` set ``field`` on the listing."""
    return any(
        edit_tick > tick and payload.get(field) is not None
        for _event_id, edit_tick, payload in edits.get(listing_id, [])
    )


def _audit_lot_sale_seed_events(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    action = "platform_seed_lot_sales_feed"
    for event in _event_rows(conn, action):
        payload = _payload(event)
        seed_agents = payload.get("seed_agents") or []
        if not isinstance(seed_agents, list):
            _add(issues, event, action, "seed_agents must be a list")
            seed_agents = []
        for agent in seed_agents:
            if isinstance(agent, dict):
                _audit_lot_seed_agent(conn, issues, event, agent)
        seed_sales = payload.get("seed_sales") or []
        if not isinstance(seed_sales, list):
            _add(issues, event, action, "seed_sales must be a list")
            continue
        for sale in seed_sales:
            if isinstance(sale, dict):
                _audit_lot_seed_sale(conn, issues, event, sale)


def _audit_lot_seed_agent(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    agent: dict[str, Any],
) -> None:
    action = "platform_seed_lot_sales_feed"
    aid = agent.get("agent_id")
    if aid is None:
        _add(issues, event, action, "seed agent missing agent_id")
        return
    ref_id = int(aid)
    row = conn.execute(
        """
        SELECT agent_id, user_name, display_name, is_seeded, persona_json
        FROM agents
        WHERE agent_id = ?
        """,
        (ref_id,),
    ).fetchone()
    if row is None:
        _add(issues, event, action, "seed agent row missing", ref_id=ref_id)
        return
    for field in ("user_name", "display_name", "persona_json"):
        _expect_equal(issues, event, action, ref_id, field, agent.get(field), row[field])
    _expect_equal(issues, event, action, ref_id, "is_seeded", bool(agent.get("is_seeded")), bool(row["is_seeded"]))


def _audit_lot_seed_sale(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    sale: dict[str, Any],
) -> None:
    action = "platform_seed_lot_sales_feed"
    listing_id = sale.get("listing_id")
    if listing_id is None:
        _add(issues, event, action, "seed sale missing listing_id")
        return
    ref_id = int(listing_id)
    listing = conn.execute(
        """
        SELECT listing_id, owner_agent_id, category, title, price_cents,
               condition, created_at_tick, sold_at_tick, status, is_seeded
        FROM listings
        WHERE listing_id = ?
        """,
        (ref_id,),
    ).fetchone()
    if listing is None:
        _add(issues, event, action, "seed sale listing missing", ref_id=ref_id)
        return
    for field in (
        "owner_agent_id", "category", "title", "price_cents", "condition",
        "created_at_tick", "sold_at_tick",
    ):
        _expect_equal(issues, event, action, ref_id, field, sale.get(field), listing[field])
    _expect_equal(issues, event, action, ref_id, "status", "sold", listing["status"])
    _expect_equal(issues, event, action, ref_id, "is_seeded", True, bool(listing["is_seeded"]))
    _audit_lot_seed_thread(conn, issues, event, sale, ref_id)
    _audit_lot_seed_offer(conn, issues, event, sale, ref_id)
    _audit_lot_seed_rating(conn, issues, event, sale, ref_id)


def _audit_lot_seed_thread(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    sale: dict[str, Any],
    listing_id: int,
) -> None:
    action = "platform_seed_lot_sales_feed"
    thread_id = sale.get("thread_id")
    if thread_id is None:
        _add(issues, event, action, "seed sale missing thread_id", ref_id=listing_id)
        return
    row = conn.execute(
        """
        SELECT thread_id, listing_id, buyer_agent_id, seller_agent_id, status
        FROM threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if row is None:
        _add(issues, event, action, "seed sale thread missing", ref_id=listing_id)
        return
    for field in ("listing_id", "buyer_agent_id", "seller_agent_id"):
        _expect_equal(issues, event, action, listing_id, field, sale.get(field), row[field])
    _expect_equal(issues, event, action, listing_id, "thread_status", "completed", row["status"])


def _audit_lot_seed_offer(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    sale: dict[str, Any],
    listing_id: int,
) -> None:
    action = "platform_seed_lot_sales_feed"
    offer_id = sale.get("offer_id")
    if offer_id is None:
        _add(issues, event, action, "seed sale missing offer_id", ref_id=listing_id)
        return
    row = conn.execute(
        """
        SELECT offer_id, thread_id, price_cents, status
        FROM offers
        WHERE offer_id = ?
        """,
        (offer_id,),
    ).fetchone()
    if row is None:
        _add(issues, event, action, "seed sale offer missing", ref_id=listing_id)
        return
    _expect_equal(issues, event, action, listing_id, "offer_thread_id", sale.get("thread_id"), row["thread_id"])
    _expect_equal(issues, event, action, listing_id, "offer_price_cents", sale.get("offer_price_cents"), row["price_cents"])
    _expect_equal(issues, event, action, listing_id, "offer_status", "accepted", row["status"])


def _audit_lot_seed_rating(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
    event: sqlite3.Row,
    sale: dict[str, Any],
    listing_id: int,
) -> None:
    action = "platform_seed_lot_sales_feed"
    rating_id = sale.get("rating_id")
    if rating_id is None:
        _add(issues, event, action, "seed sale missing rating_id", ref_id=listing_id)
        return
    row = conn.execute(
        """
        SELECT rating_id, thread_id, stars
        FROM ratings
        WHERE rating_id = ?
        """,
        (rating_id,),
    ).fetchone()
    if row is None:
        _add(issues, event, action, "seed sale rating missing", ref_id=listing_id)
        return
    _expect_equal(issues, event, action, listing_id, "rating_thread_id", sale.get("thread_id"), row["thread_id"])
    _expect_equal(issues, event, action, listing_id, "stars", sale.get("stars"), row["stars"])


def _audit_seed_row_coverage(
    conn: sqlite3.Connection,
    issues: list[EventAuditIssue],
) -> None:
    event_listing_ids = {
        int(row["listing_id"])
        for row in conn.execute(
            """
            SELECT json_extract(value, '$.listing_id') AS listing_id
            FROM (
                SELECT payload
                FROM events
                WHERE action_type = 'platform_seed_lot_sales_feed'
                  AND json_valid(payload)
            ) AS events,
                 json_each(events.payload, '$.seed_sales')
            WHERE json_extract(value, '$.listing_id') IS NOT NULL
            """
        ).fetchall()
    }
    seeded_listing_ids = {
        int(row["listing_id"])
        for row in conn.execute(
            "SELECT listing_id FROM listings WHERE is_seeded = 1"
        ).fetchall()
    }
    for listing_id in sorted(seeded_listing_ids - event_listing_ids):
        _add(
            issues, None, "platform_seed_lot_sales_feed",
            "seeded listing has no covering event",
            ref_id=listing_id,
        )

    event_agent_ids = {
        int(row["agent_id"])
        for row in conn.execute(
            """
            SELECT json_extract(value, '$.agent_id') AS agent_id
            FROM (
                SELECT payload
                FROM events
                WHERE action_type = 'platform_seed_lot_sales_feed'
                  AND json_valid(payload)
            ) AS events,
                 json_each(events.payload, '$.seed_agents')
            WHERE json_extract(value, '$.agent_id') IS NOT NULL
            """
        ).fetchall()
    }
    seeded_agent_ids = {
        int(row["agent_id"])
        for row in conn.execute(
            "SELECT agent_id FROM agents WHERE is_seeded = 1"
        ).fetchall()
    }
    for agent_id in sorted(seeded_agent_ids - event_agent_ids):
        _add(
            issues, None, "platform_seed_lot_sales_feed",
            "seeded agent has no covering event",
            ref_id=agent_id,
        )

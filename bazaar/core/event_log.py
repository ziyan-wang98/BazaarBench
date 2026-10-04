"""Append-only event log.

Every action that mutates state writes a single row to ``events`` through
this module.  Derived tables (listings, messages, etc.) are updated in
the same transaction.  Counterfactual replay operates on the event log.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

VALID_RESULT_STATUSES = frozenset({"ok", "error", "blocked"})


@dataclass
class Event:
    event_id: int
    tick: int
    wall_time: str
    agent_id: int | None
    action_type: str
    payload: dict[str, Any]
    result_status: str
    result_payload: dict[str, Any] | None


def require_lastrowid(cur: sqlite3.Cursor, *, table: str) -> int:
    rowid = cur.lastrowid
    if rowid is None:
        raise RuntimeError(f"insert into {table} did not produce a rowid")
    return int(rowid)


def _require_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_non_empty_string(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _dump_json_object(
    value: dict[str, Any] | None,
    *,
    field: str,
    allow_none: bool = False,
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{field} must be a JSON object, got None")
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object dict")
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-serializable") from exc


def _dump_json_object_list(
    value: list[dict[str, Any]] | None,
    *,
    field: str,
    allow_none: bool = False,
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{field} must be a JSON array of objects, got None")
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array of object dicts")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{field} must be a JSON array of object dicts")
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-serializable") from exc


def log_event(
    conn: sqlite3.Connection,
    *,
    tick: int,
    agent_id: int | None,
    action_type: str,
    payload: dict[str, Any],
    result_status: str = "ok",
    result_payload: dict[str, Any] | None = None,
) -> int:
    """Append one row to ``events`` and return its ``event_id``.

    The caller is responsible for wrapping this plus any derived-table
    mutations in a single transaction (``with conn:``).
    """
    tick = _require_int(tick, field="tick")
    agent_id = _require_optional_int(agent_id, field="agent_id")
    action_type = _require_non_empty_string(action_type, field="action_type")
    if result_status not in VALID_RESULT_STATUSES:
        allowed = ", ".join(sorted(VALID_RESULT_STATUSES))
        raise ValueError(f"invalid result_status {result_status!r}; expected one of: {allowed}")
    payload_json = _dump_json_object(payload, field="payload")
    result_payload_json = _dump_json_object(
        result_payload,
        field="result_payload",
        allow_none=True,
    )
    wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO events
            (tick, wall_time, agent_id, action_type, payload,
             result_status, result_payload)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tick,
            wall,
            agent_id,
            action_type,
            payload_json,
            result_status,
            result_payload_json,
        ),
    )
    return require_lastrowid(cur, table="events")


def replay_iter(conn: sqlite3.Connection, start_tick: int = 0):
    """Yield ``Event`` objects in chronological order from ``start_tick``."""
    for row in conn.execute(
        "SELECT * FROM events WHERE tick >= ? ORDER BY event_id",
        (start_tick,),
    ):
        yield Event(
            event_id=row["event_id"],
            tick=row["tick"],
            wall_time=row["wall_time"],
            agent_id=row["agent_id"],
            action_type=row["action_type"],
            payload=json.loads(row["payload"]),
            result_status=row["result_status"],
            result_payload=(
                json.loads(row["result_payload"])
                if row["result_payload"]
                else None
            ),
        )


def count_events(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    if row is None:
        raise RuntimeError("COUNT(*) over events returned no row")
    return int(row[0])


def log_llm_call(
    conn: sqlite3.Connection,
    *,
    tick: int,
    agent_id: int,
    model: str,
    backend: str,
    prompt_hash: str,
    prompt_text: str | None,
    sampling_params: dict[str, Any],
    response_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    seed: int | None = None,
    cache_hit: bool = False,
    latency_ms: int | None = None,
    reasoning_summary: str | None = None,
) -> int:
    """Append one row to ``llm_calls`` and return its ``call_id``.

    Rows are append-only and carry everything needed to deterministically
    replay the call: prompt_hash (for cache-hit detection), response_text
    (so replay can skip the real model), sampling_params + seed. The
    D13 snapshot aggregates these rows into a single ``llm_cache_hash``.
    ``reasoning_summary`` (R11) is the model's chain-of-thought summary
    when the backend exposes one (currently OpenAI ``/v1/responses``);
    None for chat/completions, Ollama, and Anthropic.
    """
    tick = _require_int(tick, field="tick")
    agent_id = _require_int(agent_id, field="agent_id")
    model = _require_non_empty_string(model, field="model")
    backend = _require_non_empty_string(backend, field="backend")
    prompt_hash = _require_non_empty_string(prompt_hash, field="prompt_hash")
    seed = _require_optional_int(seed, field="seed")
    latency_ms = _require_optional_int(latency_ms, field="latency_ms")
    if latency_ms is not None and latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if not isinstance(cache_hit, bool):
        raise TypeError("cache_hit must be a boolean")
    sampling_json = _dump_json_object(sampling_params, field="sampling_params")
    tool_calls_json = _dump_json_object_list(
        tool_calls,
        field="tool_calls",
        allow_none=True,
    )
    wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO llm_calls
            (tick, agent_id, model, backend, prompt_hash, prompt_text,
             sampling_params, response_text, tool_calls_json,
             reasoning_summary, seed, cache_hit, latency_ms, wall_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tick,
            agent_id,
            model,
            backend,
            prompt_hash,
            prompt_text,
            sampling_json,
            response_text,
            tool_calls_json,
            reasoning_summary,
            seed,
            1 if cache_hit else 0,
            latency_ms,
            wall,
        ),
    )
    return require_lastrowid(cur, table="llm_calls")


def compute_llm_cache_hash(conn: sqlite3.Connection, up_to_tick: int) -> str:
    """Rolling SHA-256 over (prompt_hash, response_text) for all LLM calls
    up to and including ``up_to_tick``. Two runs with identical LLM cache
    state produce identical hashes — this is what D13 snapshots persist
    to make Phase-4 counterfactual replay bit-reproducible.
    """
    import hashlib
    h = hashlib.sha256()
    for row in conn.execute(
        "SELECT prompt_hash, response_text FROM llm_calls "
        "WHERE tick <= ? ORDER BY call_id",
        (up_to_tick,),
    ):
        h.update(row["prompt_hash"].encode("utf-8"))
        h.update(b"\x00")
        h.update(row["response_text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

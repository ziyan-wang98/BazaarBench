"""Event log invariants."""
from __future__ import annotations

import json
import sqlite3

import pytest

from bazaar.core.event_log import (
    count_events,
    log_event,
    log_llm_call,
    replay_iter,
)


def test_log_event_assigns_monotonic_ids(fresh_conn):
    id1 = log_event(fresh_conn, tick=0, agent_id=1,
                    action_type="x", payload={"a": 1})
    id2 = log_event(fresh_conn, tick=0, agent_id=1,
                    action_type="y", payload={"b": 2})
    assert id2 == id1 + 1
    fresh_conn.commit()


def test_count_events_matches_inserts(fresh_conn):
    assert count_events(fresh_conn) == 0
    for i in range(5):
        log_event(fresh_conn, tick=i, agent_id=None,
                  action_type="tick", payload={})
    fresh_conn.commit()
    assert count_events(fresh_conn) == 5


def test_replay_iter_returns_events_in_order(fresh_conn):
    for i in range(3):
        log_event(fresh_conn, tick=i, agent_id=i,
                  action_type=f"act{i}", payload={"i": i},
                  result_status="ok", result_payload={"ok": True})
    fresh_conn.commit()
    events = list(replay_iter(fresh_conn))
    assert [e.tick for e in events] == [0, 1, 2]
    assert events[0].payload == {"i": 0}
    assert events[0].result_payload == {"ok": True}


def test_payload_json_round_trip(fresh_conn):
    payload = {"z": "hello", "n": [1, 2, 3]}
    log_event(fresh_conn, tick=0, agent_id=None,
              action_type="x", payload=payload)
    fresh_conn.commit()
    row = fresh_conn.execute(
        "SELECT payload FROM events"
    ).fetchone()
    assert json.loads(row[0]) == payload


def test_log_event_rejects_non_object_payloads(fresh_conn):
    with pytest.raises(TypeError, match="payload must be a JSON object dict"):
        log_event(
            fresh_conn,
            tick=0,
            agent_id=None,
            action_type="x",
            payload=[],  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="result_payload must be a JSON object dict"):
        log_event(
            fresh_conn,
            tick=0,
            agent_id=None,
            action_type="x",
            payload={},
            result_payload=[],  # type: ignore[arg-type]
        )


def test_log_event_rejects_invalid_status_and_unserializable_payload(fresh_conn):
    with pytest.raises(ValueError, match="invalid result_status"):
        log_event(
            fresh_conn,
            tick=0,
            agent_id=None,
            action_type="x",
            payload={},
            result_status="partial",
        )

    with pytest.raises(TypeError, match="payload must be JSON-serializable"):
        log_event(
            fresh_conn,
            tick=0,
            agent_id=None,
            action_type="x",
            payload={"bad": {1, 2}},
        )


def test_log_event_rejects_bad_identity_fields(fresh_conn):
    bad_calls = [
        {"tick": True, "agent_id": None, "action_type": "x"},
        {"tick": 0, "agent_id": False, "action_type": "x"},
        {"tick": 0, "agent_id": None, "action_type": ""},
        {"tick": 0, "agent_id": None, "action_type": "   "},
    ]
    for kwargs in bad_calls:
        with pytest.raises((TypeError, ValueError)):
            log_event(fresh_conn, payload={}, **kwargs)


def _seed_agent(conn: sqlite3.Connection, agent_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (agent_id, f"u{agent_id}", f"U{agent_id}", "00000", 0.0, 0.0, "{}"),
    )
    conn.commit()


def test_log_llm_call_persists_reasoning_summary(fresh_conn):
    """R11 B.5: passing reasoning_summary writes it to the column."""
    _seed_agent(fresh_conn)
    cid = log_llm_call(
        fresh_conn,
        tick=0, agent_id=1, model="gpt-5.2", backend="OpenAIBackend",
        prompt_hash="hash1", prompt_text=None,
        sampling_params={"temperature": 0.4},
        response_text="ok", tool_calls=[],
        reasoning_summary="agent decided to buy because deadline is close",
    )
    fresh_conn.commit()
    row = fresh_conn.execute(
        "SELECT reasoning_summary FROM llm_calls WHERE call_id=?", (cid,),
    ).fetchone()
    assert row[0] == "agent decided to buy because deadline is close"


def test_log_llm_call_reasoning_summary_backward_compat(fresh_conn):
    """R11 B.5: omitting the kwarg defaults to None (NULL in db) — old
    callers (chat/completions, Ollama, Anthropic) keep working."""
    _seed_agent(fresh_conn)
    cid = log_llm_call(
        fresh_conn,
        tick=0, agent_id=1, model="llama3.2:3b", backend="OllamaBackend",
        prompt_hash="hash2", prompt_text=None,
        sampling_params={"temperature": 0.0},
        response_text="ok", tool_calls=None,
    )
    fresh_conn.commit()
    row = fresh_conn.execute(
        "SELECT reasoning_summary FROM llm_calls WHERE call_id=?", (cid,),
    ).fetchone()
    assert row[0] is None


def test_log_llm_call_rejects_invalid_json_shapes(fresh_conn):
    _seed_agent(fresh_conn)
    with pytest.raises(TypeError, match="sampling_params must be a JSON object dict"):
        log_llm_call(
            fresh_conn,
            tick=0,
            agent_id=1,
            model="gpt-test",
            backend="OpenAIBackend",
            prompt_hash="hash3",
            prompt_text=None,
            sampling_params=[],  # type: ignore[arg-type]
            response_text="ok",
        )
    with pytest.raises(TypeError, match="tool_calls must be a JSON array"):
        log_llm_call(
            fresh_conn,
            tick=0,
            agent_id=1,
            model="gpt-test",
            backend="OpenAIBackend",
            prompt_hash="hash4",
            prompt_text=None,
            sampling_params={},
            response_text="ok",
            tool_calls=["bad"],  # type: ignore[list-item]
        )
    with pytest.raises(TypeError, match="tool_calls must be JSON-serializable"):
        log_llm_call(
            fresh_conn,
            tick=0,
            agent_id=1,
            model="gpt-test",
            backend="OpenAIBackend",
            prompt_hash="hash5",
            prompt_text=None,
            sampling_params={},
            response_text="ok",
            tool_calls=[{"bad": {1, 2}}],
        )


def test_log_llm_call_rejects_bad_identity_fields(fresh_conn):
    _seed_agent(fresh_conn)
    base = {
        "tick": 0,
        "agent_id": 1,
        "model": "gpt-test",
        "backend": "OpenAIBackend",
        "prompt_hash": "hash",
        "prompt_text": None,
        "sampling_params": {},
        "response_text": "",
    }
    bad_updates = [
        {"tick": True},
        {"agent_id": False},
        {"model": ""},
        {"backend": "   "},
        {"prompt_hash": ""},
        {"seed": True},
        {"latency_ms": False},
        {"latency_ms": -1},
        {"cache_hit": 1},
    ]
    for update in bad_updates:
        kwargs = {**base, **update}
        with pytest.raises((TypeError, ValueError)):
            log_llm_call(fresh_conn, **kwargs)

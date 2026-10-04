"""Unit tests for bazaar.agents.llm_tools (T28c-6)."""
from __future__ import annotations

import json

from bazaar.actions.types import ActionType
from bazaar.agents.llm_tools import (
    build_tool_specs,
    llm_accessible_actions,
    tool_index,
)
from bazaar.core.env import BazaarEnv


def test_every_accessible_action_has_spec() -> None:
    specs = build_tool_specs()
    names = {s.name for s in specs}
    expected = {a.value for a in llm_accessible_actions()}
    assert expected == names


def test_quote_agent_note_requires_explicit_extra_action() -> None:
    off = {s.name for s in build_tool_specs(allow_cross_agent_notes=False)}
    on = {s.name for s in build_tool_specs(allow_cross_agent_notes=True)}
    assert "quote_agent_note" not in off
    assert "quote_agent_note" not in on
    explicit = {
        s.name
        for s in build_tool_specs(extra_actions=[ActionType.QUOTE_AGENT_NOTE])
    }
    assert "quote_agent_note" in explicit


def test_each_spec_has_openai_shape() -> None:
    for s in build_tool_specs():
        payload = s.to_openai_tool()
        assert payload["type"] == "function"
        fn = payload["function"]
        assert fn["name"] == s.name
        assert isinstance(fn["description"], str) and fn["description"]
        assert isinstance(fn["parameters"], dict)
        # JSON Schema must at least declare a type.
        assert fn["parameters"].get("type") in {"object", None}


def test_tool_index_roundtrip() -> None:
    specs = build_tool_specs()
    idx = tool_index(specs)
    assert idx["search"].action == ActionType.SEARCH
    assert "do_nothing" in idx
    # Duplicate-free.
    assert len(idx) == len(specs)


def test_descriptions_are_nonempty() -> None:
    for s in build_tool_specs():
        assert len(s.description) >= 30, (
            f"Description for {s.name} is suspiciously short — that's "
            f"bad for LLM tool selection."
        )


def _desc_for(action_name: str) -> str:
    """Return the LLM-facing description for ``action_name``."""
    from bazaar.agents.llm_tools import all_descriptions
    return all_descriptions()[action_name].lower()


def test_make_offer_description_signals_purchase_intent() -> None:
    """R11 A.1: gpt-5.2 reasoned correctly about a deadline buy but
    never fired make_offer because the description called it
    ``propose a price`` instead of ``commit to buying``. Lock the
    new wording so a future edit doesn't silently regress."""
    desc = _desc_for("make_offer")
    assert "buy" in desc or "purchas" in desc, (
        "make_offer description must signal 'this is what actually "
        "buys the item' — the old 'propose a price' phrasing left "
        "gpt-5.2 unable to map intent to tool."
    )


def test_accept_offer_description_mentions_seller_role() -> None:
    """R11 A.1: accept_offer is a seller-side action; the description
    must say so or buyers will misfire it."""
    desc = _desc_for("accept_offer")
    assert "seller" in desc


def test_counter_offer_description_says_either_party_can_call() -> None:
    """R11 A.1: counter is symmetric (buyer or seller). Description
    must make that explicit so the LLM doesn't think it's seller-only."""
    desc = _desc_for("counter_offer")
    assert "buyer" in desc and "seller" in desc


def test_complete_transaction_description_mentions_meeting_in_person() -> None:
    """R11 A.1: complete_transaction is the post-meetup / post-shipment
    confirmation step. Description must convey the temporal precondition
    (already met or already shipped) so an agent doesn't call it before
    scheduling. v2: also mentions inspection-first for meetup mode.
    """
    desc = _desc_for("complete_transaction")
    assert "meet" in desc and (
        "payment" in desc or "exchang" in desc or "inspect" in desc
    )


def test_toolspec_invoke_returns_payload_field_not_result_payload(tmp_path):
    """Regression guard (T16 / R4-Tfix).

    ``ActionResult`` exposes ``payload``, not ``result_payload`` —
    an earlier copy of ``ToolSpec.invoke`` accessed the wrong
    attribute. That path only runs when an LLM emits ≥2 tool_calls
    in one turn (the trailing ones are dispatched inline through
    ``spec.invoke``), so our scripted smoke tests missed it and
    qwen3:1.7b hit it at runtime with an AttributeError. This test
    locks the contract by driving ``invoke`` directly.
    """
    env = BazaarEnv(db_path=tmp_path / "invoke.db")
    conn = env.platform.conn
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "a1", "Alice", "00001", 0.0, 0.0, "{}"),
    )
    conn.commit()

    specs = tool_index(build_tool_specs())
    result = specs["do_nothing"].invoke(
        conn, agent_id=1, tick=0, raw_args={},
    )

    assert result["status"] == "ok"
    # The key matters — before the fix this raised AttributeError
    # on ``result.result_payload`` instead of returning the dict.
    assert "result" in result
    assert isinstance(result["event_id"], int)
    env.close()


def test_toolspec_invoke_treats_none_args_as_empty_object(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "invoke_none.db")
    conn = env.platform.conn
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "a1", "Alice", "00001", 0.0, 0.0, "{}"),
    )
    conn.commit()

    specs = tool_index(build_tool_specs())
    result = specs["do_nothing"].invoke(
        conn, agent_id=1, tick=0, raw_args=None,
    )

    assert result["status"] == "ok"
    row = conn.execute(
        "SELECT payload FROM events WHERE event_id = ?",
        (result["event_id"],),
    ).fetchone()
    assert json.loads(row["payload"]) == {}
    env.close()


def test_toolspec_invoke_logs_malformed_args_through_dispatch(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "invoke_malformed.db")
    conn = env.platform.conn
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "a1", "Alice", "00001", 0.0, 0.0, "{}"),
    )
    conn.commit()

    specs = tool_index(build_tool_specs())
    result = specs["create_listing"].invoke(
        conn,
        agent_id=1,
        tick=0,
        raw_args=["not", "a", "dict"],
    )

    assert result["status"] == "error"
    assert isinstance(result["event_id"], int)
    row = conn.execute(
        "SELECT payload, result_payload FROM events WHERE event_id = ?",
        (result["event_id"],),
    ).fetchone()
    payload = json.loads(row["payload"])
    result_payload = json.loads(row["result_payload"])
    assert payload["_malformed_args"] is True
    assert payload["raw_args_type"] == "list"
    assert result_payload["error"] == "validation"
    env.close()

"""R4 fix — _parse_tool_calls must drop action-name-only calls.

Small local models (3B, 7B) routinely emit shapes like
``{"action":"create_listing"}`` or
``{"action":"create_listing","arguments":{}}`` — they name an
action but don't commit to the required arguments. Before T9 /
R3-T1 the policy would forward these to the dispatcher where
pydantic would reject every required field and log a noisy
``error`` event. The fix is to pre-filter these cases so the
policy falls back to DO_NOTHING cleanly.
"""
from __future__ import annotations

from bazaar.agents.llm_tools import build_tool_specs, tool_index
from bazaar.agents.policies import (
    _parse_tool_calls,
    _parse_tool_calls_with_rejections,
    _tool_calls_from_native_with_rejections,
)


def _tools_idx():
    return tool_index(build_tool_specs())


def test_empty_args_dict_for_required_schema_is_dropped():
    idx = _tools_idx()
    # create_listing has 5 required fields; empty args must not
    # produce a tool call.
    calls = _parse_tool_calls(
        '{"action":"create_listing","arguments":{}}', idx,
    )
    assert calls == []

    calls, rejected = _parse_tool_calls_with_rejections(
        '{"action":"create_listing","arguments":{}}', idx,
    )
    assert calls == []
    assert rejected[0]["_rejected"] is True
    assert rejected[0]["reason"] == "schema_invalid"
    assert rejected[0]["name"] == "create_listing"


def test_missing_arguments_key_for_required_schema_is_dropped():
    idx = _tools_idx()
    calls = _parse_tool_calls('{"action":"create_listing"}', idx)
    assert calls == []


def test_partial_args_missing_required_field_is_dropped():
    idx = _tools_idx()
    # create_listing requires category, title, description, price_cents,
    # condition. A partial payload still misses fields → drop.
    calls = _parse_tool_calls(
        '{"action":"create_listing","arguments":{"category":"books"}}',
        idx,
    )
    assert calls == []


def test_empty_args_ok_when_schema_has_no_required_fields():
    """do_nothing has no required fields — empty args must still
    produce a tool call. This is the common idle path and must
    not regress."""
    idx = _tools_idx()
    calls = _parse_tool_calls(
        '{"action":"do_nothing","arguments":{}}', idx,
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "do_nothing"
    # Also the even-sparser shape with no arguments key.
    calls = _parse_tool_calls('{"action":"do_nothing"}', idx)
    assert len(calls) == 1
    assert calls[0]["name"] == "do_nothing"


def test_full_args_survive():
    """A well-formed call must still pass through unchanged."""
    idx = _tools_idx()
    calls = _parse_tool_calls(
        '{"action":"search","arguments":{"query":"bike"}}', idx,
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["arguments"] == {"query": "bike"}


def test_top_level_actions_wrapper_survives():
    """Qwen/TRAPI chat fallback may emit visible {"actions": [...]} JSON."""
    idx = _tools_idx()
    calls = _parse_tool_calls(
        """
        {
          "actions": [
            {"action": "bump_listing", "arguments": {"listing_id": 40}},
            {"action": "view_listing", "arguments": {"listing_id": 2480}}
          ]
        }
        """,
        idx,
    )
    assert calls == [
        {"name": "bump_listing", "arguments": {"listing_id": 40}},
        {"name": "view_listing", "arguments": {"listing_id": 2480}},
    ]


def test_malformed_repeated_action_argument_pairs_survive():
    """Qwen/TRAPI may omit the outer object close around repeated pairs."""
    idx = _tools_idx()
    calls = _parse_tool_calls(
        """
        [{"action": "search", "arguments": {"query": "headphones"},
          "action": "make_offer", "arguments": {
            "listing_id": 2254,
            "price_cents": 1200,
            "terms": {}
          }]
        """,
        idx,
    )
    assert calls == [
        {"name": "search", "arguments": {"query": "headphones"}},
        {
            "name": "make_offer",
            "arguments": {
                "listing_id": 2254,
                "price_cents": 1200,
                "terms": {},
            },
        },
    ]


def test_balanced_json_inside_visible_thinking_survives():
    idx = _tools_idx()
    response = """
Thinking Process:
The user asked for `{"action":"do_nothing","arguments":{}}`.
I should output that JSON object exactly.
"""
    calls = _parse_tool_calls(response, idx)
    assert calls == [{"name": "do_nothing", "arguments": {}}]


def test_message_without_thread_or_listing_is_dropped():
    idx = _tools_idx()
    calls = _parse_tool_calls(
        '{"action":"message","arguments":{"body":"Do you have any deals?"}}',
        idx,
    )
    assert calls == []


def test_native_message_without_thread_or_listing_is_dropped():
    from bazaar.agents.policies import _tool_calls_from_native

    idx = _tools_idx()
    calls = _tool_calls_from_native(
        [{
            "function": {
                "name": "message",
                "arguments": {"body": "Do you have any deals?"},
            },
        }],
        idx,
    )
    assert calls == []


def test_native_rejections_include_reason_records():
    idx = _tools_idx()
    calls, rejected = _tool_calls_from_native_with_rejections(
        [
            {
                "function": {
                    "name": "unknown_action",
                    "arguments": {},
                },
            },
            {
                "function": {
                    "name": "search",
                    "arguments": "{bad-json",
                },
            },
            {
                "function": {
                    "name": "search",
                    "arguments": {"query": "bike"},
                },
            },
        ],
        idx,
    )

    assert calls == [{"name": "search", "arguments": {"query": "bike"}}]
    assert [r["reason"] for r in rejected] == [
        "unknown_tool_name",
        "malformed_arguments_json",
    ]
    assert all(r["_rejected"] is True for r in rejected)

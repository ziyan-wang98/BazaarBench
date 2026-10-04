"""Every ActionType × dispatcher sanity matrix (P6 of T23).

Three goals:

1. **Dispatcher never raises.** For every ActionType, dispatching
   valid args yields status in {ok, blocked, error} and dispatching
   empty args yields 'error' (validation). No exception bubbles up.

2. **Every enum entry is in ACTION_SCHEMAS and reachable.** Catches
   the "added a new ActionType and forgot to register its schema or
   handler" class of bug.

3. **Real vs. stub registry is intentional.** Any ActionType not in
   the real-handler registry resolves to the blocked stub, which returns
   blocked + {stubbed: true, action: <name>}. That's still a contract.
"""
from __future__ import annotations

from typing import Any

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions.dispatch import dispatch, stubbed_actions
from bazaar.actions.schemas import ACTION_SCHEMAS
from bazaar.actions.types import ActionType
from bazaar.memory import HashEncoder, NarrativeStore, install_store


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=1700 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    yield env
    env.close()


def test_every_action_has_a_schema():
    missing = [a for a in ActionType if a not in ACTION_SCHEMAS]
    assert missing == [], f"ActionTypes without schema: {missing}"


def test_every_action_has_either_a_real_handler_or_stub_fallback(env):
    # Dispatch each action with empty args — validation will reject
    # malformed args and we get status='error' with error='validation'.
    # This proves the dispatcher can at least route every ActionType
    # without raising.
    for a in ActionType:
        r = dispatch(env.platform.conn, agent_id=1,
                     action=a, raw_args={}, tick=0)
        assert r.status in ("ok", "error", "blocked"), (
            f"unexpected status for {a.value}: {r.status}"
        )
        assert r.event_id > 0, (
            f"dispatcher failed to log event for {a.value}"
        )


# A canonical set of well-formed args for each action, enough to at
# least get past pydantic validation. Some will then be 'blocked' or
# 'error' at handler-time (no such listing / self-op / etc.) but the
# dispatcher must still return structured results.
VALID_ARGS: dict[ActionType, dict[str, Any]] = {
    ActionType.DO_NOTHING:           {},
    ActionType.SEARCH:               {"query": "book"},
    ActionType.REFINE_SEARCH:        {"delta": {"max_price_cents": 1000}},
    ActionType.BROWSE_CATEGORY:      {"category": "books"},
    ActionType.VIEW_LISTING:         {"listing_id": 99999},
    ActionType.INSPECT_PHOTO:        {"photo_id": 99999},
    ActionType.PIN:                  {"listing_id": 99999},
    ActionType.UNPIN:                {"listing_id": 99999},
    ActionType.COMPARE:              {"listing_ids": [99999, 88888]},
    ActionType.CREATE_LISTING:       {"category": "books", "title": "Foo",
                                       "description": "", "price_cents": 100,
                                       "condition": "good"},
    ActionType.EDIT_LISTING:         {"listing_id": 99999, "title": "x"},
    ActionType.BUMP_LISTING:         {"listing_id": 99999},
    ActionType.MARK_SOLD:            {"listing_id": 99999},
    ActionType.RELIST:               {"listing_id": 99999},
    ActionType.CROSS_POST:           {"listing_id": 99999, "group_id": 1},
    ActionType.MESSAGE:              {"thread_id": 99999, "body": "hi"},
    ActionType.SEND_PHOTO:           {"thread_id": 99999, "listing_id": 99999,
                                       "focus": "x"},
    ActionType.SEND_CRAFTED_PHOTO:   {"thread_id": 99999,
                                       "subject_attrs": {"item": "x"}},
    ActionType.SEND_STOCK_PHOTO:     {"thread_id": 99999, "listing_id": 99999},
    ActionType.REQUEST_PHOTO:        {"thread_id": 99999, "focus_hint": "x"},
    ActionType.READ:                 {"thread_id": 99999},
    ActionType.WAIT:                 {"ticks": 1},
    ActionType.LEAVE_THREAD:         {"thread_id": 99999},
    ActionType.GHOST:                {"thread_id": 99999},
    ActionType.MAKE_OFFER:           {"listing_id": 99999, "price_cents": 100},
    ActionType.COUNTER_OFFER:        {"offer_id": 99999, "price_cents": 100},
    ActionType.ACCEPT_OFFER:         {"offer_id": 99999},
    ActionType.WITHDRAW_OFFER:       {"offer_id": 99999},
    ActionType.SCHEDULE_MEETUP:      {"thread_id": 99999,
                                       "location_desc": "x",
                                       "scheduled_tick": 10,
                                       "payment_method": "cash"},
    ActionType.SCHEDULE_SHIPMENT:    {"thread_id": 99999,
                                       "delivery_lag_ticks": 6,
                                       "payment_method": "venmo"},
    ActionType.INSPECT_AT_MEETUP:    {"meetup_id": 99999},
    ActionType.COMPLETE_TRANSACTION: {"meetup_id": 99999},
    ActionType.CANCEL_MEETUP:        {"meetup_id": 99999, "reason": "x"},
    ActionType.RATE:                 {"ratee_agent_id": 2, "stars": 4},
    ActionType.REPORT_LISTING:       {"listing_id": 99999, "reason": "x"},
    ActionType.REPORT_USER:          {"user_agent_id": 2, "reason": "x"},
    ActionType.BLOCK_USER:           {"user_agent_id": 2},
    ActionType.VIEW_PROFILE:         {"user_agent_id": 2},
    ActionType.CREATE_SUBACCOUNT:    {},
    ActionType.SWITCH_ACTIVE_ACCOUNT: {"account_id": 1},
    ActionType.LINK_ACCOUNTS:        {"account_a": 1, "account_b": 2},
    ActionType.JOIN_GROUP:           {"group_id": 1},
    ActionType.LEAVE_GROUP:          {"group_id": 1},
    ActionType.LIST_MUTUALS:         {"user_agent_id": 2},
    ActionType.SUMMARIZE_SESSION:    {"scope": "self", "content": "hi"},
    ActionType.RECALL:               {"query": "hi"},
    # QUOTE_AGENT_NOTE is env-gated — the default BazaarEnv fixture has
    # ``allow_cross_agent_notes=False``, so dispatch returns ``blocked``.
    # The matrix test asserts dispatch doesn't raise, which still holds.
    ActionType.QUOTE_AGENT_NOTE:     {"source_agent_id": 999,
                                      "content": "note from elsewhere"},
}


def test_valid_args_table_covers_every_action():
    """If someone adds a new ActionType they must also add an entry
    here. Catches drift at test-collection time."""
    missing = [a for a in ActionType if a not in VALID_ARGS]
    assert missing == [], f"VALID_ARGS missing entries for: {missing}"


@pytest.mark.parametrize("action", list(ActionType), ids=lambda a: a.value)
def test_valid_args_dispatch_without_raising(env, action):
    """Every valid-arg dispatch returns a structured result —
    never an exception. Handler-time 'error' (missing listing /
    self-op / etc.) is expected for most of these since the
    fixture DB is empty."""
    r = dispatch(env.platform.conn, agent_id=1,
                 action=action, raw_args=VALID_ARGS[action], tick=0)
    assert r.status in ("ok", "error", "blocked"), (
        f"bad status for {action.value}: {r.status} · {r.payload}"
    )
    assert isinstance(r.payload, (dict, type(None))), (
        f"{action.value} returned non-dict payload: {type(r.payload)}"
    )
    assert r.event_id > 0


@pytest.mark.parametrize(
    "action",
    sorted(stubbed_actions(), key=lambda a: a.value),
    ids=lambda a: a.value,
)
def test_stubbed_actions_are_explicitly_blocked(env, action):
    r = dispatch(
        env.platform.conn,
        agent_id=1,
        action=action,
        raw_args=VALID_ARGS[action],
        tick=0,
    )
    assert r.status == "blocked"
    assert r.payload is not None
    assert r.payload["stubbed"] is True
    assert r.payload["error"] == "handler_not_implemented"

"""Action dispatcher: validate → route → event-log → return result.

Handlers themselves are in ``bazaar.actions.handlers``.  For Phase 1 we
implement a *partial* set of handlers and route every unimplemented
action to a ``_stub_handler`` that logs ``result_status='blocked'`` with
a payload marking the handler gap. This keeps the dispatcher fail-safe
without making unsupported actions look successful.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from bazaar.actions import handlers
from bazaar.actions.schemas import get_schema
from bazaar.actions.types import ActionType
from bazaar.core.event_log import log_event


@dataclass
class ActionResult:
    """Uniform return type for every action handler."""
    status: str   # 'ok' | 'error' | 'blocked'
    payload: dict[str, Any] | None
    event_id: int


class _Stub:
    """Placeholder for handlers we haven't written yet."""

    def __init__(self, action: ActionType) -> None:
        self.action = action

    def __call__(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
        args: Any,
        *,
        tick: int,
    ) -> tuple[str, dict[str, Any] | None]:
        return "blocked", {
            "stubbed": True,
            "action": self.action.value,
            "error": "handler_not_implemented",
        }


# Handlers returning (status, result_payload).  Full set lives in
# ``bazaar.actions.handlers``; actions without a real handler resolve
# to ``_Stub`` so the dispatcher remains fail-safe.
HandlerFn = Callable[[sqlite3.Connection, int, Any], tuple[str, dict[str, Any] | None]]
_MAX_EVENT_ARG_REPR = 500

# Real implementations.  Actions not listed here fall through to ``_Stub``.
_REAL_HANDLERS: dict[ActionType, Callable] = {
    ActionType.DO_NOTHING:         handlers.do_nothing,
    ActionType.CREATE_LISTING:     handlers.create_listing,
    ActionType.VIEW_LISTING:       handlers.view_listing,
    ActionType.PIN:                handlers.pin_listing,
    ActionType.MESSAGE:            handlers.send_message,
    ActionType.MAKE_OFFER:         handlers.make_offer,
    ActionType.WAIT:               handlers.wait_action,
    ActionType.VIEW_PROFILE:       handlers.view_profile,
    ActionType.BLOCK_USER:         handlers.block_user,
    # Group 4 photo actions (T10).
    ActionType.SEND_PHOTO:         handlers.send_photo,
    ActionType.SEND_CRAFTED_PHOTO: handlers.send_crafted_photo,
    ActionType.SEND_STOCK_PHOTO:   handlers.send_stock_photo,
    ActionType.REQUEST_PHOTO:      handlers.request_photo,
    ActionType.INSPECT_PHOTO:      handlers.inspect_photo,
    # Group 9 reflective / memory actions (T13 + T28c).
    ActionType.SUMMARIZE_SESSION:  handlers.summarize_session,
    ActionType.RECALL:             handlers.recall,
    ActionType.QUOTE_AGENT_NOTE:   handlers.quote_agent_note,
    # Group 5 negotiation lifecycle (P1 of T23).
    ActionType.COUNTER_OFFER:         handlers.counter_offer,
    ActionType.ACCEPT_OFFER:          handlers.accept_offer,
    ActionType.WITHDRAW_OFFER:        handlers.withdraw_offer,
    ActionType.SCHEDULE_MEETUP:       handlers.schedule_meetup,
    ActionType.SCHEDULE_SHIPMENT:     handlers.schedule_shipment,
    ActionType.INSPECT_AT_MEETUP:     handlers.inspect_at_meetup,
    ActionType.COMPLETE_TRANSACTION:  handlers.complete_transaction,
    ActionType.CANCEL_MEETUP:         handlers.cancel_meetup,
    # Group 3 listing lifecycle + Group 6 reputation (P2 of T23).
    ActionType.EDIT_LISTING:          handlers.edit_listing,
    ActionType.BUMP_LISTING:          handlers.bump_listing,
    ActionType.MARK_SOLD:             handlers.mark_sold,
    ActionType.RELIST:                handlers.relist,
    ActionType.RATE:                  handlers.rate,
    ActionType.REPORT_LISTING:        handlers.report_listing,
    ActionType.REPORT_USER:           handlers.report_user,
    # Discovery + shortlist + thread termination (P3 of T23).
    ActionType.SEARCH:                handlers.search,
    ActionType.REFINE_SEARCH:         handlers.refine_search,
    ActionType.BROWSE_CATEGORY:       handlers.browse_category,
    ActionType.UNPIN:                 handlers.unpin,
    ActionType.COMPARE:               handlers.compare,
    ActionType.READ:                  handlers.read,
    ActionType.LEAVE_THREAD:          handlers.leave_thread,
    ActionType.GHOST:                 handlers.ghost,
}


def _resolve(action: ActionType) -> Callable:
    return _REAL_HANDLERS.get(action, _Stub(action))


def real_handler_actions() -> set[ActionType]:
    """Actions with concrete handler implementations."""
    return set(_REAL_HANDLERS)


def stubbed_actions() -> set[ActionType]:
    """Actions that validate but intentionally resolve to the blocked stub."""
    return set(ActionType) - set(_REAL_HANDLERS)


def _safe_repr(value: Any) -> str:
    try:
        out = repr(value)
    except Exception as exc:  # noqa: BLE001
        out = f"<repr failed: {type(exc).__name__}>"
    if len(out) > _MAX_EVENT_ARG_REPR:
        return out[:_MAX_EVENT_ARG_REPR] + "..."
    return out


def _event_payload_from_args(raw_args: Any, *, source: str) -> dict[str, Any]:
    """Coerce arbitrary external args into a JSON-object event payload."""
    if isinstance(raw_args, dict):
        try:
            json.dumps(raw_args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return {
                "_malformed_args": True,
                "source": source,
                "raw_args_type": type(raw_args).__name__,
                "raw_args_repr": _safe_repr(raw_args),
            }
        return raw_args
    return {
        "_malformed_args": True,
        "source": source,
        "raw_args_type": type(raw_args).__name__,
        "raw_args_repr": _safe_repr(raw_args),
    }


def dispatch(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    action: ActionType,
    raw_args: Any,
    tick: int,
) -> ActionResult:
    """Validate args, route to a handler, log the event, return result.

    Never raises on invalid input: returns an ``error`` ``ActionResult``
    with a logged event instead. The fail-safe rule keeps the
    simulation loop robust to malformed LLM output.
    """
    schema = get_schema(action)
    try:
        validated = schema.model_validate(raw_args)
    except ValidationError as exc:
        # ``include_context=False`` strips the raw exception object
        # pydantic stashes under ``ctx`` for ``model_validator``-raised
        # errors. ``include_input=False`` keeps arbitrary raw LLM input
        # out of result_payload, preserving the event-log JSON contract.
        details = exc.errors(
            include_context=False,
            include_input=False,
            include_url=False,
        )
        event_id = log_event(
            conn,
            tick=tick,
            agent_id=agent_id,
            action_type=action.value,
            payload=_event_payload_from_args(raw_args, source="raw_args"),
            result_status="error",
            result_payload={"error": "validation", "detail": details},
        )
        conn.commit()
        return ActionResult(status="error",
                            payload={"error": "validation",
                                     "detail": details},
                            event_id=event_id)

    handler = _resolve(action)
    try:
        with conn:
            status, result_payload = handler(conn, agent_id, validated, tick=tick)
            event_id = log_event(
                conn,
                tick=tick,
                agent_id=agent_id,
                action_type=action.value,
                payload=_event_payload_from_args(
                    validated.model_dump(),
                    source="validated_args",
                ),
                result_status=status,
                result_payload=result_payload,
            )
    except Exception as exc:  # noqa: BLE001 — final safety net
        event_id = log_event(
            conn,
            tick=tick,
            agent_id=agent_id,
            action_type=action.value,
            payload=_event_payload_from_args(
                validated.model_dump(),
                source="validated_args",
            ),
            result_status="error",
            result_payload={"error": "handler_exception", "detail": str(exc)},
        )
        conn.commit()
        return ActionResult(status="error",
                            payload={"error": "handler_exception",
                                     "detail": str(exc)},
                            event_id=event_id)

    return ActionResult(status=status, payload=result_payload, event_id=event_id)

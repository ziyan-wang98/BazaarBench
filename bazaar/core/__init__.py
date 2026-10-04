"""Core primitives: schema, event log, tick clock, environment.

NOTE ON IMPORT ORDER
--------------------
``BazaarEnv`` is defined in ``bazaar.core.env`` but it depends on
``bazaar.actions.dispatch``, which in turn imports ``log_event`` from
this package.  To avoid a circular import at package init time we do
NOT re-export ``BazaarEnv`` here; import it from its submodule
(``from bazaar.core.env import BazaarEnv``) or from the top-level
package (``from bazaar import BazaarEnv``), which imports it lazily
after the actions module is fully loaded.
"""
from bazaar.core.event_audit import (
    EventAuditIssue,
    audit_agent_action_event_consistency,
    audit_marketplace_state_invariants,
    audit_platform_event_consistency,
)
from bazaar.core.event_log import Event, count_events, log_event, replay_iter
from bazaar.core.schema import SCHEMA_VERSION, connect, initialize_db, schema_version
from bazaar.core.tick_clock import (
    HOURS_PER_TICK,
    MINUTES_PER_TICK,
    TICKS_PER_DAY,
    TICKS_PER_HOUR,
    TICKS_PER_WEEK,
    WALL_START_HOUR,
    TickClock,
    tick_to_days_from_zero,
    tick_to_wall,
)

__all__ = [
    "Event",
    "EventAuditIssue",
    "HOURS_PER_TICK",
    "MINUTES_PER_TICK",
    "SCHEMA_VERSION",
    "TICKS_PER_DAY",
    "TICKS_PER_HOUR",
    "TICKS_PER_WEEK",
    "TickClock",
    "WALL_START_HOUR",
    "audit_agent_action_event_consistency",
    "audit_marketplace_state_invariants",
    "audit_platform_event_consistency",
    "connect",
    "count_events",
    "initialize_db",
    "log_event",
    "replay_iter",
    "schema_version",
    "tick_to_days_from_zero",
    "tick_to_wall",
]

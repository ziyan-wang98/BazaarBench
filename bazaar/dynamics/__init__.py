"""Environment dynamics — scheduled platform-side callbacks.

Thirteen dynamics (D1-D13) advance the world independently of agent
actions, responsible for making the marketplace "feel alive".

Phase-2 scope of T15 is D1-D10 plus D13 snapshot scheduling; D11
(memory consolidation) needs an LLM and is deferred, and D12
(self-portrait checkpoint) is covered by the self_portraits table
wiring in a later phase. D7 is already seeded by
:meth:`MarketplacePlatform.seed_phantom_listings`; this module adds
the per-tick callbacks that sit alongside it.

A ``Dynamic`` is any callable that matches the protocol::

    def callback(conn, *, tick, rng) -> int | None

Returning the number of side-effect rows written is idiomatic but
optional. The ``DynamicRegistry`` handles tick scheduling (interval
and optional phase offset) and deterministic RNG seeding so a replay
that restores to tick T and re-runs forward reproduces byte-for-byte.

Invariants:

* **Event-log-first.** Every side-effect a dynamic performs logs a
  corresponding ``platform_*`` event (``agent_id IS NULL`` in the
  events table). Counterfactual replay therefore sees the same
  deterministic sequence.
* **Schema-additive.** Dynamics never rewrite existing rows; they
  either insert new ones, mutate nullable status columns, or update
  append-only counters (view_count, save_count, …) introduced with
  this task.
"""
from __future__ import annotations

from typing import Any

from bazaar.core.tick_clock import TICKS_PER_DAY, TICKS_PER_WEEK
from bazaar.dynamics.registry import (
    Dynamic,
    DynamicRegistry,
    DynamicSpec,
)

__all__ = [
    "Dynamic",
    "DynamicRegistry",
    "DynamicSpec",
    "default_registry",
    "with_agent_summary_dynamic",
    "with_llm_dynamics",
]


def _deferred_d3():
    """Import-level defer to break the recsys → dynamics cycle."""
    from bazaar.recsys import D3_recsys_refresh
    return D3_recsys_refresh


def default_registry(
    *,
    llm_backend: Any | None = None,
    llm_model: str | None = None,
    reflection_backend: Any | None = None,
    reflection_model: str | None = None,
    d11_interval: int = TICKS_PER_DAY,
    d12_interval: int = 4 * TICKS_PER_DAY,
    d14_interval: int = 1,
    d11_start_tick: int = 0,
    d12_start_tick: int = 0,
    d14_start_tick: int = 0,
    agent_ids: set[int] | None = None,
) -> DynamicRegistry:
    """Return a registry wired with the Phase-2 default dynamics.

    ``llm_backend`` + ``llm_model`` (R12 Gap 1): when both are
    provided, D11 memory consolidation and D12 self-portrait are
    also registered so LLM-driven runs get bounded narrative memory
    growth out-of-the-box. Without a backend the LLM dynamics stay
    unregistered — the offline smoke path doesn't pay for template
    fallbacks it won't inspect. Default intervals are expressed in
    2h ticks (``TICKS_PER_DAY=12``): D11
    fires once per simulated day; D12 fires once every four days;
    D14 rolling-summary fires every tick (the agent rewrites its
    self-narrative each turn so next-tick prompts see a current
    ``## MY CURRENT STATE`` block).

    ``reflection_backend`` + ``reflection_model`` (R14a): the
    reflection loop is a *separate* LLM call from the action loop
    so cheap action models can share the run with a slightly more
    capable reflection model (generative-agents pattern). Falls back
    to ``llm_backend`` + ``llm_model`` when unset, so the common case
    of "one model does everything" keeps working.
    """
    from bazaar.agents.moderator import ModeratorPolicy, make_d9_callback
    from bazaar.dynamics import callbacks
    reg = DynamicRegistry()
    reg.register(DynamicSpec(
        name="D2_message_delivery",
        interval=1,
        callback=callbacks.D2_message_delivery,
    ))
    reg.register(DynamicSpec(
        name="D3_recsys_refresh",
        interval=5,
        callback=_deferred_d3(),
    ))
    reg.register(DynamicSpec(
        name="D4_listing_aging",
        interval=TICKS_PER_DAY,
        callback=callbacks.D4_listing_aging,
    ))
    reg.register(DynamicSpec(
        name="D6_rating_decay",
        interval=TICKS_PER_DAY,
        callback=callbacks.D6_rating_decay,
    ))
    reg.register(DynamicSpec(
        name="D7_phantom_tripwire",
        interval=1,
        callback=callbacks.D7_phantom_tripwire,
    ))
    reg.register(DynamicSpec(
        name="D9_moderator",
        interval=4,  # run every simulated hour
        callback=make_d9_callback(ModeratorPolicy()),
    ))
    reg.register(DynamicSpec(
        name="D10_public_metric_aggregation",
        interval=1,
        callback=callbacks.D10_public_metric_aggregation,
    ))
    # v2 D_restock: weekly inventory restock weighted by recent sales.
    reg.register(DynamicSpec(
        name="D_restock",
        interval=TICKS_PER_WEEK,
        callback=callbacks.D_restock,
        # First pulse after one simulated week, then weekly. Short smoke tests
        # should validate the action loop without injecting extra supply on day 1.
        phase=TICKS_PER_WEEK,
    ))
    if llm_backend is not None and llm_model:
        with_llm_dynamics(
            reg,
            backend=llm_backend,
            model=llm_model,
            d11_interval=d11_interval,
            d12_interval=d12_interval,
            d11_start_tick=d11_start_tick,
            d12_start_tick=d12_start_tick,
            agent_ids=agent_ids,
        )
    # R14a: D14 rolling self-summary. Uses reflection backend/model
    # when provided, else falls back to the action model. Register
    # even in offline mode — the fallback path writes a deterministic
    # template so the ## MY CURRENT STATE block always has content.
    with_agent_summary_dynamic(
        reg,
        backend=(reflection_backend if reflection_backend is not None
                 else llm_backend),
        model=(reflection_model if reflection_model is not None
               else llm_model),
        interval=d14_interval,
        start_tick=d14_start_tick,
        agent_ids=agent_ids,
    )
    return reg

def with_llm_dynamics(
    reg: DynamicRegistry,
    *,
    backend: Any | None = None,
    model: str | None = None,
    d11_interval: int = TICKS_PER_DAY,
    d12_interval: int = 4 * TICKS_PER_DAY,
    d11_start_tick: int = 0,
    d12_start_tick: int = 0,
    agent_ids: set[int] | None = None,
) -> DynamicRegistry:
    """Extend a registry with D11 (memory consolidation) + D12
    (self-portrait), optionally LLM-backed.

    When ``backend`` or ``model`` is None, the dynamics use their
    fallback paths (concatenation for D11, templated sentence for
    D12) so runs stay offline-friendly. Passing a real backend +
    model engages the LLM at every firing.

    The helper is idempotent per name — re-calling it on a registry
    that already carries D11/D12 is a no-op.
    """
    from bazaar.dynamics.llm_dynamics import (
        make_d11_memory_consolidation,
        make_d12_self_portrait,
    )
    existing = {s.name for s in reg.specs}
    if "D11_memory_consolidation" not in existing:
        reg.register(DynamicSpec(
            name="D11_memory_consolidation",
            interval=d11_interval,
            start_tick=d11_start_tick,
            callback=make_d11_memory_consolidation(
                backend=backend, model=model, agent_ids=agent_ids,
            ),
        ))
    if "D12_self_portrait" not in existing:
        reg.register(DynamicSpec(
            name="D12_self_portrait",
            interval=d12_interval,
            start_tick=d12_start_tick,
            callback=make_d12_self_portrait(
                backend=backend, model=model, agent_ids=agent_ids,
            ),
        ))
    return reg


def with_agent_summary_dynamic(
    reg: DynamicRegistry,
    *,
    backend: Any | None = None,
    model: str | None = None,
    interval: int = TICKS_PER_DAY,
    start_tick: int = 0,
    agent_ids: set[int] | None = None,
) -> DynamicRegistry:
    """Register D14 rolling self-summary (R14a Layer-1).

    Separate from :func:`with_llm_dynamics` because D14 is load-bearing
    even without a live LLM: its fallback template keeps the agent-
    summary table populated so ``PromptBuilder`` can always inject a
    ``## MY CURRENT STATE`` block.

    Idempotent per name — re-calling is a no-op when D14 is already
    registered.
    """
    from bazaar.dynamics.llm_dynamics import make_d14_agent_summary
    existing = {s.name for s in reg.specs}
    if "D14_agent_summary" not in existing:
        reg.register(DynamicSpec(
            name="D14_agent_summary",
            interval=interval,
            start_tick=start_tick,
            callback=make_d14_agent_summary(
                backend=backend, model=model, agent_ids=agent_ids,
            ),
        ))
    return reg

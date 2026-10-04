"""BazaarEnv: the top-level simulation orchestrator.

Synchronous ``reset`` / ``step`` / ``close`` API: per-agent action
dicts are validated through pydantic schemas before reaching the
platform handler. The handler runs inside the dispatcher's transaction
so the action's effect and its event-log row commit together.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bazaar.actions.dispatch import ActionResult, dispatch
from bazaar.actions.types import ActionType
from bazaar.agents.market_agent import AgentAction, MarketAgent
from bazaar.agents.persona import PersonaCard
from bazaar.core.event_log import count_events, log_event
from bazaar.core.handoff_checks import (
    HANDOFF_CHECK_MODES,
    HANDOFF_CHECKS_SET_ACTION,
    HandoffCheckResumeWarning,
    resolve_handoff_checks,
)
from bazaar.core.tick_clock import TickClock
from bazaar.dynamics import DynamicRegistry, default_registry
from bazaar.platform.marketplace import MarketplacePlatform


@dataclass
class StepReport:
    """Outcome of one tick."""
    tick: int
    actions_attempted: int = 0
    actions_ok: int = 0
    actions_blocked: int = 0
    actions_error: int = 0
    per_agent: dict[int, ActionResult] = field(default_factory=dict)
    dynamics_fired: dict[str, int] = field(default_factory=dict)


_MISSING = object()
_MAX_POLICY_ERROR_REPR = 500


@dataclass(frozen=True)
class _LoggedPolicyError:
    result: ActionResult


@dataclass(frozen=True)
class _PendingPolicyError:
    error: str
    detail: str | None = None
    raw_value: Any = _MISSING


def _safe_policy_repr(value: Any) -> str:
    try:
        out = repr(value)
    except Exception as exc:  # noqa: BLE001
        out = f"<repr failed: {type(exc).__name__}>"
    if len(out) > _MAX_POLICY_ERROR_REPR:
        return out[:_MAX_POLICY_ERROR_REPR] + "..."
    return out


def _exception_detail(exc: BaseException) -> str:
    try:
        detail = str(exc)
    except Exception as str_exc:  # noqa: BLE001
        detail = f"<str failed: {type(str_exc).__name__}>"
    return f"{type(exc).__name__}: {detail}"


class BazaarEnv:
    """Synchronous Phase 1 environment."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        agents: list[MarketAgent] | None = None,
        seed_phantom_listings: int = 0,
        seed_real_listings: int = 0,
        seed_lot_sales: int = 0,
        dynamics: DynamicRegistry | None = None,
        allow_cross_agent_notes: bool = False,
        disable_r20_nudge: bool = False,
        disable_seller_inventory_guard: bool = False,
        require_handoff_proof: bool = False,
        inventory_validator_mode: str = "off",
        meetup_ownership_check_mode: str = "off",
        handoff_checks: str = "legacy",
        inspection_truth_mode: str | None = None,
        commitment_lock_mode: str | None = None,
        completion_integrity_mode: str | None = None,
        shipment_inspection_mode: str | None = None,
        resume: bool = False,
        parallel_decide: bool = False,
        parallel_workers: int = 32,
    ) -> None:
        """Construct the environment.

        ``resume=True`` (R13): open an existing db in place instead of
        wiping it. The tick clock is restored to ``max(events.tick)+1``
        so the next ``step()`` picks up where the previous run left
        off, and ``reset()`` becomes a no-op (seed counts are honored
        only on fresh runs — the db already contains the seed corpus).
        Agents are *not* auto-reconstructed here; callers use
        :func:`reconstruct_agents_from_db` to rehydrate the persona
        list, then pass the result via ``agents=...`` or ``add_agent``.

        ``handoff_checks`` selects the truthful handoff-check preset
        (``legacy`` or ``truthful``).
        ``inspection_truth_mode``, ``commitment_lock_mode``,
        ``completion_integrity_mode`` and ``shipment_inspection_mode``
        override single checks when not ``None``. The legacy preset
        reproduces the reported runs.
        """
        # Validate before touching the database so a bad value leaves
        # no half-initialised file behind.
        self.handoff_checks = resolve_handoff_checks(
            handoff_checks,
            inspection_truth_mode=inspection_truth_mode,
            commitment_lock_mode=commitment_lock_mode,
            completion_integrity_mode=completion_integrity_mode,
            shipment_inspection_mode=shipment_inspection_mode,
        )
        self.clock = TickClock()
        self.platform = MarketplacePlatform(
            db_path, clock=self.clock, resume=resume,
        )
        self.agents: list[MarketAgent] = []
        self._phantom_seed_count = seed_phantom_listings
        self._real_seed_count = seed_real_listings
        self._lot_sale_seed_count = seed_lot_sales
        self._resume = resume
        # ``None`` ⇒ default Phase-2 registry; pass an empty registry
        # to disable dynamics entirely (useful for reproducing Phase-1
        # behaviour in tests).
        self.dynamics = dynamics if dynamics is not None else default_registry()
        self._parallel_decide = parallel_decide
        self._parallel_workers = max(1, parallel_workers)
        # NOTE on conn safety in parallel mode: only the LLM network
        # call (dispatch_llm_call) runs on worker threads, and that
        # method reads no DB. Phase A (prepare_decision) and Phase C
        # (apply_decision) both use the main thread on platform.conn,
        # so we never need a cross-thread sqlite connection — leaving
        # check_same_thread at the python default eliminates the WAL
        # corruption we hit with multi-conn parallel decide.

        if resume:
            row = self.platform.conn.execute(
                "SELECT COALESCE(MAX(tick), -1) FROM events "
                "WHERE action_type NOT IN (?, ?)",
                ("experiment_config", HANDOFF_CHECKS_SET_ACTION),
            ).fetchone()
            max_tick = int(row[0]) if row is not None else -1
            self.clock.current = max_tick + 1

        # QUOTE_AGENT_NOTE gate (entry-point-E inheritance experiments).
        # Default off so H1 runs stay confound-free. Stored in ``meta``
        # so handlers can read it without env coupling.
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("allow_cross_agent_notes", "1" if allow_cross_agent_notes else "0"),
        )
        # R22: ablation flag for the scheduled_meetups_awaiting_
        # confirmation footer bullet. When set, the prompt builder
        # suppresses the bullet so we can test whether the closing-
        # loop hallucination observed in benign victims (Case C.2)
        # is causally attributable to that nudge.
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("disable_r20_nudge", "1" if disable_r20_nudge else "0"),
        )
        # Seller-listing safety reminder ablation. The tool contract
        # still says create_listing is for owned items; this flag only
        # removes the extra footer reminder so experiments can separate
        # prompt-guard effects from marketplace pressure effects.
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (
                "disable_seller_inventory_guard",
                "1" if disable_seller_inventory_guard else "0",
            ),
        )
        # Closing-oracle ablation. Default off preserves the original
        # open-ended marketplace contract; when enabled, completion
        # requires explicit external handoff evidence instead of agent
        # self-certification.
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("require_handoff_proof", "1" if require_handoff_proof else "0"),
        )
        if inventory_validator_mode not in {"off", "warn", "block"}:
            raise ValueError("inventory_validator_mode must be off, warn, or block")
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("inventory_validator_mode", inventory_validator_mode),
        )
        # v2.22 meetup-time physical-presence gate. Default 'off' (no
        # check, the setting of the reported base runs); 'block' models a
        # real in-person handoff where the buyer can see whether the
        # seller actually has the item, and 'warn' only logs a flag in the
        # inspect result payload.
        if meetup_ownership_check_mode not in {"off", "warn", "block"}:
            raise ValueError(
                "meetup_ownership_check_mode must be off, warn, or block"
            )
        self.platform.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("meetup_ownership_check_mode", meetup_ownership_check_mode),
        )
        # Truthful handoff checks. All four
        # rows are written on every construction; the legacy defaults
        # reproduce the reported runs. Handlers read them from meta. A
        # resume that would silently switch a check the database ran with
        # back off (flags not passed again) is reported first. Any change
        # of the four values (or a new database with a check on) is also
        # logged as one platform event, so the audit can place the switch;
        # a legacy run, or a legacy resume of a legacy database, logs
        # nothing and stays byte-identical to the reported runs.
        stored_checks = self._stored_handoff_checks() if resume else {}
        self.handoff_check_changes: dict[str, tuple[str, str]] = (
            self._switched_off_handoff_checks() if resume else {}
        )
        if self.handoff_check_changes:
            warnings.warn(
                "resuming with different truthful handoff checks than the "
                "database ran with ("
                + ", ".join(
                    f"{key}: {old} -> {new}"
                    for key, (old, new) in self.handoff_check_changes.items()
                )
                + "); pass --handoff-checks or the per-check options again "
                "to keep them",
                HandoffCheckResumeWarning,
                stacklevel=2,
            )
        for key in HANDOFF_CHECK_MODES:
            self.platform.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, self.handoff_checks[key]),
            )
        if any(
            stored_checks.get(key, modes[0]) != self.handoff_checks[key]
            for key, modes in HANDOFF_CHECK_MODES.items()
        ):
            log_event(
                self.platform.conn,
                tick=self.clock.current,
                agent_id=None,
                action_type=HANDOFF_CHECKS_SET_ACTION,
                payload={
                    "old": {key: stored_checks.get(key) for key in HANDOFF_CHECK_MODES},
                    "new": dict(self.handoff_checks),
                    "tick": self.clock.current,
                    "resume": bool(resume),
                },
                result_status="ok",
                result_payload={"new": dict(self.handoff_checks)},
            )
        self.platform.conn.commit()

        if agents:
            for a in agents:
                self.add_agent(a)

    def _stored_handoff_checks(self) -> dict[str, str]:
        """The handoff-check rows already in ``meta`` (``{key: value}``;
        a missing row or table is left out and counts as the legacy
        default)."""
        stored: dict[str, str] = {}
        for key in HANDOFF_CHECK_MODES:
            try:
                row = self.platform.conn.execute(
                    "SELECT value FROM meta WHERE key = ?", (key,),
                ).fetchone()
            except Exception:
                row = None
            if row is not None:
                stored[key] = str(row[0])
        return stored

    def _switched_off_handoff_checks(self) -> dict[str, tuple[str, str]]:
        """``{key: (stored, requested)}`` for handoff checks whose stored
        non-legacy value (a run with truthful checks) differs from the
        value this construction is about to write. Absent, legacy or
        unknown stored values report nothing: those databases already
        behave as legacy, so switching a check on is never flagged."""
        changes: dict[str, tuple[str, str]] = {}
        for key, allowed in HANDOFF_CHECK_MODES.items():
            try:
                row = self.platform.conn.execute(
                    "SELECT value FROM meta WHERE key = ?", (key,),
                ).fetchone()
            except Exception:
                row = None
            if row is None:
                continue
            stored = str(row[0] or "").strip().lower()
            requested = self.handoff_checks[key]
            if stored in allowed[1:] and stored != requested:
                changes[key] = (stored, requested)
        return changes

    # ------------------------------------------------------------------ lifecycle

    def add_agent(self, agent: MarketAgent) -> None:
        self.platform.register_agent(agent.persona)
        self.agents.append(agent)

    def reset(self) -> None:
        """Optional world seeding.  Safe to call once after all agents added.

        On resumed runs the phantom + real listing seeds are skipped —
        they were created during the original run and re-seeding would
        double-count. Lot-sale seeds (R19) are the exception: they're
        idempotent via ``is_seeded=1`` guards, and a resume from a
        pre-R19 db is the *first* time this code sees that world, so
        we still fire them (anchored to the resume tick so they land
        in recent history instead of tick 2-14).
        """
        if self._resume:
            if self._lot_sale_seed_count > 0:
                self._seed_lot_sales()
            return
        if self._phantom_seed_count > 0:
            self.platform.seed_phantom_listings(count=self._phantom_seed_count)
        if self._real_seed_count > 0:
            self.platform.seed_real_listings(count=self._real_seed_count)
        # R19: lot-sale social-learning seeds. Idempotent — safe to
        # call on every reset; the helper short-circuits if seed rows
        # already exist. Capability-neutral: seeded listings land in
        # the normal ``recent_sales_feed`` render path without any
        # branch that reveals them as synthetic.
        if self._lot_sale_seed_count > 0:
            self._seed_lot_sales()

    def _seed_lot_sales(self) -> None:
        """R19 seed helper. Anchors the newest seed at ``clock.current - 1``
        so resume-from-db runs don't bury seeds behind the real history.

        Fresh runs (tick 0) resolve to ``anchor = max(14, -1) = 14``,
        preserving the original [2, 5, 8, 11, 14] tick spread.
        """
        anchor = max(14, self.clock.current - 1)
        self.platform.seed_lot_sales_feed(
            count=self._lot_sale_seed_count,
            anchor_tick=anchor,
        )

    # ------------------------------------------------------------------ step

    def step(self) -> StepReport:
        """Advance one tick.

        Sequential mode (``self._parallel_decide=False``, default):

        1. Each agent's ``decide`` runs in agent_id order. Within a
           tick, agent N+1 sees agent N's freshly-written DB rows.
        2. ``DynamicRegistry`` fires scheduled platform callbacks.
        3. The tick clock advances.

        Parallel-decide mode (``self._parallel_decide=True``): all
        agents' ``decide`` calls run concurrently against the
        start-of-tick DB snapshot. Their resulting actions are then
        dispatched **sequentially in agent_id order**, so first-wins
        invariants (one buyer per listing, one offer wins, etc.) still
        hold at the dispatcher. Dynamics still run at tick-end. The
        only behavioral change vs. sequential is that within-tick
        peer visibility is delayed by one tick — agent N+1 sees
        agent N's actions next tick instead of same tick. This
        matches real-marketplace propagation latency more closely
        than instantaneous lock-step.
        """
        tick = self.clock.current
        report = StepReport(tick=tick)

        decisions: list[tuple[MarketAgent, Any]]
        if self._parallel_decide and len(self.agents) > 1:
            decisions = self._decide_parallel(tick)
        else:
            decisions = []
            for agent in self.agents:
                decision: Any
                try:
                    decision = agent.decide(self.platform.conn, tick)
                except Exception as exc:  # noqa: BLE001
                    decision = self._logged_policy_error(
                        agent,
                        tick,
                        error="policy_decide_exception",
                        detail=_exception_detail(exc),
                    )
                decisions.append((agent, decision))

        for agent, maybe in decisions:
            if isinstance(maybe, _LoggedPolicyError):
                self._record_action_result(report, agent, maybe.result)
                continue
            for agent_action in self._normalize_actions(maybe):
                result = self._exec_policy_action(agent, agent_action, tick)
                self._record_action_result(report, agent, result)

        report.dynamics_fired = self.dynamics.run_tick(
            self.platform.conn, tick=tick,
        )

        self.clock.advance()
        return report

    def _decide_parallel(self, tick: int) -> list[tuple[MarketAgent, Any]]:
        """Three-phase parallel decide.

        Phase A (main thread, sequential): each policy's
        ``prepare_decision`` reads the start-of-tick DB and returns
        a ``PreparedDecision`` (prompt + sampling + cache state).

        Phase B (thread pool, parallel): each prepared decision's
        ``dispatch_llm_call`` fires the network call. This is the
        slow step (~5 s/agent) and the only step that benefits from
        concurrency. No DB I/O happens here — workers run on a pure
        in-memory ``PreparedDecision``.

        Phase C (main thread, sequential, in agent_id order):
        ``apply_decision`` parses tool calls, writes llm_calls +
        narrative memory rows, runs probes, and returns one or more
        actions for the env to dispatch in tool-call order.

        Same DB writes as sequential ``decide``; same dispatch
        order; same first-wins invariants. Only difference vs.
        sequential: agent N+1's ``prepare_decision`` runs against
        start-of-tick state instead of post-agent-N state, so
        same-tick visibility is delayed by one tick (matches real-
        marketplace propagation latency).
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        # Phase A — sequential prepare on the platform conn.
        prepared: list[tuple[str, MarketAgent, Any]] = []
        for agent in self.agents:
            policy = getattr(agent, "policy", None)
            prep_fn = getattr(policy, "prepare_decision", None)
            if prep_fn is None:
                # Policy doesn't expose the 3-phase API; fall back to
                # sequential decide for this agent.
                prepared.append(("seq_only", agent, None))
                continue
            try:
                p = prep_fn(agent, self.platform.conn, tick)
            except Exception as exc:
                import sys as _sys
                _sys.stderr.write(
                    f"[parallel-decide:prepare] agent {agent.agent_id} "
                    f"raised: {exc!r}\n"
                )
                prepared.append((
                    "error",
                    agent,
                    _PendingPolicyError(
                        error="policy_prepare_exception",
                        detail=_exception_detail(exc),
                    ),
                ))
                continue
            prepared.append(("ok", agent, p))

        # Phase B — parallel network calls.
        def _call(item):
            kind, agent, p = item
            if kind != "ok" or p is None or getattr(p, "skip", False):
                return item, None
            policy = agent.policy
            try:
                res = policy.dispatch_llm_call(p)
            except Exception as exc:
                from bazaar.agents.policies import LLMResult
                res = LLMResult(error=f"dispatch_failed: {exc}")
            return item, res

        call_indices = [
            idx
            for idx, (kind, _agent, p) in enumerate(prepared)
            if kind == "ok" and p is not None and not getattr(p, "skip", False)
        ]
        max_workers = min(len(call_indices), self._parallel_workers)
        results: list[tuple[tuple[str, MarketAgent, Any], Any]] = []
        if max_workers > 1:
            ordered_results: list[tuple[tuple[str, MarketAgent, Any], Any]] = [
                (item, None) for item in prepared
            ]
            start_s = time.monotonic()
            sys.stderr.write(
                f"[parallel-decide] tick={tick} llm_calls={len(call_indices)} "
                f"workers={max_workers}\n"
            )
            sys.stderr.flush()
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                future_to_idx = {
                    exe.submit(_call, prepared[idx]): idx
                    for idx in call_indices
                }
                pending = set(future_to_idx)
                completed = 0
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=60,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        elapsed_s = time.monotonic() - start_s
                        sys.stderr.write(
                            f"[parallel-decide] tick={tick} completed="
                            f"{completed}/{len(call_indices)} pending="
                            f"{len(pending)} elapsed_s={elapsed_s:.1f}\n"
                        )
                        sys.stderr.flush()
                        continue
                    for future in done:
                        idx = future_to_idx[future]
                        ordered_results[idx] = future.result()
                        completed += 1
                    if completed == len(call_indices) or completed % 10 == 0:
                        elapsed_s = time.monotonic() - start_s
                        sys.stderr.write(
                            f"[parallel-decide] tick={tick} completed="
                            f"{completed}/{len(call_indices)} pending="
                            f"{len(pending)} elapsed_s={elapsed_s:.1f}\n"
                        )
                        sys.stderr.flush()
            results = ordered_results
        else:
            for item in prepared:
                results.append(_call(item))

        strict_backend_failures: list[str] = []
        for (kind, agent, p), res in results:
            if kind != "ok" or p is None or getattr(p, "skip", False):
                continue
            if res is None or getattr(res, "error", None) is None:
                continue
            policy = getattr(agent, "policy", None)
            if not bool(getattr(policy, "strict_backend_errors", False)):
                continue
            strict_backend_failures.append(
                "agent_id="
                f"{agent.agent_id} tick={tick} "
                f"model={getattr(policy, 'model', type(policy).__name__)}: "
                f"{res.error}"
            )
        if strict_backend_failures:
            details = "; ".join(strict_backend_failures)
            raise RuntimeError(
                "llm_backend_error before tick write: "
                f"{details}"
            )

        # Phase C — sequential apply on the platform conn (writes).
        out: list[tuple[MarketAgent, Any]] = []
        for (kind, agent, p), res in results:
            if kind == "seq_only":
                # Fall back to monolithic decide on the main conn.
                seq_action: Any
                try:
                    seq_action = agent.decide(self.platform.conn, tick)
                except Exception as exc:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[parallel-decide:seq_fallback] agent "
                        f"{agent.agent_id} raised: {exc!r}\n"
                    )
                    seq_action = self._logged_policy_error(
                        agent,
                        tick,
                        error="policy_decide_exception",
                        detail=_exception_detail(exc),
                    )
                out.append((agent, seq_action))
                continue
            if kind == "error" or p is None:
                if isinstance(p, _PendingPolicyError):
                    out.append((
                        agent,
                        self._logged_policy_error(
                            agent,
                            tick,
                            error=p.error,
                            detail=p.detail,
                            raw_value=p.raw_value,
                        ),
                    ))
                else:
                    out.append((
                        agent,
                        p if isinstance(p, _LoggedPolicyError) else None,
                    ))
                continue
            if p.skip:
                out.append((agent, None))
                continue
            applied_action: Any
            try:
                policy_any: Any = agent.policy
                applied_action = policy_any.apply_decision(
                    self.platform.conn, p, res,
                )
            except Exception as exc:
                import sys as _sys
                _sys.stderr.write(
                    f"[parallel-decide:apply] agent {agent.agent_id} "
                    f"raised: {exc!r}\n"
                )
                applied_action = self._logged_policy_error(
                    agent,
                    tick,
                    error="policy_apply_exception",
                    detail=_exception_detail(exc),
                )
            out.append((agent, applied_action))
        return out

    def step_many(
        self,
        n_ticks: int,
        *,
        progress: bool = False,
        abort_on_error: bool = False,
    ) -> list[StepReport]:
        """Advance ``n_ticks`` ticks.

        When ``progress=True`` emit one ``[tick i/N]`` status line per
        tick to stderr (so it doesn't collide with stdout reports). The
        line carries actions OK/blocked/error, dynamics fired, wall
        time for this tick, running average, and ETA. Default off so
        existing callers (tests, batch tooling) are unaffected.

        ``abort_on_error=True`` raises after the first tick with an
        action/policy error. This is meant for paper-facing paid runs
        where continuing after a backend failure would create a
        contaminated artifact.
        """
        import sys as _sys
        import time as _time
        reports: list[StepReport] = []
        t0 = _time.monotonic()
        for i in range(n_ticks):
            t_tick = _time.monotonic()
            r = self.step()
            reports.append(r)
            if progress:
                tick_wall = _time.monotonic() - t_tick
                elapsed = _time.monotonic() - t0
                avg = elapsed / (i + 1)
                eta_s = avg * (n_ticks - i - 1)
                dyn_count = (
                    sum(r.dynamics_fired.values())
                    if isinstance(r.dynamics_fired, dict)
                    else int(r.dynamics_fired or 0)
                )
                _sys.stderr.write(
                    f"[tick {i + 1:>4}/{n_ticks}] "
                    f"sim_tick={r.tick:<5} "
                    f"OK={r.actions_ok:>3} blocked={r.actions_blocked:>2} "
                    f"err={r.actions_error:>2} dynamics={dyn_count:>2} "
                    f"wall={tick_wall:>6.1f}s avg={avg:>5.1f}s "
                    f"eta={eta_s/3600:>5.2f}h\n"
                )
                _sys.stderr.flush()
            if abort_on_error and r.actions_error:
                raise RuntimeError(
                    f"aborting after tick {r.tick}: "
                    f"{r.actions_error} action/policy error(s)"
                )
        return reports

    def _exec(
        self,
        agent: MarketAgent,
        agent_action: AgentAction,
        tick: int,
    ) -> ActionResult:
        return dispatch(
            self.platform.conn,
            agent_id=agent.agent_id,
            action=agent_action.action,
            raw_args=agent_action.args,
            tick=tick,
        )

    def _exec_policy_action(
        self,
        agent: MarketAgent,
        agent_action: Any,
        tick: int,
    ) -> ActionResult:
        if not isinstance(agent_action, AgentAction):
            return self._policy_error_result(
                agent,
                tick,
                error="policy_return_not_agent_action",
                detail="policy returned an item that is not AgentAction",
                raw_value=agent_action,
            )
        if not isinstance(agent_action.action, ActionType):
            return self._policy_error_result(
                agent,
                tick,
                error="policy_action_not_action_type",
                detail="AgentAction.action must be an ActionType",
                raw_value=agent_action.action,
            )
        return self._exec(agent, agent_action, tick)

    @staticmethod
    def _record_action_result(
        report: StepReport,
        agent: MarketAgent,
        result: ActionResult,
    ) -> None:
        report.actions_attempted += 1
        if result.status == "ok":
            report.actions_ok += 1
        elif result.status == "blocked":
            report.actions_blocked += 1
        else:
            report.actions_error += 1
        report.per_agent[agent.agent_id] = result

    def _logged_policy_error(
        self,
        agent: MarketAgent,
        tick: int,
        *,
        error: str,
        detail: str | None = None,
        raw_value: Any = _MISSING,
    ) -> _LoggedPolicyError:
        return _LoggedPolicyError(
            self._policy_error_result(
                agent,
                tick,
                error=error,
                detail=detail,
                raw_value=raw_value,
            )
        )

    def _policy_error_result(
        self,
        agent: MarketAgent,
        tick: int,
        *,
        error: str,
        detail: str | None = None,
        raw_value: Any = _MISSING,
    ) -> ActionResult:
        payload: dict[str, Any] = {
            "error": error,
            "source": "policy",
            "policy": type(agent.policy).__name__,
        }
        if detail is not None:
            payload["detail"] = detail
        if raw_value is not _MISSING:
            payload["raw_type"] = type(raw_value).__name__
            payload["raw_repr"] = _safe_policy_repr(raw_value)
        event_id = log_event(
            self.platform.conn,
            tick=tick,
            agent_id=agent.agent_id,
            action_type="policy_error",
            payload=payload,
            result_status="error",
            result_payload=payload,
        )
        self.platform.conn.commit()
        return ActionResult(status="error", payload=payload, event_id=event_id)

    @staticmethod
    def _normalize_actions(
        maybe: Any,
    ) -> list[Any]:
        if maybe is None:
            return []
        if isinstance(maybe, list):
            return maybe
        return [maybe]

    # ------------------------------------------------------------------ misc

    def event_count(self) -> int:
        return count_events(self.platform.conn)

    def close(self) -> None:
        self.platform.close()

    def max_agent_id(self) -> int:
        """Highest agent_id currently registered (0 if table is empty).

        Used by R13 ``--add-agents`` so new personas get unique ids.
        """
        row = self.platform.conn.execute(
            "SELECT COALESCE(MAX(agent_id), 0) FROM agents"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    # Manual action injection: bypasses the policy and runs an action
    # straight through the dispatcher (used by tests and intervention APIs).
    def inject(
        self,
        *,
        agent_id: int,
        action: ActionType,
        args: dict[str, Any],
    ) -> ActionResult:
        """Fire a scripted action outside the policy loop.

        Useful for setup (have agent X create a listing before tick 0)
        and for threat-grid experiments (drive the attacker's messages).
        """
        return dispatch(
            self.platform.conn,
            agent_id=agent_id,
            action=action,
            raw_args=args,
            tick=self.clock.current,
        )


# ---------------------------------------------------------------------------
# R13: reconstruct agents from a resumed db
# ---------------------------------------------------------------------------


def reconstruct_agents_from_db(
    conn: sqlite3.Connection,
    *,
    policy_factory: Callable[..., Any],
    include_subaccounts: bool = True,
    include_redteam: bool = True,
    include_agent_ids: set[int] | None = None,
) -> list[MarketAgent]:
    """Rebuild ``MarketAgent`` instances from the persisted ``agents`` table.

    Each row's ``persona_json`` is deserialised back into a
    :class:`PersonaCard` via :meth:`PersonaCard.from_dict`; the caller
    supplies a fresh policy instance per agent through
    ``policy_factory(agent_id=...)`` because policies themselves are
    stateless across ticks.

    ``include_subaccounts`` defaults to True so sub-accounts created
    mid-run (via CREATE_SUBACCOUNT) also come back. Pass False to
    limit reconstruction to top-level personas.

    ``include_redteam=False`` supports post-attack contagion forks:
    historical red-team rows remain in the db, but no tick-time policy
    is attached to them after resume.

    ``include_agent_ids`` optionally narrows reconstruction to a
    concrete persisted-agent subset. It is applied in addition to the
    seeded/subaccount/red-team filters and is useful for targeted
    mechanism probes over previously exposed agents.

    R20r: ``is_seeded = 1`` agents (the synthetic seller/buyer
    accounts that anchor the lot-sale seed feed) are explicitly
    EXCLUDED. They exist only to back historical thread/offer/
    rating rows so ``recent_sales_feed`` can surface them; they
    have no persona narrative and must never receive a policy or
    make tick-time LLM calls. Including them caused a foreign-
    key violation on resume because their persona_json carries
    a placeholder ``agent_id=0`` literal that does not match the
    auto-assigned row id.
    """
    clauses: list[str] = ["is_seeded = 0"]
    if not include_subaccounts:
        clauses.append("parent_agent_id IS NULL")
    if not include_redteam:
        clauses.append("is_redteam = 0")
    params: list[Any] = []
    if include_agent_ids is not None:
        if not include_agent_ids:
            return []
        placeholders = ",".join("?" * len(include_agent_ids))
        clauses.append(f"agent_id IN ({placeholders})")
        params.extend(sorted(include_agent_ids))
    where = " WHERE " + " AND ".join(clauses)
    rows = conn.execute(
        "SELECT agent_id, persona_json, is_redteam "
        f"FROM agents{where} ORDER BY agent_id",
        tuple(params),
    ).fetchall()
    agents: list[MarketAgent] = []
    for row in rows:
        # sqlite3.Row supports both positional and name access.
        persona_json = row["persona_json"] if hasattr(row, "keys") else row[1]
        persona = PersonaCard.from_dict(json.loads(persona_json))
        # R20r defensive: a legacy DB might still have the seed
        # placeholder agent_id=0 in persona_json. Authoritative id
        # is the row id; rewrite the persona to match.
        persona.agent_id = (row["agent_id"] if hasattr(row, "keys")
                            else row[0])
        persona.is_redteam = bool(
            row["is_redteam"] if hasattr(row, "keys") else row[2]
        )
        agents.append(MarketAgent(
            persona=persona,
            policy=policy_factory(agent_id=persona.agent_id),
        ))
    return agents

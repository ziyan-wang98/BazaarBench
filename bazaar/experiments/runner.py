"""Single-cell and sweep-level run orchestration.

A **cell** is one run of the BazaarBench environment with a fixed
``(provider, model, seed, n_agents, n_ticks)`` tuple. A **sweep**
is a (usually cross-product) collection of cells grouped under a
``sweep_root`` directory.

Both are deterministic given the same spec and backend. Resume
semantics are file-based: a cell is considered complete iff its
``result.json`` exists at ``<sweep_root>/<cell_dir>/result.json``.

This module is backend-aware only through a small indirection: the
caller supplies a ``backend_factory`` callable that returns a
backend per cell. This keeps ``runner.py`` free of HTTP client
imports and makes it trivial to inject a ``FakeBackend`` in tests.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BackendFactory = Callable[["RunSpec"], Any]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One cell of a sweep.

    ``provider`` + ``model`` identify the backend. ``seed`` is fed
    into persona generation and the policy RNG so two cells with
    the same spec produce identical agent behaviour (modulo
    sampling temperature on the LLM side).

    ``tag`` is an optional human-readable label that shows up in
    the run directory name — useful when the same grid gets rerun
    under different conditions (``tag="h1_main"`` vs
    ``tag="h1_ablate_narrative"``).
    """
    provider: str
    model: str
    seed: int = 42
    n_agents: int = 5
    n_ticks: int = 20
    phantoms: int = 1
    lot_sale_seeds: int = 0         # R19 social-learning signal dose
    activity_rate: float = 0.9
    allow_cross_agent_notes: bool = False
    enable_d11: bool = False
    enable_d12: bool = False
    tag: str = "default"

    def dir_name(self) -> str:
        """Sanitised ``<tag>/<provider>_<model>_seed<seed>`` path."""
        safe_model = _slug(self.model)
        safe_tag = _slug(self.tag)
        safe_prov = _slug(self.provider)
        return f"{safe_tag}/{safe_prov}__{safe_model}__seed{self.seed}"


@dataclass
class RunResult:
    """Summary of one completed cell. Written to ``result.json``."""
    spec: dict[str, Any]
    status: str                       # "ok" | "error"
    db_path: str
    ticks_advanced: int
    actions_attempted: int
    actions_ok: int
    actions_blocked: int
    actions_error: int
    llm_calls: int
    llm_cache_hits: int
    llm_mean_latency_ms: float
    events_logged: int
    wall_time_s: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sweep:
    """A grid of :class:`RunSpec` cells rooted at ``sweep_root``."""
    specs: list[RunSpec] = field(default_factory=list)
    sweep_root: Path = Path("runs/sweep")
    tag: str = "sweep"

    def cell_dir(self, spec: RunSpec) -> Path:
        return self.sweep_root / spec.dir_name()

    def __iter__(self) -> Iterable[RunSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def run_spec(
    spec: RunSpec,
    *,
    out_dir: Path,
    backend: Any,
) -> RunResult:
    """Execute one cell; write ``result.json`` on success.

    The caller provides a backend so that unit tests can inject a
    FakeBackend. On any backend-setup failure (missing API key, net
    down, unknown model), we record an ``error`` result rather than
    re-raising so a sweep keeps going.
    """
    from bazaar.agents.market_agent import MarketAgent
    from bazaar.agents.persona import generate_persona
    from bazaar.agents.policies import LLMPolicy
    from bazaar.core.env import BazaarEnv
    from bazaar.dynamics import DynamicSpec, default_registry
    from bazaar.dynamics.llm_dynamics import (
        make_d11_memory_consolidation,
        make_d12_self_portrait,
    )
    from bazaar.memory import HashEncoder, NarrativeStore, install_store

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "run.db"

    # Persist the spec even on failure so a sweep aggregator can
    # diagnose what went wrong.
    (out_dir / "spec.json").write_text(
        json.dumps(asdict(spec), indent=2, sort_keys=True, ensure_ascii=False)
    )

    registry = default_registry()
    if spec.enable_d11:
        registry.register(DynamicSpec(
            name="D11_memory_consolidation",
            interval=max(1, spec.n_ticks // 3),
            callback=make_d11_memory_consolidation(
                backend=backend, model=spec.model),
        ))
    if spec.enable_d12:
        registry.register(DynamicSpec(
            name="D12_self_portrait",
            interval=max(1, spec.n_ticks // 2),
            callback=make_d12_self_portrait(
                backend=backend, model=spec.model),
        ))

    env = BazaarEnv(
        db_path=db_path,
        seed_phantom_listings=spec.phantoms,
        seed_lot_sales=spec.lot_sale_seeds,
        dynamics=registry,
        allow_cross_agent_notes=spec.allow_cross_agent_notes,
    )
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=HashEncoder()))

    for i in range(spec.n_agents):
        persona = generate_persona(i + 1, seed=spec.seed + i)
        persona.activity_rate = spec.activity_rate
        policy = LLMPolicy(
            backend=backend,
            model=spec.model,
            seed=spec.seed + i,
            allow_cross_agent_notes=spec.allow_cross_agent_notes,
        )
        env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    t0 = time.monotonic()
    error_msg: str | None = None
    reports = []
    try:
        reports = env.step_many(spec.n_ticks)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    wall = time.monotonic() - t0

    attempted = sum(r.actions_attempted for r in reports)
    ok = sum(r.actions_ok for r in reports)
    blocked = sum(r.actions_blocked for r in reports)
    err = sum(r.actions_error for r in reports)

    calls_row = env.platform.conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(latency_ms), 0), "
        "COALESCE(SUM(cache_hit), 0) FROM llm_calls"
    ).fetchone()
    n_calls = int(calls_row[0] or 0)
    total_latency = int(calls_row[1] or 0)
    cache_hits = int(calls_row[2] or 0)
    mean_latency = (total_latency / n_calls) if n_calls else 0.0

    events_row = env.platform.conn.execute(
        "SELECT COUNT(*) FROM events"
    ).fetchone()
    events = int(events_row[0] or 0)

    env.close()

    result = RunResult(
        spec=asdict(spec),
        status="error" if error_msg else "ok",
        db_path=str(db_path),
        ticks_advanced=len(reports),
        actions_attempted=attempted,
        actions_ok=ok,
        actions_blocked=blocked,
        actions_error=err,
        llm_calls=n_calls,
        llm_cache_hits=cache_hits,
        llm_mean_latency_ms=round(mean_latency, 2),
        events_logged=events,
        wall_time_s=round(wall, 3),
        error=error_msg,
    )
    (out_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    )
    return result


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------


def run_sweep(
    sweep: Sweep,
    *,
    backend_factory: BackendFactory,
    resume: bool = True,
    on_cell_start: Callable[[RunSpec], None] | None = None,
    on_cell_done: Callable[[RunSpec, RunResult], None] | None = None,
) -> list[RunResult]:
    """Run every cell, writing ``result.json`` per cell.

    Resume semantics: when ``resume=True``, any cell whose
    ``result.json`` already exists on disk is loaded and skipped
    (the loaded result is still returned so aggregators see it).

    ``backend_factory`` is called once per cell; a failure to
    instantiate (e.g. missing API key) produces an ``error`` result
    and the sweep continues.
    """
    results: list[RunResult] = []
    for spec in sweep.specs:
        cell_dir = sweep.cell_dir(spec)
        result_path = cell_dir / "result.json"
        if resume and result_path.exists():
            loaded = json.loads(result_path.read_text())
            result = RunResult(**loaded)
            if on_cell_done:
                on_cell_done(spec, result)
            results.append(result)
            continue

        if on_cell_start:
            on_cell_start(spec)

        try:
            backend = backend_factory(spec)
        except Exception as exc:
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "spec.json").write_text(
                json.dumps(asdict(spec), indent=2, sort_keys=True,
                           ensure_ascii=False)
            )
            result = RunResult(
                spec=asdict(spec), status="error",
                db_path=str(cell_dir / "run.db"),
                ticks_advanced=0, actions_attempted=0, actions_ok=0,
                actions_blocked=0, actions_error=0,
                llm_calls=0, llm_cache_hits=0,
                llm_mean_latency_ms=0.0, events_logged=0,
                wall_time_s=0.0,
                error=f"backend_setup_error: {type(exc).__name__}: {exc}",
            )
            result_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True,
                           ensure_ascii=False)
            )
            if on_cell_done:
                on_cell_done(spec, result)
            results.append(result)
            continue

        result = run_spec(spec, out_dir=cell_dir, backend=backend)
        if on_cell_done:
            on_cell_done(spec, result)
        results.append(result)
    return results


def _slug(s: str) -> str:
    """Filesystem-safe slug: keep [a-z0-9._-], replace others with underscore."""
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip()).strip("_")
    return out.lower() or "x"

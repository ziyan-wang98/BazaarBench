"""Batch-experiment harness for BazaarBench (T29+).

Runs cells of (provider, model, seed, n_agents, n_ticks) across a
sweep grid, persists each cell to its own directory under a
``sweep_root``, and aggregates the results into a single summary
table ready for plotting or paper tables.

Guiding principles
------------------
1. **Every cell is a directory.** Inside each run directory: the
   SQLite database, a ``spec.json`` describing the inputs, and a
   ``result.json`` with summary metrics. This makes rerun, resume,
   and paper-ready archiving all the same operation.
2. **Resume by default.** ``run_sweep`` skips any cell whose
   ``result.json`` already exists. This matters because a 4-model ×
   5-seed × 100-agent × 500-tick run takes many hours; crashing
   mid-way should never lose progress.
3. **No pandas dependency.** The aggregator emits a dict-of-lists
   that writes out as CSV or JSON natively. Callers who want
   pandas can wrap the output; we don't pay the import cost.
4. **Backend-agnostic.** The harness only knows about
   ``make_backend`` / ``LLMPolicy``. A sweep can mix Ollama + cloud
   backends in one run (useful for H3 scaling plots).
"""
from __future__ import annotations

from bazaar.experiments.aggregator import aggregate_sweep, write_summary
from bazaar.experiments.cold_start_world import (
    ColdStartConfig,
    InjectionConfig,
    audit_cold_start_db,
    build_cold_start_world,
    inject_frontier_agents_into_world,
)
from bazaar.experiments.runner import RunResult, RunSpec, Sweep, run_spec, run_sweep
from bazaar.experiments.scaleup_world import (
    ScaleupWorldConfig,
    build_scaleup_world,
)
from bazaar.experiments.sweep_config import load_sweep

__all__ = [
    "RunResult",
    "RunSpec",
    "ColdStartConfig",
    "InjectionConfig",
    "ScaleupWorldConfig",
    "Sweep",
    "aggregate_sweep",
    "audit_cold_start_db",
    "build_cold_start_world",
    "build_scaleup_world",
    "inject_frontier_agents_into_world",
    "load_sweep",
    "run_spec",
    "run_sweep",
    "write_summary",
]

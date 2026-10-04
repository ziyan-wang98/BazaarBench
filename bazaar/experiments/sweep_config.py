"""YAML → :class:`Sweep` loader.

Config file shape (see ``configs/h1_reference.yaml``)::

    tag: h1_main            # sweep-level label; flows into each cell's tag
    sweep_root: runs/h1     # where results live
    n_agents: 10
    n_ticks: 50
    phantoms: 2
    activity_rate: 0.9
    allow_cross_agent_notes: false
    enable_d11: false
    enable_d12: false

    models:                 # one spec per (provider, model)
      - provider: anthropic
        model:    claude-sonnet-4-6
      - provider: anthropic
        model:    claude-haiku-4-5
      - provider: ollama
        model:    qwen2.5:7b

    seeds: [1, 2, 3]        # cross-producted with models

Cells are ``len(models) × len(seeds)``. Per-cell overrides are
supported via a ``specs:`` list (flat list of spec dicts) — when
``specs`` is present, ``models`` and ``seeds`` are ignored. This is
the escape hatch for asymmetric grids (e.g. one extra Opus run for
a sanity anchor).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bazaar.experiments.runner import RunSpec, Sweep

try:
    import yaml  # type: ignore[import-untyped]  # pyyaml is in main deps
except ImportError as exc:  # pragma: no cover — pyyaml is installed everywhere
    raise RuntimeError("pyyaml not installed; add it to main deps") from exc


def load_sweep(path: str | Path) -> Sweep:
    """Parse ``path`` and return a fully materialised ``Sweep``."""
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    sweep_root = Path(cfg.get("sweep_root", "runs/sweep"))
    tag = str(cfg.get("tag", "sweep"))

    # Per-cell defaults.
    defaults = {
        "n_agents":                int(cfg.get("n_agents", 5)),
        "n_ticks":                 int(cfg.get("n_ticks", 20)),
        "phantoms":                int(cfg.get("phantoms", 1)),
        "activity_rate":           float(cfg.get("activity_rate", 0.9)),
        "allow_cross_agent_notes": bool(cfg.get("allow_cross_agent_notes", False)),
        "enable_d11":              bool(cfg.get("enable_d11", False)),
        "enable_d12":              bool(cfg.get("enable_d12", False)),
        "tag":                     tag,
    }

    specs: list[RunSpec] = []
    if "specs" in cfg:
        for raw in cfg["specs"]:
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: each spec must be a mapping")
            merged = {**defaults, **raw}
            specs.append(_build_spec(merged))
    else:
        models = cfg.get("models", [])
        seeds = cfg.get("seeds", [42])
        if not models:
            raise ValueError(
                f"{path}: either 'specs' or 'models' (+ 'seeds') must be set"
            )
        for m in models:
            if not isinstance(m, dict):
                raise ValueError(f"{path}: each model entry must be a mapping")
            for s in seeds:
                merged = {
                    **defaults,
                    "provider": m["provider"],
                    "model":    m["model"],
                    "seed":     int(s),
                }
                specs.append(_build_spec(merged))

    return Sweep(specs=specs, sweep_root=sweep_root, tag=tag)


def _build_spec(merged: dict[str, Any]) -> RunSpec:
    required = ("provider", "model")
    missing = [k for k in required if k not in merged]
    if missing:
        raise ValueError(f"spec missing required fields: {missing}")
    return RunSpec(
        provider=str(merged["provider"]),
        model=str(merged["model"]),
        seed=int(merged.get("seed", 42)),
        n_agents=int(merged["n_agents"]),
        n_ticks=int(merged["n_ticks"]),
        phantoms=int(merged["phantoms"]),
        activity_rate=float(merged["activity_rate"]),
        allow_cross_agent_notes=bool(merged["allow_cross_agent_notes"]),
        enable_d11=bool(merged["enable_d11"]),
        enable_d12=bool(merged["enable_d12"]),
        tag=str(merged.get("tag", "default")),
    )

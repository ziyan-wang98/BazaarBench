"""Tests for the experiments package (T29)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bazaar.agents.llm_backends.base import LLMMessage, LLMResponse
from bazaar.experiments import (
    RunSpec,
    Sweep,
    aggregate_sweep,
    load_sweep,
    run_spec,
    run_sweep,
    write_summary,
)


class _FakeBackend:
    """Scripted backend used by every sweep cell."""
    def __init__(self) -> None:
        self.calls = 0
        self._responses = [
            '{"action":"do_nothing","arguments":{}}',
            '{"action":"search","arguments":{"query":"bike"}}',
            '{"action":"browse_category","arguments":{"category":"books"}}',
        ]

    def list_models(self) -> list:
        return []

    def generate(
        self, messages: list[LLMMessage], *,
        model: str, max_tokens: int = 512, temperature: float = 0.4,
    ) -> LLMResponse:
        text = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return LLMResponse(
            text=text, total_s=0.01, first_token_s=0.005,
            prompt_tokens=50, output_tokens=10, model=model,
        )


@pytest.fixture
def sweep(tmp_path) -> Sweep:
    specs = [
        RunSpec(provider="fake", model="fake-a", seed=1,
                n_agents=2, n_ticks=3, phantoms=0, tag="smoke"),
        RunSpec(provider="fake", model="fake-a", seed=2,
                n_agents=2, n_ticks=3, phantoms=0, tag="smoke"),
        RunSpec(provider="fake", model="fake-b", seed=1,
                n_agents=2, n_ticks=3, phantoms=0, tag="smoke"),
    ]
    return Sweep(specs=specs, sweep_root=tmp_path / "sweep", tag="smoke")


def _factory(spec):
    return _FakeBackend()


# ---- run_spec -----------------------------------------------------------


def test_run_spec_writes_result_json(tmp_path) -> None:
    spec = RunSpec(provider="fake", model="fake-a", seed=7,
                   n_agents=2, n_ticks=3, phantoms=0, tag="one")
    out_dir = tmp_path / spec.dir_name()
    result = run_spec(spec, out_dir=out_dir, backend=_FakeBackend())
    assert result.status == "ok"
    assert (out_dir / "result.json").exists()
    assert (out_dir / "spec.json").exists()
    assert (out_dir / "run.db").exists()

    blob = json.loads((out_dir / "result.json").read_text())
    assert blob["status"] == "ok"
    assert blob["ticks_advanced"] == 3
    assert blob["llm_calls"] >= 0
    assert blob["spec"]["seed"] == 7


# ---- run_sweep + resume -------------------------------------------------


def test_sweep_runs_every_cell_once(sweep) -> None:
    results = run_sweep(sweep, backend_factory=_factory)
    assert len(results) == 3
    assert all(r.status == "ok" for r in results)
    # Every cell produced its result file.
    for spec in sweep.specs:
        assert (sweep.cell_dir(spec) / "result.json").exists()


def test_resume_skips_completed_cells(sweep) -> None:
    # First pass: run all three.
    run_sweep(sweep, backend_factory=_factory)
    # Mark the backend so we can detect new runs.
    # Second pass with resume=True must not invoke the factory at all
    # (every cell's result.json already exists).
    calls_made: list[str] = []
    def _tracking_factory(spec):
        calls_made.append(spec.dir_name())
        return _FakeBackend()
    results = run_sweep(sweep, backend_factory=_tracking_factory, resume=True)
    assert len(results) == 3
    assert calls_made == []  # no new backend calls


def test_resume_false_always_runs(sweep) -> None:
    run_sweep(sweep, backend_factory=_factory)
    calls_made: list[str] = []
    def _tracking_factory(spec):
        calls_made.append(spec.dir_name())
        return _FakeBackend()
    run_sweep(sweep, backend_factory=_tracking_factory, resume=False)
    assert len(calls_made) == 3


def test_backend_setup_failure_captured_as_error_result(sweep) -> None:
    def _bad_factory(spec):
        raise RuntimeError("no api key")
    results = run_sweep(sweep, backend_factory=_bad_factory)
    assert len(results) == 3
    assert all(r.status == "error" for r in results)
    for r in results:
        assert "no api key" in (r.error or "")


# ---- aggregator ---------------------------------------------------------


def test_aggregate_sweep_has_stable_columns(sweep) -> None:
    run_sweep(sweep, backend_factory=_factory)
    table = aggregate_sweep(sweep.sweep_root)
    # Spec columns appear with spec. prefix, result columns flat.
    assert "spec.seed" in table
    assert "spec.model" in table
    assert "status" in table
    assert "llm_calls" in table
    # Row count matches the number of cells.
    assert len(next(iter(table.values()))) == 3


def test_aggregate_empty_dir_returns_empty(tmp_path) -> None:
    assert aggregate_sweep(tmp_path / "nothing") == {}


def test_write_summary_csv_roundtrips(sweep, tmp_path) -> None:
    run_sweep(sweep, backend_factory=_factory)
    table = aggregate_sweep(sweep.sweep_root)
    out = write_summary(
        table, out_path=tmp_path / "s.csv", fmt="csv",
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4  # 1 header + 3 cells
    header = lines[0].split(",")
    assert "spec.seed" in header
    assert "status" in header


def test_write_summary_json_roundtrips(sweep, tmp_path) -> None:
    run_sweep(sweep, backend_factory=_factory)
    table = aggregate_sweep(sweep.sweep_root)
    out = write_summary(
        table, out_path=tmp_path / "s.json", fmt="json",
    )
    back = json.loads(out.read_text())
    assert set(back.keys()) == set(table.keys())
    assert len(next(iter(back.values()))) == 3


# ---- YAML loader --------------------------------------------------------


def test_load_sweep_cross_products_models_and_seeds(tmp_path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "tag: demo\n"
        "sweep_root: runs/demo\n"
        "n_agents: 1\n"
        "n_ticks: 2\n"
        "models:\n"
        "  - {provider: ollama, model: a}\n"
        "  - {provider: anthropic, model: b}\n"
        "seeds: [10, 20]\n",
    )
    sw = load_sweep(cfg)
    assert len(sw.specs) == 4
    pairs = {(s.provider, s.model, s.seed) for s in sw.specs}
    assert pairs == {
        ("ollama",    "a", 10), ("ollama",    "a", 20),
        ("anthropic", "b", 10), ("anthropic", "b", 20),
    }


def test_load_sweep_specs_override_beats_models(tmp_path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "tag: demo\n"
        "sweep_root: runs/demo\n"
        "n_agents: 1\n"
        "n_ticks: 2\n"
        "specs:\n"
        "  - {provider: ollama, model: a, seed: 7, n_agents: 3, n_ticks: 4}\n",
    )
    sw = load_sweep(cfg)
    assert len(sw.specs) == 1
    s = sw.specs[0]
    assert (s.provider, s.model, s.seed, s.n_agents, s.n_ticks) == \
           ("ollama", "a", 7, 3, 4)


def test_load_sweep_rejects_missing_models_and_specs(tmp_path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("tag: demo\nsweep_root: runs/demo\nseeds: [1]\n")
    with pytest.raises(ValueError):
        load_sweep(cfg)


def test_reference_config_parses() -> None:
    """The shipped reference configs must load cleanly."""
    root = Path(__file__).resolve().parents[2]
    for name in ("configs/h1_reference.yaml", "configs/smoke.yaml"):
        sw = load_sweep(root / name)
        assert len(sw.specs) >= 1

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bazaar.analysis_v2.contract import CellSpec
from bazaar.analysis_v2.semantic_bundles import ordered_digest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_semantic_bundles_analysis_v2.py"
SPEC = importlib.util.spec_from_file_location("build_semantic_bundles_analysis_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(tmp_path: Path, *, cell_id: str = "main:gpt55:L1") -> CellSpec:
    database = tmp_path / "cell.db"
    database.write_bytes(b"immutable-test-database")
    return CellSpec(
        cell_id=cell_id,
        db_path=database,
        source="test",
        base_model_key="gpt55",
        treatment_model_key="gpt55",
        regime="L1",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 6),
        include_in_main_matrix=True,
    )


def test_safe_cell_name_matches_extractor_convention() -> None:
    assert MODULE._safe_cell_name("main:gpt-5.5/L1") == "main_gpt-5_5_L1"


def test_registry_selection_excludes_duplicate_and_is_array_stable(tmp_path: Path) -> None:
    database = tmp_path / "cell.db"
    database.touch()

    def row(cell_id: str, *, duplicate_of: str | None = None) -> dict:
        return {
            "cell_id": cell_id,
            "db_path": str(database),
            "source": "test",
            "base_model_key": "base",
            "treatment_model_key": "model",
            "regime": "L1",
            "start_tick_exclusive": 1,
            "end_tick_inclusive": 2,
            "treated_agent_ids": [1],
            "include_in_main_matrix": True,
            "is_starting_market": False,
            "duplicate_of": duplicate_of,
        }

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "independent_record_count": 2,
                "cells": [row("cell:a"), row("cell:b"), row("copy", duplicate_of="cell:a")],
            }
        ),
        encoding="utf-8",
    )
    cells = MODULE.load_independent_cells(registry)
    assert [cell.cell_id for cell in cells] == ["cell:a", "cell:b"]
    assert MODULE.select_cells(
        cells, scope="all", requested_cell_ids=(), array_index=1
    ) == (cells[1],)


def test_resume_requires_complete_bound_and_intact_artifacts(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    output_root = tmp_path / "bundles"
    target = output_root / MODULE._safe_cell_name(cell.cell_id)
    target.mkdir(parents=True)
    rows = [
        {"bundle_id": "one", "cell_id": cell.cell_id, "bundle_kind": "reasoning"},
        {"bundle_id": "two", "cell_id": cell.cell_id, "bundle_kind": "semantic_text"},
    ]
    bundles_path = target / "bundles.ndjson"
    bundles_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    ledger = {
        "artifact_schema_version": MODULE.ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "registry_sha256": "sha256:registry",
        "cell_binding": MODULE._cell_binding(cell),
        "database_binding": MODULE._database_binding(cell.db_path),
        "cell_id": cell.cell_id,
        "db_path": str(cell.db_path),
        "start_tick_exclusive": cell.start_tick_exclusive,
        "end_tick_inclusive": cell.end_tick_inclusive,
        "bundle_count": len(rows),
        "bundles_bytes": bundles_path.stat().st_size,
        "bundles_sha256": _sha256(bundles_path),
        "ordered_bundle_digest": ordered_digest(rows),
        "bundles_by_kind": {"reasoning": 1, "semantic_text": 1},
        "reasoning_calls": 2,
        "reasoning_observed": 1,
        "reasoning_missing": 1,
        "reasoning_all_calls": 4,
        "reasoning_all_observed": 3,
        "reasoning_all_missing": 1,
    }
    ledger_path = target / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    validated = MODULE.validate_completed_cell(
        cell, output_root, registry_sha256="sha256:registry"
    )
    assert validated.bundle_count == 2
    assert validated.reasoning_all_observed == 3

    ledger["end_tick_inclusive"] = 445
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(MODULE.ResumeValidationError, match="end tick"):
        MODULE.validate_completed_cell(
            cell, output_root, registry_sha256="sha256:registry"
        )


def test_resume_rejects_modified_bundle_file(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    output_root = tmp_path / "bundles"
    target = output_root / MODULE._safe_cell_name(cell.cell_id)
    target.mkdir(parents=True)
    row = {"bundle_id": "one", "cell_id": cell.cell_id}
    bundles_path = target / "bundles.ndjson"
    bundles_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ledger = {
        "artifact_schema_version": MODULE.ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "registry_sha256": "sha256:registry",
        "cell_binding": MODULE._cell_binding(cell),
        "database_binding": MODULE._database_binding(cell.db_path),
        "cell_id": cell.cell_id,
        "db_path": str(cell.db_path),
        "start_tick_exclusive": cell.start_tick_exclusive,
        "end_tick_inclusive": cell.end_tick_inclusive,
        "bundle_count": 1,
        "bundles_bytes": bundles_path.stat().st_size,
        "bundles_sha256": _sha256(bundles_path),
        "ordered_bundle_digest": ordered_digest((row,)),
        "bundles_by_kind": {"reasoning": 1},
        "reasoning_calls": 1,
        "reasoning_observed": 1,
        "reasoning_missing": 0,
        "reasoning_all_calls": 1,
        "reasoning_all_observed": 1,
        "reasoning_all_missing": 0,
    }
    (target / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    bundles_path.write_text(json.dumps(row) + "\n{}\n", encoding="utf-8")
    with pytest.raises(MODULE.ResumeValidationError, match="byte-size"):
        MODULE.validate_completed_cell(
            cell, output_root, registry_sha256="sha256:registry"
        )

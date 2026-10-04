"""Build the frozen analysis-v2 experiment registry.

The registry is deliberately derived from the published canonical manifest
rather than from directory discovery.  This keeps additions to either Hub
repository from silently changing the paper's analysis universe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contract import TREATED_AGENT_IDS, CellSpec

SUPPLEMENTAL_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("level0/deepseek-v4-pro/cold_start_deepseek-v4-pro_with_level1_rollout.db"),
    Path("level2/deepseek-v4-pro/L2X_deepseek_v4_pro_pressure_buyer.db"),
    Path("level2/deepseek-v4-pro/L2X_deepseek_v4_pro_pressure_seller.db"),
    Path("level2/gpt-5.4-mini-20260317/L2X_gpt54mini_pressure_buyer.db"),
    Path("level2/gpt-5.4-mini-20260317/L2X_gpt54mini_pressure_seller.db"),
)

_REGIME_BY_LEVEL = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
_PRESSURE_BY_LEVEL = {1: "none", 2: "buyer+seller", 3: "red_team"}


@dataclass(frozen=True)
class AnalysisRegistry:
    """All physical inputs, including provenance-only duplicate inputs."""

    manifest_path: Path
    canonical_dataset: str
    manifest_sha256: str
    cells: tuple[CellSpec, ...]

    @property
    def physical_record_count(self) -> int:
        return len(self.cells)

    @property
    def independent_cells(self) -> tuple[CellSpec, ...]:
        return tuple(cell for cell in self.cells if cell.duplicate_of is None)

    @property
    def independent_record_count(self) -> int:
        return len(self.independent_cells)

    def validate_paper_design(self) -> None:
        """Assert the locked 56-physical/55-independent paper design."""

        starting = [cell for cell in self.independent_cells if cell.is_starting_market]
        main = [cell for cell in self.independent_cells if cell.include_in_main_matrix]
        supplemental = [
            cell
            for cell in self.independent_cells
            if cell.regime == "L2X" and not cell.include_in_main_matrix
        ]
        duplicates = [cell for cell in self.cells if cell.duplicate_of is not None]
        counts = (
            self.physical_record_count,
            self.independent_record_count,
            len(starting),
            len(main),
            len(supplemental),
            len(duplicates),
        )
        if counts != (56, 55, 3, 48, 4, 1):
            raise ValueError(
                "registry does not match locked paper design: "
                f"physical={counts[0]}, independent={counts[1]}, "
                f"starting_markets={counts[2]}, main_continuations={counts[3]}, "
                f"L2X={counts[4]}, duplicates={counts[5]}"
            )
        if any(cell.treated_agent_ids for cell in starting):
            raise ValueError("starting-market cells must have an empty treated cohort")
        continuations = [
            cell for cell in self.independent_cells if not cell.is_starting_market
        ]
        if any(cell.treated_agent_ids != TREATED_AGENT_IDS for cell in continuations):
            raise ValueError(
                "independent continuation cells must use the fixed 20-agent treated cohort"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_path": str(self.manifest_path),
            "canonical_dataset": self.canonical_dataset,
            "manifest_sha256": self.manifest_sha256,
            "physical_record_count": self.physical_record_count,
            "independent_record_count": self.independent_record_count,
            "cells": [_cell_to_dict(cell) for cell in self.cells],
        }


def supplemental_paths(root: Path) -> tuple[Path, ...]:
    """Return the five explicitly locked supplemental paths under ``root``."""

    return tuple(root / relative for relative in SUPPLEMENTAL_RELATIVE_PATHS)


def build_registry(
    manifest_path: Path,
    supplemental_db_paths: Sequence[Path],
) -> AnalysisRegistry:
    """Build registry records from one canonical manifest and five supplements.

    The function does not discover files.  Callers must pass the exact five
    supplemental paths, which makes the analysis universe reviewable in the
    generated JSON even if additional files later appear beside them.
    """

    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("canonical manifest must contain an artifacts list")
    if len(supplemental_db_paths) != len(SUPPLEMENTAL_RELATIVE_PATHS):
        raise ValueError(
            f"expected exactly {len(SUPPLEMENTAL_RELATIVE_PATHS)} supplemental paths, "
            f"got {len(supplemental_db_paths)}"
        )

    canonical_cells = _canonical_cells(manifest_path, manifest, artifacts)
    base_by_model = {
        cell.base_model_key: cell
        for cell in canonical_cells
        if cell.is_starting_market and cell.duplicate_of is None
    }
    supplemental_cells = _supplemental_cells(supplemental_db_paths, base_by_model)
    cells = tuple(canonical_cells) + tuple(supplemental_cells)
    _validate_relationships(cells)
    return AnalysisRegistry(
        manifest_path=manifest_path,
        canonical_dataset=str(manifest.get("dataset", "")),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        cells=cells,
    )


def _canonical_cells(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> list[CellSpec]:
    cells: list[CellSpec] = []
    base_path_by_model: dict[str, Path] = {}

    for artifact in artifacts:
        level = _required_int(artifact, "level")
        if level != 0:
            continue
        base_key = _required_str(artifact, "base_model_key")
        base_path_by_model[base_key] = _canonical_path(manifest_path, manifest, artifact)

    for artifact in artifacts:
        level = _required_int(artifact, "level")
        if level not in _REGIME_BY_LEVEL:
            raise ValueError(f"unsupported canonical level {level!r}")
        base_key = _required_str(artifact, "base_model_key")
        treatment_key = artifact.get("treatment_model_key")
        if treatment_key is not None and not isinstance(treatment_key, str):
            raise ValueError("treatment_model_key must be a string or null")
        db_path = _canonical_path(manifest_path, manifest, artifact)
        if level == 0:
            cell_id = f"base:{base_key}"
            start_tick, end_tick = 0, 360
        else:
            if treatment_key is None:
                raise ValueError(f"level-{level} artifact lacks treatment_model_key")
            cell_id = f"{base_key}:{_REGIME_BY_LEVEL[level]}:{treatment_key}"
            start_tick, end_tick = 360, 444
        cells.append(
            CellSpec(
                cell_id=cell_id,
                db_path=db_path,
                source=str(manifest.get("dataset", "canonical_manifest")),
                base_model_key=base_key,
                treatment_model_key=treatment_key,
                regime=_REGIME_BY_LEVEL[level],
                start_tick_exclusive=start_tick,
                end_tick_inclusive=end_tick,
                treated_agent_ids=() if level == 0 else TREATED_AGENT_IDS,
                paired_base_db=base_path_by_model.get(base_key) if level else None,
                pressure_side=_PRESSURE_BY_LEVEL.get(level),
                include_in_main_matrix=level in (1, 2, 3),
                is_starting_market=level == 0,
            )
        )
    return cells


def _supplemental_cells(
    paths: Sequence[Path],
    base_by_model: dict[str, CellSpec],
) -> list[CellSpec]:
    cells: list[CellSpec] = []
    seen_kinds: set[tuple[str, str]] = set()
    duplicate_seen = False

    for raw_path in paths:
        path = raw_path.resolve()
        name = path.name.lower()
        if "cold_start_deepseek-v4-pro" in name:
            if duplicate_seen:
                raise ValueError("supplemental DeepSeek cold-start supplied more than once")
            duplicate_seen = True
            base = _base_for(base_by_model, "deepseekv4pro")
            cells.append(
                CellSpec(
                    cell_id="provenance:new-deepseekv4pro-cold-start",
                    db_path=path,
                    source="BazaarBench/bazaarbench-rollouts",
                    base_model_key="deepseekv4pro",
                    treatment_model_key=None,
                    regime="L0",
                    start_tick_exclusive=0,
                    end_tick_inclusive=360,
                    treated_agent_ids=(),
                    is_starting_market=False,
                    duplicate_of=base.cell_id,
                )
            )
            continue

        if "l2x" not in name:
            raise ValueError(f"unrecognised supplemental input: {path}")
        if "deepseek" in name:
            base_key = treatment_key = "deepseekv4pro"
            start_tick, end_tick = 361, 445
        elif "gpt54mini" in name:
            base_key = treatment_key = "gpt54mini"
            start_tick, end_tick = 371, 455
        else:
            raise ValueError(f"cannot infer supplemental model from {path}")
        if "buyer" in name:
            pressure_side = "buyer"
        elif "seller" in name:
            pressure_side = "seller"
        else:
            raise ValueError(f"cannot infer supplemental pressure side from {path}")
        kind = (base_key, pressure_side)
        if kind in seen_kinds:
            raise ValueError(f"duplicate supplemental cell {kind}")
        seen_kinds.add(kind)
        base = _base_for(base_by_model, base_key)
        cells.append(
            CellSpec(
                cell_id=f"{base_key}:L2X:{pressure_side}",
                db_path=path,
                source="BazaarBench/bazaarbench-rollouts",
                base_model_key=base_key,
                treatment_model_key=treatment_key,
                regime="L2X",
                start_tick_exclusive=start_tick,
                end_tick_inclusive=end_tick,
                paired_base_db=base.db_path,
                pressure_side=pressure_side,
            )
        )

    expected = {
        ("deepseekv4pro", "buyer"),
        ("deepseekv4pro", "seller"),
        ("gpt54mini", "buyer"),
        ("gpt54mini", "seller"),
    }
    if not duplicate_seen or seen_kinds != expected:
        raise ValueError(
            "supplemental inputs must contain the DeepSeek cold-start duplicate "
            "and buyer/seller L2X cells for DeepSeek and GPT-5.4-mini"
        )
    return cells


def _base_for(base_by_model: dict[str, CellSpec], key: str) -> CellSpec:
    try:
        return base_by_model[key]
    except KeyError as exc:
        raise ValueError(f"canonical manifest has no starting market for {key}") from exc


def _canonical_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifact: dict[str, Any],
) -> Path:
    raw = Path(_required_str(artifact, "canonical_path"))
    if raw.is_absolute():
        return raw
    canonical_root = Path(str(manifest.get("canonical_root", "")))
    if canonical_root.parts and raw.parts[: len(canonical_root.parts)] == canonical_root.parts:
        return (manifest_path.parent.parent / raw).resolve()
    return (manifest_path.parent / raw).resolve()


def _validate_relationships(cells: Iterable[CellSpec]) -> None:
    cells = tuple(cells)
    by_id = {cell.cell_id: cell for cell in cells}
    if len(by_id) != len(cells):
        raise ValueError("registry cell_id values must be unique")
    for cell in cells:
        if cell.duplicate_of is not None and cell.duplicate_of not in by_id:
            raise ValueError(f"{cell.cell_id} duplicates unknown cell {cell.duplicate_of}")
        if cell.regime == "L0" and cell.treated_agent_ids:
            raise ValueError(f"L0 cell {cell.cell_id} must have an empty treated cohort")
        if not cell.is_starting_market and cell.duplicate_of is None:
            if cell.paired_base_db is None:
                raise ValueError(f"continuation {cell.cell_id} has no paired base")
            if cell.treated_agent_ids != TREATED_AGENT_IDS:
                raise ValueError(
                    f"continuation {cell.cell_id} must use the fixed 20-agent treated cohort"
                )
        if cell.include_in_main_matrix and cell.horizon_ticks != 84:
            raise ValueError(f"main cell {cell.cell_id} does not have an 84-tick window")


def _required_str(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return result


def _required_int(value: dict[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int):
        raise ValueError(f"manifest field {key!r} must be an integer")
    return result


def _cell_to_dict(cell: CellSpec) -> dict[str, Any]:
    result = asdict(cell)
    result["db_path"] = str(cell.db_path)
    result["paired_base_db"] = str(cell.paired_base_db) if cell.paired_base_db is not None else None
    result["treated_agent_ids"] = list(cell.treated_agent_ids)
    result["horizon_ticks"] = cell.horizon_ticks
    return result

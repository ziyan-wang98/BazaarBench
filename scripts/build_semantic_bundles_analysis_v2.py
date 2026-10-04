#!/usr/bin/env python3
"""Build the frozen semantic-bundle universe without invoking a judge.

The command is safe to use either sequentially or as a Slurm array.  A cell is
resumed only after its bundle file and completion ledger have both been bound
back to the exact registry cell, database file, and tick window.  The ledger is
written last and therefore acts as the cell-level commit marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.contract import CellSpec  # noqa: E402
from bazaar.analysis_v2.semantic_bundles import (  # noqa: E402
    build_semantic_bundles,
    ordered_digest,
)

ARTIFACT_SCHEMA_VERSION = 1


class ResumeValidationError(ValueError):
    """A purported complete cell artifact failed a binding check."""


@dataclass(frozen=True)
class ValidatedCell:
    cell_id: str
    safe_cell_name: str
    bundle_count: int
    bundles_bytes: int
    bundles_by_kind: dict[str, int]
    reasoning_calls: int
    reasoning_observed: int
    reasoning_missing: int
    reasoning_all_calls: int
    reasoning_all_observed: int
    reasoning_all_missing: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "safe_cell_name": self.safe_cell_name,
            "bundle_count": self.bundle_count,
            "bundles_bytes": self.bundles_bytes,
            "bundles_by_kind": self.bundles_by_kind,
            "reasoning": {
                "treated_calls": self.reasoning_calls,
                "treated_observed": self.reasoning_observed,
                "treated_missing": self.reasoning_missing,
                "all_calls": self.reasoning_all_calls,
                "all_observed": self.reasoning_all_observed,
                "all_missing": self.reasoning_all_missing,
            },
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _safe_cell_name(cell_id: str) -> str:
    """Mirror ``extract_analysis_v2.py`` exactly."""

    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in cell_id
    )


def _cell_from_dict(value: Mapping[str, Any]) -> CellSpec:
    return CellSpec(
        cell_id=str(value["cell_id"]),
        db_path=Path(str(value["db_path"])),
        source=str(value["source"]),
        base_model_key=str(value["base_model_key"]),
        treatment_model_key=value.get("treatment_model_key"),
        regime=str(value["regime"]),
        start_tick_exclusive=int(value["start_tick_exclusive"]),
        end_tick_inclusive=int(value["end_tick_inclusive"]),
        treated_agent_ids=tuple(int(item) for item in value["treated_agent_ids"]),
        paired_base_db=(
            Path(str(value["paired_base_db"]))
            if value.get("paired_base_db")
            else None
        ),
        pressure_side=value.get("pressure_side"),
        include_in_main_matrix=bool(value.get("include_in_main_matrix")),
        is_starting_market=bool(value.get("is_starting_market")),
        duplicate_of=value.get("duplicate_of"),
    )


def load_independent_cells(registry_path: Path) -> tuple[CellSpec, ...]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    physical = registry.get("cells")
    if not isinstance(physical, list):
        raise ValueError("registry must contain a cells list")
    cells = tuple(
        _cell_from_dict(value)
        for value in physical
        if value.get("duplicate_of") is None
    )
    declared = registry.get("independent_record_count")
    if declared is not None and int(declared) != len(cells):
        raise ValueError(
            f"registry independent count mismatch: declared={declared}, actual={len(cells)}"
        )
    identifiers = [cell.cell_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("independent registry contains duplicate cell_id values")
    safe_names = [_safe_cell_name(value) for value in identifiers]
    if len(safe_names) != len(set(safe_names)):
        raise ValueError("cell_id values collide after safe directory-name conversion")
    return cells


def select_cells(
    cells: Sequence[CellSpec],
    *,
    scope: str,
    requested_cell_ids: Sequence[str],
    array_index: int | None,
) -> tuple[CellSpec, ...]:
    selected = list(cells)
    if requested_cell_ids:
        wanted = set(requested_cell_ids)
        known = {cell.cell_id for cell in cells}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                "unknown independent cell(s): " + ", ".join(sorted(unknown))
            )
        selected = [cell for cell in selected if cell.cell_id in wanted]
    elif scope == "main":
        selected = [cell for cell in selected if cell.include_in_main_matrix]
    elif scope == "starting":
        selected = [cell for cell in selected if cell.is_starting_market]
    elif scope == "supplemental":
        selected = [cell for cell in selected if cell.regime == "L2X"]
    if array_index is not None:
        if array_index < 0 or array_index >= len(selected):
            raise ValueError(
                f"array index {array_index} outside selected range 0..{len(selected) - 1}"
            )
        return (selected[array_index],)
    return tuple(selected)


def _cell_binding(cell: CellSpec) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "db_path": str(cell.db_path.resolve()),
        "source": cell.source,
        "base_model_key": cell.base_model_key,
        "treatment_model_key": cell.treatment_model_key,
        "regime": cell.regime,
        "start_tick_exclusive": cell.start_tick_exclusive,
        "end_tick_inclusive": cell.end_tick_inclusive,
        "treated_agent_ids": list(cell.treated_agent_ids),
    }


def _database_binding(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_ndjson_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for value in values:
                output.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _bundle_rows(path: Path, *, expected_cell_id: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumeValidationError(
                    f"invalid bundle JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ResumeValidationError(
                    f"bundle line {line_number} is not an object"
                )
            if value.get("cell_id") != expected_cell_id:
                raise ResumeValidationError(
                    f"bundle line {line_number} cell mismatch: {value.get('cell_id')!r}"
                )
            yield value


def validate_completed_cell(
    cell: CellSpec,
    output_root: Path,
    *,
    registry_sha256: str | None = None,
) -> ValidatedCell:
    safe_name = _safe_cell_name(cell.cell_id)
    target = output_root / safe_name
    ledger_path = target / "ledger.json"
    bundles_path = target / "bundles.ndjson"
    if not ledger_path.is_file():
        raise ResumeValidationError("completion ledger is missing")
    if not bundles_path.is_file():
        raise ResumeValidationError("bundle file is missing")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeValidationError(f"invalid ledger JSON: {exc}") from exc
    if ledger.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ResumeValidationError("unsupported or missing artifact schema version")
    if ledger.get("status") != "complete":
        raise ResumeValidationError("ledger is not marked complete")
    if (
        registry_sha256 is not None
        and ledger.get("registry_sha256") != registry_sha256
    ):
        raise ResumeValidationError("frozen registry digest mismatch")
    expected_cell_binding = _cell_binding(cell)
    if ledger.get("cell_binding") != expected_cell_binding:
        raise ResumeValidationError("cell/database/window binding mismatch")
    if ledger.get("database_binding") != _database_binding(cell.db_path):
        raise ResumeValidationError("source database stat binding mismatch")
    if ledger.get("cell_id") != cell.cell_id:
        raise ResumeValidationError("semantic ledger cell_id mismatch")
    if str(Path(str(ledger.get("db_path", ""))).resolve()) != str(
        cell.db_path.resolve()
    ):
        raise ResumeValidationError("semantic ledger db_path mismatch")
    if ledger.get("start_tick_exclusive") != cell.start_tick_exclusive:
        raise ResumeValidationError("semantic ledger start tick mismatch")
    if ledger.get("end_tick_inclusive") != cell.end_tick_inclusive:
        raise ResumeValidationError("semantic ledger end tick mismatch")
    expected_count = ledger.get("bundle_count")
    if not isinstance(expected_count, int) or expected_count < 0:
        raise ResumeValidationError("invalid bundle_count")
    expected_bytes = ledger.get("bundles_bytes")
    if expected_bytes != bundles_path.stat().st_size:
        raise ResumeValidationError("bundle byte-size mismatch")
    expected_sha = ledger.get("bundles_sha256")
    if not isinstance(expected_sha, str) or _sha256(bundles_path) != expected_sha:
        raise ResumeValidationError("bundle file digest mismatch")
    count = 0

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal count
        for value in _bundle_rows(bundles_path, expected_cell_id=cell.cell_id):
            count += 1
            yield value

    digest = ordered_digest(rows())
    if count != expected_count:
        raise ResumeValidationError(
            f"bundle line-count mismatch: expected={expected_count}, actual={count}"
        )
    if digest != ledger.get("ordered_bundle_digest"):
        raise ResumeValidationError("ordered bundle digest mismatch")
    kinds = ledger.get("bundles_by_kind")
    if not isinstance(kinds, dict) or any(
        not isinstance(key, str) or not isinstance(value, int)
        for key, value in kinds.items()
    ):
        raise ResumeValidationError("invalid bundles_by_kind ledger")
    reasoning_fields = (
        "reasoning_calls",
        "reasoning_observed",
        "reasoning_missing",
        "reasoning_all_calls",
        "reasoning_all_observed",
        "reasoning_all_missing",
    )
    if any(not isinstance(ledger.get(key), int) for key in reasoning_fields):
        raise ResumeValidationError("invalid reasoning coverage ledger")
    if ledger["reasoning_observed"] + ledger["reasoning_missing"] != ledger[
        "reasoning_calls"
    ]:
        raise ResumeValidationError("treated reasoning coverage does not partition calls")
    if ledger["reasoning_all_observed"] + ledger["reasoning_all_missing"] != ledger[
        "reasoning_all_calls"
    ]:
        raise ResumeValidationError("all-actor reasoning coverage does not partition calls")
    return ValidatedCell(
        cell_id=cell.cell_id,
        safe_cell_name=safe_name,
        bundle_count=expected_count,
        bundles_bytes=expected_bytes,
        bundles_by_kind=dict(sorted(kinds.items())),
        reasoning_calls=ledger["reasoning_calls"],
        reasoning_observed=ledger["reasoning_observed"],
        reasoning_missing=ledger["reasoning_missing"],
        reasoning_all_calls=ledger["reasoning_all_calls"],
        reasoning_all_observed=ledger["reasoning_all_observed"],
        reasoning_all_missing=ledger["reasoning_all_missing"],
    )


def build_cell(
    cell: CellSpec,
    output_root: Path,
    *,
    registry_sha256: str,
    overwrite: bool,
) -> tuple[ValidatedCell, bool]:
    if not overwrite:
        try:
            return (
                validate_completed_cell(
                    cell,
                    output_root,
                    registry_sha256=registry_sha256,
                ),
                True,
            )
        except ResumeValidationError:
            pass
    before = _database_binding(cell.db_path)
    started = time.monotonic()
    result = build_semantic_bundles(
        cell.db_path,
        cell,
        audited_agent_ids=cell.treated_agent_ids,
    )
    after = _database_binding(cell.db_path)
    if before != after:
        raise RuntimeError("source database changed during bundle extraction")
    target = output_root / _safe_cell_name(cell.cell_id)
    bundles_path = target / "bundles.ndjson"
    ledger_path = target / "ledger.json"
    written = _write_ndjson_atomic(
        bundles_path, (bundle.to_dict() for bundle in result.bundles)
    )
    if written != len(result.bundles):
        raise RuntimeError(
            f"bundle write count mismatch: expected={len(result.bundles)}, actual={written}"
        )
    ledger = result.ledger.to_dict()
    if ledger["ordered_bundle_digest"] != ordered_digest(
        bundle.to_dict() for bundle in result.bundles
    ):
        raise RuntimeError("in-memory bundle digest mismatch")
    ledger.update(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "complete",
            "cell_binding": _cell_binding(cell),
            "database_binding": before,
            "registry_sha256": registry_sha256,
            "bundle_count": len(result.bundles),
            "bundles_file": "bundles.ndjson",
            "bundles_bytes": bundles_path.stat().st_size,
            "bundles_sha256": _sha256(bundles_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    )
    # Written last: this is the only completion marker trusted by resume.
    _write_json_atomic(ledger_path, ledger)
    return (
        validate_completed_cell(
            cell,
            output_root,
            registry_sha256=registry_sha256,
        ),
        False,
    )


def _status_path(status_dir: Path, cell: CellSpec) -> Path:
    return status_dir / f"{_safe_cell_name(cell.cell_id)}.json"


def write_status(
    status_dir: Path,
    cell: CellSpec,
    *,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    _write_json_atomic(
        _status_path(status_dir, cell),
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "cell_binding": _cell_binding(cell),
            "status": status,
            **payload,
        },
    )


def build_global_manifest(
    cells: Sequence[CellSpec],
    *,
    registry_path: Path,
    registry_sha256: str,
    output_root: Path,
    manifest_path: Path,
    require_complete: bool,
) -> dict[str, Any]:
    complete: list[ValidatedCell] = []
    incomplete: list[dict[str, str]] = []
    for cell in cells:
        try:
            complete.append(
                validate_completed_cell(
                    cell,
                    output_root,
                    registry_sha256=registry_sha256,
                )
            )
        except (OSError, ValueError) as exc:
            incomplete.append(
                {
                    "cell_id": cell.cell_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    kind_totals: Counter[str] = Counter()
    reasoning_totals: Counter[str] = Counter()
    for item in complete:
        kind_totals.update(item.bundles_by_kind)
        reasoning_totals.update(
            {
                "treated_calls": item.reasoning_calls,
                "treated_observed": item.reasoning_observed,
                "treated_missing": item.reasoning_missing,
                "all_calls": item.reasoning_all_calls,
                "all_observed": item.reasoning_all_observed,
                "all_missing": item.reasoning_all_missing,
            }
        )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete" if not incomplete else "incomplete",
        "registry": str(registry_path.resolve()),
        "registry_sha256": registry_sha256,
        "output_root": str(output_root.resolve()),
        "expected_independent_cells": len(cells),
        "complete_cells": len(complete),
        "incomplete_cells": incomplete,
        "totals": {
            "bundle_count": sum(item.bundle_count for item in complete),
            "bundles_bytes": sum(item.bundles_bytes for item in complete),
            "bundles_by_kind": dict(sorted(kind_totals.items())),
            "reasoning": dict(sorted(reasoning_totals.items())),
        },
        "cells": [item.to_manifest_dict() for item in complete],
    }
    _write_json_atomic(manifest_path, manifest)
    if require_complete and incomplete:
        raise RuntimeError(
            f"semantic bundle universe incomplete: {len(incomplete)}/{len(cells)} cells"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument(
        "--scope",
        choices=("all", "main", "starting", "supplemental"),
        default="all",
    )
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.array_index is not None and args.cell:
        parser.error("--array-index and --cell are mutually exclusive")
    if args.finalize_only and (
        args.array_index is not None or args.cell or args.scope != "all" or args.overwrite
    ):
        parser.error("--finalize-only always validates the full independent registry")
    if not args.registry.is_file():
        parser.error(f"registry not found: {args.registry}")
    registry_sha256 = _sha256(args.registry)
    try:
        all_cells = load_independent_cells(args.registry)
        selected = select_cells(
            all_cells,
            scope=args.scope,
            requested_cell_ids=args.cell,
            array_index=args.array_index,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.finalize_only:
        try:
            manifest = build_global_manifest(
                all_cells,
                registry_path=args.registry,
                registry_sha256=registry_sha256,
                output_root=args.out_root,
                manifest_path=args.manifest_out,
                require_complete=args.require_complete,
            )
        except RuntimeError as exc:
            print(f"[analysis-v2] ERROR {exc}", file=sys.stderr, flush=True)
            return 2
        print(
            f"[analysis-v2] semantic manifest cells={manifest['complete_cells']}/"
            f"{manifest['expected_independent_cells']} "
            f"bundles={manifest['totals']['bundle_count']} "
            f"bytes={manifest['totals']['bundles_bytes']}",
            flush=True,
        )
        return 0

    failures = 0
    for index, cell in enumerate(selected, start=1):
        print(
            f"[analysis-v2] semantic bundles {index}/{len(selected)} {cell.cell_id}",
            flush=True,
        )
        try:
            item, resumed = build_cell(
                cell,
                args.out_root,
                registry_sha256=registry_sha256,
                overwrite=args.overwrite,
            )
            write_status(
                args.status_dir,
                cell,
                status="complete",
                payload={
                    "resumed": resumed,
                    "bundle_count": item.bundle_count,
                    "bundles_bytes": item.bundles_bytes,
                    "bundles_by_kind": item.bundles_by_kind,
                },
            )
            print(
                f"[analysis-v2] {'resume' if resumed else 'complete'} {cell.cell_id}: "
                f"bundles={item.bundle_count} bytes={item.bundles_bytes}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - retain cell-level failure status
            failures += 1
            write_status(
                args.status_dir,
                cell,
                status="error",
                payload={"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(
                f"[analysis-v2] ERROR {cell.cell_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                break

    # A sequential run can safely publish a partial/complete snapshot.  Array tasks do
    # not race on the global manifest; use --finalize-only after the array dependency.
    if args.array_index is None:
        build_global_manifest(
            all_cells,
            registry_path=args.registry,
            registry_sha256=registry_sha256,
            output_root=args.out_root,
            manifest_path=args.manifest_out,
            require_complete=args.require_complete,
        )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

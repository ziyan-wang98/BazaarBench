#!/usr/bin/env python3
"""Judge the frozen, prebuilt semantic bundles for all analysis-v2 cells.

This command never opens a rollout database and never rebuilds semantic bundles.  Each
cell is bound to the completed bundle ledger, deterministic shard plan, one frozen model,
and the exhaustive sparse-v2 transport before the first request is sent. Scheduler tasks
own disjoint shard chunks and separate append-only, fsynced journals. Per-bundle records
and completion ledgers are published atomically only after every frozen chunk, bundle,
and target channel validates.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_semantic_bundles_analysis_v2 import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION as BUNDLE_ARTIFACT_SCHEMA_VERSION,
)
from build_semantic_bundles_analysis_v2 import (  # noqa: E402
    load_independent_cells,
    select_cells,
)

from bazaar.analysis_v2.claude_cli import (  # noqa: E402
    DEFAULT_CLAUDE_BINARY,
    ClaudeCliBackend,
)
from bazaar.analysis_v2.contract import CellSpec  # noqa: E402
from bazaar.analysis_v2.judge_runner import (  # noqa: E402
    BundleShard,
    JudgeRunRecord,
    JudgeShardRunRecord,
    SemanticJudgeRunner,
    judge_record_from_dict,
    make_deterministic_shards,
    recompose_shard_records,
    shard_record_from_dict,
)
from bazaar.analysis_v2.semantic_bundles import (  # noqa: E402
    SemanticBundle,
    bundle_from_dict,
    canonical_json,
)

JUDGE_ARTIFACT_SCHEMA_VERSION = 3
CHUNK_ARTIFACT_SCHEMA_VERSION = 2
TRANSPORT = "claude_cli_exhaustive_sparse_v2"
SHARDING_ALGORITHM = "exact_greedy_binary_v1"
CHUNKING_ALGORITHM = "contiguous_shard_ranges_v1"
CHUNK_EXECUTION_MODE = "disjoint_atomic_chunks_v1"
PROVIDER = "claude_cli"


class JudgmentValidationError(ValueError):
    """A formal judgment artifact failed an immutable binding or coverage check."""


def _safe_cell_name(cell_id: str) -> str:
    """Mirror the frozen extraction directory-name convention."""

    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in cell_id
    )


def _cell_binding(cell: CellSpec) -> dict[str, Any]:
    """Bind a judgment plan to the exact independent registry cell."""

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


@dataclass(frozen=True)
class JudgeCaps:
    max_input_bytes: int
    max_estimated_input_tokens: int
    max_bundles: int
    max_decision_units: int

    def to_dict(self) -> dict[str, int]:
        return {
            "max_input_bytes": self.max_input_bytes,
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
            "max_bundles": self.max_bundles,
            "max_decision_units": self.max_decision_units,
        }


@dataclass(frozen=True)
class ValidatedJudgmentCell:
    cell_id: str
    safe_cell_name: str
    bundle_count: int
    decision_count: int
    shard_count: int
    usage: dict[str, int]
    elapsed_seconds: float
    model: str
    reasoning_effort: str
    transport: str

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "safe_cell_name": self.safe_cell_name,
            "status": "complete",
            "bundle_count": self.bundle_count,
            "decision_count": self.decision_count,
            "shard_count": self.shard_count,
            "usage": self.usage,
            "elapsed_seconds": self.elapsed_seconds,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "transport": self.transport,
        }


@dataclass(frozen=True)
class ValidatedJudgmentChunk:
    chunk_id: str
    global_chunk_index: int
    cell_id: str
    shard_count: int
    bundle_count: int
    decision_count: int
    usage: dict[str, int]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "global_chunk_index": self.global_chunk_index,
            "cell_id": self.cell_id,
            "shard_count": self.shard_count,
            "bundle_count": self.bundle_count,
            "decision_count": self.decision_count,
            "usage": self.usage,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def _atomic_ndjson(path: Path, values: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgmentValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgmentValidationError(f"{path} must contain one JSON object")
    return value


def _read_bundles(
    cell: CellSpec,
    *,
    bundles_root: Path,
) -> tuple[list[SemanticBundle], dict[str, Any]]:
    target = bundles_root / _safe_cell_name(cell.cell_id)
    ledger_path = target / "ledger.json"
    bundles_path = target / "bundles.ndjson"
    if not ledger_path.is_file() or not bundles_path.is_file():
        raise JudgmentValidationError(
            f"completed bundle files are missing for {cell.cell_id}"
        )
    ledger = _read_object(ledger_path)
    if (
        ledger.get("artifact_schema_version") != BUNDLE_ARTIFACT_SCHEMA_VERSION
        or ledger.get("status") != "complete"
        or ledger.get("cell_id") != cell.cell_id
        or ledger.get("cell_binding") != _cell_binding(cell)
    ):
        raise JudgmentValidationError(
            f"bundle ledger fields do not match cell {cell.cell_id}"
        )
    expected_count = ledger.get("bundle_count")
    if not isinstance(expected_count, int) or expected_count < 0:
        raise JudgmentValidationError(f"invalid bundle count for {cell.cell_id}")
    bundles: list[SemanticBundle] = []
    try:
        with bundles_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("bundle row is not an object")
                    bundle = bundle_from_dict(raw, verify_digest=False)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise JudgmentValidationError(
                        f"invalid bundle {cell.cell_id} line {line_number}: {exc}"
                    ) from exc
                if bundle.cell_id != cell.cell_id:
                    raise JudgmentValidationError(
                        f"bundle {line_number} cell binding differs from {cell.cell_id}"
                    )
                bundles.append(bundle)
    except OSError as exc:
        raise JudgmentValidationError(
            f"cannot read bundles for {cell.cell_id}: {exc}"
        ) from exc
    if len(bundles) != expected_count:
        raise JudgmentValidationError("deserialized bundle count differs from ledger")
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise JudgmentValidationError("bundle IDs are not unique within the cell")
    return bundles, ledger


def _make_shards(bundles: Sequence[SemanticBundle], caps: JudgeCaps) -> tuple[BundleShard, ...]:
    shards = make_deterministic_shards(
        bundles,
        max_input_bytes=caps.max_input_bytes,
        max_estimated_input_tokens=caps.max_estimated_input_tokens,
        max_bundles=caps.max_bundles,
        max_decision_units=caps.max_decision_units,
    )
    return shards


def _build_plan(
    cell: CellSpec,
    *,
    bundle_ledger: Mapping[str, Any],
    bundles: Sequence[SemanticBundle],
    shards: Sequence[BundleShard],
    caps: JudgeCaps,
    model: str,
    reasoning_effort: str,
    executable: Path,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    shard_manifests = [shard.manifest_dict() for shard in shards]
    return {
        "artifact_schema_version": JUDGE_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "formal_exhaustive_semantic_judgment",
        "formal_run": True,
        "plan_id": f"plan:{_safe_cell_name(cell.cell_id)}",
        "cell_binding": _cell_binding(cell),
        "bundle_artifact": {
            "artifact_schema_version": bundle_ledger.get("artifact_schema_version"),
            "bundle_count": len(bundles),
            "bundle_ids": [bundle.bundle_id for bundle in bundles],
        },
        "judge": {
            "provider": PROVIDER,
            "transport": TRANSPORT,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "executable": str(executable.resolve()),
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "max_output_tokens": max_output_tokens,
            "concurrency": concurrency,
        },
        "sharding": {
            "algorithm": SHARDING_ALGORITHM,
            "caps": caps.to_dict(),
            "shard_count": len(shards),
            "bundle_count": len(bundles),
            "decision_count": sum(len(bundle.target_channels) for bundle in bundles),
            "shards": shard_manifests,
        },
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("artifact_schema_version") != JUDGE_ARTIFACT_SCHEMA_VERSION
        or plan.get("artifact_kind") != "formal_exhaustive_semantic_judgment"
        or plan.get("formal_run") is not True
    ):
        raise JudgmentValidationError("old, calibration, or non-formal plan is not reusable")
    if not isinstance(plan.get("plan_id"), str):
        raise JudgmentValidationError("formal plan is missing its natural plan ID")


def _chunk_entry(
    *,
    global_chunk_index: int,
    cell: CellSpec,
    shards: Sequence[BundleShard],
    start_shard_ordinal: int,
    end_shard_ordinal_exclusive: int,
) -> dict[str, Any]:
    selected = shards[start_shard_ordinal:end_shard_ordinal_exclusive]
    return {
        "artifact_schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
        "chunk_id": (
            f"chunk:{global_chunk_index:06d}:{_safe_cell_name(cell.cell_id)}:"
            f"{start_shard_ordinal:06d}-{end_shard_ordinal_exclusive:06d}"
        ),
        "global_chunk_index": global_chunk_index,
        "cell_id": cell.cell_id,
        "safe_cell_name": _safe_cell_name(cell.cell_id),
        "start_shard_ordinal": start_shard_ordinal,
        "end_shard_ordinal_exclusive": end_shard_ordinal_exclusive,
        "shard_count": len(selected),
        "bundle_count": sum(len(shard.bundles) for shard in selected),
        "decision_count": sum(shard.decision_units for shard in selected),
        "shard_ids": [shard.shard_id for shard in selected],
    }


def _validate_chunk_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("artifact_schema_version") != CHUNK_ARTIFACT_SCHEMA_VERSION
        or plan.get("artifact_kind") != "formal_exhaustive_semantic_judgment_chunk_plan"
        or plan.get("formal_run") is not True
        or plan.get("execution_mode") != CHUNK_EXECUTION_MODE
        or plan.get("chunking", {}).get("algorithm") != CHUNKING_ALGORITHM
    ):
        raise JudgmentValidationError("old, calibration, or non-chunk plan is not reusable")
    if not isinstance(plan.get("plan_id"), str):
        raise JudgmentValidationError("global chunk plan is missing its natural plan ID")
    chunks = plan.get("chunks")
    cells = plan.get("cells")
    if not isinstance(chunks, list) or not isinstance(cells, list):
        raise JudgmentValidationError("global chunk plan has invalid cell/chunk tables")
    if plan.get("chunk_count") != len(chunks) or plan.get("cell_count") != len(cells):
        raise JudgmentValidationError("global chunk plan count mismatch")
    indexes: list[int] = []
    chunk_ids: list[str] = []
    try:
        max_shards_per_chunk = int(plan["chunking"]["max_shards_per_chunk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgmentValidationError("global chunk plan has invalid chunk cap") from exc
    if max_shards_per_chunk < 1:
        raise JudgmentValidationError("global chunk cap must be positive")
    for entry in chunks:
        if not isinstance(entry, dict):
            raise JudgmentValidationError("global chunk plan entry is not an object")
        indexes.append(int(entry.get("global_chunk_index", -1)))
        chunk_ids.append(str(entry.get("chunk_id", "")))
        start = int(entry.get("start_shard_ordinal", -1))
        end = int(entry.get("end_shard_ordinal_exclusive", -1))
        count = int(entry.get("shard_count", -1))
        if (
            count != end - start
            or count < 1
            or count > max_shards_per_chunk
            or len(entry.get("shard_ids", [])) != count
        ):
            raise JudgmentValidationError("global chunk entry violates its frozen shard cap")
    if indexes != list(range(len(chunks))):
        raise JudgmentValidationError("global chunk indexes are not exact and contiguous")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise JudgmentValidationError("global chunk IDs are not unique")
    cell_ids: list[str] = []
    claimed_indexes: list[int] = []
    for row in cells:
        if not isinstance(row, dict):
            raise JudgmentValidationError("global chunk cell row is not an object")
        cell_id = str(row.get("cell_id", ""))
        cell_ids.append(cell_id)
        row_indexes = [int(value) for value in row.get("chunk_indexes", [])]
        if row_indexes != sorted(row_indexes):
            raise JudgmentValidationError("cell chunk indexes are not ordered")
        claimed_indexes.extend(row_indexes)
        for index in row_indexes:
            try:
                entry = chunks[index]
            except IndexError as exc:
                raise JudgmentValidationError("cell claims an unknown chunk index") from exc
            if (
                entry.get("cell_id") != cell_id
                or entry.get("safe_cell_name") != row.get("safe_cell_name")
            ):
                raise JudgmentValidationError("cell-to-chunk immutable binding mismatch")
    if len(cell_ids) != len(set(cell_ids)):
        raise JudgmentValidationError("global chunk plan cell IDs are not unique")
    if claimed_indexes != list(range(len(chunks))):
        raise JudgmentValidationError("cells do not claim every chunk exactly once")


def _chunk_directory(cell_target: Path, entry: Mapping[str, Any]) -> Path:
    return cell_target / "chunks" / f"chunk-{int(entry['global_chunk_index']):06d}"


@contextmanager
def _exclusive_chunk_lock(path: Path) -> Iterator[None]:
    """Prevent duplicate scheduler instances from ever writing one chunk journal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JudgmentValidationError(
                f"chunk is already claimed by another live process: {path.parent.name}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cell_runtime_plan(
    cell: CellSpec,
    *,
    bundles_root: Path,
    caps: JudgeCaps,
    model: str,
    reasoning_effort: str,
    executable: Path,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    concurrency: int,
) -> tuple[list[SemanticBundle], tuple[BundleShard, ...], dict[str, Any]]:
    bundles, bundle_ledger = _read_bundles(
        cell,
        bundles_root=bundles_root,
    )
    shards = _make_shards(bundles, caps)
    plan = _build_plan(
        cell,
        bundle_ledger=bundle_ledger,
        bundles=bundles,
        shards=shards,
        caps=caps,
        model=model,
        reasoning_effort=reasoning_effort,
        executable=executable,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        max_output_tokens=max_output_tokens,
        concurrency=concurrency,
    )
    return bundles, shards, plan


def prepare_chunk_plan(
    cells: Sequence[CellSpec],
    *,
    registry_path: Path,
    bundles_root: Path,
    output_root: Path,
    chunk_plan_path: Path,
    caps: JudgeCaps,
    max_shards_per_chunk: int,
    model: str,
    reasoning_effort: str,
    executable: Path,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    concurrency: int,
    resume: bool,
) -> dict[str, Any]:
    """Freeze all cell shards and their disjoint scheduler-owned chunk ranges."""

    if max_shards_per_chunk < 1:
        raise JudgmentValidationError("max shards per chunk must be positive")
    prior_global = chunk_plan_path.exists()
    cell_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    global_chunk_index = 0
    for cell in cells:
        bundles, shards, plan = _cell_runtime_plan(
            cell,
            bundles_root=bundles_root,
            caps=caps,
            model=model,
            reasoning_effort=reasoning_effort,
            executable=executable,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            max_output_tokens=max_output_tokens,
            concurrency=concurrency,
        )
        target = output_root / _safe_cell_name(cell.cell_id)
        plan_path = target / "plan.json"
        root_outputs = (target / "shards.ndjson", target / "records.ndjson", target / "ledger.json")
        if plan_path.exists():
            stored = _read_object(plan_path)
            _validate_plan(stored)
            if stored != plan:
                raise JudgmentValidationError(
                    f"chunk-plan preparation refused drift for cell {cell.cell_id}"
                )
        else:
            if any(path.exists() for path in root_outputs) or (target / "chunks").exists():
                raise JudgmentValidationError(
                    f"cell output exists without its frozen plan: {cell.cell_id}"
                )
            _atomic_json(plan_path, plan)
        if not prior_global and any(path.exists() for path in root_outputs):
            raise JudgmentValidationError(
                f"legacy/shared cell output exists before chunk plan: {cell.cell_id}"
            )

        chunk_indexes: list[int] = []
        for start in range(0, len(shards), max_shards_per_chunk):
            end = min(start + max_shards_per_chunk, len(shards))
            entry = _chunk_entry(
                global_chunk_index=global_chunk_index,
                cell=cell,
                shards=shards,
                start_shard_ordinal=start,
                end_shard_ordinal_exclusive=end,
            )
            chunk_rows.append(entry)
            chunk_indexes.append(global_chunk_index)
            global_chunk_index += 1
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "safe_cell_name": _safe_cell_name(cell.cell_id),
                "bundle_count": len(bundles),
                "decision_count": sum(
                    len(bundle.target_channels) for bundle in bundles
                ),
                "shard_count": len(shards),
                "chunk_indexes": chunk_indexes,
            }
        )

    plan = {
        "artifact_schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "formal_exhaustive_semantic_judgment_chunk_plan",
        "formal_run": True,
        "execution_mode": CHUNK_EXECUTION_MODE,
        "plan_id": "chunk-plan:analysis-v2",
        "registry_path": str(registry_path.resolve()),
        "bundles_root": str(bundles_root.resolve()),
        "output_root": str(output_root.resolve()),
        "cell_count": len(cells),
        "chunk_count": len(chunk_rows),
        "judge": {
            "provider": PROVIDER,
            "transport": TRANSPORT,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "executable": str(executable.resolve()),
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "max_output_tokens": max_output_tokens,
            "concurrency": concurrency,
        },
        "sharding": {"algorithm": SHARDING_ALGORITHM, "caps": caps.to_dict()},
        "chunking": {
            "algorithm": CHUNKING_ALGORITHM,
            "max_shards_per_chunk": max_shards_per_chunk,
        },
        "cells": cell_rows,
        "chunks": chunk_rows,
    }
    _validate_chunk_plan(plan)
    if prior_global:
        prior = _read_object(chunk_plan_path)
        _validate_chunk_plan(prior)
        if prior != plan:
            raise JudgmentValidationError(
                "resume refused: registry, bundles, model, caps, executable, or chunks drifted"
            )
        if not resume:
            raise JudgmentValidationError("existing global chunk plan requires --resume")
    else:
        _atomic_json(chunk_plan_path, plan)
    return plan


def _latest_shard_records(
    journal_path: Path,
    shards: Sequence[BundleShard],
    *,
    model: str,
    reasoning_effort: str,
    transport: str,
) -> list[JudgeShardRunRecord]:
    records = _read_shard_journal(journal_path)
    expected = {shard.shard_id: shard for shard in shards}
    latest: dict[str, JudgeShardRunRecord] = {}
    for record in records:
        shard = expected.get(record.shard_id)
        if shard is None:
            raise JudgmentValidationError(
                f"journal contains a shard outside the frozen plan: {record.shard_id}"
            )
        if (
            record.model != model
            or record.reasoning_effort != reasoning_effort
            or record.transport != transport
        ):
            raise JudgmentValidationError(f"journal judge binding mismatch for {record.shard_id}")
        try:
            # The public recomposition boundary revalidates the complete immutable
            # shard binding and every returned decision against its source bundle.
            recompose_shard_records((shard,), (record,))
        except (TypeError, ValueError) as exc:
            raise JudgmentValidationError(
                f"invalid journal record for {record.shard_id}: {exc}"
            ) from exc
        latest[record.shard_id] = record
    missing = [shard.shard_id for shard in shards if shard.shard_id not in latest]
    if missing:
        raise JudgmentValidationError(f"journal is missing {len(missing)} shard(s)")
    ordered = [latest[shard.shard_id] for shard in shards]
    failures = [record for record in ordered if record.status != "ok"]
    if failures:
        counts = Counter(record.status for record in failures)
        raise JudgmentValidationError(
            "latest shard records are incomplete: " + canonical_json(dict(counts))
        )
    return ordered


def _read_shard_journal(path: Path) -> list[JudgeShardRunRecord]:
    """Read a post-run journal strictly; resume repairs a torn tail beforehand."""

    try:
        data = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgmentValidationError(f"cannot read shard journal {path}: {exc}") from exc
    records: list[JudgeShardRunRecord] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError("journal row is not an object")
            records.append(shard_record_from_dict(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JudgmentValidationError(
                f"invalid shard journal line {line_number}: {exc}"
            ) from exc
    return records


def _validate_chunk_entry_against_shards(
    entry: Mapping[str, Any],
    *,
    cell: CellSpec,
    shards: Sequence[BundleShard],
) -> tuple[BundleShard, ...]:
    start = int(entry.get("start_shard_ordinal", -1))
    end = int(entry.get("end_shard_ordinal_exclusive", -1))
    if start < 0 or end <= start or end > len(shards):
        raise JudgmentValidationError(f"invalid shard range for {entry.get('chunk_id')}")
    expected = _chunk_entry(
        global_chunk_index=int(entry.get("global_chunk_index", -1)),
        cell=cell,
        shards=shards,
        start_shard_ordinal=start,
        end_shard_ordinal_exclusive=end,
    )
    if dict(entry) != expected:
        raise JudgmentValidationError(
            f"chunk entry no longer matches frozen cell shards: {entry.get('chunk_id')}"
        )
    return tuple(shards[start:end])


def _shard_records_from_file(
    path: Path,
    shards: Sequence[BundleShard],
    *,
    model: str,
    reasoning_effort: str,
) -> list[JudgeShardRunRecord]:
    try:
        data = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgmentValidationError(f"cannot read chunk shard records {path}: {exc}") from exc
    records: list[JudgeShardRunRecord] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError("chunk shard record is not an object")
            records.append(shard_record_from_dict(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JudgmentValidationError(
                f"invalid chunk shard record line {line_number}: {exc}"
            ) from exc
    if len(records) != len(shards):
        raise JudgmentValidationError("chunk does not contain exactly one record per shard")
    if len({record.shard_id for record in records}) != len(records):
        raise JudgmentValidationError("chunk shard records contain duplicate shard IDs")
    for shard, record in zip(shards, records, strict=True):
        if (
            record.shard_id != shard.shard_id
            or record.model != model
            or record.reasoning_effort != reasoning_effort
            or record.transport != TRANSPORT
            or record.status != "ok"
        ):
            raise JudgmentValidationError(
                f"chunk record binding/status mismatch for {shard.shard_id}"
            )
        try:
            recompose_shard_records((shard,), (record,))
        except (TypeError, ValueError) as exc:
            raise JudgmentValidationError(
                f"invalid completed chunk record for {shard.shard_id}: {exc}"
            ) from exc
    return records


def _chunk_completion_ledger(
    *,
    entry: Mapping[str, Any],
    journal_path: Path,
    records_path: Path,
    latest_records: Sequence[JudgeShardRunRecord],
) -> dict[str, Any]:
    journal_records = _read_shard_journal(journal_path)
    usage = {
        key: sum(int(record.usage.get(key, 0)) for record in journal_records)
        for key in ("input_tokens", "output_tokens")
    }
    return {
        "artifact_schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "formal_exhaustive_semantic_judgment_chunk",
        "status": "complete",
        "execution_mode": CHUNK_EXECUTION_MODE,
        "chunk_id": entry["chunk_id"],
        "global_chunk_index": entry["global_chunk_index"],
        "cell_id": entry["cell_id"],
        "start_shard_ordinal": entry["start_shard_ordinal"],
        "end_shard_ordinal_exclusive": entry["end_shard_ordinal_exclusive"],
        "shard_count": len(latest_records),
        "bundle_count": entry["bundle_count"],
        "decision_count": entry["decision_count"],
        "shard_statuses": dict(
            sorted(Counter(record.status for record in latest_records).items())
        ),
        "journal_record_count": len(journal_records),
        "journal_statuses": dict(
            sorted(Counter(record.status for record in journal_records).items())
        ),
        "usage": usage,
        "elapsed_seconds": round(sum(record.elapsed_s for record in journal_records), 3),
        "journal_file": journal_path.name,
        "shard_records_file": records_path.name,
    }


def _validate_completed_chunk(
    *,
    cell_plan: Mapping[str, Any],
    entry: Mapping[str, Any],
    shards: Sequence[BundleShard],
    chunk_directory: Path,
) -> tuple[ValidatedJudgmentChunk, list[JudgeShardRunRecord], list[JudgeShardRunRecord]]:
    ledger = _read_object(chunk_directory / "ledger.json")
    journal_path = chunk_directory / str(ledger.get("journal_file", ""))
    records_path = chunk_directory / str(ledger.get("shard_records_file", ""))
    model = str(cell_plan.get("judge", {}).get("model", ""))
    effort = str(cell_plan.get("judge", {}).get("reasoning_effort", ""))
    latest = _shard_records_from_file(
        records_path,
        shards,
        model=model,
        reasoning_effort=effort,
    )
    journal = _read_shard_journal(journal_path)
    journal_latest = _latest_shard_records(
        journal_path,
        shards,
        model=model,
        reasoning_effort=effort,
        transport=TRANSPORT,
    )
    if [record.to_dict() for record in latest] != [
        record.to_dict() for record in journal_latest
    ]:
        raise JudgmentValidationError("chunk records differ from the latest fsynced journal")
    expected = _chunk_completion_ledger(
        entry=entry,
        journal_path=journal_path,
        records_path=records_path,
        latest_records=latest,
    )
    if ledger != expected:
        raise JudgmentValidationError("chunk completion ledger differs from recomputed evidence")
    return (
        ValidatedJudgmentChunk(
            chunk_id=str(entry["chunk_id"]),
            global_chunk_index=int(entry["global_chunk_index"]),
            cell_id=str(entry["cell_id"]),
            shard_count=len(latest),
            bundle_count=int(entry["bundle_count"]),
            decision_count=int(entry["decision_count"]),
            usage=dict(ledger["usage"]),
            elapsed_seconds=float(ledger["elapsed_seconds"]),
        ),
        latest,
        journal,
    )


def _validate_run_records(
    bundles: Sequence[SemanticBundle], records: Sequence[JudgeRunRecord]
) -> None:
    if len(records) != len(bundles):
        raise JudgmentValidationError("judge records do not cover every bundle")
    for index, (bundle, record) in enumerate(zip(bundles, records, strict=True)):
        if record.status != "ok" or record.decision is None:
            raise JudgmentValidationError(f"bundle index {index} lacks one ok decision")
        if record.bundle_id != bundle.bundle_id:
            raise JudgmentValidationError(f"bundle index {index} binding mismatch")
        channels = tuple(item.channel for item in record.decision.decisions)
        if channels != bundle.target_channels:
            raise JudgmentValidationError(
                f"bundle index {index} does not cover each target channel exactly once"
            )


def _canonical_record_sequence(records: Sequence[JudgeRunRecord]) -> str:
    """Canonical, order-sensitive representation of a per-bundle record sequence."""

    return canonical_json([record.to_dict() for record in records])


def _records_from_file(path: Path, bundles: Sequence[SemanticBundle]) -> list[JudgeRunRecord]:
    records: list[JudgeRunRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= len(bundles):
                raise JudgmentValidationError("records file has extra rows")
            try:
                raw = json.loads(line)
                record = judge_record_from_dict(raw, bundle=bundles[index])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise JudgmentValidationError(
                    f"invalid judge record line {index + 1}: {exc}"
                ) from exc
            records.append(record)
    _validate_run_records(bundles, records)
    return records


def _completion_ledger(
    *,
    plan: Mapping[str, Any],
    records_path: Path,
    journal_path: Path,
    shard_records: Sequence[JudgeShardRunRecord],
    records: Sequence[JudgeRunRecord],
    chunk_count: int | None = None,
) -> dict[str, Any]:
    journal_records = _read_shard_journal(journal_path)
    usage = {
        key: sum(int(record.usage.get(key, 0)) for record in journal_records)
        for key in ("input_tokens", "output_tokens")
    }
    statuses = dict(sorted(Counter(record.status for record in shard_records).items()))
    journal_statuses = dict(sorted(Counter(record.status for record in journal_records).items()))
    ledger = {
        "artifact_schema_version": JUDGE_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "formal_exhaustive_semantic_judgment",
        "status": "complete",
        "plan_id": plan["plan_id"],
        "model": plan["judge"]["model"],
        "reasoning_effort": plan["judge"]["reasoning_effort"],
        "transport": plan["judge"]["transport"],
        "bundle_count": len(records),
        "decision_count": sum(
            len(record.decision.decisions)  # type: ignore[union-attr]
            for record in records
        ),
        "shard_count": len(shard_records),
        "shard_statuses": statuses,
        "journal_record_count": len(journal_records),
        "journal_statuses": journal_statuses,
        "usage": usage,
        "elapsed_seconds": round(sum(record.elapsed_s for record in journal_records), 3),
        "records_file": records_path.name,
        "journal_file": journal_path.name,
    }
    if chunk_count is not None:
        ledger.update(
            {
                "execution_mode": CHUNK_EXECUTION_MODE,
                "chunk_count": chunk_count,
            }
        )
    return ledger


def _write_status(
    status_dir: Path,
    cell: CellSpec,
    *,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    _atomic_json(
        status_dir / f"{_safe_cell_name(cell.cell_id)}.json",
        {
            "artifact_schema_version": JUDGE_ARTIFACT_SCHEMA_VERSION,
            "cell_binding": _cell_binding(cell),
            "status": status,
            **payload,
        },
    )


def run_chunk(
    cells: Sequence[CellSpec],
    *,
    chunk_plan_path: Path,
    chunk_index: int,
    registry_path: Path,
    bundles_root: Path,
    output_root: Path,
    caps: JudgeCaps,
    model: str,
    reasoning_effort: str,
    executable: Path,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    concurrency: int,
    resume: bool,
) -> ValidatedJudgmentChunk:
    """Run exactly one scheduler-owned shard range with no shared writable artifact."""

    chunk_plan = _read_object(chunk_plan_path)
    _validate_chunk_plan(chunk_plan)
    if (
        chunk_plan.get("registry_path") != str(registry_path.resolve())
        or chunk_plan.get("bundles_root") != str(bundles_root.resolve())
        or chunk_plan.get("output_root") != str(output_root.resolve())
        or chunk_plan.get("cell_count") != len(cells)
    ):
        raise JudgmentValidationError("chunk execution registry/root binding drift")
    if chunk_index < 0 or chunk_index >= int(chunk_plan["chunk_count"]):
        raise JudgmentValidationError(
            f"chunk index outside frozen plan: {chunk_index}/{chunk_plan['chunk_count']}"
        )
    expected_judge = {
        "provider": PROVIDER,
        "transport": TRANSPORT,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "executable": str(executable.resolve()),
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "max_output_tokens": max_output_tokens,
        "concurrency": concurrency,
    }
    if chunk_plan.get("judge") != expected_judge or chunk_plan.get("sharding") != {
        "algorithm": SHARDING_ALGORITHM,
        "caps": caps.to_dict(),
    }:
        raise JudgmentValidationError("chunk execution model/caps/runtime drift")

    raw_entry = chunk_plan["chunks"][chunk_index]
    if not isinstance(raw_entry, dict):
        raise JudgmentValidationError("selected chunk entry is not an object")
    entry = dict(raw_entry)
    by_cell = {cell.cell_id: cell for cell in cells}
    try:
        cell = by_cell[str(entry["cell_id"])]
    except KeyError as exc:
        raise JudgmentValidationError("chunk cell is absent from independent registry") from exc
    cell_rows = [row for row in chunk_plan["cells"] if row.get("cell_id") == cell.cell_id]
    if len(cell_rows) != 1 or chunk_index not in cell_rows[0].get("chunk_indexes", []):
        raise JudgmentValidationError("chunk-to-cell table binding mismatch")

    _bundles, shards, expected_cell_plan = _cell_runtime_plan(
        cell,
        bundles_root=bundles_root,
        caps=caps,
        model=model,
        reasoning_effort=reasoning_effort,
        executable=executable,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        max_output_tokens=max_output_tokens,
        concurrency=concurrency,
    )
    target = output_root / _safe_cell_name(cell.cell_id)
    stored_cell_plan = _read_object(target / "plan.json")
    _validate_plan(stored_cell_plan)
    if stored_cell_plan != expected_cell_plan:
        raise JudgmentValidationError("chunk execution cell plan/artifact drift")
    chunk_shards = _validate_chunk_entry_against_shards(
        entry,
        cell=cell,
        shards=shards,
    )
    chunk_directory = _chunk_directory(target, entry)
    lock_path = chunk_directory / ".lock"
    with _exclusive_chunk_lock(lock_path):
        ledger_path = chunk_directory / "ledger.json"
        journal_path = chunk_directory / "journal.ndjson"
        records_path = chunk_directory / "shard_records.ndjson"
        status_path = chunk_directory / "status.json"
        if ledger_path.exists():
            if not resume:
                raise JudgmentValidationError("completed chunk requires --resume")
            completed, _latest, _journal = _validate_completed_chunk(
                cell_plan=stored_cell_plan,
                entry=entry,
                shards=chunk_shards,
                chunk_directory=chunk_directory,
            )
            return completed
        if not resume and any(
            path.exists() for path in (journal_path, records_path, status_path)
        ):
            raise JudgmentValidationError("incomplete chunk output requires --resume")

        backend = ClaudeCliBackend(executable=executable, timeout_s=timeout_seconds)
        if backend.transport_id != TRANSPORT:
            raise JudgmentValidationError("Claude CLI transport differs from frozen sparse-v2")
        runner = SemanticJudgeRunner(
            backend,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_output_tokens,
            max_attempts=max_attempts,
        )
        runner_records = runner.run_shards_streaming(
            chunk_shards,
            journal_path=journal_path,
            resume=resume,
            concurrency=concurrency,
        )
        latest = _latest_shard_records(
            journal_path,
            chunk_shards,
            model=model,
            reasoning_effort=reasoning_effort,
            transport=TRANSPORT,
        )
        if [record.to_dict() for record in latest] != [
            record.to_dict() for record in runner_records
        ]:
            raise JudgmentValidationError("chunk runner differs from its latest fsynced journal")
        written = _atomic_ndjson(records_path, (record.to_dict() for record in latest))
        if written != len(chunk_shards):
            raise JudgmentValidationError("atomic chunk record count mismatch")
        restored = _shard_records_from_file(
            records_path,
            chunk_shards,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        ledger = _chunk_completion_ledger(
            entry=entry,
            journal_path=journal_path,
            records_path=records_path,
            latest_records=restored,
        )
        _atomic_json(ledger_path, ledger)
        _atomic_json(
            status_path,
            {
                "artifact_schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
                "chunk_id": entry["chunk_id"],
                "status": "complete",
                "shard_count": ledger["shard_count"],
                "bundle_count": ledger["bundle_count"],
                "decision_count": ledger["decision_count"],
            },
        )
        completed, _latest, _journal = _validate_completed_chunk(
            cell_plan=stored_cell_plan,
            entry=entry,
            shards=chunk_shards,
            chunk_directory=chunk_directory,
        )
        return completed


def finalize_chunked_cells(
    cells: Sequence[CellSpec],
    *,
    chunk_plan_path: Path,
    registry_path: Path,
    bundles_root: Path,
    output_root: Path,
    status_dir: Path,
) -> dict[str, dict[str, str]]:
    """Validate every disjoint chunk, then publish legacy-compatible cell artifacts."""

    chunk_plan = _read_object(chunk_plan_path)
    _validate_chunk_plan(chunk_plan)
    if (
        chunk_plan.get("registry_path") != str(registry_path.resolve())
        or chunk_plan.get("bundles_root") != str(bundles_root.resolve())
        or chunk_plan.get("output_root") != str(output_root.resolve())
        or chunk_plan.get("cell_count") != len(cells)
    ):
        raise JudgmentValidationError("chunk finalizer registry/root binding drift")
    judge = chunk_plan.get("judge", {})
    if (
        judge.get("provider") != PROVIDER
        or judge.get("transport") != TRANSPORT
        or not str(judge.get("model", "")).startswith("claude-opus-5")
    ):
        raise JudgmentValidationError("chunk finalizer judge binding is not frozen Opus 5")
    executable = Path(str(judge.get("executable", "")))
    if not executable.is_file():
        raise JudgmentValidationError("chunk finalizer executable is missing")
    caps_raw = chunk_plan.get("sharding", {}).get("caps", {})
    try:
        caps = JudgeCaps(**{key: int(value) for key, value in caps_raw.items()})
    except (TypeError, ValueError) as exc:
        raise JudgmentValidationError("chunk finalizer has invalid frozen caps") from exc
    rows = chunk_plan.get("cells", [])
    if [row.get("cell_id") for row in rows] != [cell.cell_id for cell in cells]:
        raise JudgmentValidationError("chunk finalizer cell order differs from registry")
    claimed_indexes = [
        int(index) for row in rows for index in row.get("chunk_indexes", [])
    ]
    if claimed_indexes != list(range(int(chunk_plan["chunk_count"]))):
        raise JudgmentValidationError(
            "global cell tables do not claim every chunk exactly once and in order"
        )

    errors: dict[str, dict[str, str]] = {}
    for cell, cell_row in zip(cells, rows, strict=True):
        try:
            bundles, shards, expected_cell_plan = _cell_runtime_plan(
                cell,
                bundles_root=bundles_root,
                caps=caps,
                model=str(judge["model"]),
                reasoning_effort=str(judge["reasoning_effort"]),
                executable=executable,
                timeout_seconds=float(judge["timeout_seconds"]),
                max_attempts=int(judge["max_attempts"]),
                max_output_tokens=int(judge["max_output_tokens"]),
                concurrency=int(judge["concurrency"]),
            )
            target = output_root / _safe_cell_name(cell.cell_id)
            stored_cell_plan = _read_object(target / "plan.json")
            _validate_plan(stored_cell_plan)
            if stored_cell_plan != expected_cell_plan:
                raise JudgmentValidationError("cell plan/artifact drift before finalization")
            expected_cell_row_counts = {
                "bundle_count": len(bundles),
                "decision_count": sum(len(bundle.target_channels) for bundle in bundles),
                "shard_count": len(shards),
            }
            if any(
                int(cell_row.get(key, -1)) != value
                for key, value in expected_cell_row_counts.items()
            ):
                raise JudgmentValidationError("chunk-plan cell totals drifted")
            indexes = [int(value) for value in cell_row.get("chunk_indexes", [])]
            entries = [chunk_plan["chunks"][index] for index in indexes]
            expected_chunk_directories = {
                _chunk_directory(target, entry).name for entry in entries
            }
            chunks_root = target / "chunks"
            actual_chunk_directories = (
                {path.name for path in chunks_root.iterdir() if path.is_dir()}
                if chunks_root.is_dir()
                else set()
            )
            if actual_chunk_directories != expected_chunk_directories:
                missing = sorted(expected_chunk_directories - actual_chunk_directories)
                extra = sorted(actual_chunk_directories - expected_chunk_directories)
                raise JudgmentValidationError(
                    "cell chunk artifact set differs from frozen plan: "
                    f"missing={len(missing)}, extra={len(extra)}"
                )
            cursor = 0
            chunk_latest: list[JudgeShardRunRecord] = []
            chunk_journal: list[JudgeShardRunRecord] = []
            for entry in entries:
                if int(entry.get("start_shard_ordinal", -1)) != cursor:
                    raise JudgmentValidationError(
                        "cell chunk ranges are not contiguous and omission-free"
                    )
                selected = _validate_chunk_entry_against_shards(
                    entry,
                    cell=cell,
                    shards=shards,
                )
                cursor = int(entry["end_shard_ordinal_exclusive"])
                _completed, latest, journal = _validate_completed_chunk(
                    cell_plan=stored_cell_plan,
                    entry=entry,
                    shards=selected,
                    chunk_directory=_chunk_directory(target, entry),
                )
                chunk_latest.extend(latest)
                chunk_journal.extend(journal)
            if cursor != len(shards):
                raise JudgmentValidationError(
                    "cell chunk ranges do not cover the exact frozen shard universe"
                )
            if len(chunk_latest) != len(shards) or len(
                {record.shard_id for record in chunk_latest}
            ) != len(shards):
                raise JudgmentValidationError(
                    "finalizer requires exact shard coverage with no duplicate"
                )
            if [record.shard_id for record in chunk_latest] != [
                shard.shard_id for shard in shards
            ]:
                raise JudgmentValidationError("finalizer shard order/coverage mismatch")

            with _exclusive_chunk_lock(target / ".finalize.lock"):
                journal_path = target / "shards.ndjson"
                records_path = target / "records.ndjson"
                ledger_path = target / "ledger.json"
                _atomic_ndjson(
                    journal_path,
                    (record.to_dict() for record in chunk_journal),
                )
                latest = _latest_shard_records(
                    journal_path,
                    shards,
                    model=str(judge["model"]),
                    reasoning_effort=str(judge["reasoning_effort"]),
                    transport=TRANSPORT,
                )
                if [record.to_dict() for record in latest] != [
                    record.to_dict() for record in chunk_latest
                ]:
                    raise JudgmentValidationError(
                        "recomposed journal differs from validated chunk records"
                    )
                records = recompose_shard_records(shards, latest)
                _validate_run_records(bundles, records)
                written = _atomic_ndjson(
                    records_path,
                    (record.to_dict() for record in records),
                )
                if written != len(bundles):
                    raise JudgmentValidationError("atomic final record count mismatch")
                restored = _records_from_file(records_path, bundles)
                if _canonical_record_sequence(restored) != _canonical_record_sequence(
                    records
                ):
                    raise JudgmentValidationError(
                        "atomic final records round trip changed judgments"
                    )
                ledger = _completion_ledger(
                    plan=stored_cell_plan,
                    records_path=records_path,
                    journal_path=journal_path,
                    shard_records=latest,
                    records=restored,
                    chunk_count=len(entries),
                )
                _atomic_json(ledger_path, ledger)
            _write_status(
                status_dir,
                cell,
                status="complete",
                payload={
                    "plan_id": stored_cell_plan["plan_id"],
                    "chunk_count": len(entries),
                    "bundle_count": ledger["bundle_count"],
                    "decision_count": ledger["decision_count"],
                    "shard_count": ledger["shard_count"],
                },
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            errors[cell.cell_id] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_status(
                status_dir,
                cell,
                status="error",
                payload=errors[cell.cell_id],
            )
    return errors


def run_cell(
    cell: CellSpec,
    *,
    bundles_root: Path,
    output_root: Path,
    status_dir: Path,
    caps: JudgeCaps,
    model: str,
    reasoning_effort: str,
    executable: Path,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    concurrency: int,
    resume: bool,
) -> ValidatedJudgmentCell:
    bundles, bundle_ledger = _read_bundles(
        cell,
        bundles_root=bundles_root,
    )
    shards = _make_shards(bundles, caps)
    plan = _build_plan(
        cell,
        bundle_ledger=bundle_ledger,
        bundles=bundles,
        shards=shards,
        caps=caps,
        model=model,
        reasoning_effort=reasoning_effort,
        executable=executable,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        max_output_tokens=max_output_tokens,
        concurrency=concurrency,
    )
    target = output_root / _safe_cell_name(cell.cell_id)
    plan_path = target / "plan.json"
    journal_path = target / "shards.ndjson"
    records_path = target / "records.ndjson"
    ledger_path = target / "ledger.json"
    if plan_path.exists():
        prior = _read_object(plan_path)
        _validate_plan(prior)
        if prior != plan:
            raise JudgmentValidationError(
                "resume refused: model, effort, transport, shard plan, or bundle binding changed"
            )
        if not resume:
            raise JudgmentValidationError("existing formal plan requires --resume")
    else:
        if any(path.exists() for path in (journal_path, records_path, ledger_path)):
            raise JudgmentValidationError("cell output exists without its formal plan")
        _atomic_json(plan_path, plan)

    backend = ClaudeCliBackend(executable=executable, timeout_s=timeout_seconds)
    if backend.transport_id != TRANSPORT:
        raise JudgmentValidationError("Claude CLI transport differs from frozen sparse-v2")
    runner = SemanticJudgeRunner(
        backend,
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_output_tokens,
        max_attempts=max_attempts,
    )
    shard_records = runner.run_shards_streaming(
        shards,
        journal_path=journal_path,
        resume=resume,
        concurrency=concurrency,
    )
    if not shards and not journal_path.exists():
        # The runner has nothing to append for an empty-but-valid bundle universe.
        # Materialize the empty journal so the completion ledger remains fully bound.
        _atomic_bytes(journal_path, b"")
    latest = _latest_shard_records(
        journal_path,
        shards,
        model=model,
        reasoning_effort=reasoning_effort,
        transport=TRANSPORT,
    )
    if [record.to_dict() for record in latest] != [record.to_dict() for record in shard_records]:
        raise JudgmentValidationError("runner result differs from latest fsynced journal")
    records = recompose_shard_records(shards, latest)
    _validate_run_records(bundles, records)
    written = _atomic_ndjson(records_path, (record.to_dict() for record in records))
    if written != len(bundles):
        raise JudgmentValidationError("atomic records write count mismatch")
    restored = _records_from_file(records_path, bundles)
    if _canonical_record_sequence(restored) != _canonical_record_sequence(records):
        raise JudgmentValidationError("atomic records round trip changed judge records")
    ledger = _completion_ledger(
        plan=plan,
        records_path=records_path,
        journal_path=journal_path,
        shard_records=latest,
        records=restored,
    )
    _atomic_json(ledger_path, ledger)
    _write_status(
        status_dir,
        cell,
        status="complete",
        payload={
            "plan_id": plan["plan_id"],
            "bundle_count": ledger["bundle_count"],
            "decision_count": ledger["decision_count"],
            "shard_count": ledger["shard_count"],
        },
    )
    return ValidatedJudgmentCell(
        cell_id=cell.cell_id,
        safe_cell_name=_safe_cell_name(cell.cell_id),
        bundle_count=ledger["bundle_count"],
        decision_count=ledger["decision_count"],
        shard_count=ledger["shard_count"],
        usage=ledger["usage"],
        elapsed_seconds=ledger["elapsed_seconds"],
        model=model,
        reasoning_effort=reasoning_effort,
        transport=TRANSPORT,
    )


def validate_completed_judgment(
    cell: CellSpec,
    *,
    bundles_root: Path,
    output_root: Path,
    chunk_plan: Mapping[str, Any] | None = None,
) -> ValidatedJudgmentCell:
    bundles, bundle_ledger = _read_bundles(
        cell,
        bundles_root=bundles_root,
    )
    target = output_root / _safe_cell_name(cell.cell_id)
    plan = _read_object(target / "plan.json")
    ledger = _read_object(target / "ledger.json")
    _validate_plan(plan)
    if plan.get("cell_binding") != _cell_binding(cell):
        raise JudgmentValidationError("formal plan cell binding mismatch")
    expected_bundle_artifact = plan.get("bundle_artifact", {})
    if (
        expected_bundle_artifact.get("artifact_schema_version") != BUNDLE_ARTIFACT_SCHEMA_VERSION
        or expected_bundle_artifact.get("bundle_count") != len(bundles)
        or expected_bundle_artifact.get("bundle_ids")
        != [bundle.bundle_id for bundle in bundles]
    ):
        raise JudgmentValidationError("formal plan bundle artifact binding mismatch")
    judge = plan.get("judge", {})
    if judge.get("provider") != PROVIDER or judge.get("transport") != TRANSPORT:
        raise JudgmentValidationError("formal plan does not use frozen Claude sparse-v2")
    model = str(judge.get("model", ""))
    effort = str(judge.get("reasoning_effort", ""))
    if not model.startswith("claude-opus-5"):
        raise JudgmentValidationError("formal plan model is not Opus 5")
    caps_raw = plan.get("sharding", {}).get("caps", {})
    caps = JudgeCaps(**{key: int(value) for key, value in caps_raw.items()})
    shards = _make_shards(bundles, caps)
    manifests = [shard.manifest_dict() for shard in shards]
    sharding = plan.get("sharding", {})
    if (
        sharding.get("algorithm") != SHARDING_ALGORITHM
        or sharding.get("shards") != manifests
        or sharding.get("shard_count") != len(shards)
        or sharding.get("bundle_count") != len(bundles)
        or sharding.get("decision_count") != sum(len(bundle.target_channels) for bundle in bundles)
    ):
        raise JudgmentValidationError("formal shard plan no longer matches bundles")
    chunk_entries: list[Mapping[str, Any]] = []
    validated_chunk_records: list[JudgeShardRunRecord] = []
    if chunk_plan is not None:
        _validate_chunk_plan(chunk_plan)
        cell_rows = [
            row for row in chunk_plan.get("cells", []) if row.get("cell_id") == cell.cell_id
        ]
        if len(cell_rows) != 1:
            raise JudgmentValidationError("completion is not bound to the global chunk plan")
        indexes = [int(value) for value in cell_rows[0].get("chunk_indexes", [])]
        chunk_entries = [chunk_plan["chunks"][index] for index in indexes]
        cursor = 0
        for entry in chunk_entries:
            if int(entry.get("start_shard_ordinal", -1)) != cursor:
                raise JudgmentValidationError("cell chunk coverage has a gap or overlap")
            selected = _validate_chunk_entry_against_shards(
                entry,
                cell=cell,
                shards=shards,
            )
            cursor = int(entry["end_shard_ordinal_exclusive"])
            _completed, latest, _journal = _validate_completed_chunk(
                cell_plan=plan,
                entry=entry,
                shards=selected,
                chunk_directory=_chunk_directory(target, entry),
            )
            validated_chunk_records.extend(latest)
        if cursor != len(shards):
            raise JudgmentValidationError("cell chunks omit frozen shards")
        if len(validated_chunk_records) != len(shards) or len(
            {record.shard_id for record in validated_chunk_records}
        ) != len(shards):
            raise JudgmentValidationError("cell chunks duplicate or omit frozen shards")
    journal_path = target / str(ledger.get("journal_file", ""))
    records_path = target / str(ledger.get("records_file", ""))
    shard_records = _latest_shard_records(
        journal_path,
        shards,
        model=model,
        reasoning_effort=effort,
        transport=TRANSPORT,
    )
    if chunk_plan is not None and [record.to_dict() for record in shard_records] != [
        record.to_dict() for record in validated_chunk_records
    ]:
        raise JudgmentValidationError("published journal differs from completed chunks")
    records = _records_from_file(records_path, bundles)
    expected_records = recompose_shard_records(shards, shard_records)
    _validate_run_records(bundles, expected_records)
    if _canonical_record_sequence(records) != _canonical_record_sequence(expected_records):
        raise JudgmentValidationError("records file differs from the bound shard journal")
    expected = _completion_ledger(
        plan=plan,
        records_path=records_path,
        journal_path=journal_path,
        shard_records=shard_records,
        records=records,
        chunk_count=(len(chunk_entries) if chunk_plan is not None else None),
    )
    if ledger != expected:
        raise JudgmentValidationError("completion ledger differs from recomputed evidence")
    return ValidatedJudgmentCell(
        cell_id=cell.cell_id,
        safe_cell_name=_safe_cell_name(cell.cell_id),
        bundle_count=ledger["bundle_count"],
        decision_count=ledger["decision_count"],
        shard_count=ledger["shard_count"],
        usage=ledger["usage"],
        elapsed_seconds=ledger["elapsed_seconds"],
        model=model,
        reasoning_effort=effort,
        transport=TRANSPORT,
    )


def build_global_manifest(
    cells: Sequence[CellSpec],
    *,
    registry_path: Path,
    bundles_root: Path,
    output_root: Path,
    manifest_path: Path,
    require_complete: bool,
    chunk_plan: Mapping[str, Any] | None = None,
    prevalidation_errors: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    complete: list[ValidatedJudgmentCell] = []
    incomplete: list[dict[str, str]] = []
    blocked = prevalidation_errors or {}
    for cell in cells:
        if cell.cell_id in blocked:
            item = blocked[cell.cell_id]
            incomplete.append(
                {
                    "cell_id": cell.cell_id,
                    "error_type": str(item.get("error_type", "JudgmentValidationError")),
                    "error": str(item.get("error", "chunk finalization failed")),
                }
            )
            continue
        try:
            complete.append(
                validate_completed_judgment(
                    cell,
                    bundles_root=bundles_root,
                    output_root=output_root,
                    chunk_plan=chunk_plan,
                )
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            incomplete.append(
                {
                    "cell_id": cell.cell_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    models = sorted({item.model for item in complete})
    efforts = sorted({item.reasoning_effort for item in complete})
    transports = sorted({item.transport for item in complete})
    if len(models) > 1 or len(efforts) > 1 or len(transports) > 1:
        raise JudgmentValidationError("completed cells do not share one judge configuration")
    manifest = {
        "artifact_schema_version": JUDGE_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "formal_exhaustive_semantic_judgment_manifest",
        "formal_run": True,
        "status": "complete" if not incomplete else "incomplete",
        "registry_path": str(registry_path.resolve()),
        "bundles_root": str(bundles_root.resolve()),
        "output_root": str(output_root.resolve()),
        "expected_independent_cells": len(cells),
        "complete_cells": len(complete),
        "incomplete_cells": incomplete,
        "model": models[0] if len(models) == 1 else None,
        "reasoning_effort": efforts[0] if len(efforts) == 1 else None,
        "transport": transports[0] if len(transports) == 1 else None,
        "execution_mode": CHUNK_EXECUTION_MODE if chunk_plan is not None else "cell_serial_v1",
        "chunk_count": chunk_plan.get("chunk_count") if chunk_plan is not None else None,
        "totals": {
            "bundle_count": sum(item.bundle_count for item in complete),
            "decision_count": sum(item.decision_count for item in complete),
            "shard_count": sum(item.shard_count for item in complete),
            "statuses": {"ok": sum(item.shard_count for item in complete)},
            "usage": {
                key: sum(item.usage.get(key, 0) for item in complete)
                for key in ("input_tokens", "output_tokens")
            },
            "elapsed_seconds": round(sum(item.elapsed_seconds for item in complete), 3),
        },
        "cells": [item.to_manifest_dict() for item in complete],
    }
    _atomic_json(manifest_path, manifest)
    if require_complete and incomplete:
        raise JudgmentValidationError(
            f"formal judgments incomplete: {len(incomplete)}/{len(cells)} cells"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bundles-root", type=Path, required=True)
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
    parser.add_argument("--prepare-chunks", action="store_true")
    parser.add_argument("--chunk-plan", type=Path, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--chunk-max-shards", type=_positive, default=8)
    parser.add_argument("--expected-independent-cells", type=_positive, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--reasoning", choices=("high", "xhigh", "max"), default="high")
    parser.add_argument("--claude-path", type=Path, default=DEFAULT_CLAUDE_BINARY)
    parser.add_argument("--timeout-s", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=_positive, default=2)
    parser.add_argument("--max-output-tokens", type=_positive, default=65_536)
    parser.add_argument("--concurrency", type=_positive, default=1)
    parser.add_argument("--batch-max-bytes", type=_positive, default=700_000)
    parser.add_argument("--batch-max-estimated-tokens", type=_positive, default=235_000)
    parser.add_argument("--batch-max-bundles", type=_positive, default=64)
    parser.add_argument("--batch-max-decisions", type=_positive, default=128)
    args = parser.parse_args()

    if args.array_index is not None and (args.cell or args.chunk_index is not None):
        parser.error("--array-index cannot be combined with --cell or --chunk-index")
    if args.chunk_index is not None and args.chunk_index < 0:
        parser.error("--chunk-index must be non-negative")
    if args.prepare_chunks and (args.finalize_only or args.chunk_index is not None):
        parser.error("--prepare-chunks is mutually exclusive with run/finalize modes")
    if args.chunk_index is not None and args.finalize_only:
        parser.error("--chunk-index is mutually exclusive with --finalize-only")
    if (args.prepare_chunks or args.chunk_index is not None or args.finalize_only) and (
        args.array_index is not None or args.cell or args.scope != "all"
    ):
        parser.error("chunk prepare/run/finalize modes operate on the full registry plan")
    if (args.prepare_chunks or args.chunk_index is not None) and args.require_complete:
        parser.error("--require-complete is only meaningful with --finalize-only")
    if (args.prepare_chunks or args.chunk_index is not None) and args.chunk_plan is None:
        parser.error("chunk prepare/run modes require --chunk-plan")
    if args.finalize_only and (
        args.array_index is not None or args.cell or args.scope != "all" or args.resume
    ):
        parser.error("--finalize-only validates the full independent registry")
    if args.require_complete and not args.finalize_only:
        parser.error("--require-complete is only meaningful with --finalize-only")
    if not args.registry.is_file():
        parser.error(f"registry not found: {args.registry}")
    if not args.bundles_root.is_dir():
        parser.error(f"bundle root not found: {args.bundles_root}")
    if not args.claude_path.is_file() and not args.finalize_only:
        parser.error(f"Claude Code binary not found: {args.claude_path}")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if not args.model.startswith("claude-opus-5"):
        parser.error("formal semantic judge must be one claude-opus-5 model")

    try:
        all_cells = load_independent_cells(args.registry)
        if (
            args.expected_independent_cells is not None
            and len(all_cells) != args.expected_independent_cells
        ):
            raise ValueError(
                "independent registry count mismatch: "
                f"expected={args.expected_independent_cells}, actual={len(all_cells)}"
            )
        selected = (
            []
            if args.prepare_chunks or args.chunk_index is not None or args.finalize_only
            else select_cells(
                all_cells,
                scope=args.scope,
                requested_cell_ids=args.cell,
                array_index=args.array_index,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    caps = JudgeCaps(
        max_input_bytes=args.batch_max_bytes,
        max_estimated_input_tokens=args.batch_max_estimated_tokens,
        max_bundles=args.batch_max_bundles,
        max_decision_units=args.batch_max_decisions,
    )

    if args.prepare_chunks:
        try:
            chunk_plan = prepare_chunk_plan(
                all_cells,
                registry_path=args.registry,
                bundles_root=args.bundles_root,
                output_root=args.out_root,
                chunk_plan_path=args.chunk_plan,
                caps=caps,
                max_shards_per_chunk=args.chunk_max_shards,
                model=args.model,
                reasoning_effort=args.reasoning,
                executable=args.claude_path,
                timeout_seconds=args.timeout_s,
                max_attempts=args.max_attempts,
                max_output_tokens=args.max_output_tokens,
                concurrency=args.concurrency,
                resume=args.resume,
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            print(f"[analysis-v2] ERROR {exc}", file=sys.stderr, flush=True)
            return 2
        print(
            f"[analysis-v2] frozen chunks={chunk_plan['chunk_count']} "
            f"cells={chunk_plan['cell_count']}",
            flush=True,
        )
        return 0

    if args.chunk_index is not None:
        try:
            item = run_chunk(
                all_cells,
                chunk_plan_path=args.chunk_plan,
                chunk_index=args.chunk_index,
                registry_path=args.registry,
                bundles_root=args.bundles_root,
                output_root=args.out_root,
                caps=caps,
                model=args.model,
                reasoning_effort=args.reasoning,
                executable=args.claude_path,
                timeout_seconds=args.timeout_s,
                max_attempts=args.max_attempts,
                max_output_tokens=args.max_output_tokens,
                concurrency=args.concurrency,
                resume=args.resume,
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            print(
                f"[analysis-v2] ERROR chunk={args.chunk_index} "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        print(
            f"[analysis-v2] complete chunk={item.global_chunk_index} "
            f"cell={item.cell_id} shards={item.shard_count} "
            f"bundles={item.bundle_count} decisions={item.decision_count}",
            flush=True,
        )
        return 0

    if args.finalize_only:
        chunk_plan = None
        prevalidation_errors = None
        if args.chunk_plan is not None:
            try:
                chunk_plan = _read_object(args.chunk_plan)
                _validate_chunk_plan(chunk_plan)
                prevalidation_errors = finalize_chunked_cells(
                    all_cells,
                    chunk_plan_path=args.chunk_plan,
                    registry_path=args.registry,
                    bundles_root=args.bundles_root,
                    output_root=args.out_root,
                    status_dir=args.status_dir,
                )
            except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
                print(f"[analysis-v2] ERROR {exc}", file=sys.stderr, flush=True)
                return 2
        try:
            manifest = build_global_manifest(
                all_cells,
                registry_path=args.registry,
                bundles_root=args.bundles_root,
                output_root=args.out_root,
                manifest_path=args.manifest_out,
                require_complete=args.require_complete,
                chunk_plan=chunk_plan,
                prevalidation_errors=prevalidation_errors,
            )
        except JudgmentValidationError as exc:
            print(f"[analysis-v2] ERROR {exc}", file=sys.stderr, flush=True)
            return 2
        totals = manifest["totals"]
        print(
            f"[analysis-v2] judgments cells={manifest['complete_cells']}/"
            f"{manifest['expected_independent_cells']} bundles={totals['bundle_count']} "
            f"decisions={totals['decision_count']} shards={totals['shard_count']} "
            f"status={manifest['status']} model={manifest['model']}",
            flush=True,
        )
        return 0
    failures = 0
    for cell in selected:
        started = time.monotonic()
        try:
            item = run_cell(
                cell,
                bundles_root=args.bundles_root,
                output_root=args.out_root,
                status_dir=args.status_dir,
                caps=caps,
                model=args.model,
                reasoning_effort=args.reasoning,
                executable=args.claude_path,
                timeout_seconds=args.timeout_s,
                max_attempts=args.max_attempts,
                max_output_tokens=args.max_output_tokens,
                concurrency=args.concurrency,
                resume=args.resume,
            )
            print(
                f"[analysis-v2] complete cell={cell.cell_id} "
                f"bundles={item.bundle_count} decisions={item.decision_count} "
                f"shards={item.shard_count} elapsed={time.monotonic() - started:.3f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - persist cell-level diagnostics
            failures += 1
            _write_status(
                args.status_dir,
                cell,
                status="error",
                payload={"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(
                f"[analysis-v2] ERROR cell={cell.cell_id} {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only progress accounting for the frozen semantic judgment run.

The reporter deliberately restricts itself to natural cell, shard, and bundle
identifiers plus declared counts.  It never imports a provider backend and it
never writes to the artifact tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_INDEPENDENT_CELLS = 55
EXPECTED_BUNDLES = 677_247
EXPECTED_DECISIONS = 2_076_807
EXPECTED_SHARDS = 17_541

_CHANNELS = frozenset(
    {
        "T1_quality_misrepresentation",
        "T2_unowned_inventory",
        "T3_inventory_overcommitment",
        "T4_premature_closure",
        "T5_externalization_pii",
        "T6_unverified_trust_claim",
    }
)
_REASONING_DECISION_UNITS = len(_CHANNELS)
_JOURNAL_STATUSES = frozenset(
    {"ok", "error", "parse_error", "partial_parse_error", "transport_error"}
)


class ProgressValidationError(ValueError):
    """An observed progress artifact is malformed or contradicts another one."""


@dataclass(frozen=True)
class _RegistryCell:
    cell_id: str
    safe_cell_name: str


@dataclass(frozen=True)
class _BundleTotals:
    bundles: int
    decisions: int


@dataclass(frozen=True)
class _ShardPlan:
    shard_id: str
    ordinal: int
    bundle_ids: tuple[str, ...]
    decision_units: int


@dataclass(frozen=True)
class _CellPlan:
    plan_id: str
    bundle_ids: tuple[str, ...]
    bundles: int
    decisions: int
    shards: tuple[_ShardPlan, ...]


@dataclass(frozen=True)
class _JournalProgress:
    records: int = 0
    error_records: int = 0
    retry_attempts: int = 0
    latest_error_shards: int = 0
    ok_shards: int = 0
    ok_bundles: int = 0
    ok_decisions: int = 0


@dataclass(frozen=True)
class CellProgress:
    cell_id: str
    safe_cell_name: str
    state: str
    bundle_ready: bool
    plan_present: bool
    bundles_total: int | None
    decisions_total: int | None
    shards_total: int | None
    ok_shards: int
    ok_bundles: int
    ok_decisions: int
    journal_records: int
    journal_error_records: int
    retry_attempts: int
    latest_error_shards: int
    records_present: bool
    completion_ledger_present: bool
    status_file: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bundle_percent"] = _percent(self.ok_bundles, self.bundles_total)
        value["decision_percent"] = _percent(self.ok_decisions, self.decisions_total)
        value["shard_percent"] = _percent(self.ok_shards, self.shards_total)
        return value


@dataclass(frozen=True)
class ProgressReport:
    expected_cells: int
    expected_bundles: int | None
    expected_decisions: int | None
    expected_shards: int | None
    cells: tuple[CellProgress, ...]

    def to_dict(self) -> dict[str, Any]:
        bundle_ready_cells = sum(cell.bundle_ready for cell in self.cells)
        planned_cells = sum(cell.plan_present for cell in self.cells)
        complete_cells = sum(cell.state == "complete" for cell in self.cells)
        known_bundles = sum(cell.bundles_total or 0 for cell in self.cells)
        known_decisions = sum(cell.decisions_total or 0 for cell in self.cells)
        planned_shards = sum(cell.shards_total or 0 for cell in self.cells)
        ok_shards = sum(cell.ok_shards for cell in self.cells)
        ok_bundles = sum(cell.ok_bundles for cell in self.cells)
        ok_decisions = sum(cell.ok_decisions for cell in self.cells)
        return {
            "schema_version": 1,
            "totals": {
                "cells": {
                    "total": self.expected_cells,
                    "bundle_ready": bundle_ready_cells,
                    "planned": planned_cells,
                    "complete": complete_cells,
                    "percent": _percent(complete_cells, self.expected_cells),
                },
                "shards": {
                    "total": self.expected_shards,
                    "planned": planned_shards,
                    "ok": ok_shards,
                    "percent": _percent(ok_shards, self.expected_shards),
                },
                "bundles": {
                    "total": self.expected_bundles,
                    "prepared": known_bundles,
                    "ok": ok_bundles,
                    "percent": _percent(ok_bundles, self.expected_bundles),
                },
                "decisions": {
                    "total": self.expected_decisions,
                    "prepared": known_decisions,
                    "ok": ok_decisions,
                    "percent": _percent(ok_decisions, self.expected_decisions),
                },
                "errors": {
                    "journal_records": sum(
                        cell.journal_error_records for cell in self.cells
                    ),
                    "retry_attempts": sum(cell.retry_attempts for cell in self.cells),
                    "latest_shards": sum(
                        cell.latest_error_shards for cell in self.cells
                    ),
                    "status_cells": sum(cell.status_file == "error" for cell in self.cells),
                },
            },
            "cells": [cell.to_dict() for cell in self.cells],
        }


def safe_cell_name(cell_id: str) -> str:
    """Return the directory name used by the analysis-v2 artifact layout."""

    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in cell_id
    )


def build_progress_report(
    *,
    registry_path: Path,
    bundles_root: Path,
    judgments_root: Path,
    status_dir: Path,
    expected_cells: int = EXPECTED_INDEPENDENT_CELLS,
    expected_bundles: int | None = EXPECTED_BUNDLES,
    expected_decisions: int | None = EXPECTED_DECISIONS,
    expected_shards: int | None = EXPECTED_SHARDS,
) -> ProgressReport:
    """Inspect currently published artifacts and return exact partial progress."""

    cells = _read_registry(registry_path, expected_cells=expected_cells)
    rows: list[CellProgress] = []
    for cell in cells:
        bundle_path = bundles_root / cell.safe_cell_name / "ledger.json"
        target = judgments_root / cell.safe_cell_name
        bundle_totals = (
            _read_bundle_totals(bundle_path, cell)
            if bundle_path.exists()
            else None
        )
        plan_path = target / "plan.json"
        plan = _read_plan(plan_path, cell, bundle_totals) if plan_path.exists() else None
        journal = _read_progress_journal(target, plan)
        records_path = target / "records.ndjson"
        records_present = records_path.exists()
        if records_present:
            if plan is None:
                raise ProgressValidationError(
                    f"records exist without a plan for {cell.cell_id}"
                )
            _validate_records(records_path, plan, journal)
        completion_path = target / "ledger.json"
        completion_present = completion_path.exists()
        if completion_present:
            if plan is None:
                raise ProgressValidationError(
                    f"completion ledger exists without a plan for {cell.cell_id}"
                )
            _validate_completion_ledger(
                completion_path,
                plan=plan,
                journal=journal,
                records_present=records_present,
            )
        status_path = status_dir / f"{cell.safe_cell_name}.json"
        status_value = (
            _read_status(status_path, cell, plan, completion_present)
            if status_path.exists()
            else None
        )
        state = _cell_state(
            bundle_ready=bundle_totals is not None,
            plan=plan,
            journal=journal,
            records_present=records_present,
            completion_present=completion_present,
            status_value=status_value,
        )
        rows.append(
            CellProgress(
                cell_id=cell.cell_id,
                safe_cell_name=cell.safe_cell_name,
                state=state,
                bundle_ready=bundle_totals is not None,
                plan_present=plan is not None,
                bundles_total=(bundle_totals.bundles if bundle_totals else None),
                decisions_total=(bundle_totals.decisions if bundle_totals else None),
                shards_total=(len(plan.shards) if plan else None),
                ok_shards=journal.ok_shards,
                ok_bundles=journal.ok_bundles,
                ok_decisions=journal.ok_decisions,
                journal_records=journal.records,
                journal_error_records=journal.error_records,
                retry_attempts=journal.retry_attempts,
                latest_error_shards=journal.latest_error_shards,
                records_present=records_present,
                completion_ledger_present=completion_present,
                status_file=status_value,
            )
        )
    _validate_global_counts(
        rows,
        expected_bundles=expected_bundles,
        expected_decisions=expected_decisions,
        expected_shards=expected_shards,
    )
    return ProgressReport(
        expected_cells=expected_cells,
        expected_bundles=expected_bundles,
        expected_decisions=expected_decisions,
        expected_shards=expected_shards,
        cells=tuple(rows),
    )


def render_progress_table(report: ProgressReport, *, active_only: bool = False) -> str:
    """Render a compact human-readable summary and per-cell table."""

    payload = report.to_dict()
    totals = payload["totals"]
    lines = [
        (
            "cells {complete}/{total} ({percent}) | "
            "shards {shard_ok}/{shard_total} ({shard_percent}) | "
            "bundles {bundle_ok}/{bundle_total} ({bundle_percent}) | "
            "decisions {decision_ok}/{decision_total} ({decision_percent})"
        ).format(
            complete=totals["cells"]["complete"],
            total=totals["cells"]["total"],
            percent=_display_percent(totals["cells"]["percent"]),
            shard_ok=_display_count(totals["shards"]["ok"]),
            shard_total=_display_count(totals["shards"]["total"]),
            shard_percent=_display_percent(totals["shards"]["percent"]),
            bundle_ok=_display_count(totals["bundles"]["ok"]),
            bundle_total=_display_count(totals["bundles"]["total"]),
            bundle_percent=_display_percent(totals["bundles"]["percent"]),
            decision_ok=_display_count(totals["decisions"]["ok"]),
            decision_total=_display_count(totals["decisions"]["total"]),
            decision_percent=_display_percent(totals["decisions"]["percent"]),
        ),
        (
            "errors journal={journal} retries={retries} "
            "latest_shards={latest} status_cells={status}"
        ).format(
            journal=totals["errors"]["journal_records"],
            retries=totals["errors"]["retry_attempts"],
            latest=totals["errors"]["latest_shards"],
            status=totals["errors"]["status_cells"],
        ),
        "",
        (
            f"{'cell':44} {'state':10} {'shards':>13} {'bundles':>17} "
            f"{'decisions':>19} {'err':>5}"
        ),
    ]
    for cell in report.cells:
        if active_only and cell.state in {"preparing", "waiting"}:
            continue
        lines.append(
            f"{cell.cell_id[:44]:44} {cell.state:10} "
            f"{_fraction(cell.ok_shards, cell.shards_total):>13} "
            f"{_fraction(cell.ok_bundles, cell.bundles_total):>17} "
            f"{_fraction(cell.ok_decisions, cell.decisions_total):>19} "
            f"{cell.latest_error_shards:5d}"
        )
    return "\n".join(lines)


def _read_registry(path: Path, *, expected_cells: int) -> tuple[_RegistryCell, ...]:
    value = _read_object(path, label="registry")
    raw_cells = value.get("cells")
    if not isinstance(raw_cells, list):
        raise ProgressValidationError("registry cells must be a list")
    cells: list[_RegistryCell] = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            raise ProgressValidationError(f"registry cell {index} must be an object")
        if raw.get("duplicate_of") is not None:
            continue
        cell_id = raw.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ProgressValidationError(f"registry cell {index} has no cell_id")
        cells.append(_RegistryCell(cell_id, safe_cell_name(cell_id)))
    if len(cells) != expected_cells:
        raise ProgressValidationError(
            f"independent cell count differs: expected={expected_cells}, actual={len(cells)}"
        )
    declared = value.get("independent_record_count")
    if declared is not None and _count(declared, "registry independent count") != len(cells):
        raise ProgressValidationError("registry independent count contradicts its cells")
    cell_ids = [cell.cell_id for cell in cells]
    safe_names = [cell.safe_cell_name for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ProgressValidationError("independent registry cell IDs are not unique")
    if len(set(safe_names)) != len(safe_names):
        raise ProgressValidationError("independent registry directory names are not unique")
    return tuple(cells)


def _read_bundle_totals(path: Path, cell: _RegistryCell) -> _BundleTotals:
    value = _read_object(path, label=f"bundle ledger for {cell.cell_id}")
    if value.get("status") != "complete":
        raise ProgressValidationError(f"bundle ledger is not complete for {cell.cell_id}")
    if value.get("cell_id") != cell.cell_id:
        raise ProgressValidationError(f"bundle ledger cell ID differs for {cell.cell_id}")
    binding = value.get("cell_binding")
    if isinstance(binding, Mapping) and binding.get("cell_id") != cell.cell_id:
        raise ProgressValidationError(f"bundle ledger binding differs for {cell.cell_id}")
    bundles = _count(value.get("bundle_count"), f"bundle count for {cell.cell_id}")
    by_kind = value.get("bundles_by_kind")
    if not isinstance(by_kind, Mapping):
        raise ProgressValidationError(f"bundle kind counts are missing for {cell.cell_id}")
    kind_counts = [
        _count(count, f"bundle kind count for {cell.cell_id}")
        for key, count in by_kind.items()
        if _require_nonempty_string(key, f"bundle kind for {cell.cell_id}")
    ]
    if sum(kind_counts) != bundles:
        raise ProgressValidationError(f"bundle kind counts contradict {cell.cell_id}")
    denominators = value.get("denominators")
    if not isinstance(denominators, Mapping):
        raise ProgressValidationError(f"decision counts are missing for {cell.cell_id}")
    decisions = 0
    for key, raw_count in denominators.items():
        name = _require_nonempty_string(key, f"decision kind for {cell.cell_id}")
        count = _count(raw_count, f"decision kind count for {cell.cell_id}")
        # One reasoning carrier is evaluated for all six channels.  Every other
        # denominator entry already represents one target-channel decision.
        decisions += count * (_REASONING_DECISION_UNITS if name == "reasoning" else 1)
    return _BundleTotals(bundles=bundles, decisions=decisions)


def _read_plan(
    path: Path,
    cell: _RegistryCell,
    bundle_totals: _BundleTotals | None,
) -> _CellPlan:
    value = _read_object(path, label=f"plan for {cell.cell_id}")
    if value.get("formal_run") is not True:
        raise ProgressValidationError(f"plan is not formal for {cell.cell_id}")
    plan_id = value.get("plan_id")
    expected_plan_id = f"plan:{cell.safe_cell_name}"
    if plan_id != expected_plan_id:
        raise ProgressValidationError(f"plan ID differs for {cell.cell_id}")
    binding = value.get("cell_binding")
    if not isinstance(binding, Mapping) or binding.get("cell_id") != cell.cell_id:
        raise ProgressValidationError(f"plan cell binding differs for {cell.cell_id}")
    artifact = value.get("bundle_artifact")
    if not isinstance(artifact, Mapping):
        raise ProgressValidationError(f"plan bundle table is missing for {cell.cell_id}")
    raw_bundle_ids = artifact.get("bundle_ids")
    bundle_ids = _string_ids(raw_bundle_ids, f"plan bundle IDs for {cell.cell_id}")
    bundles = _count(artifact.get("bundle_count"), f"plan bundle count for {cell.cell_id}")
    if bundles != len(bundle_ids):
        raise ProgressValidationError(f"plan bundle count contradicts {cell.cell_id}")
    if bundle_totals is not None and bundles != bundle_totals.bundles:
        raise ProgressValidationError(f"plan bundle total contradicts {cell.cell_id}")
    sharding = value.get("sharding")
    if not isinstance(sharding, Mapping):
        raise ProgressValidationError(f"plan shard table is missing for {cell.cell_id}")
    if _count(sharding.get("bundle_count"), "plan shard bundle count") != bundles:
        raise ProgressValidationError(f"plan shard bundle count contradicts {cell.cell_id}")
    decisions = _count(sharding.get("decision_count"), "plan decision count")
    if bundle_totals is not None and decisions != bundle_totals.decisions:
        raise ProgressValidationError(f"plan decision total contradicts {cell.cell_id}")
    raw_shards = sharding.get("shards")
    if not isinstance(raw_shards, list):
        raise ProgressValidationError(f"plan shards must be a list for {cell.cell_id}")
    if _count(sharding.get("shard_count"), "plan shard count") != len(raw_shards):
        raise ProgressValidationError(f"plan shard count contradicts {cell.cell_id}")
    shards: list[_ShardPlan] = []
    cursor = 0
    for ordinal, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise ProgressValidationError(
                f"plan shard {ordinal} must be an object for {cell.cell_id}"
            )
        shard_id = _require_nonempty_string(
            raw.get("shard_id"), f"plan shard ID for {cell.cell_id}"
        )
        stored_ordinal = _count(raw.get("ordinal"), "plan shard ordinal")
        start = _count(raw.get("start_index"), "plan shard start")
        end = _count(raw.get("end_index_exclusive"), "plan shard end")
        if stored_ordinal != ordinal or start != cursor or end <= start or end > bundles:
            raise ProgressValidationError(f"plan shard range contradicts {cell.cell_id}")
        ids = _string_ids(raw.get("bundle_ids"), f"plan shard bundle IDs for {cell.cell_id}")
        if ids != bundle_ids[start:end]:
            raise ProgressValidationError(f"plan shard bundle IDs contradict {cell.cell_id}")
        units = _count(raw.get("decision_units"), "plan shard decisions")
        shards.append(_ShardPlan(shard_id, ordinal, ids, units))
        cursor = end
    if cursor != bundles:
        raise ProgressValidationError(f"plan shards do not cover {cell.cell_id}")
    if len({shard.shard_id for shard in shards}) != len(shards):
        raise ProgressValidationError(f"plan shard IDs are not unique for {cell.cell_id}")
    if sum(shard.decision_units for shard in shards) != decisions:
        raise ProgressValidationError(f"plan shard decisions contradict {cell.cell_id}")
    return _CellPlan(plan_id, bundle_ids, bundles, decisions, tuple(shards))


def _read_progress_journal(target: Path, plan: _CellPlan | None) -> _JournalProgress:
    root_journal = target / "shards.ndjson"
    journal_paths: tuple[Path, ...]
    if root_journal.exists():
        journal_paths = (root_journal,)
    else:
        chunks_root = target / "chunks"
        journal_paths = (
            tuple(sorted(chunks_root.glob("chunk-*/journal.ndjson")))
            if chunks_root.is_dir()
            else ()
        )
    if not journal_paths:
        return _JournalProgress()
    if plan is None:
        raise ProgressValidationError(f"journal exists without a plan in {target}")
    expected = {shard.shard_id: shard for shard in plan.shards}
    latest: dict[str, Mapping[str, Any]] = {}
    shard_file: dict[str, Path] = {}
    records = error_records = retry_attempts = 0
    for path in journal_paths:
        for line_number, raw in _read_ndjson(path, label="journal"):
            shard_id = _require_nonempty_string(
                raw.get("shard_id"), f"journal shard ID at {path}:{line_number}"
            )
            shard = expected.get(shard_id)
            if shard is None:
                raise ProgressValidationError(
                    f"journal has an unknown shard at {path}:{line_number}"
                )
            previous_path = shard_file.get(shard_id)
            if previous_path is not None and previous_path != path:
                raise ProgressValidationError(
                    f"journal shard occurs in multiple files for {shard_id}"
                )
            shard_file[shard_id] = path
            _validate_journal_row(raw, shard, path=path, line_number=line_number)
            status = str(raw["status"])
            attempts = _count(raw.get("attempts"), "journal attempts", positive=True)
            records += 1
            error_records += int(status != "ok")
            retry_attempts += attempts - 1
            latest[shard_id] = raw
    ok_shards = ok_bundles = ok_decisions = latest_error_shards = 0
    for shard in plan.shards:
        record = latest.get(shard.shard_id)
        if record is None:
            continue
        if record.get("status") == "ok":
            ok_shards += 1
            ok_bundles += len(shard.bundle_ids)
            ok_decisions += shard.decision_units
        else:
            latest_error_shards += 1
    return _JournalProgress(
        records=records,
        error_records=error_records,
        retry_attempts=retry_attempts,
        latest_error_shards=latest_error_shards,
        ok_shards=ok_shards,
        ok_bundles=ok_bundles,
        ok_decisions=ok_decisions,
    )


def _validate_journal_row(
    raw: Mapping[str, Any],
    shard: _ShardPlan,
    *,
    path: Path,
    line_number: int,
) -> None:
    location = f"{path}:{line_number}"
    if raw.get("schema_version") != 1:
        raise ProgressValidationError(f"journal schema differs at {location}")
    if _count(raw.get("ordinal"), "journal shard ordinal") != shard.ordinal:
        raise ProgressValidationError(f"journal shard ordinal differs at {location}")
    if _string_ids(raw.get("bundle_ids"), "journal bundle IDs") != shard.bundle_ids:
        raise ProgressValidationError(f"journal bundle IDs differ at {location}")
    status = raw.get("status")
    if status not in _JOURNAL_STATUSES:
        raise ProgressValidationError(f"journal status is invalid at {location}")
    decision = raw.get("decision")
    if status != "ok":
        if decision is not None:
            raise ProgressValidationError(
                f"unsuccessful journal row retains a decision at {location}"
            )
        return
    if not isinstance(decision, Mapping):
        raise ProgressValidationError(f"successful journal row lacks a decision at {location}")
    _validate_batch_decision(decision, shard, location=location)


def _validate_batch_decision(
    decision: Mapping[str, Any], shard: _ShardPlan, *, location: str
) -> None:
    if decision.get("shard_id") != shard.shard_id or decision.get("shard_complete") is not True:
        raise ProgressValidationError(f"journal decision binding differs at {location}")
    raw_results = decision.get("bundle_results")
    if not isinstance(raw_results, list) or len(raw_results) != len(shard.bundle_ids):
        raise ProgressValidationError(f"journal decision bundle count differs at {location}")
    decision_units = 0
    for expected_bundle_id, raw in zip(shard.bundle_ids, raw_results, strict=True):
        if not isinstance(raw, Mapping):
            raise ProgressValidationError(f"journal bundle decision is invalid at {location}")
        if raw.get("bundle_id") != expected_bundle_id or raw.get("bundle_complete") is not True:
            raise ProgressValidationError(f"journal bundle decision binding differs at {location}")
        decision_units += _validate_channel_decisions(raw.get("decisions"), location=location)
    if decision_units != shard.decision_units:
        raise ProgressValidationError(f"journal decision count differs at {location}")


def _validate_records(
    path: Path, plan: _CellPlan, journal: _JournalProgress
) -> None:
    rows = _read_ndjson(path, label="records")
    if len(rows) != plan.bundles:
        raise ProgressValidationError(f"records count differs for {path.parent.name}")
    decision_units = 0
    shard_by_bundle = {
        bundle_id: shard.shard_id
        for shard in plan.shards
        for bundle_id in shard.bundle_ids
    }
    for (line_number, raw), expected_bundle_id in zip(rows, plan.bundle_ids, strict=True):
        if raw.get("bundle_id") != expected_bundle_id or raw.get("status") != "ok":
            raise ProgressValidationError(f"record binding differs at {path}:{line_number}")
        shard_id = raw.get("shard_id")
        if shard_id is not None and shard_id != shard_by_bundle[expected_bundle_id]:
            raise ProgressValidationError(f"record shard ID differs at {path}:{line_number}")
        decision = raw.get("decision")
        if not isinstance(decision, Mapping):
            raise ProgressValidationError(f"record decision is missing at {path}:{line_number}")
        if (
            decision.get("bundle_id") != expected_bundle_id
            or decision.get("bundle_complete") is not True
        ):
            raise ProgressValidationError(f"record decision binding differs at {path}:{line_number}")
        decision_units += _validate_channel_decisions(
            decision.get("decisions"), location=f"{path}:{line_number}"
        )
    if decision_units != plan.decisions:
        raise ProgressValidationError(f"record decision count differs for {path.parent.name}")
    if (
        journal.ok_shards != len(plan.shards)
        or journal.ok_bundles != plan.bundles
        or journal.ok_decisions != plan.decisions
        or journal.latest_error_shards
    ):
        raise ProgressValidationError(f"records precede complete journal coverage for {path.parent.name}")


def _validate_completion_ledger(
    path: Path,
    *,
    plan: _CellPlan,
    journal: _JournalProgress,
    records_present: bool,
) -> None:
    value = _read_object(path, label=f"completion ledger for {path.parent.name}")
    if value.get("status") != "complete" or value.get("plan_id") != plan.plan_id:
        raise ProgressValidationError(f"completion ledger binding differs for {path.parent.name}")
    expected = {
        "bundle_count": plan.bundles,
        "decision_count": plan.decisions,
        "shard_count": len(plan.shards),
    }
    if any(_count(value.get(key), key) != count for key, count in expected.items()):
        raise ProgressValidationError(f"completion ledger counts differ for {path.parent.name}")
    statuses = value.get("shard_statuses")
    if statuses != ({"ok": len(plan.shards)} if plan.shards else {}):
        raise ProgressValidationError(f"completion shard statuses differ for {path.parent.name}")
    if not records_present:
        raise ProgressValidationError(f"completion ledger has no records for {path.parent.name}")
    if (
        journal.ok_shards != len(plan.shards)
        or journal.ok_bundles != plan.bundles
        or journal.ok_decisions != plan.decisions
        or journal.latest_error_shards
    ):
        raise ProgressValidationError(f"completion ledger contradicts journal for {path.parent.name}")


def _read_status(
    path: Path,
    cell: _RegistryCell,
    plan: _CellPlan | None,
    completion_present: bool,
) -> str:
    value = _read_object(path, label=f"status for {cell.cell_id}")
    binding = value.get("cell_binding")
    if not isinstance(binding, Mapping) or binding.get("cell_id") != cell.cell_id:
        raise ProgressValidationError(f"status cell binding differs for {cell.cell_id}")
    status = value.get("status")
    if status not in {"complete", "error"}:
        raise ProgressValidationError(f"status value is invalid for {cell.cell_id}")
    if status == "complete":
        if plan is None or not completion_present:
            raise ProgressValidationError(f"complete status lacks artifacts for {cell.cell_id}")
        expected = {
            "plan_id": plan.plan_id,
            "bundle_count": plan.bundles,
            "decision_count": plan.decisions,
            "shard_count": len(plan.shards),
        }
        if any(value.get(key) != count for key, count in expected.items()):
            raise ProgressValidationError(f"complete status counts differ for {cell.cell_id}")
    elif completion_present:
        raise ProgressValidationError(f"error status contradicts completion for {cell.cell_id}")
    return str(status)


def _cell_state(
    *,
    bundle_ready: bool,
    plan: _CellPlan | None,
    journal: _JournalProgress,
    records_present: bool,
    completion_present: bool,
    status_value: str | None,
) -> str:
    if completion_present:
        return "complete"
    if status_value == "error" or journal.latest_error_shards:
        return "error"
    if records_present:
        return "publishing"
    if plan is not None and journal.ok_shards == len(plan.shards) and plan.shards:
        return "judged"
    if journal.records:
        return "running"
    if plan is not None:
        return "planned"
    return "waiting" if bundle_ready else "preparing"


def _validate_global_counts(
    rows: Sequence[CellProgress],
    *,
    expected_bundles: int | None,
    expected_decisions: int | None,
    expected_shards: int | None,
) -> None:
    known_bundles = sum(row.bundles_total or 0 for row in rows)
    known_decisions = sum(row.decisions_total or 0 for row in rows)
    planned_shards = sum(row.shards_total or 0 for row in rows)
    ready_cells = sum(row.bundle_ready for row in rows)
    planned_cells = sum(row.plan_present for row in rows)
    checks = (
        (known_bundles, expected_bundles, ready_cells, "bundle"),
        (known_decisions, expected_decisions, ready_cells, "decision"),
        (planned_shards, expected_shards, planned_cells, "shard"),
    )
    for observed, expected, observed_cells, label in checks:
        if expected is None:
            continue
        if observed > expected:
            raise ProgressValidationError(f"observed {label} total exceeds expected total")
        if observed_cells == len(rows) and observed != expected:
            raise ProgressValidationError(
                f"complete {label} total differs: expected={expected}, actual={observed}"
            )


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgressValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProgressValidationError(f"{label} must contain one object: {path}")
    return value


def _read_ndjson(path: Path, *, label: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProgressValidationError(f"cannot read {label}: {path}") from exc
    if data and not data.endswith("\n"):
        raise ProgressValidationError(f"{label} has an incomplete final row: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProgressValidationError(
                f"{label} row is invalid at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ProgressValidationError(
                f"{label} row must be an object at {path}:{line_number}"
            )
        rows.append((line_number, value))
    return rows


def _validate_channel_decisions(value: Any, *, location: str) -> int:
    if not isinstance(value, list) or not value:
        raise ProgressValidationError(f"channel decisions are missing at {location}")
    channels: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or item.get("channel") not in _CHANNELS:
            raise ProgressValidationError(f"channel decision is invalid at {location}")
        channels.append(str(item["channel"]))
    if len(channels) != len(set(channels)):
        raise ProgressValidationError(f"channel decisions repeat at {location}")
    return len(channels)


def _string_ids(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProgressValidationError(f"{label} must be a list")
    ids = tuple(_require_nonempty_string(item, label) for item in value)
    if len(ids) != len(set(ids)):
        raise ProgressValidationError(f"{label} are not unique")
    return ids


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProgressValidationError(f"{label} must be a non-empty string")
    return value


def _count(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgressValidationError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        raise ProgressValidationError(f"{label} is outside its valid range")
    return value


def _percent(numerator: int, denominator: int | None) -> float | None:
    if denominator is None:
        return None
    if denominator == 0:
        return 100.0 if numerator == 0 else None
    return round(100.0 * numerator / denominator, 6)


def _display_percent(value: float | None) -> str:
    return "?" if value is None else f"{value:.3f}%"


def _display_count(value: int | None) -> str:
    return "?" if value is None else f"{value:,}"


def _fraction(numerator: int, denominator: int | None) -> str:
    return f"{numerator:,}/?" if denominator is None else f"{numerator:,}/{denominator:,}"

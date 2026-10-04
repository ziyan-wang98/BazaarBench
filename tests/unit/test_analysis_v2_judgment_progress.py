from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bazaar.analysis_v2.judgment_progress import (
    ProgressValidationError,
    build_progress_report,
    render_progress_table,
    safe_cell_name,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_ndjson(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _registry(path: Path, cell_ids: tuple[str, ...]) -> None:
    _write_json(
        path,
        {
            "independent_record_count": len(cell_ids),
            "cells": [
                {"cell_id": cell_id, "duplicate_of": None} for cell_id in cell_ids
            ]
            + [{"cell_id": "provenance-copy", "duplicate_of": cell_ids[0]}],
        },
    )


def _bundle_ledger(
    root: Path,
    cell_id: str,
    *,
    bundle_count: int,
    decisions: dict[str, int],
) -> None:
    _write_json(
        root / safe_cell_name(cell_id) / "ledger.json",
        {
            "status": "complete",
            "cell_id": cell_id,
            "cell_binding": {"cell_id": cell_id},
            "bundle_count": bundle_count,
            "bundles_by_kind": {"fixture": bundle_count},
            "denominators": decisions,
        },
    )


def _plan(
    root: Path,
    cell_id: str,
    *,
    bundle_ids: tuple[str, ...],
    shard_sizes: tuple[int, ...],
    shard_decisions: tuple[int, ...],
) -> dict[str, Any]:
    assert sum(shard_sizes) == len(bundle_ids)
    assert len(shard_sizes) == len(shard_decisions)
    cursor = 0
    shards = []
    for ordinal, (size, decision_units) in enumerate(
        zip(shard_sizes, shard_decisions, strict=True)
    ):
        end = cursor + size
        shards.append(
            {
                "shard_id": f"shard:{ordinal:06d}:{cursor:09d}-{end:09d}",
                "ordinal": ordinal,
                "start_index": cursor,
                "end_index_exclusive": end,
                "bundle_ids": list(bundle_ids[cursor:end]),
                "decision_units": decision_units,
            }
        )
        cursor = end
    value = {
        "formal_run": True,
        "plan_id": f"plan:{safe_cell_name(cell_id)}",
        "cell_binding": {"cell_id": cell_id},
        "bundle_artifact": {
            "bundle_count": len(bundle_ids),
            "bundle_ids": list(bundle_ids),
        },
        "sharding": {
            "bundle_count": len(bundle_ids),
            "decision_count": sum(shard_decisions),
            "shard_count": len(shards),
            "shards": shards,
        },
    }
    _write_json(root / safe_cell_name(cell_id) / "plan.json", value)
    return value


def _journal_row(
    shard: dict[str, Any],
    *,
    status: str,
    channels: tuple[tuple[str, ...], ...] = (),
    attempts: int = 1,
) -> dict[str, Any]:
    decision = None
    if status == "ok":
        assert len(channels) == len(shard["bundle_ids"])
        decision = {
            "shard_id": shard["shard_id"],
            "shard_complete": True,
            "bundle_results": [
                {
                    "bundle_id": bundle_id,
                    "bundle_complete": True,
                    "decisions": [{"channel": channel} for channel in channel_names],
                }
                for bundle_id, channel_names in zip(
                    shard["bundle_ids"], channels, strict=True
                )
            ],
        }
    return {
        "schema_version": 1,
        "shard_id": shard["shard_id"],
        "ordinal": shard["ordinal"],
        "bundle_ids": shard["bundle_ids"],
        "status": status,
        "attempts": attempts,
        "decision": decision,
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "registry_path": tmp_path / "registry.json",
        "bundles_root": tmp_path / "bundles",
        "judgments_root": tmp_path / "judgments",
        "status_dir": tmp_path / "status",
    }


def test_missing_files_are_waiting_and_prepared_totals_are_exact(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _registry(paths["registry_path"], ("cell:a", "cell:b"))
    _bundle_ledger(
        paths["bundles_root"], "cell:a", bundle_count=3, decisions={"t5_text": 5}
    )
    _bundle_ledger(
        paths["bundles_root"], "cell:b", bundle_count=1, decisions={"reasoning": 1}
    )

    report = build_progress_report(
        **paths,
        expected_cells=2,
        expected_bundles=4,
        expected_decisions=11,
        expected_shards=3,
    )
    payload = report.to_dict()

    assert payload["totals"]["bundles"] == {
        "total": 4,
        "prepared": 4,
        "ok": 0,
        "percent": 0.0,
    }
    assert payload["totals"]["decisions"]["prepared"] == 11
    assert [cell.state for cell in report.cells] == ["waiting", "waiting"]


def test_partial_journal_reports_latest_ok_work_and_errors(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _registry(paths["registry_path"], ("cell:a", "cell:b"))
    _bundle_ledger(
        paths["bundles_root"], "cell:a", bundle_count=3, decisions={"t5_text": 5}
    )
    _bundle_ledger(
        paths["bundles_root"], "cell:b", bundle_count=1, decisions={"reasoning": 1}
    )
    plan = _plan(
        paths["judgments_root"],
        "cell:a",
        bundle_ids=("a:1", "a:2", "a:3"),
        shard_sizes=(2, 1),
        shard_decisions=(3, 2),
    )
    target = paths["judgments_root"] / safe_cell_name("cell:a")
    _write_ndjson(
        target / "shards.ndjson",
        [
            _journal_row(plan["sharding"]["shards"][0], status="transport_error", attempts=2),
            _journal_row(
                plan["sharding"]["shards"][0],
                status="ok",
                channels=(
                    ("T5_externalization_pii", "T6_unverified_trust_claim"),
                    ("T4_premature_closure",),
                ),
            ),
            _journal_row(plan["sharding"]["shards"][1], status="parse_error", attempts=2),
        ],
    )
    _write_json(
        paths["status_dir"] / f"{safe_cell_name('cell:a')}.json",
        {"cell_binding": {"cell_id": "cell:a"}, "status": "error"},
    )

    report = build_progress_report(
        **paths,
        expected_cells=2,
        expected_bundles=4,
        expected_decisions=11,
        expected_shards=3,
    )
    first = report.cells[0]
    totals = report.to_dict()["totals"]

    assert first.state == "error"
    assert (first.ok_shards, first.ok_bundles, first.ok_decisions) == (1, 2, 3)
    assert (first.journal_error_records, first.retry_attempts) == (2, 2)
    assert first.latest_error_shards == 1
    assert totals["shards"]["percent"] == pytest.approx(33.333333)
    assert totals["bundles"]["percent"] == 50.0
    assert totals["decisions"]["percent"] == pytest.approx(27.272727)
    assert totals["errors"] == {
        "journal_records": 2,
        "retry_attempts": 2,
        "latest_shards": 1,
        "status_cells": 1,
    }


def test_complete_cell_requires_matching_journal_records_ledger_and_status(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _registry(paths["registry_path"], ("cell:a",))
    _bundle_ledger(
        paths["bundles_root"], "cell:a", bundle_count=2, decisions={"t5_text": 3}
    )
    plan = _plan(
        paths["judgments_root"],
        "cell:a",
        bundle_ids=("a:1", "a:2"),
        shard_sizes=(2,),
        shard_decisions=(3,),
    )
    target = paths["judgments_root"] / safe_cell_name("cell:a")
    _write_ndjson(
        target / "shards.ndjson",
        [
            _journal_row(
                plan["sharding"]["shards"][0],
                status="ok",
                channels=(
                    ("T5_externalization_pii", "T6_unverified_trust_claim"),
                    ("T4_premature_closure",),
                ),
            )
        ],
    )
    records = [
        {
            "bundle_id": "a:1",
            "shard_id": plan["sharding"]["shards"][0]["shard_id"],
            "status": "ok",
            "decision": {
                "bundle_id": "a:1",
                "bundle_complete": True,
                "decisions": [
                    {"channel": "T5_externalization_pii"},
                    {"channel": "T6_unverified_trust_claim"},
                ],
            },
        },
        {
            "bundle_id": "a:2",
            "shard_id": plan["sharding"]["shards"][0]["shard_id"],
            "status": "ok",
            "decision": {
                "bundle_id": "a:2",
                "bundle_complete": True,
                "decisions": [{"channel": "T4_premature_closure"}],
            },
        },
    ]
    _write_ndjson(target / "records.ndjson", records)
    _write_json(
        target / "ledger.json",
        {
            "status": "complete",
            "plan_id": plan["plan_id"],
            "bundle_count": 2,
            "decision_count": 3,
            "shard_count": 1,
            "shard_statuses": {"ok": 1},
        },
    )
    _write_json(
        paths["status_dir"] / f"{safe_cell_name('cell:a')}.json",
        {
            "cell_binding": {"cell_id": "cell:a"},
            "status": "complete",
            "plan_id": plan["plan_id"],
            "bundle_count": 2,
            "decision_count": 3,
            "shard_count": 1,
        },
    )

    report = build_progress_report(
        **paths,
        expected_cells=1,
        expected_bundles=2,
        expected_decisions=3,
        expected_shards=1,
    )

    assert report.cells[0].state == "complete"
    assert report.to_dict()["totals"]["cells"]["percent"] == 100.0
    assert "decisions 3/3 (100.000%)" in render_progress_table(report)


def test_malformed_observed_journal_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _registry(paths["registry_path"], ("cell:a",))
    _bundle_ledger(
        paths["bundles_root"], "cell:a", bundle_count=1, decisions={"t4_thread": 1}
    )
    _plan(
        paths["judgments_root"],
        "cell:a",
        bundle_ids=("a:1",),
        shard_sizes=(1,),
        shard_decisions=(1,),
    )
    journal = paths["judgments_root"] / safe_cell_name("cell:a") / "shards.ndjson"
    journal.write_text("{", encoding="utf-8")

    with pytest.raises(ProgressValidationError, match="incomplete final row"):
        build_progress_report(
            **paths,
            expected_cells=1,
            expected_bundles=1,
            expected_decisions=1,
            expected_shards=1,
        )


def test_plan_count_contradiction_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _registry(paths["registry_path"], ("cell:a",))
    _bundle_ledger(
        paths["bundles_root"], "cell:a", bundle_count=1, decisions={"t4_thread": 1}
    )
    _plan(
        paths["judgments_root"],
        "cell:a",
        bundle_ids=("a:1",),
        shard_sizes=(1,),
        shard_decisions=(2,),
    )

    with pytest.raises(ProgressValidationError, match="plan decision total contradicts"):
        build_progress_report(
            **paths,
            expected_cells=1,
            expected_bundles=1,
            expected_decisions=1,
            expected_shards=1,
        )

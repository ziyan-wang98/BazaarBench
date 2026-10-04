#!/usr/bin/env python3
"""Extract the frozen BazaarBench analysis universe on the cluster.

The command is read-only with respect to rollout databases.  It writes one
atomic, resumable artifact directory per independent cell so a long 55-record
run can continue after a scheduler or network interruption.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.contract import CellSpec  # noqa: E402
from bazaar.analysis_v2.inventory import replay_inventory  # noqa: E402
from bazaar.analysis_v2.structural import extract_structural_episodes  # noqa: E402
from bazaar.analysis_v2.transactions import replay_transactions  # noqa: E402

SCHEMA_VERSION = 3


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_gz(path: Path, values: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as output:
        for value in values:
            output.write(
                json.dumps(
                    _json_value(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def _cell_from_dict(value: dict[str, Any]) -> CellSpec:
    return CellSpec(
        cell_id=str(value["cell_id"]),
        db_path=Path(value["db_path"]),
        source=str(value["source"]),
        base_model_key=str(value["base_model_key"]),
        treatment_model_key=value.get("treatment_model_key"),
        regime=str(value["regime"]),
        start_tick_exclusive=int(value["start_tick_exclusive"]),
        end_tick_inclusive=int(value["end_tick_inclusive"]),
        treated_agent_ids=tuple(int(item) for item in value["treated_agent_ids"]),
        paired_base_db=(
            Path(value["paired_base_db"]) if value.get("paired_base_db") else None
        ),
        pressure_side=value.get("pressure_side"),
        include_in_main_matrix=bool(value.get("include_in_main_matrix")),
        is_starting_market=bool(value.get("is_starting_market")),
        duplicate_of=value.get("duplicate_of"),
    )


def _safe_cell_name(cell_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in cell_id)


def _connection(path: Path) -> sqlite3.Connection:
    # ``immutable=1`` is appropriate for the frozen rollout snapshot and keeps
    # SQLite from creating ``-wal``/``-shm`` sidecars beside the downloaded DB.
    uri = f"file:{quote(path.resolve().as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _raw_completion_count(conn: sqlite3.Connection, cell: CellSpec) -> int:
    """Count raw closure events that are eligible for canonical replay.

    Older Level-0 databases can contain successful transactions against the
    synthetic seeded catalogue.  ``replay_transactions`` deliberately excludes
    those historical-provenance listings, so the independent raw-event check
    must apply the same eligibility boundary before comparing counts.  It also
    mirrors replay's exact thread/meetup identity fallbacks without editing the
    source database.
    """

    seeded_listing_ids = {
        int(row["listing_id"])
        for row in conn.execute(
            "SELECT listing_id FROM listings WHERE COALESCE(is_seeded, 0) != 0"
        )
    }
    listing_by_thread = {
        int(row["thread_id"]): int(row["listing_id"])
        for row in conn.execute("SELECT thread_id, listing_id FROM threads")
    }
    thread_by_meetup = {
        int(row["meetup_id"]): int(row["thread_id"])
        for row in conn.execute("SELECT meetup_id, thread_id FROM meetups")
    }

    def optional_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    count = 0
    for row in conn.execute(
        """
        SELECT tick, result_status, payload, result_payload
        FROM events
        WHERE action_type='complete_transaction'
          AND tick > ? AND tick <= ?
        """,
        (cell.start_tick_exclusive, cell.end_tick_inclusive),
    ):
        if row["result_status"] != "ok" or not row["result_payload"]:
            continue
        try:
            result = json.loads(row["result_payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(result, dict) or result.get("completed") is not True:
            continue
        if not isinstance(payload, dict):
            payload = {}
        meetup_id = optional_int(result.get("meetup_id")) or optional_int(
            payload.get("meetup_id")
        )
        if meetup_id is None:
            continue
        thread_id = optional_int(result.get("thread_id"))
        if thread_id is None:
            thread_id = thread_by_meetup.get(meetup_id)
        if thread_id is None:
            continue
        listing_id = listing_by_thread.get(thread_id)
        if listing_id is None or listing_id in seeded_listing_ids:
            continue
        count += 1
    return count


def _completed_summary(completed: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "delivery_method": dict(Counter(item.delivery_method for item in completed)),
        "payment_method": dict(Counter(item.payment_method for item in completed)),
        "platform_completion_observed": sum(
            item.platform_completion_observed for item in completed
        ),
        "inspection_observed": sum(item.inspection_observed for item in completed),
        "handoff_proof_present": sum(item.handoff_proof_present for item in completed),
        "handoff_proof_verified": sum(item.handoff_proof_verified for item in completed),
        "platform_completion_before_scheduled_meetup": sum(
            item.platform_completion_before_scheduled_meetup for item in completed
        ),
        "platform_completion_at_or_after_scheduled_meetup": sum(
            item.platform_completion_at_or_after_scheduled_meetup for item in completed
        ),
        "platform_completion_before_eta": sum(
            item.platform_completion_before_eta for item in completed
        ),
        "platform_completion_at_or_after_eta": sum(
            item.platform_completion_at_or_after_eta for item in completed
        ),
        "delivery_evidence_basis": dict(
            Counter(item.delivery_evidence_basis.value for item in completed)
        ),
        "fraud_discovered": sum(item.fraud_event_id is not None for item in completed),
        "treated_party": sum(bool(item.treated_party_ids) for item in completed),
        "price_observed": sum(item.price_observed for item in completed),
        "acquisition_cost_observed": sum(
            item.acquisition_cost_observed for item in completed
        ),
        "inventory_asking_reference_observed": sum(
            item.reference_price_observed for item in completed
        ),
    }


def _extract_cell(cell: CellSpec, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    target = output_root / "cells" / _safe_cell_name(cell.cell_id)
    summary_path = target / "summary.json"
    if summary_path.exists() and not overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("schema_version") == SCHEMA_VERSION:
            return existing | {"resumed": True}
        raise ValueError(f"partial cell output exists; pass --overwrite: {target}")
    if not cell.db_path.is_file():
        raise FileNotFoundError(cell.db_path)

    started = time.monotonic()
    with _connection(cell.db_path) as conn:
        inventory = replay_inventory(conn)
        transactions = replay_transactions(
            conn,
            start_tick_exclusive=cell.start_tick_exclusive,
            end_tick_inclusive=cell.end_tick_inclusive,
            treated_agent_ids=cell.treated_agent_ids,
            inventory=inventory,
        )
        raw_completions = _raw_completion_count(conn, cell)
        if raw_completions != len(transactions.completed):
            raise ValueError(
                f"completion replay mismatch: raw={raw_completions} "
                f"replay={len(transactions.completed)}"
            )
        opportunity_threads = {item.thread_id for item in transactions.opportunities}
        missing = {
            item.thread_id for item in transactions.completed if item.thread_id not in opportunity_threads
        }
        if missing:
            raise ValueError(f"completed threads absent from opportunity set: {sorted(missing)[:10]}")
        structural = extract_structural_episodes(
            conn,
            cell_id=cell.cell_id,
            start_tick_exclusive=cell.start_tick_exclusive,
            end_tick_inclusive=cell.end_tick_inclusive,
            inventory=inventory,
            transactions=transactions,
            treated_agent_ids=cell.treated_agent_ids,
        )

    files = {
        "inventory_units": "inventory_units.jsonl.gz",
        "inventory_links": "inventory_links.jsonl.gz",
        "transaction_opportunities": "transaction_opportunities.jsonl.gz",
        "completed_transactions": "completed_transactions.jsonl.gz",
        "structural_episodes": "structural_episodes.jsonl.gz",
        "structural_opportunity_counts": "structural_opportunity_counts.jsonl.gz",
        "structural_opportunity_keys": "structural_opportunity_keys.jsonl.gz",
    }
    counts = {
        "inventory_units": _write_jsonl_gz(target / files["inventory_units"], inventory.units),
        "inventory_links": _write_jsonl_gz(target / files["inventory_links"], inventory.links),
        "transaction_opportunities": _write_jsonl_gz(
            target / files["transaction_opportunities"], transactions.opportunities
        ),
        "completed_transactions": _write_jsonl_gz(
            target / files["completed_transactions"], transactions.completed
        ),
        "structural_episodes": _write_jsonl_gz(
            target / files["structural_episodes"], structural.episodes
        ),
        "structural_opportunity_counts": _write_jsonl_gz(
            target / files["structural_opportunity_counts"], structural.opportunity_counts
        ),
        "structural_opportunity_keys": _write_jsonl_gz(
            target / files["structural_opportunity_keys"], structural.opportunity_keys
        ),
    }
    link_counts = Counter(link.link_confidence.value for link in inventory.links)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "cell": _json_value(cell),
        "files": files,
        "counts": counts,
        "raw_successful_completion_events": raw_completions,
        "completed_summary": _completed_summary(transactions.completed),
        "inventory_link_confidence": dict(sorted(link_counts.items())),
        "inventory_links_with_deterministic_backing": sum(
            link.create_backing_id is not None for link in inventory.links
        ),
        "inventory_ambiguous_with_deterministic_backing": sum(
            link.link_confidence.value == "ambiguous" and link.create_backing_id is not None
            for link in inventory.links
        ),
        "structural_coverage": structural.coverage,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell", action="append", help="exact cell_id; repeat as needed")
    parser.add_argument(
        "--scope",
        choices=("all", "main", "starting", "supplemental"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    cells = [
        _cell_from_dict(value)
        for value in registry["cells"]
        if value.get("duplicate_of") is None
    ]
    if args.cell:
        selected = set(args.cell)
        cells = [cell for cell in cells if cell.cell_id in selected]
        unknown = selected - {cell.cell_id for cell in cells}
        if unknown:
            parser.error(f"unknown independent cell(s): {', '.join(sorted(unknown))}")
    elif args.scope == "main":
        cells = [cell for cell in cells if cell.include_in_main_matrix]
    elif args.scope == "starting":
        cells = [cell for cell in cells if cell.is_starting_market]
    elif args.scope == "supplemental":
        cells = [cell for cell in cells if cell.regime == "L2X"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, cell in enumerate(cells, start=1):
        print(f"[analysis-v2] extract {index}/{len(cells)} {cell.cell_id}", flush=True)
        try:
            summary = _extract_cell(cell, args.out_dir, overwrite=args.overwrite)
            summaries.append(summary)
            print(
                f"[analysis-v2] complete {cell.cell_id}: "
                f"opportunities={summary['counts']['transaction_opportunities']} "
                f"completed={summary['counts']['completed_transactions']} "
                f"episodes={summary['counts']['structural_episodes']} "
                f"seconds={summary['elapsed_seconds']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - retain cell-level failure ledger
            failure = {
                "cell_id": cell.cell_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[analysis-v2] ERROR {cell.cell_id}: {failure}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "registry": str(args.registry.resolve()),
        "scope": args.scope,
        "requested_cells": len(cells),
        "completed_cells": len(summaries),
        "failed_cells": failures,
        "cell_summaries": [
            {
                "cell_id": summary["cell"]["cell_id"],
                "counts": summary["counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "resumed": summary.get("resumed", False),
            }
            for summary in summaries
        ],
    }
    _write_json(args.out_dir / "extraction_run.json", run_summary)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

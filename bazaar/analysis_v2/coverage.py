"""Streaming row-coverage ledger for the frozen analysis-v2 registry."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .contract import CellSpec


@dataclass(frozen=True)
class TableScanSpec:
    table: str
    tick_column: str
    primary_key: tuple[str, ...]


# This is intentionally explicit: schema additions must not silently expand
# the paper's extraction universe.  State tables without an event-time column
# (for example ``meetups``) are reached through their timeline events instead.
RELEVANT_TABLES: tuple[TableScanSpec, ...] = (
    TableScanSpec("llm_calls", "tick", ("call_id",)),
    TableScanSpec("events", "tick", ("event_id",)),
    TableScanSpec("messages", "tick", ("message_id",)),
    TableScanSpec("listings", "created_at_tick", ("listing_id",)),
    TableScanSpec("threads", "created_at_tick", ("thread_id",)),
    TableScanSpec("offers", "tick", ("offer_id",)),
    TableScanSpec("ratings", "tick", ("rating_id",)),
    TableScanSpec("reports", "tick", ("report_id",)),
    TableScanSpec("photos", "created_at_tick", ("photo_id",)),
    TableScanSpec("blocks", "tick", ("block_id",)),
    TableScanSpec("ledger_entries", "tick", ("entry_id",)),
    TableScanSpec("mental_prices", "tick", ("entry_id",)),
    TableScanSpec("transaction_utility", "accept_tick", ("entry_id",)),
    TableScanSpec("agent_summary", "tick", ("summary_id",)),
    TableScanSpec("self_portraits", "tick", ("portrait_id",)),
    TableScanSpec("snapshots", "tick", ("snapshot_id",)),
    TableScanSpec("narrative_memories", "created_tick", ("memory_id",)),
    TableScanSpec("agents", "created_at_tick", ("agent_id",)),
)

# Nullable columns that later schema migrations add to databases written
# before them (the truthful handoff checks).
# ``bazaar.core.schema.connect`` adds them whenever newer code opens a
# database, so a NULL value is left out of the row digest: the digest of a
# reported database is the same before and after such an open. A non-NULL
# value is digested like any other column.
NULL_OMITTED_DIGEST_COLUMNS: dict[str, frozenset[str]] = {
    "listings": frozenset({"backing_unit_uid"}),
    "meetups": frozenset({"inspection_outcome"}),
}


@dataclass(frozen=True)
class ExtractionDecision:
    """Per-row accounting returned by an extractor callback."""

    bundles_or_candidates_emitted: int = 0
    unknown: bool = False


Extractor = Callable[[Mapping[str, Any]], ExtractionDecision | None]


@dataclass
class CoverageEntry:
    cell_id: str
    db_path: str
    table: str
    extractor: str
    start_tick_exclusive: int
    end_tick_inclusive: int
    rows_seen: int = 0
    rows_eligible: int = 0
    rows_excluded_base_duplicate: int = 0
    decisions: int = 0
    errors: int = 0
    bundles_or_candidates_emitted: int = 0
    unknown: int = 0
    missing_reasoning: int = 0
    ordered_digest: str = ""
    error_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_coverage_entries(
    cells: Sequence[CellSpec],
    *,
    tables: Sequence[TableScanSpec] = RELEVANT_TABLES,
    extractor_name: str = "coverage",
    extractor: Extractor | None = None,
    batch_size: int = 1_000,
) -> Iterator[CoverageEntry]:
    """Yield one bounded-memory coverage record per cell and table."""

    for cell in cells:
        for table in tables:
            yield scan_table(
                cell,
                table,
                extractor_name=extractor_name,
                extractor=extractor,
                batch_size=batch_size,
            )


def scan_table(
    cell: CellSpec,
    table: TableScanSpec,
    *,
    extractor_name: str,
    extractor: Extractor | None = None,
    batch_size: int = 1_000,
) -> CoverageEntry:
    """Stream one locked table/window and return auditable coverage counts.

    For continuation ``llm_calls``, inherited rows are excluded with a SQL
    ``NOT EXISTS`` anti-join against the paired base.  A row is inherited only
    when its fork-stable ``call_id`` *and every common column* agree; this does
    not collapse two legitimate calls that happen to share a prompt hash,
    agent, tick, model, and seed.  Base rows are never materialised in Python.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    entry = CoverageEntry(
        cell_id=cell.cell_id,
        db_path=str(cell.db_path),
        table=table.table,
        extractor=extractor_name,
        start_tick_exclusive=cell.start_tick_exclusive,
        end_tick_inclusive=cell.end_tick_inclusive,
    )
    digest = hashlib.sha256()

    try:
        with _read_only_connection(cell.db_path) as conn:
            columns = _table_columns(conn, "main", table.table)
            _require_scan_columns(columns, table)
            anti_join = ""
            if table.table == "llm_calls" and cell.paired_base_db is not None:
                conn.execute(
                    "ATTACH DATABASE ? AS paired",
                    (_read_only_uri(cell.paired_base_db),),
                )
                base_columns = _table_columns(conn, "paired", table.table)
                if "call_id" not in columns or "call_id" not in base_columns:
                    raise ValueError("llm_calls paired-base anti-join requires call_id")
                common_columns = tuple(column for column in columns if column in base_columns)
                comparisons = " AND ".join(
                    f"b.{_ident(column)} IS c.{_ident(column)}" for column in common_columns
                )
                anti_join = (
                    f" AND NOT EXISTS (SELECT 1 FROM paired.llm_calls AS b WHERE {comparisons})"
                )

            window_sql = f"c.{_ident(table.tick_column)} > ? AND c.{_ident(table.tick_column)} <= ?"
            params = (cell.start_tick_exclusive, cell.end_tick_inclusive)
            entry.rows_seen = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {_ident(table.table)} AS c WHERE {window_sql}",
                    params,
                ).fetchone()[0]
            )
            order_columns = (table.tick_column,) + tuple(
                column for column in table.primary_key if column != table.tick_column
            )
            order_sql = ", ".join(f"c.{_ident(column)}" for column in order_columns)
            cursor = conn.execute(
                f"SELECT c.* FROM {_ident(table.table)} AS c "
                f"WHERE {window_sql}{anti_join} ORDER BY {order_sql}",
                params,
            )
            names = tuple(description[0] for description in cursor.description)
            omit_if_null = NULL_OMITTED_DIGEST_COLUMNS.get(table.table, frozenset())
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                for raw_row in batch:
                    row = dict(zip(names, raw_row, strict=True))
                    entry.rows_eligible += 1
                    _update_ordered_digest(
                        digest, row, table.primary_key, omit_if_null=omit_if_null,
                    )
                    if (
                        table.table == "llm_calls"
                        and not str(row.get("reasoning_summary") or "").strip()
                    ):
                        entry.missing_reasoning += 1
                    try:
                        decision = extractor(row) if extractor is not None else None
                    except Exception as exc:  # noqa: BLE001 - ledger must retain row coverage
                        entry.errors += 1
                        if len(entry.error_messages) < 20:
                            entry.error_messages.append(f"{type(exc).__name__}: {exc}")
                        continue
                    entry.decisions += 1
                    if decision is not None:
                        emitted = decision.bundles_or_candidates_emitted
                        if emitted < 0:
                            raise ValueError("bundles_or_candidates_emitted cannot be negative")
                        entry.bundles_or_candidates_emitted += emitted
                        entry.unknown += int(decision.unknown)
            entry.rows_excluded_base_duplicate = entry.rows_seen - entry.rows_eligible
    except Exception as exc:  # noqa: BLE001 - serialise failures into the ledger
        entry.errors += 1
        if len(entry.error_messages) < 20:
            entry.error_messages.append(f"{type(exc).__name__}: {exc}")

    entry.ordered_digest = f"sha256:{digest.hexdigest()}"
    return entry


def table_specs(names: Sequence[str]) -> tuple[TableScanSpec, ...]:
    """Resolve CLI table names without permitting implicit schema discovery."""

    by_name = {spec.table: spec for spec in RELEVANT_TABLES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown relevant table(s): {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_read_only_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _read_only_uri(path: Path) -> str:
    absolute = path.resolve().as_posix()
    return f"file:{quote(absolute, safe='/')}?mode=ro"


def _table_columns(conn: sqlite3.Connection, schema: str, table: str) -> tuple[str, ...]:
    exists = conn.execute(
        f"SELECT 1 FROM {_ident(schema)}.sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        raise ValueError(f"missing table {schema}.{table}")
    rows = conn.execute(f"PRAGMA {_ident(schema)}.table_info({_ident(table)})").fetchall()
    return tuple(str(row[1]) for row in rows)


def _require_scan_columns(columns: Sequence[str], table: TableScanSpec) -> None:
    required = {table.tick_column, *table.primary_key}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{table.table} missing scan column(s): {', '.join(missing)}")


def _update_ordered_digest(
    digest: Any,
    row: Mapping[str, Any],
    primary_key: Sequence[str],
    *,
    omit_if_null: frozenset[str] = frozenset(),
) -> None:
    """Add one row to the ordered digest. Columns in ``omit_if_null``
    (see :data:`NULL_OMITTED_DIGEST_COLUMNS`) are left out while NULL."""
    payload = {
        "primary_key": [_normalise(row[column]) for column in primary_key],
        "content": {
            key: _normalise(row[key])
            for key in sorted(row)
            if not (key in omit_if_null and row[key] is None)
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _normalise(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            try:
                return {"__json__": _normalise_json(json.loads(stripped))}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return text
    return {"__repr__": repr(value)}


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalise_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    return _normalise(value)


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'

#!/usr/bin/env python3
"""Audit SQLite rollout DBs for LLM backend/API-error contamination."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ERROR_PATTERNS = (
    "__backend_error__",
    "backend_error:",
    "llm_backend_error",
    "invalid_api_key",
    "incorrect api key",
    "api key provided",
)


@dataclass
class _PatternStats:
    count: int = 0
    min_tick: int | None = None
    max_tick: int | None = None

    def add(self, tick: int | None) -> None:
        self.count += 1
        if tick is None:
            return
        self.min_tick = tick if self.min_tick is None else min(self.min_tick, tick)
        self.max_tick = tick if self.max_tick is None else max(self.max_tick, tick)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _scan_text_column(
    conn: sqlite3.Connection,
    *,
    path: Path,
    table: str,
    column: str,
    min_tick: int | None = None,
) -> list[str]:
    issues: list[str] = []
    table_cols = _columns(conn, table)
    tick_expr = "tick" if "tick" in table_cols else "NULL AS tick"
    where_parts = [f"{column} IS NOT NULL"]
    params: list[int] = []
    if min_tick is not None and "tick" in table_cols:
        where_parts.append("tick >= ?")
        params.append(int(min_tick))
    where_sql = " AND ".join(where_parts)
    stats = {pattern: _PatternStats() for pattern in ERROR_PATTERNS}
    rows = conn.execute(
        f"SELECT {tick_expr}, {column} AS scanned_text FROM {table} "
        f"WHERE {where_sql}",
        params,
    )
    for row in rows:
        text = str(row["scanned_text"]).lower()
        if not text:
            continue
        tick = row["tick"]
        tick_int = int(tick) if tick is not None else None
        for pattern, pattern_stats in stats.items():
            if pattern in text:
                pattern_stats.add(tick_int)

    for pattern, pattern_stats in stats.items():
        if pattern_stats.count == 0:
            continue
        tick_part = (
            f" ticks {pattern_stats.min_tick}..{pattern_stats.max_tick}"
            if pattern_stats.min_tick is not None and pattern_stats.max_tick is not None
            else ""
        )
        issues.append(
            f"{path}: {table}.{column} contains {pattern!r} "
            f"in {pattern_stats.count} row(s){tick_part}"
        )
    return issues


def scan_db(
    path: Path,
    *,
    include_prompts: bool = False,
    min_tick: int | None = None,
) -> list[str]:
    """Return backend/API-error contamination issues for one rollout DB."""
    if not path.exists():
        return [f"{path}: file does not exist"]
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return [f"{path}: could not open SQLite DB: {exc}"]

    issues: list[str] = []
    try:
        if _table_exists(conn, "llm_calls"):
            llm_cols = _columns(conn, "llm_calls")
            llm_text_columns = ["response_text", "reasoning_summary"]
            if include_prompts:
                llm_text_columns.append("prompt_text")
            for column in llm_text_columns:
                if column in llm_cols:
                    issues.extend(
                        _scan_text_column(
                            conn,
                            path=path,
                            table="llm_calls",
                            column=column,
                            min_tick=min_tick,
                        )
                    )
        if _table_exists(conn, "events"):
            event_cols = _columns(conn, "events")
            for column in ("payload", "result_payload"):
                if column in event_cols:
                    issues.extend(
                        _scan_text_column(
                            conn,
                            path=path,
                            table="events",
                            column=column,
                            min_tick=min_tick,
                        )
                    )
    except sqlite3.Error as exc:
        issues.append(f"{path}: could not scan DB for backend errors: {exc}")
    finally:
        conn.close()
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path, nargs="+", help="SQLite rollout DB path(s)")
    ap.add_argument(
        "--include-prompts",
        action="store_true",
        help=(
            "Also scan llm_calls.prompt_text. This is much slower on large "
            "paper rollouts; response_text and events are enough to detect "
            "backend/API failures."
        ),
    )
    ap.add_argument(
        "--min-tick",
        type=int,
        default=None,
        help="Only scan rows with tick >= this value.",
    )
    args = ap.parse_args(argv)

    issues: list[str] = []
    for db in args.db:
        issues.extend(
            scan_db(
                db,
                include_prompts=args.include_prompts,
                min_tick=args.min_tick,
            )
        )

    if issues:
        print(f"[audit-llm-backend-errors] FAIL: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 2
    print(f"[audit-llm-backend-errors] OK: {len(args.db)} DB(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Restore a tick-0 cold-start base DB from the archived warmup DB.

The public cold-start artifact contains the original persona/task seed plus
an old qwen3.6 warmup through positive ticks.  Level-0 reruns need the same
seeded personas and negative/history rows, but no positive-tick rollout state.
This script copies the artifact and strips positive-tick state from the copy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

DELETE_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("transaction_utility", "accept_tick > 0", ("accept_tick",)),
    ("messages", "tick > 0", ("tick",)),
    ("ratings", "tick > 0", ("tick",)),
    ("meetups", "scheduled_tick > 0", ("scheduled_tick",)),
    ("offers", "tick > 0", ("tick",)),
    ("threads", "created_at_tick > 0", ("created_at_tick",)),
    ("listings", "created_at_tick > 0", ("created_at_tick",)),
    ("events", "tick > 0", ("tick",)),
    ("llm_calls", "tick > 0", ("tick",)),
    ("agent_summary", "tick > 0", ("tick",)),
    ("narrative_memories", "created_tick > 0", ("created_tick",)),
    ("mental_prices", "tick > 0", ("tick",)),
    ("photos", "created_at_tick > 0", ("created_at_tick",)),
    ("reports", "tick > 0", ("tick",)),
    ("self_portraits", "tick > 0", ("tick",)),
    ("snapshots", "tick > 0", ("tick",)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip VACUUM after deleting old positive-tick state.",
    )
    args = parser.parse_args()

    summary = restore_cold_start_base(
        source=args.source,
        out=args.out,
        force=args.force,
        vacuum=not args.no_vacuum,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def restore_cold_start_base(
    *,
    source: Path,
    out: Path,
    force: bool = False,
    vacuum: bool = True,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    out = out.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source == out:
        raise ValueError("--source and --out must be different paths")
    if out.exists():
        if not force:
            raise FileExistsError(out)
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out)

    conn = sqlite3.connect(out)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        before = _db_summary(conn)
        deleted: dict[str, int] = {}
        updates: dict[str, int] = {}
        with conn:
            for table, where, required_columns in DELETE_SPECS:
                if not _has_columns(conn, table, required_columns):
                    continue
                deleted[table] = _delete_where(conn, table, where)
            updates.update(_restore_initial_listing_state(conn))
            updates.update(_restore_initial_persona_inventory_state(conn))
            _write_restore_meta(
                conn,
                source=source,
                deleted=deleted,
                updates=updates,
            )
        if vacuum:
            conn.execute("VACUUM")
        after = _db_summary(conn)
        return {
            "source": str(source),
            "out": str(out),
            "vacuum": vacuum,
            "before": before,
            "after": after,
            "deleted": deleted,
            "updates": updates,
        }
    finally:
        conn.close()


def _delete_where(conn: sqlite3.Connection, table: str, where: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()
    count = int(row[0] if row else 0)
    if count:
        conn.execute(f"DELETE FROM {table} WHERE {where}")
    return count


def _restore_initial_listing_state(conn: sqlite3.Connection) -> dict[str, int]:
    if not _has_columns(
        conn,
        "listings",
        (
            "created_at_tick",
            "status",
            "sold_at_tick",
            "last_bumped_tick",
            "view_count",
            "save_count",
            "inquiry_count",
        ),
    ):
        return {}
    cur = conn.execute(
        """
        UPDATE listings
        SET status = 'active',
            sold_at_tick = NULL,
            last_bumped_tick = NULL,
            view_count = 0,
            save_count = 0,
            inquiry_count = 0
        WHERE created_at_tick = 0
        """
    )
    cur2 = conn.execute(
        """
        UPDATE listings
        SET last_bumped_tick = NULL
        WHERE last_bumped_tick > 0
        """
    )
    return {
        "initial_listings_reset": int(cur.rowcount),
        "positive_bumps_cleared": int(cur2.rowcount),
    }


def _restore_initial_persona_inventory_state(conn: sqlite3.Connection) -> dict[str, int]:
    if not _has_columns(conn, "agents", ("agent_id", "persona_json")):
        return {}
    rows = conn.execute("SELECT agent_id, persona_json FROM agents").fetchall()
    personas_updated = 0
    bought_items_removed = 0
    sold_markers_cleared = 0
    for agent_id, raw in rows:
        try:
            persona = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        inv = persona.get("inventory_items")
        if not isinstance(inv, list):
            continue
        restored: list[Any] = []
        changed = False
        for item in inv:
            if not isinstance(item, dict):
                restored.append(item)
                continue
            if item.get("source") == "bought" or item.get("bought_from_listing_id") is not None:
                bought_items_removed += 1
                changed = True
                continue
            cleaned = dict(item)
            for key in ("sold_at_tick", "sold_listing_id", "sold_price_cents"):
                if key in cleaned:
                    cleaned.pop(key, None)
                    sold_markers_cleared += 1
                    changed = True
            restored.append(cleaned)
        if not changed:
            continue
        persona["inventory_items"] = restored
        conn.execute(
            "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
            (json.dumps(persona, sort_keys=True), agent_id),
        )
        personas_updated += 1
    return {
        "personas_inventory_reset": personas_updated,
        "bought_inventory_items_removed": bought_items_removed,
        "sold_inventory_markers_cleared": sold_markers_cleared,
    }


def _write_restore_meta(
    conn: sqlite3.Connection,
    *,
    source: Path,
    deleted: dict[str, int],
    updates: dict[str, int],
) -> None:
    if not _has_columns(conn, "meta", ("key", "value")):
        return
    payload = {
        "restored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_db": str(source),
        "deleted_positive_tick_rows": deleted,
        "updates": updates,
        "note": (
            "Restored from archived cold-start DB by preserving seeded personas, "
            "initial listings, negative/history rows, and tick-0 summaries while "
            "removing old positive-tick qwen3.6 rollout state."
        ),
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("level0_restore_summary", json.dumps(payload, sort_keys=True)),
    )


def _db_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "agents": _count(conn, "agents"),
        "active_benign_agents": _count_where(
            conn,
            "agents",
            "is_seeded = 0 AND is_redteam = 0 AND status = 'active'",
        ),
        "max_event_tick": _scalar(
            conn,
            "events",
            "SELECT COALESCE(MAX(tick), -1) FROM events",
        ),
        "events": _count(conn, "events"),
        "llm_calls": _count(conn, "llm_calls"),
        "messages": _count(conn, "messages"),
        "offers": _count(conn, "offers"),
        "threads": _count(conn, "threads"),
        "listings": _count(conn, "listings"),
        "active_initial_listings": _count_where(
            conn,
            "listings",
            "created_at_tick = 0 AND status = 'active'",
        ),
        "positive_tick_events": _count_where(conn, "events", "tick > 0"),
        "positive_created_listings": _count_where(
            conn,
            "listings",
            "created_at_tick > 0",
        ),
        "positive_llm_calls": _count_where(conn, "llm_calls", "tick > 0"),
    }


def _count(conn: sqlite3.Connection, table: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0] if row else 0)


def _count_where(conn: sqlite3.Connection, table: str, where: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()
    return int(row[0] if row else 0)


def _scalar(conn: sqlite3.Connection, table: str, sql: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    row = conn.execute(sql).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _has_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    if not _table_exists(conn, table):
        return False
    have = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    return all(column in have for column in columns)


if __name__ == "__main__":
    main()

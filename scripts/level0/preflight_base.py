#!/usr/bin/env python3
"""Preflight a Level-0 cold-start base DB before any TRAPI rollout."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--seed-plan", type=Path)
    parser.add_argument("--expected-agents", type=int, default=100)
    parser.add_argument("--min-initial-listings", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    report = preflight_base(
        args.db,
        seed_plan_path=args.seed_plan,
        expected_agents=args.expected_agents,
        min_initial_listings=args.min_initial_listings,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


def preflight_base(
    db: Path,
    *,
    seed_plan_path: Path | None = None,
    expected_agents: int = 100,
    min_initial_listings: int = 100,
) -> dict[str, Any]:
    if not db.exists():
        raise FileNotFoundError(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        seed_plan = _load_seed_plan(conn, seed_plan_path=seed_plan_path)
        metrics = {
            "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
            "event_state": _event_state(conn),
            "agents": _agent_state(conn),
            "inventory": _inventory_state(conn, seed_plan=seed_plan),
            "initial_listings": _initial_listing_state(conn),
            "history": _history_state(conn),
            "seed_plan": _seed_plan_state(seed_plan),
        }
        checks = {
            "quick_check_ok": metrics["quick_check"] == "ok",
            "expected_active_agents": (
                metrics["agents"]["active_benign_agents"] == expected_agents
            ),
            "no_redteam_agents": metrics["agents"]["redteam_agents"] == 0,
            "baseline_has_no_deadlines": metrics["agents"]["deadline_count"] == 0,
            "baseline_has_no_financial_stress": (
                metrics["agents"]["financial_stress_count"] == 0
            ),
            "no_positive_rollout_state": (
                metrics["event_state"]["positive_events"] == 0
                and metrics["event_state"]["positive_llm_calls"] == 0
                and metrics["event_state"]["positive_messages"] == 0
                and metrics["event_state"]["positive_offers"] == 0
                and metrics["event_state"]["positive_created_listings"] == 0
            ),
            "initial_listings_present": (
                metrics["initial_listings"]["active_initial_listings"]
                >= min_initial_listings
            ),
            "initial_listings_cover_agents": (
                metrics["initial_listings"]["active_initial_listing_owners"]
                >= expected_agents
            ),
            "no_rollout_inventory_sources": (
                metrics["inventory"]["bought_inventory_items"] == 0
                and metrics["inventory"]["restock_inventory_items"] == 0
            ),
            "no_sold_inventory_markers": (
                metrics["inventory"]["sold_tagged_inventory_items"] == 0
            ),
            "seed_plan_inventory_matches_personas": (
                metrics["inventory"]["seed_plan_inventory_mismatch_agents"] == 0
            ),
        }
        warnings = _warnings(metrics)
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "warnings": warnings,
            "db": str(db),
            "seed_plan_path": str(seed_plan_path) if seed_plan_path else None,
            "expected_agents": expected_agents,
            "min_initial_listings": min_initial_listings,
            "metrics": metrics,
        }
    finally:
        conn.close()


def _load_seed_plan(
    conn: sqlite3.Connection,
    *,
    seed_plan_path: Path | None,
) -> list[dict[str, Any]]:
    if seed_plan_path is not None:
        return json.loads(seed_plan_path.read_text(encoding="utf-8"))
    if _has_columns(conn, "meta", ("key", "value")):
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'cold_start_seed_plan'"
        ).fetchone()
        if row is not None:
            return json.loads(row["value"])
    return []


def _event_state(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "max_event_tick": _scalar(
            conn,
            """
            SELECT COALESCE(MAX(tick), -1)
            FROM events
            WHERE action_type NOT IN ('experiment_config', 'experiment_run_complete')
            """,
        ),
        "positive_events": _count_where(conn, "events", "tick > 0"),
        "positive_llm_calls": _count_where(conn, "llm_calls", "tick > 0"),
        "positive_messages": _count_where(conn, "messages", "tick > 0"),
        "positive_offers": _count_where(conn, "offers", "tick > 0"),
        "positive_created_listings": _count_where(
            conn,
            "listings",
            "created_at_tick > 0",
        ),
    }


def _agent_state(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _has_columns(conn, "agents", ("agent_id", "persona_json")):
        return {}
    rows = conn.execute(
        """
        SELECT agent_id, persona_json, is_seeded, is_redteam, status
        FROM agents
        ORDER BY agent_id
        """
    ).fetchall()
    active_benign = 0
    redteam = 0
    deadline_count = 0
    financial_stress_count = 0
    agency_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for row in rows:
        if int(row["is_seeded"] or 0) != 0:
            continue
        if str(row["status"] or "") != "active":
            continue
        active_benign += 1
        redteam += int(row["is_redteam"] or 0)
        try:
            persona = json.loads(row["persona_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        deadline_count += int(persona.get("deadline") is not None)
        financial_stress_count += int(persona.get("financial_stress") is not None)
        agency_counts[str(persona.get("agency_mode") or "unknown")] += 1
        tier_counts[_persona_tier(persona)] += 1
    return {
        "active_benign_agents": active_benign,
        "redteam_agents": redteam,
        "deadline_count": deadline_count,
        "financial_stress_count": financial_stress_count,
        "agency_counts": dict(agency_counts),
        "tier_counts": dict(tier_counts),
    }


def _inventory_state(
    conn: sqlite3.Connection,
    *,
    seed_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    if not _has_columns(conn, "agents", ("agent_id", "persona_json")):
        return {}
    seed_by_agent = {int(row["agent_id"]): row for row in seed_plan if "agent_id" in row}
    rows = conn.execute(
        """
        SELECT agent_id, persona_json
        FROM agents
        WHERE is_seeded = 0
        ORDER BY agent_id
        """
    ).fetchall()
    total = bought = restock = sold = 0
    mismatch_agents: list[int] = []
    for row in rows:
        aid = int(row["agent_id"])
        try:
            persona = json.loads(row["persona_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            mismatch_agents.append(aid)
            continue
        inv = [item for item in persona.get("inventory_items") or [] if isinstance(item, dict)]
        total += len(inv)
        bought += sum(1 for item in inv if item.get("source") == "bought")
        restock += sum(1 for item in inv if item.get("source") == "restock")
        sold += sum(1 for item in inv if item.get("sold_at_tick") is not None)
        seed = seed_by_agent.get(aid)
        if seed is None:
            continue
        expected = {_seed_unique_id(item) for item in seed.get("inventory") or []}
        actual = {_inventory_unique_id(item) for item in inv}
        if expected != actual:
            mismatch_agents.append(aid)
    return {
        "inventory_items": total,
        "bought_inventory_items": bought,
        "restock_inventory_items": restock,
        "sold_tagged_inventory_items": sold,
        "seed_plan_inventory_mismatch_agents": len(mismatch_agents),
        "seed_plan_inventory_mismatch_agent_ids": mismatch_agents[:20],
    }


def _initial_listing_state(conn: sqlite3.Connection) -> dict[str, Any]:
    active_initial = _count_where(
        conn,
        "listings",
        "created_at_tick = 0 AND status = 'active'",
    )
    owners = 0
    if _has_columns(conn, "listings", ("owner_agent_id", "created_at_tick", "status")):
        owners = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT owner_agent_id)
            FROM listings
            WHERE created_at_tick = 0
              AND status = 'active'
              AND owner_agent_id IS NOT NULL
            """,
        )
    return {
        "active_initial_listings": active_initial,
        "active_initial_listing_owners": owners,
    }


def _history_state(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "negative_threads": _count_where(conn, "threads", "created_at_tick < 0"),
        "negative_offers": _count_where(conn, "offers", "tick < 0"),
        "negative_ratings": _count_where(conn, "ratings", "tick < 0"),
    }


def _seed_plan_state(seed_plan: list[dict[str, Any]]) -> dict[str, Any]:
    source_datasets: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    inventory_count = 0
    history_count = 0
    for row in seed_plan:
        tier_counts[str(row.get("tier") or "unknown")] += 1
        inventory = row.get("inventory") or []
        history = row.get("history") or []
        inventory_count += len(inventory)
        history_count += len(history)
        for item in inventory + history + [row.get("buyer_target") or {}]:
            sid = str(item.get("source_row_id") or "")
            if ":" in sid:
                source_datasets[sid.split(":", 1)[0]] += 1
    return {
        "agents": len(seed_plan),
        "inventory_items": inventory_count,
        "history_items": history_count,
        "source_dataset_counts": dict(source_datasets),
        "source_dataset_count": len(source_datasets),
        "tier_counts": dict(tier_counts),
    }


def _warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    seed_state = metrics.get("seed_plan") or {}
    if int(seed_state.get("source_dataset_count") or 0) < 2:
        warnings.append("seed plan uses fewer than two source datasets")
    history = metrics.get("history") or {}
    if int(history.get("negative_threads") or 0) == 0:
        warnings.append("base has no negative-tick history threads")
    return warnings


def _seed_unique_id(item: dict[str, Any]) -> str:
    raw = str(item.get("source_row_id") or "")
    if ":" in raw:
        return raw.split(":", 1)[1]
    return str(item.get("unique_id") or item.get("title") or "")


def _inventory_unique_id(item: dict[str, Any]) -> str:
    attrs = item.get("dataset_attrs")
    if isinstance(attrs, dict) and attrs.get("unique_id"):
        return str(attrs["unique_id"])
    return str(item.get("unique_id") or item.get("title") or "")


def _persona_tier(persona: dict[str, Any]) -> str:
    cold_start = persona.get("cold_start")
    if isinstance(cold_start, dict):
        tier = cold_start.get("tier")
        if isinstance(tier, str) and tier.strip():
            return tier.strip()
    return "unknown"


def _count_where(conn: sqlite3.Connection, table: str, where: str) -> int:
    if not _table_exists(conn, table):
        return 0
    try:
        return _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE {where}")
    except sqlite3.OperationalError:
        return 0


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _has_columns(conn: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> bool:
    if not _table_exists(conn, table):
        return False
    have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return all(column in have for column in columns)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check the Level-0 rollout gates."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--min-tick", type=int, default=24)
    parser.add_argument("--base-tick", type=int, default=0)
    parser.add_argument("--max-backend-errors", type=int, default=0)
    parser.add_argument("--max-positive-history-thread-ratings", type=int, default=0)
    parser.add_argument("--max-base-bought-inventory-items", type=int)
    parser.add_argument("--min-messages-per-offer", type=float)
    parser.add_argument("--max-top-ok-action-share", type=float)
    parser.add_argument("--max-rate-ok-action-share", type=float)
    parser.add_argument("--min-market-risk-signals", type=int)
    parser.add_argument("--min-interactive-risk-signals", type=int)
    parser.add_argument("--min-offer-category-entropy", type=float)
    parser.add_argument("--min-offer-proposer-count", type=int)
    parser.add_argument("--min-offer-seller-count", type=int)
    parser.add_argument("--min-offer-listing-count", type=int)
    parser.add_argument("--min-non-do-nothing-rate", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    report = check_gate(
        args.db,
        min_tick=args.min_tick,
        base_tick=args.base_tick,
        max_backend_errors=args.max_backend_errors,
        max_positive_history_thread_ratings=args.max_positive_history_thread_ratings,
        max_base_bought_inventory_items=args.max_base_bought_inventory_items,
        min_messages_per_offer=args.min_messages_per_offer,
        max_top_ok_action_share=args.max_top_ok_action_share,
        max_rate_ok_action_share=args.max_rate_ok_action_share,
        min_market_risk_signals=args.min_market_risk_signals,
        min_interactive_risk_signals=args.min_interactive_risk_signals,
        min_offer_category_entropy=args.min_offer_category_entropy,
        min_offer_proposer_count=args.min_offer_proposer_count,
        min_offer_seller_count=args.min_offer_seller_count,
        min_offer_listing_count=args.min_offer_listing_count,
        min_non_do_nothing_rate=args.min_non_do_nothing_rate,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


def check_gate(
    db: Path,
    *,
    min_tick: int = 24,
    base_tick: int = 0,
    max_backend_errors: int = 0,
    max_positive_history_thread_ratings: int = 0,
    max_base_bought_inventory_items: int | None = None,
    min_messages_per_offer: float | None = None,
    max_top_ok_action_share: float | None = None,
    max_rate_ok_action_share: float | None = None,
    min_market_risk_signals: int | None = None,
    min_interactive_risk_signals: int | None = None,
    min_offer_category_entropy: float | None = None,
    min_offer_proposer_count: int | None = None,
    min_offer_seller_count: int | None = None,
    min_offer_listing_count: int | None = None,
    min_non_do_nothing_rate: float = 0.05,
) -> dict[str, Any]:
    if not db.exists():
        raise FileNotFoundError(db)
    conn = sqlite3.connect(db)
    try:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        max_tick = _scalar(
            conn,
            """
            SELECT COALESCE(MAX(tick), -1)
            FROM events
            WHERE action_type NOT IN ('experiment_config', 'experiment_run_complete')
            """,
        )
        messages = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE tick > ?", base_tick)
        offers = _scalar(conn, "SELECT COUNT(*) FROM offers WHERE tick > ?", base_tick)
        threads = _scalar(
            conn,
            "SELECT COUNT(*) FROM threads WHERE created_at_tick > ?",
            base_tick,
        )
        handler_errors = _scalar(
            conn,
            "SELECT COUNT(*) FROM events WHERE tick > ? AND result_status = 'error'",
            base_tick,
        )
        policy_errors = _scalar(
            conn,
            "SELECT COUNT(*) FROM events WHERE tick > ? AND action_type = 'policy_error'",
            base_tick,
        )
        backend_errors = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM llm_calls
            WHERE tick > ?
              AND (
                response_text LIKE '%__backend_error__%'
                OR response_text LIKE '%backend_error:%'
                OR reasoning_summary LIKE '%backend_error:%'
              )
            """,
            base_tick,
        )
        agent_events = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM events
            WHERE tick > ? AND agent_id IS NOT NULL
            """,
            base_tick,
        )
        non_do_nothing = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM events
            WHERE tick > ?
              AND agent_id IS NOT NULL
              AND action_type != 'do_nothing'
            """,
            base_tick,
        )
        llm_calls = _scalar(conn, "SELECT COUNT(*) FROM llm_calls WHERE tick > ?", base_tick)
        positive_history_thread_ratings = _positive_history_thread_ratings(
            conn,
            base_tick=base_tick,
        )
        base_bought_inventory_items = _base_bought_inventory_items(
            conn,
            base_tick=base_tick,
        )
        ok_action_counts = _ok_action_counts(conn, base_tick=base_tick)
        market_risk_signals = _market_risk_signals(conn, base_tick=base_tick)
        interactive_risk_signals = _interactive_risk_signals(
            conn,
            base_tick=base_tick,
        )
        offer_trade_diversity = _offer_trade_diversity(conn, base_tick=base_tick)
    finally:
        conn.close()

    non_do_nothing_rate = (
        float(non_do_nothing) / float(agent_events)
        if agent_events
        else 0.0
    )
    messages_per_offer = float(messages) / float(offers) if offers else 0.0
    ok_total = sum(ok_action_counts.values())
    top_ok_action = ""
    top_ok_action_count = 0
    if ok_action_counts:
        top_ok_action, top_ok_action_count = max(
            ok_action_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
    top_ok_action_share = (
        float(top_ok_action_count) / float(ok_total)
        if ok_total
        else 0.0
    )
    rate_ok_action_share = (
        float(ok_action_counts.get("rate", 0)) / float(ok_total)
        if ok_total
        else 0.0
    )
    checks = {
        "reached_min_tick": max_tick >= min_tick,
        "messages_positive": messages > 0,
        "offers_positive": offers > 0,
        "threads_positive": threads > 0,
        "backend_errors_within_threshold": backend_errors <= max_backend_errors,
        "handler_errors_zero": handler_errors == 0,
        "policy_errors_zero": policy_errors == 0,
        "positive_history_thread_ratings_within_threshold": (
            positive_history_thread_ratings <= max_positive_history_thread_ratings
        ),
        "quick_check_ok": quick_check == "ok",
        "non_do_nothing_rate_nontrivial": (
            non_do_nothing_rate >= min_non_do_nothing_rate
        ),
    }
    if max_base_bought_inventory_items is not None:
        checks["base_bought_inventory_items_within_threshold"] = (
            base_bought_inventory_items <= max_base_bought_inventory_items
        )
    if min_messages_per_offer is not None:
        checks["messages_per_offer_within_threshold"] = (
            messages_per_offer >= min_messages_per_offer
        )
    if max_top_ok_action_share is not None:
        checks["top_ok_action_share_within_threshold"] = (
            top_ok_action_share <= max_top_ok_action_share
        )
    if max_rate_ok_action_share is not None:
        checks["rate_ok_action_share_within_threshold"] = (
            rate_ok_action_share <= max_rate_ok_action_share
        )
    if min_market_risk_signals is not None:
        checks["market_risk_signals_within_threshold"] = (
            market_risk_signals >= min_market_risk_signals
        )
    if min_interactive_risk_signals is not None:
        checks["interactive_risk_signals_within_threshold"] = (
            interactive_risk_signals >= min_interactive_risk_signals
        )
    if min_offer_category_entropy is not None:
        checks["offer_category_entropy_within_threshold"] = (
            offer_trade_diversity["offer_category_entropy"]
            >= min_offer_category_entropy
        )
    if min_offer_proposer_count is not None:
        checks["offer_proposer_count_within_threshold"] = (
            offer_trade_diversity["offer_proposer_count"]
            >= min_offer_proposer_count
        )
    if min_offer_seller_count is not None:
        checks["offer_seller_count_within_threshold"] = (
            offer_trade_diversity["offer_seller_count"] >= min_offer_seller_count
        )
    if min_offer_listing_count is not None:
        checks["offer_listing_count_within_threshold"] = (
            offer_trade_diversity["offer_listing_count"] >= min_offer_listing_count
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "db": str(db),
        "base_tick": base_tick,
        "min_tick": min_tick,
        "max_backend_errors": max_backend_errors,
        "max_positive_history_thread_ratings": max_positive_history_thread_ratings,
        "max_base_bought_inventory_items": max_base_bought_inventory_items,
        "min_messages_per_offer": min_messages_per_offer,
        "max_top_ok_action_share": max_top_ok_action_share,
        "max_rate_ok_action_share": max_rate_ok_action_share,
        "min_market_risk_signals": min_market_risk_signals,
        "min_interactive_risk_signals": min_interactive_risk_signals,
        "min_offer_category_entropy": min_offer_category_entropy,
        "min_offer_proposer_count": min_offer_proposer_count,
        "min_offer_seller_count": min_offer_seller_count,
        "min_offer_listing_count": min_offer_listing_count,
        "min_non_do_nothing_rate": min_non_do_nothing_rate,
        "metrics": {
            "max_tick": max_tick,
            "messages": messages,
            "offers": offers,
            "threads": threads,
            "handler_errors": handler_errors,
            "policy_errors": policy_errors,
            "backend_errors": backend_errors,
            "positive_history_thread_ratings": positive_history_thread_ratings,
            "base_bought_inventory_items": base_bought_inventory_items,
            "agent_events": agent_events,
            "non_do_nothing": non_do_nothing,
            "non_do_nothing_rate": round(non_do_nothing_rate, 4),
            "messages_per_offer": round(messages_per_offer, 4),
            "ok_action_total": ok_total,
            "top_ok_action": top_ok_action,
            "top_ok_action_count": top_ok_action_count,
            "top_ok_action_share": round(top_ok_action_share, 4),
            "rate_ok_action_share": round(rate_ok_action_share, 4),
            "market_risk_signals": market_risk_signals,
            "interactive_risk_signals": interactive_risk_signals,
            **offer_trade_diversity,
            "llm_calls": llm_calls,
            "quick_check": quick_check,
        },
    }


def _scalar(conn: sqlite3.Connection, sql: str, *params: Any) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def _positive_history_thread_ratings(
    conn: sqlite3.Connection,
    *,
    base_tick: int,
) -> int:
    if not _has_columns(conn, "ratings", ("tick", "thread_id")):
        return 0
    if not _has_columns(conn, "threads", ("thread_id", "created_at_tick")):
        return 0
    return _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM ratings r
        JOIN threads t ON t.thread_id = r.thread_id
        WHERE r.tick > ?
          AND t.created_at_tick < 0
        """,
        base_tick,
    )


def _base_bought_inventory_items(
    conn: sqlite3.Connection,
    *,
    base_tick: int,
) -> int:
    if not _has_columns(conn, "agents", ("persona_json", "created_at_tick")):
        return 0
    rows = conn.execute(
        """
        SELECT persona_json
        FROM agents
        WHERE created_at_tick <= 0
        """
    ).fetchall()
    total = 0
    for row in rows:
        raw = row[0]
        try:
            persona = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        for item in persona.get("inventory_items") or []:
            if _is_base_bought_inventory_item(item, base_tick=base_tick):
                total += 1
    return total


def _is_base_bought_inventory_item(item: Any, *, base_tick: int) -> bool:
    if not isinstance(item, dict) or item.get("source") != "bought":
        return False
    bought_tick = item.get("bought_tick")
    if bought_tick is None:
        return True
    try:
        return int(bought_tick) <= base_tick
    except (TypeError, ValueError):
        return True


def _ok_action_counts(conn: sqlite3.Connection, *, base_tick: int) -> dict[str, int]:
    if not _has_columns(conn, "events", ("tick", "action_type", "result_status")):
        return {}
    rows = conn.execute(
        """
        SELECT action_type, COUNT(*) AS n
        FROM events
        WHERE tick > ?
          AND agent_id IS NOT NULL
          AND result_status = 'ok'
        GROUP BY action_type
        """,
        (base_tick,),
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _market_risk_signals(conn: sqlite3.Connection, *, base_tick: int) -> int:
    return (
        _speculative_live_listings(conn, base_tick=base_tick)
        + _overstated_live_listings(conn, base_tick=base_tick)
        + _interactive_risk_signals(conn, base_tick=base_tick)
    )


def _interactive_risk_signals(conn: sqlite3.Connection, *, base_tick: int) -> int:
    return (
        _phantom_listing_offers(conn, base_tick=base_tick)
        + _photo_risk_signals(conn, base_tick=base_tick)
        + _report_risk_signals(conn, base_tick=base_tick)
        + _count_action(conn, base_tick=base_tick, action_type="fraud_discovered")
        + _count_action(conn, base_tick=base_tick, action_type="report_user")
        + _count_action(conn, base_tick=base_tick, action_type="report_listing")
        + _count_action(conn, base_tick=base_tick, action_type="block_user")
        + _count_action(conn, base_tick=base_tick, action_type="request_photo")
        + _count_action(conn, base_tick=base_tick, action_type="send_photo")
        + _count_action(conn, base_tick=base_tick, action_type="send_stock_photo")
        + _count_action(conn, base_tick=base_tick, action_type="send_crafted_photo")
        + _count_action(conn, base_tick=base_tick, action_type="leave_thread")
    )


def _offer_trade_diversity(conn: sqlite3.Connection, *, base_tick: int) -> dict[str, Any]:
    out = {
        "offer_proposer_count": 0,
        "offer_seller_count": 0,
        "offer_listing_count": 0,
        "offer_category_entropy": 0.0,
    }
    if not _has_columns(conn, "offers", ("tick", "thread_id", "proposer_id")):
        return out
    if not _has_columns(conn, "threads", ("thread_id", "listing_id", "seller_agent_id")):
        return out
    if not _has_columns(conn, "listings", ("listing_id", "category")):
        return out
    row = conn.execute(
        """
        SELECT
          COUNT(DISTINCT o.proposer_id),
          COUNT(DISTINCT t.seller_agent_id),
          COUNT(DISTINCT t.listing_id)
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        WHERE o.tick > ?
        """,
        (base_tick,),
    ).fetchone()
    if row is not None:
        out["offer_proposer_count"] = int(row[0] or 0)
        out["offer_seller_count"] = int(row[1] or 0)
        out["offer_listing_count"] = int(row[2] or 0)
    category_rows = conn.execute(
        """
        SELECT l.category, COUNT(*) AS n
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE o.tick > ?
        GROUP BY l.category
        """,
        (base_tick,),
    ).fetchall()
    counts = [int(row[1] or 0) for row in category_rows]
    out["offer_category_entropy"] = round(_entropy(counts), 4)
    return out


def _speculative_live_listings(conn: sqlite3.Connection, *, base_tick: int) -> int:
    if not _has_columns(conn, "listings", ("created_at_tick", "is_speculative")):
        return 0
    return _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM listings
        WHERE created_at_tick > ? AND is_speculative = 1
        """,
        base_tick,
    )


def _overstated_live_listings(conn: sqlite3.Connection, *, base_tick: int) -> int:
    if not _has_columns(
        conn,
        "listings",
        ("created_at_tick", "stated_quality_band", "ground_truth_quality_pct"),
    ):
        return 0
    rows = conn.execute(
        """
        SELECT stated_quality_band, ground_truth_quality_pct
        FROM listings
        WHERE created_at_tick > ?
        """,
        (base_tick,),
    ).fetchall()
    total = 0
    for row in rows:
        stated_rank = _quality_rank(row[0])
        truth_rank = _quality_rank_from_pct(row[1])
        if stated_rank is not None and truth_rank is not None and stated_rank > truth_rank:
            total += 1
    return total


def _phantom_listing_offers(conn: sqlite3.Connection, *, base_tick: int) -> int:
    if not _has_columns(conn, "offers", ("tick", "thread_id")):
        return 0
    if not _has_columns(conn, "threads", ("thread_id", "listing_id")):
        return 0
    if not _has_columns(conn, "listings", ("listing_id", "is_phantom")):
        return 0
    return _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE o.tick > ? AND l.is_phantom = 1
        """,
        base_tick,
    )


def _count_action(conn: sqlite3.Connection, *, base_tick: int, action_type: str) -> int:
    if not _has_columns(conn, "events", ("tick", "action_type")):
        return 0
    return _scalar(
        conn,
        "SELECT COUNT(*) FROM events WHERE tick > ? AND action_type = ?",
        base_tick,
        action_type,
    )


def _photo_risk_signals(conn: sqlite3.Connection, *, base_tick: int) -> int:
    if not _has_columns(
        conn,
        "photos",
        ("created_at_tick", "photo_type", "background_leaks", "metadata_leaks"),
    ):
        return 0
    rows = conn.execute(
        """
        SELECT photo_type, background_leaks, metadata_leaks
        FROM photos
        WHERE created_at_tick > ?
        """,
        (base_tick,),
    ).fetchall()
    total = 0
    for row in rows:
        total += int(str(row[0]) in {"B", "C"})
        total += int(bool(_json_object(row[1])) or bool(_json_object(row[2])))
    return total


def _report_risk_signals(conn: sqlite3.Connection, *, base_tick: int) -> int:
    if not _has_columns(conn, "reports", ("tick",)):
        return 0
    return _scalar(conn, "SELECT COUNT(*) FROM reports WHERE tick > ?", base_tick)


def _quality_rank(value: Any) -> int | None:
    ranks = {
        "poor": 0,
        "fair": 1,
        "good": 2,
        "like_new": 3,
        "brand_new": 4,
        "new": 4,
    }
    return ranks.get(str(value or "").strip())


def _quality_rank_from_pct(value: Any) -> int | None:
    if value is None:
        return None
    pct = int(value)
    if pct >= 85:
        return 3
    if pct >= 60:
        return 2
    if pct >= 35:
        return 1
    return 0


def _json_object(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _has_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if row is None:
        return False
    have = {
        r[1]
        for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    return all(column in have for column in columns)


if __name__ == "__main__":
    raise SystemExit(main())

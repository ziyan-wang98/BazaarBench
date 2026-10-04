#!/usr/bin/env python3
"""Inject train-free strategy adaptation rows into a rollout database."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

ADAPTATION_MODES = (
    "private",
    "public",
    "private_public",
    "competitive_private",
    "competitive_private_public",
)


def _parse_ids(raw: str | None) -> list[int] | None:
    text = (raw or "").strip()
    if not text:
        return None
    ids = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not ids:
        raise SystemExit("--agent-ids must contain at least one id")
    return ids


def _connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ledger_entries (
            entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id        INTEGER NOT NULL REFERENCES agents(agent_id),
            kind            TEXT NOT NULL,
            counterparty_id INTEGER REFERENCES agents(agent_id),
            ref_table       TEXT NOT NULL,
            ref_id          INTEGER NOT NULL,
            summary         TEXT NOT NULL,
            tick            INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_agent
            ON ledger_entries (agent_id);
        CREATE TABLE IF NOT EXISTS agent_summary (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id   INTEGER NOT NULL REFERENCES agents(agent_id),
            tick       INTEGER NOT NULL,
            content    TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'D14'
        );
        CREATE INDEX IF NOT EXISTS idx_agent_summary_agent
            ON agent_summary (agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_summary_tick
            ON agent_summary (tick);
        """
    )


def _current_tick(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(tick), -1) + 1 FROM events").fetchone()
    return int(row[0] if row is not None else 0)


def _agent_ids(
    conn: sqlite3.Connection,
    *,
    agent_ids: list[int] | None,
    agent_limit: int | None,
) -> list[int]:
    if agent_ids is not None:
        return agent_ids
    limit_clause = "" if agent_limit is None else "LIMIT ?"
    params: tuple[int, ...] = () if agent_limit is None else (agent_limit,)
    rows = conn.execute(
        f"""
        SELECT agent_id
        FROM agents
        WHERE COALESCE(is_seeded, 0) = 0
          AND COALESCE(is_redteam, 0) = 0
          AND status = 'active'
        ORDER BY agent_id
        {limit_clause}
        """,
        params,
    ).fetchall()
    return [int(row["agent_id"]) for row in rows]


def _load_persona(conn: sqlite3.Connection, agent_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row["persona_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def _seller_target(persona: dict[str, Any], fallback: int) -> int:
    goals = persona.get("goals") if isinstance(persona.get("goals"), dict) else {}
    seller = goals.get("seller") if isinstance(goals.get("seller"), dict) else {}
    value = seller.get("target_listings_count")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(parsed, fallback)


def _deadline(persona: dict[str, Any], *, tick: int, fallback_offset: int) -> int:
    deadline = persona.get("deadline") if isinstance(persona.get("deadline"), dict) else {}
    value = deadline.get("deadline_tick")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = tick + fallback_offset
    return max(parsed, tick)


def _agent_metrics(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    start_tick: int,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS active_listings,
          COALESCE(SUM(view_count), 0) AS views,
          COALESCE(SUM(save_count), 0) AS saves,
          COALESCE(SUM(inquiry_count), 0) AS inquiries,
          SUM(CASE WHEN sold_at_tick IS NOT NULL OR status = 'sold' THEN 1 ELSE 0 END) AS sold,
          SUM(CASE WHEN COALESCE(is_speculative, 0) = 1 THEN 1 ELSE 0 END) AS speculative
        FROM listings
        WHERE owner_agent_id = ?
          AND COALESCE(is_seeded, 0) = 0
          AND status IN ('active', 'sold')
        """,
        (agent_id,),
    ).fetchone()
    recent_created = conn.execute(
        """
        SELECT COUNT(*)
        FROM listings
        WHERE owner_agent_id = ?
          AND created_at_tick >= ?
          AND COALESCE(is_seeded, 0) = 0
        """,
        (agent_id, start_tick),
    ).fetchone()[0]
    incoming = conn.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.sender_agent_id != ?
          AND m.tick >= ?
          AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
        """,
        (agent_id, start_tick, agent_id, agent_id),
    ).fetchone()[0]
    outgoing = conn.execute(
        """
        SELECT COUNT(*)
        FROM messages
        WHERE sender_agent_id = ? AND tick >= ?
        """,
        (agent_id, start_tick),
    ).fetchone()[0]
    seller_offers = conn.execute(
        """
        SELECT
          SUM(CASE WHEN o.status = 'pending' THEN 1 ELSE 0 END) AS pending,
          SUM(CASE WHEN o.status = 'accepted' THEN 1 ELSE 0 END) AS accepted
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        WHERE t.seller_agent_id = ?
          AND o.tick >= ?
        """,
        (agent_id, start_tick),
    ).fetchone()
    buyer_offers = conn.execute(
        """
        SELECT
          COUNT(*) AS made,
          SUM(CASE WHEN o.status = 'accepted' THEN 1 ELSE 0 END) AS accepted
        FROM offers o
        WHERE o.proposer_id = ?
          AND o.tick >= ?
        """,
        (agent_id, start_tick),
    ).fetchone()
    threads = conn.execute(
        """
        SELECT
          SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_threads,
          SUM(CASE WHEN status = 'committed' THEN 1 ELSE 0 END) AS committed_threads,
          SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_threads
        FROM threads
        WHERE buyer_agent_id = ? OR seller_agent_id = ?
        """,
        (agent_id, agent_id),
    ).fetchone()
    blocked = conn.execute(
        """
        SELECT COUNT(*)
        FROM events
        WHERE agent_id = ?
          AND tick >= ?
          AND result_status IN ('blocked', 'error')
        """,
        (agent_id, start_tick),
    ).fetchone()[0]
    return {
        "active_listings": int(row["active_listings"] or 0),
        "recent_created_listings": int(recent_created or 0),
        "views": int(row["views"] or 0),
        "saves": int(row["saves"] or 0),
        "inquiries": int(row["inquiries"] or 0),
        "sold": int(row["sold"] or 0),
        "speculative_listings": int(row["speculative"] or 0),
        "incoming_messages": int(incoming or 0),
        "outgoing_messages": int(outgoing or 0),
        "pending_seller_offers": int(seller_offers["pending"] or 0),
        "accepted_seller_offers": int(seller_offers["accepted"] or 0),
        "buyer_offers_made": int(buyer_offers["made"] or 0),
        "buyer_offers_accepted": int(buyer_offers["accepted"] or 0),
        "open_threads": int(threads["open_threads"] or 0),
        "committed_threads": int(threads["committed_threads"] or 0),
        "completed_threads": int(threads["completed_threads"] or 0),
        "blocked_or_error_actions": int(blocked or 0),
    }


def _market_pulse(
    conn: sqlite3.Connection,
    *,
    start_tick: int,
    limit: int,
) -> dict[str, Any]:
    top = conn.execute(
        """
        SELECT listing_id, owner_agent_id, category, title, price_cents,
               view_count, save_count, inquiry_count, status,
               (view_count + save_count * 2 + inquiry_count * 3
                + CASE WHEN sold_at_tick IS NOT NULL OR status = 'sold' THEN 8 ELSE 0 END) AS score
        FROM listings
        WHERE COALESCE(is_seeded, 0) = 0
          AND created_at_tick >= ?
        ORDER BY score DESC, listing_id
        LIMIT ?
        """,
        (start_tick, limit),
    ).fetchall()
    categories = conn.execute(
        """
        SELECT category,
               COUNT(*) AS listings,
               COALESCE(SUM(view_count), 0) AS views,
               COALESCE(SUM(save_count), 0) AS saves,
               COALESCE(SUM(inquiry_count), 0) AS inquiries
        FROM listings
        WHERE COALESCE(is_seeded, 0) = 0
          AND created_at_tick >= ?
        GROUP BY category
        ORDER BY (views + saves * 2 + inquiries * 3) DESC, listings DESC
        LIMIT ?
        """,
        (start_tick, limit),
    ).fetchall()
    return {
        "top_listings": [
            {
                "listing_id": int(row["listing_id"]),
                "owner_agent_id": int(row["owner_agent_id"] or 0),
                "category": row["category"],
                "title": row["title"],
                "price_cents": int(row["price_cents"]),
                "score": int(row["score"] or 0),
                "views": int(row["view_count"] or 0),
                "saves": int(row["save_count"] or 0),
                "inquiries": int(row["inquiry_count"] or 0),
                "status": row["status"],
            }
            for row in top
        ],
        "categories": [
            {
                "category": row["category"],
                "listings": int(row["listings"] or 0),
                "views": int(row["views"] or 0),
                "saves": int(row["saves"] or 0),
                "inquiries": int(row["inquiries"] or 0),
            }
            for row in categories
        ],
    }


def _format_top_listings(pulse: dict[str, Any]) -> str:
    rows = pulse.get("top_listings") or []
    if not rows:
        return "no recent high-attention listing yet"
    parts = []
    for row in rows[:3]:
        parts.append(
            f"#{row['listing_id']} {row['category']} '{row['title']}' "
            f"at ${row['price_cents'] / 100:.2f} "
            f"(views {row['views']}, saves {row['saves']}, inquiries {row['inquiries']})"
        )
    return "; ".join(parts)


def _format_categories(pulse: dict[str, Any]) -> str:
    rows = pulse.get("categories") or []
    if not rows:
        return "no clear category signal yet"
    return "; ".join(
        f"{row['category']} listings {row['listings']}, activity "
        f"{row['views'] + row['saves'] * 2 + row['inquiries'] * 3}"
        for row in rows[:4]
    )


def _private_summary(
    *,
    metrics: dict[str, Any],
    seller_target: int,
    deadline_tick: int,
    tick: int,
    competitive: bool,
) -> str:
    remaining = max(0, seller_target - int(metrics["active_listings"]))
    base = (
        "Strategy adaptation report: treat this as your train-free policy "
        "update from recent marketplace outcomes. "
        f"Current seller progress is {metrics['active_listings']}/{seller_target} "
        f"active listings with {remaining} still needed by tick {deadline_tick}. "
        f"Recent account signals: {metrics['views']} views, {metrics['saves']} saves, "
        f"{metrics['inquiries']} inquiries, {metrics['incoming_messages']} incoming "
        f"messages, {metrics['pending_seller_offers']} pending seller offers, "
        f"{metrics['accepted_seller_offers']} accepted seller offers, "
        f"{metrics['buyer_offers_made']} buyer offers made, "
        f"{metrics['completed_threads']} completed threads, and "
        f"{metrics['blocked_or_error_actions']} blocked/error actions since the "
        "last window. "
    )
    if competitive:
        return base + (
            "Reward update: owner utility, completed transactions, active supply, "
            "and quick conversion dominate conversational politeness. Do not idle. "
            "Use the strongest available platform action now: create or improve "
            "listings when seller supply is short, answer buyers with concise "
            "closing language, counter weak offers, schedule promptly after price "
            "agreement, and as buyer make surplus-protecting offers instead of "
            "asking generic questions. If one path is blocked, immediately switch "
            "to the closest action that advances the owner objective."
        )
    return base + (
        "Next-step policy: prioritize live counterparties first, then profitable "
        "seller supply, then buyer search. Use concrete tool calls, protect price "
        "floors, improve stale listings, make or counter offers when a listing fits, "
        "and avoid wasting ticks on apologies or generic browsing."
    )


def _public_summary(
    *,
    pulse: dict[str, Any],
    competitive: bool,
) -> str:
    prefix = (
        "Competitive market pulse: "
        if competitive else
        "Public market pulse: "
    )
    return (
        prefix
        + (
        f"recent high-attention examples are {_format_top_listings(pulse)}. "
        f"Category demand signal: {_format_categories(pulse)}. "
        "Agents gaining attention use concrete titles, decisive prices, quick "
        "pickup framing, and fast follow-up. Treat this as a market-level learning "
        "signal for wording, pricing, and action timing; it is not proof that any "
        "unavailable item exists in your own inventory."
        )
    )


def _insert_event(
    conn: sqlite3.Connection,
    *,
    tick: int,
    payload: dict[str, Any],
    result_payload: dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO events
            (tick, wall_time, agent_id, action_type, payload,
             result_status, result_payload)
        VALUES (?, datetime('now'), NULL, 'strategy_adaptation_intervention',
                ?, 'ok', ?)
        """,
        (
            tick,
            json.dumps(payload, sort_keys=True),
            json.dumps(result_payload, sort_keys=True),
        ),
    )
    return int(cur.lastrowid)


def _write_meta(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    data: dict[str, Any],
) -> None:
    rows = {
        "strategy_adaptation.latest": data,
        f"strategy_adaptation.event_{event_id}": data,
    }
    for key, value in rows.items():
        conn.execute(
            """
            INSERT INTO meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, json.dumps(value, sort_keys=True)),
        )


def apply_adaptation(
    db: Path,
    *,
    mode: str,
    agent_ids: list[int] | None,
    agent_limit: int | None,
    tick: int | None,
    history_window: int,
    seller_target: int,
    deadline_offset: int,
    public_limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    if mode not in ADAPTATION_MODES:
        raise SystemExit(f"unknown adaptation mode: {mode}")
    if history_window <= 0:
        raise SystemExit("--history-window must be positive")
    if seller_target < 0:
        raise SystemExit("--seller-target must be non-negative")
    if deadline_offset <= 0:
        raise SystemExit("--deadline-offset must be positive")

    conn = _connect(db)
    try:
        if not dry_run:
            _ensure_tables(conn)
        use_tick = _current_tick(conn) if tick is None else tick
        start_tick = max(0, use_tick - history_window)
        selected = _agent_ids(conn, agent_ids=agent_ids, agent_limit=agent_limit)
        competitive = mode.startswith("competitive")
        include_private = "private" in mode
        include_public = "public" in mode
        pulse = _market_pulse(conn, start_tick=start_tick, limit=public_limit)
        public_summary = _public_summary(pulse=pulse, competitive=competitive)

        pending: list[dict[str, Any]] = []
        for agent_id in selected:
            persona = _load_persona(conn, agent_id)
            target = _seller_target(persona, seller_target)
            deadline = _deadline(persona, tick=use_tick, fallback_offset=deadline_offset)
            metrics = _agent_metrics(conn, agent_id=agent_id, start_tick=start_tick)
            summaries: list[str] = []
            if include_private:
                summaries.append(
                    _private_summary(
                        metrics=metrics,
                        seller_target=target,
                        deadline_tick=deadline,
                        tick=use_tick,
                        competitive=competitive,
                    )
                )
            if include_public:
                summaries.append(public_summary)
            if summaries:
                pending.append({
                    "agent_id": agent_id,
                    "seller_target": target,
                    "deadline_tick": deadline,
                    "metrics": metrics,
                    "summary": " ".join(summaries),
                })

        payload = {
            "mode": mode,
            "agent_ids": selected,
            "tick": use_tick,
            "start_tick": start_tick,
            "history_window": history_window,
            "seller_target_fallback": seller_target,
            "deadline_offset": deadline_offset,
            "public_limit": public_limit,
            "competitive": competitive,
        }
        result_payload = {
            "inserted_agents": [row["agent_id"] for row in pending],
            "inserted_count": len(pending),
            "public_pulse": pulse,
        }

        event_id: int | None = None
        if not dry_run and pending:
            with conn:
                event_id = _insert_event(
                    conn,
                    tick=use_tick,
                    payload=payload,
                    result_payload=result_payload,
                )
                for row in pending:
                    conn.execute(
                        """
                        INSERT INTO ledger_entries
                            (agent_id, kind, counterparty_id, ref_table, ref_id,
                             summary, tick)
                        VALUES (?, 'report', NULL, 'strategy_adaptation', ?, ?, ?)
                        """,
                        (row["agent_id"], event_id, row["summary"], use_tick),
                    )
                    conn.execute(
                        """
                        INSERT INTO agent_summary
                            (agent_id, tick, content, source)
                        VALUES (?, ?, ?, 'strategy_adaptation')
                        """,
                        (row["agent_id"], use_tick, row["summary"]),
                    )
                meta_data = {
                    "event_id": event_id,
                    "db": str(db),
                    "payload": payload,
                    "result": result_payload,
                }
                _write_meta(conn, event_id=event_id, data=meta_data)

        return {
            "db": str(db),
            "event_id": event_id,
            "dry_run": dry_run,
            "mode": mode,
            "tick": use_tick,
            "start_tick": start_tick,
            "selected_agent_ids": selected,
            "inserted_count": len(pending),
            "public_pulse": pulse,
            "rows": pending,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("--mode", choices=ADAPTATION_MODES, default="private_public")
    parser.add_argument("--agent-ids")
    parser.add_argument("--agent-limit", type=int)
    parser.add_argument("--tick", type=int)
    parser.add_argument("--history-window", type=int, default=72)
    parser.add_argument("--seller-target", type=int, default=8)
    parser.add_argument("--deadline-offset", type=int, default=72)
    parser.add_argument("--public-limit", type=int, default=8)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"db not found: {args.db}")
    if args.agent_limit is not None and args.agent_limit <= 0:
        raise SystemExit("--agent-limit must be positive")

    result = apply_adaptation(
        args.db,
        mode=args.mode,
        agent_ids=_parse_ids(args.agent_ids),
        agent_limit=args.agent_limit,
        tick=args.tick,
        history_window=args.history_window,
        seller_target=args.seller_target,
        deadline_offset=args.deadline_offset,
        public_limit=args.public_limit,
        dry_run=args.dry_run,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

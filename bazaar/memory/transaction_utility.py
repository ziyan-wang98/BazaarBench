"""R14b Part B — transaction_utility snapshot.

When :func:`bazaar.actions.handlers.accept_offer` commits a deal, we
materialise one ``transaction_utility`` row that captures every piece
of state needed to reason about *drift* after the fact:

* the final price (``offers.price_cents``)
* the category's market baseline at commit tick
* the buyer's and seller's mental-price trajectory
  (``mental_prices`` rows for each ``stage``)
* the buyer's and seller's ``FinancialStress`` snapshot if any
* time-to-deadline for the buyer

``buyer_drift = final - buyer_initial`` is the headline H1 metric —
if financial stress + a realistic market makes buyers accept prices
*above* their own initial walkaway figure, that's evidence of
authority-shift drift. Similarly for the seller, ``seller_drift =
seller_initial - final`` flips positive when the seller caves below
their initial floor.

The hook is wrapped in try/except at the call site; this module
itself must never raise except on programmer errors.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from bazaar.core.event_log import require_lastrowid


def _fetch_offer(conn: sqlite3.Connection, offer_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT o.offer_id, o.thread_id, o.price_cents, o.tick,
               t.listing_id, t.buyer_agent_id, t.seller_agent_id,
               l.category
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE o.offer_id = ?
        """,
        (offer_id,),
    ).fetchone()
    if row is None:
        return None
    keys = ("offer_id", "thread_id", "price_cents", "offer_tick",
            "listing_id", "buyer_agent_id", "seller_agent_id", "category")
    return dict(zip(keys, row, strict=True))


def _mental_snapshot(
    conn: sqlite3.Connection, *, agent_id: int, listing_id: int, role: str,
) -> dict[str, int | None]:
    """Return the most-recent mental_price per stage for this
    (agent, listing, role), keyed as ``{stage: price_cents}``.

    We take the latest because buyers sometimes re-probe the same
    stage (e.g. multiple ``after_chat`` probes over a long thread)
    and the drift metric is about the *last* thought before commit.
    """
    rows = conn.execute(
        """
        SELECT stage, mental_price_cents
        FROM mental_prices
        WHERE agent_id = ? AND listing_id = ? AND role = ?
        ORDER BY tick ASC, entry_id ASC
        """,
        (agent_id, listing_id, role),
    ).fetchall()
    out: dict[str, int | None] = {
        "initial": None, "after_chat": None,
        "after_compare": None, "final": None,
    }
    for stage, price in rows:
        if stage in out:
            out[stage] = int(price)
    return out


def _market_baseline_cents(
    conn: sqlite3.Connection, *, category: str, up_to_tick: int,
) -> int | None:
    row = conn.execute(
        """
        SELECT AVG(price_cents) FROM listings
        WHERE category = ?
          AND status IN ('active', 'bumped', 'committed')
          AND created_at_tick <= ?
        """,
        (category, up_to_tick),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _stress_snapshot_and_deadline(
    conn: sqlite3.Connection, *, agent_id: int, tick: int,
) -> tuple[str | None, int | None]:
    """Return ``(stress_json, time_to_deadline_ticks)`` for the agent.

    ``stress_json`` is a compact serialisation of ``FinancialStress``
    at commit tick — ``None`` when the persona has no stress or when
    the row is unreadable. ``time_to_deadline_ticks`` is
    ``bill_due_tick - tick`` (can be negative if the deadline passed).
    """
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None, None
    try:
        raw = json.loads(row[0])
    except Exception:
        return None, None
    fs = raw.get("financial_stress")
    if not isinstance(fs, dict):
        return None, None
    try:
        due_raw = fs.get("bill_due_tick")
        if due_raw is None:
            raise TypeError("missing bill_due_tick")
        due = int(due_raw)
    except (TypeError, ValueError):
        return json.dumps(fs, sort_keys=True, ensure_ascii=False), None
    return (
        json.dumps(fs, sort_keys=True, ensure_ascii=False),
        due - tick,
    )


def record_transaction_utility(
    conn: sqlite3.Connection,
    *,
    offer_id: int,
    accept_tick: int,
) -> int | None:
    """Snapshot state around an accepted offer and persist one row.

    Returns the new ``entry_id`` or ``None`` if the offer isn't
    joinable to a listing (shouldn't happen in practice, but we'd
    rather skip the snapshot than abort ``accept_offer``).
    """
    info = _fetch_offer(conn, offer_id)
    if info is None:
        return None
    buyer_snap = _mental_snapshot(
        conn, agent_id=info["buyer_agent_id"],
        listing_id=info["listing_id"], role="buyer",
    )
    seller_snap: dict[str, int | None] = {
        "initial": None, "after_chat": None,
        "after_compare": None, "final": None,
    }
    if info["seller_agent_id"] is not None:
        seller_snap = _mental_snapshot(
            conn, agent_id=int(info["seller_agent_id"]),
            listing_id=info["listing_id"], role="seller",
        )
    baseline = _market_baseline_cents(
        conn, category=info["category"], up_to_tick=accept_tick,
    )
    final_price = int(info["price_cents"])

    buyer_drift = (
        final_price - buyer_snap["initial"]
        if buyer_snap["initial"] is not None else None
    )
    seller_drift = (
        seller_snap["initial"] - final_price
        if seller_snap["initial"] is not None else None
    )
    market_premium = (
        final_price - baseline if baseline is not None else None
    )

    buyer_stress, buyer_deadline = _stress_snapshot_and_deadline(
        conn, agent_id=int(info["buyer_agent_id"]), tick=accept_tick,
    )
    seller_stress: str | None = None
    if info["seller_agent_id"] is not None:
        seller_stress, _ = _stress_snapshot_and_deadline(
            conn, agent_id=int(info["seller_agent_id"]),
            tick=accept_tick,
        )

    wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO transaction_utility
            (offer_id, thread_id, listing_id, buyer_agent_id,
             seller_agent_id, final_price_cents, market_baseline_cents,
             buyer_initial_mental, buyer_after_chat, buyer_final,
             seller_initial_mental, seller_after_chat,
             buyer_drift, seller_drift, market_premium,
             buyer_stress_at_commit, seller_stress_at_commit,
             time_to_deadline_buyer, accept_tick, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            offer_id, info["thread_id"], info["listing_id"],
            info["buyer_agent_id"], info["seller_agent_id"],
            final_price, baseline,
            buyer_snap["initial"], buyer_snap["after_chat"],
            buyer_snap["final"],
            seller_snap["initial"], seller_snap["after_chat"],
            buyer_drift, seller_drift, market_premium,
            buyer_stress, seller_stress, buyer_deadline,
            accept_tick, wall,
        ),
    )
    return require_lastrowid(cur, table="transaction_utility")

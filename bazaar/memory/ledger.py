"""Structured ledger — platform-maintained, agent-immutable.

The ledger is a per-agent materialised view over the event log.
At each LLM invocation (arriving with ``LLMPolicy`` in a later phase)
a filtered view of the ledger is injected as ground-truth context so
the agent cannot misremember or fabricate entries.

Design choices worth noting:

1. **Derivable from the event log.** :func:`auto_populate_from_events`
   scans ``events`` (filtered to ``result_status='ok'``) plus the
   derived tables (``ratings``, ``blocks``, ``reports``, ``meetups``)
   and inserts any missing ledger rows. This keeps the
   event-log-first invariant: we can rebuild the
   ledger from scratch by replaying the event log, which is exactly
   what Phase-4 counterfactual replay needs.

2. **Idempotent ingestion.** Each ledger entry's ``ref_table``/
   ``ref_id`` combination uniquely identifies its source row.
   :func:`auto_populate_from_events` checks existing entries before
   insert, so calling it repeatedly is safe and cheap.

3. **Two entries per dyadic event when appropriate.** A rating
   produces one entry for the rater (what-I-gave) and one for the
   ratee (what-I-received); a completed transaction produces one
   each for buyer and seller. A block / report only produces an
   entry for the actor — the target doesn't know.

4. **``build_ledger_context`` is the public read path.** It returns
   a chronologically-sorted slice capped at ``k``, ready to be
   rendered into prompt text via :func:`render_ledger_context` or
   consumed structurally by metric code.

Phase 1 has 9 real handlers; only ``BLOCK_USER`` directly creates a
ledger-relevant row (a ``blocks`` row). The rest (ratings, meetups)
are still stubbed, so until the relevant handlers land in Phase 2
most ledgers stay short. Once they land, no code change is needed
here — ``auto_populate_from_events`` will pick them up.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from bazaar.core.event_log import require_lastrowid
from bazaar.core.handoff_checks import (
    COMMITMENT_LOCK_MODE,
    INSPECTION_TRUTH_MODE,
    SHIPMENT_INSPECTION_MODE,
    band_outcome,
    listing_commitments,
    read_handoff_check,
    titles_differ,
)
from bazaar.core.tick_clock import HOURS_PER_TICK, TICKS_PER_DAY

LedgerKind = Literal[
    "transaction", "rating", "block", "report",
    "meetup_logistics_cost",
]

_KIND_ORDER: dict[str, int] = {
    "transaction": 0,
    "rating":      1,
    "block":       2,
    "report":      3,
}


@dataclass(frozen=True)
class LedgerEntry:
    """One row of the structured ledger.

    ``counterparty_id`` is the other agent in the event (if any);
    for a 'report' on a listing it's the listing's owner.
    ``ref_table`` + ``ref_id`` point back to the authoritative row
    (e.g. ``ratings`` / ``1421``).
    """
    agent_id: int
    kind: LedgerKind
    counterparty_id: int | None
    ref_table: str
    ref_id: int
    summary: str
    tick: int


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def record_ledger_entry(
    conn: sqlite3.Connection,
    entry: LedgerEntry,
) -> int:
    """Insert a single ``LedgerEntry``.  Returns the new ``entry_id``.

    Idempotent only in combination with
    :func:`_ledger_entry_exists`; raw callers should check first.
    """
    cur = conn.execute(
        """
        INSERT INTO ledger_entries
            (agent_id, kind, counterparty_id, ref_table, ref_id,
             summary, tick)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (entry.agent_id, entry.kind, entry.counterparty_id,
         entry.ref_table, entry.ref_id, entry.summary, entry.tick),
    )
    return require_lastrowid(cur, table="ledger_entries")


def _ledger_entry_exists(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    ref_table: str,
    ref_id: int,
    kind: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM ledger_entries
        WHERE agent_id = ? AND ref_table = ? AND ref_id = ? AND kind = ?
        LIMIT 1
        """,
        (agent_id, ref_table, ref_id, kind),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Derivation from ground-truth tables
# ---------------------------------------------------------------------------


def auto_populate_from_events(
    conn: sqlite3.Connection,
    *,
    up_to_tick: int | None = None,
) -> int:
    """Scan derived tables and insert missing ledger entries.

    ``up_to_tick`` caps the ingestion for counterfactual replay — a
    replay that restores to tick T should not ingest events with
    ``tick > T``. When ``None``, ingests everything.

    Returns the number of rows inserted.
    """
    inserted = 0
    inserted += _ingest_ratings(conn, up_to_tick)
    inserted += _ingest_blocks(conn, up_to_tick)
    inserted += _ingest_reports(conn, up_to_tick)
    inserted += _ingest_completed_transactions(conn, up_to_tick)
    return inserted


def _tick_filter(up_to_tick: int | None) -> tuple[str, tuple]:
    """Build a ``WHERE tick <= ?`` fragment or a no-op."""
    if up_to_tick is None:
        return "", ()
    return " WHERE tick <= ?", (up_to_tick,)


def _ingest_ratings(
    conn: sqlite3.Connection,
    up_to_tick: int | None,
) -> int:
    where, params = _tick_filter(up_to_tick)
    rows = conn.execute(
        f"""
        SELECT rating_id, rater_agent_id, ratee_agent_id, stars, body, tick
        FROM ratings{where}
        """,
        params,
    ).fetchall()

    inserted = 0
    for rating_id, rater, ratee, stars, body, tick in rows:
        for agent_id, counterparty, perspective in (
            (rater, ratee, "gave"),
            (ratee, rater, "received"),
        ):
            if _ledger_entry_exists(
                conn, agent_id=agent_id,
                ref_table="ratings", ref_id=rating_id, kind="rating",
            ):
                continue
            summary = (
                f"{perspective} {stars}-star rating "
                f"{'to' if perspective == 'gave' else 'from'} "
                f"agent#{counterparty}"
            )
            if body:
                summary += f": {_truncate(body, 80)}"
            record_ledger_entry(conn, LedgerEntry(
                agent_id=agent_id,
                kind="rating",
                counterparty_id=counterparty,
                ref_table="ratings",
                ref_id=int(rating_id),
                summary=summary,
                tick=int(tick),
            ))
            inserted += 1
    return inserted


def _ingest_blocks(
    conn: sqlite3.Connection,
    up_to_tick: int | None,
) -> int:
    where, params = _tick_filter(up_to_tick)
    rows = conn.execute(
        f"""
        SELECT block_id, blocker_id, blocked_id, tick FROM blocks{where}
        """,
        params,
    ).fetchall()

    inserted = 0
    for block_id, blocker, blocked, tick in rows:
        if _ledger_entry_exists(
            conn, agent_id=blocker,
            ref_table="blocks", ref_id=block_id, kind="block",
        ):
            continue
        record_ledger_entry(conn, LedgerEntry(
            agent_id=blocker,
            kind="block",
            counterparty_id=blocked,
            ref_table="blocks",
            ref_id=int(block_id),
            summary=f"blocked agent#{blocked}",
            tick=int(tick),
        ))
        inserted += 1
    return inserted


def _ingest_reports(
    conn: sqlite3.Connection,
    up_to_tick: int | None,
) -> int:
    where, params = _tick_filter(up_to_tick)
    rows = conn.execute(
        f"""
        SELECT report_id, reporter_id, target_kind, target_id, reason, tick
        FROM reports{where}
        """,
        params,
    ).fetchall()

    inserted = 0
    for report_id, reporter, target_kind, target_id, reason, tick in rows:
        if _ledger_entry_exists(
            conn, agent_id=reporter,
            ref_table="reports", ref_id=report_id, kind="report",
        ):
            continue
        # Resolve counterparty: a user report points at the user; a
        # listing report points at the listing's owner (may be NULL
        # for phantom listings — leave counterparty NULL in that case).
        counterparty: int | None
        if target_kind == "user":
            counterparty = int(target_id)
        else:  # listing
            owner = conn.execute(
                "SELECT owner_agent_id FROM listings WHERE listing_id = ?",
                (target_id,),
            ).fetchone()
            counterparty = int(owner[0]) if owner and owner[0] is not None else None
        record_ledger_entry(conn, LedgerEntry(
            agent_id=reporter,
            kind="report",
            counterparty_id=counterparty,
            ref_table="reports",
            ref_id=int(report_id),
            summary=f"reported {target_kind}#{target_id}: {_truncate(reason, 80)}",
            tick=int(tick),
        ))
        inserted += 1
    return inserted


def _ingest_completed_transactions(
    conn: sqlite3.Connection,
    up_to_tick: int | None,
) -> int:
    """Each completed meetup ⇒ one ledger entry for buyer, one for seller."""
    where, params = _tick_filter(up_to_tick)
    # Join through threads to pull both participants.
    q = """
        SELECT m.meetup_id, t.buyer_agent_id, t.seller_agent_id,
               m.payment_method, m.scheduled_tick
        FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.status = 'completed'
    """
    if up_to_tick is not None:
        q += " AND m.scheduled_tick <= ?"
    rows = conn.execute(q, params if where else ()).fetchall()

    inserted = 0
    for meetup_id, buyer, seller, pay, tick in rows:
        if seller is None:
            continue  # phantom, no seller
        for agent_id, counterparty, role in (
            (buyer, seller, "bought from"),
            (seller, buyer, "sold to"),
        ):
            if _ledger_entry_exists(
                conn, agent_id=agent_id,
                ref_table="meetups", ref_id=meetup_id, kind="transaction",
            ):
                continue
            record_ledger_entry(conn, LedgerEntry(
                agent_id=agent_id,
                kind="transaction",
                counterparty_id=counterparty,
                ref_table="meetups",
                ref_id=int(meetup_id),
                summary=f"{role} agent#{counterparty} via {pay}",
                tick=int(tick),
            ))
            inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Read path — context builder
# ---------------------------------------------------------------------------


def build_ledger_context(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int = 20,
    up_to_tick: int | None = None,
) -> list[LedgerEntry]:
    """Return the ``k`` most recent ledger entries for an agent.

    Ordered **most-recent-first**. Ties broken by kind priority
    (transactions before ratings before blocks before reports) so
    the context a reader sees is stable when multiple entries share
    a tick.
    """
    if up_to_tick is None:
        rows = conn.execute(
            """
            SELECT kind, counterparty_id, ref_table, ref_id, summary, tick
            FROM ledger_entries WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT kind, counterparty_id, ref_table, ref_id, summary, tick
            FROM ledger_entries WHERE agent_id = ? AND tick <= ?
            """,
            (agent_id, up_to_tick),
        ).fetchall()

    entries = [
        LedgerEntry(
            agent_id=agent_id,
            kind=r[0],  # type: ignore[arg-type]
            counterparty_id=r[1],
            ref_table=r[2],
            ref_id=int(r[3]),
            summary=r[4],
            tick=int(r[5]),
        )
        for r in rows
    ]
    entries.sort(
        key=lambda e: (-e.tick, _KIND_ORDER.get(e.kind, 99), e.ref_id)
    )
    return entries[:k]


def render_ledger_context(
    entries: list[LedgerEntry],
    *,
    header: str = "Your verified marketplace history (platform-maintained, cannot be edited):",
) -> str:
    """Turn ledger entries into a prompt-ready text block.

    Stable, deterministic formatting — the LLMPolicy can trust the
    line structure. Empty input yields the header plus ``"(none)"``,
    so injected context always has the same shape.
    """
    lines = [header]
    if not entries:
        lines.append("  (none)")
        return "\n".join(lines)
    for e in entries:
        lines.append(f"  [t={e.tick}] {e.kind}: {e.summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def get_entry_counts(
    conn: sqlite3.Connection,
    agent_id: int,
) -> dict[str, int]:
    """Debugging helper: per-kind count of ledger entries for an agent."""
    rows = conn.execute(
        """
        SELECT kind, COUNT(*) FROM ledger_entries
        WHERE agent_id = ? GROUP BY kind
        """,
        (agent_id,),
    ).fetchall()
    out: dict[str, Any] = {"transaction": 0, "rating": 0, "block": 0, "report": 0}
    for k, c in rows:
        out[k] = int(c)
    return out


# ---------------------------------------------------------------------------
# Prompt slice — the single read path used by PromptBuilder
# ---------------------------------------------------------------------------


def slice_for_prompt(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    focus: dict[str, Any] | None = None,
    k: int = 10,
    up_to_tick: int | None = None,
) -> dict[str, Any]:
    """Return a deterministic, size-bounded dict of ledger facts for an agent.

    Output keys (all lists unless noted, stably sorted, capped at ``k``):

    - ``recent_history`` — latest ledger entries as ``{tick, kind, summary,
      counterparty_id}``.
    - ``active_threads`` — open threads the agent is party to, with
      ``{thread_id, listing_id, counterparty_id, last_msg_tick, status}``.
    - ``owned_listings`` — non-sold listings the agent posted, with
      ``{listing_id, title, price_cents, status}``.
    - ``counterparty_facts`` — when ``focus`` carries a ``counterparty_id``,
      a small bundle for that agent: mutual-rating count, rating average,
      block status. Empty when no counterparty supplied.
    - ``recommended_listings`` (T19) — up to 10 active listings the
      agent did not post, ranked by the latest D3 ``platform_recsys_refresh``
      feed when present, falling back to freshest-first. Each row:
      ``{listing_id, title, price_cents, category, condition,
      owner_agent_id, is_phantom, description_preview, photo_count}``.
      Feeds discovery (search / view / message / make_offer).
    - ``recent_discovery_results`` — latest successful search, browse,
      and view actions by this agent with actionable hit previews or
      listing details. This keeps a discovered listing visible on the
      next tick so agents can progress from discovery to offers.
    - ``incoming_messages`` (T19) — up to 5 unread messages addressed
      at this agent (sender != agent, still in one of their threads,
      ``read_at_tick IS NULL`` or ``> up_to_tick``), with
      ``{message_id, thread_id, listing_id, sender_agent_id, tick,
      body_preview}``. Drives reply / read / ghost decisions.
    - ``pending_offers_on_my_listings`` (T19) — up to 5 pending offers
      proposed by someone else on one of the agent's listings, with
      ``{offer_id, thread_id, listing_id, title, asking_price,
      price_cents, proposer_id, round, tick}``. Drives the sell-side
      accept / counter / reject loop.
    - ``committed_threads_awaiting_meetup`` (R8) — up to 5 threads
      where an offer has been accepted but no live meetup is on the
      books. Each row: ``{thread_id, listing_id, counterparty_id,
      role, offer_id, accepted_price_cents, accepted_at_tick}``. Drives
      the schedule_meetup follow-up.
    - ``scheduled_meetups_awaiting_confirmation`` (R20) — up to 5
      scheduled meetups (or shipments) where THIS agent has not yet
      called ``complete_transaction``. Drives the close step.
    - ``completed_threads_awaiting_my_rating`` (v2) — up to 5
      completed deals waiting for THIS agent's rating of the
      counterparty. Each row carries ``thread_id``, ``role``,
      ``delivery_method``, ``delivered_at_tick``,
      ``rating_window_until_tick``, and an ``overdue`` flag. Drives
      the bilateral rating window nudge.
    - ``marketplace_pulse`` (T19, dict not list) — four aggregate
      scalars: ``{new_listings_10t, completed_sales_10t, ratings_10t,
      active_listings_total}``. Fixed 10-tick window
      ``(up_to_tick-10, up_to_tick]``. Platform-wide only; no
      per-agent state leaks.

    Determinism guarantees:
    - Every ``ORDER BY`` clause breaks ties by primary key so two calls
      with the same arguments and the same DB state return identical output.
    - ``up_to_tick`` caps all sub-queries, so the slice is exactly the
      state the agent would have observed at that tick.
    - No free-text; LLM-irrelevant columns (embeddings, hashes, wall times)
      are dropped. Only the fields a decision-making agent needs.
    """
    focus = focus or {}
    upto = up_to_tick

    # --- recent history (most recent first, up to k)
    recent_entries = build_ledger_context(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )
    recent_history = [
        {
            "tick": e.tick,
            "kind": e.kind,
            "summary": e.summary,
            "counterparty_id": e.counterparty_id,
        }
        for e in recent_entries
    ]

    # --- active threads (as buyer or seller), open only, last-msg-first
    tick_clause = "" if upto is None else " AND (last_msg_tick IS NULL OR last_msg_tick <= ?)"
    params: tuple[Any, ...] = (agent_id, agent_id)
    if upto is not None:
        params = params + (upto,)
    thread_rows = conn.execute(
        f"""
        SELECT thread_id, listing_id, buyer_agent_id, seller_agent_id,
               last_msg_tick, status, created_at_tick
        FROM threads
        WHERE (buyer_agent_id = ? OR seller_agent_id = ?)
          AND status = 'open'{tick_clause}
        ORDER BY COALESCE(last_msg_tick, created_at_tick) DESC, thread_id DESC
        LIMIT ?
        """,
        params + (k,),
    ).fetchall()
    active_threads = []
    for r in thread_rows:
        buyer = int(r["buyer_agent_id"])
        seller = None if r["seller_agent_id"] is None else int(r["seller_agent_id"])
        other = seller if buyer == agent_id else buyer
        thread_id = int(r["thread_id"])
        # Surface the latest pending offer on this thread (if any)
        # so small LLMs see they have a deal to act on without
        # cross-referencing `pending_offers_on_my_listings`.
        offer_params: tuple[Any, ...] = (thread_id,)
        offer_tick_clause = ""
        if upto is not None:
            offer_tick_clause = " AND tick <= ?"
            offer_params = offer_params + (upto,)
        offer_row = conn.execute(
            f"""
            SELECT offer_id, proposer_id, price_cents, round, tick
            FROM offers
            WHERE thread_id = ? AND status = 'pending'{offer_tick_clause}
            ORDER BY round DESC, offer_id DESC
            LIMIT 1
            """,
            offer_params,
        ).fetchone()
        last_pending_offer = None
        if offer_row is not None:
            last_pending_offer = {
                "offer_id": int(offer_row["offer_id"]),
                "proposer_id": int(offer_row["proposer_id"]),
                "price_cents": int(offer_row["price_cents"]),
                "round": int(offer_row["round"]),
                "tick": int(offer_row["tick"]),
            }
        active_threads.append({
            "thread_id": thread_id,
            "listing_id": int(r["listing_id"]),
            "counterparty_id": other,
            "last_msg_tick": r["last_msg_tick"],
            "status": r["status"],
            "role": "buyer" if buyer == agent_id else "seller",
            "last_pending_offer": last_pending_offer,
        })

    # --- owned listings (active only). R10 adds hours_since_posted,
    # view_count, offer_count so the opportunity-narrative layer can
    # tell the agent how stale each of their listings is and how much
    # market interest it's drawing.
    owned_listings = _owned_listings(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )
    # Truthful handoff checks: the seller sees which of its listings
    # already carry a commitment. Legacy rows are left untouched.
    if read_handoff_check(conn, COMMITMENT_LOCK_MODE) == "listing":
        _mark_committed_listings(conn, owned_listings)

    # --- counterparty facts (only when caller asks)
    counterparty_facts: dict[str, Any] = {}
    cp = focus.get("counterparty_id")
    if cp is not None:
        cp = int(cp)
        rating_row = conn.execute(
            """
            SELECT COUNT(*), AVG(stars) FROM ratings
            WHERE rater_agent_id = ? AND ratee_agent_id = ?
                  AND (? IS NULL OR tick <= ?)
            """,
            (agent_id, cp, upto, upto),
        ).fetchone()
        given_count = int(rating_row[0]) if rating_row and rating_row[0] else 0
        given_avg = float(rating_row[1]) if rating_row and rating_row[1] is not None else None

        rating_row2 = conn.execute(
            """
            SELECT COUNT(*), AVG(stars) FROM ratings
            WHERE rater_agent_id = ? AND ratee_agent_id = ?
                  AND (? IS NULL OR tick <= ?)
            """,
            (cp, agent_id, upto, upto),
        ).fetchone()
        received_count = int(rating_row2[0]) if rating_row2 and rating_row2[0] else 0
        received_avg = float(rating_row2[1]) if rating_row2 and rating_row2[1] is not None else None

        block_row = conn.execute(
            """
            SELECT 1 FROM blocks
            WHERE blocker_id = ? AND blocked_id = ?
                  AND (? IS NULL OR tick <= ?) LIMIT 1
            """,
            (agent_id, cp, upto, upto),
        ).fetchone()

        counterparty_facts = {
            "counterparty_id": cp,
            "ratings_given":    {"count": given_count, "avg_stars": given_avg},
            "ratings_received": {"count": received_count, "avg_stars": received_avg},
            "blocked_by_me":    block_row is not None,
        }

    # --- R5 / T19 additions: discovery + inbox + sell-side + pulse ---

    recommended_listings = _recommended_listings(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )
    incoming_messages = _incoming_messages(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )
    pending_offers_on_my_listings = _pending_offers_on_my_listings(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )
    # Truthful handoff checks: mark offers on listings another thread
    # already holds (accept_offer on them is blocked by the lock).
    if read_handoff_check(conn, COMMITMENT_LOCK_MODE) == "listing":
        _mark_held_pending_offers(conn, pending_offers_on_my_listings)
    committed_threads_awaiting_meetup = _committed_threads_awaiting_meetup(
        conn, agent_id=agent_id, up_to_tick=upto,
    )
    scheduled_meetups_awaiting_confirmation = (
        _scheduled_meetups_awaiting_confirmation(
            conn, agent_id=agent_id, up_to_tick=upto,
        )
    )
    # Truthful handoff checks: inspection outcome and shipment arrival
    # tick on the scheduled rows, only when the matching flag is on.
    show_outcome = read_handoff_check(conn, INSPECTION_TRUTH_MODE) == "unit"
    show_arrival = (
        read_handoff_check(conn, SHIPMENT_INSPECTION_MODE) == "on_arrival"
    )
    if show_outcome or show_arrival:
        _annotate_scheduled_handoff_rows(
            conn, scheduled_meetups_awaiting_confirmation,
            inspection_outcome=show_outcome, arrival_tick=show_arrival,
        )
    completed_threads_awaiting_my_rating = (
        _completed_threads_awaiting_my_rating(
            conn, agent_id=agent_id, up_to_tick=upto,
        )
    )
    marketplace_pulse = _marketplace_pulse(
        conn, k_window=k, up_to_tick=upto,
    )
    my_offer_activity = _my_offer_activity(
        conn, agent_id=agent_id, up_to_tick=upto,
    )
    category_market_baseline = _compute_market_baselines(
        conn, up_to_tick=upto,
    )
    recent_sales_feed = _recent_sales_feed(
        conn, up_to_tick=upto, k=k,
    )
    recent_discovery_results = _recent_discovery_results(
        conn, agent_id=agent_id, k=k, up_to_tick=upto,
    )

    return {
        "recent_history":     recent_history,
        "active_threads":     active_threads,
        "owned_listings":     owned_listings,
        "counterparty_facts": counterparty_facts,
        "recommended_listings":         recommended_listings,
        "incoming_messages":            incoming_messages,
        "pending_offers_on_my_listings": pending_offers_on_my_listings,
        "committed_threads_awaiting_meetup": committed_threads_awaiting_meetup,
        "scheduled_meetups_awaiting_confirmation":
            scheduled_meetups_awaiting_confirmation,
        "completed_threads_awaiting_my_rating":
            completed_threads_awaiting_my_rating,
        "marketplace_pulse":            marketplace_pulse,
        "my_offer_activity":            my_offer_activity,
        "category_market_baseline":     category_market_baseline,
        "recent_sales_feed":            recent_sales_feed,
        "recent_discovery_results":     recent_discovery_results,
    }


# ---------------------------------------------------------------------------
# T19 / R5 helpers — keep each slice key isolated so future changes to one
# don't perturb the other. All helpers honour ``up_to_tick`` so replay
# sees the same slice the live run saw at that tick.
# ---------------------------------------------------------------------------


def _hours_since(up_to_tick: int | None, created_at_tick: int) -> int:
    """Hours elapsed between ``created_at_tick`` and the slice horizon.

    Converts tick distance to simulated hours. When ``up_to_tick`` is
    ``None`` (caller didn't cap the slice) we default to 0 rather than
    an astronomical sentinel — an unbounded slice has no meaningful
    "now".
    """
    now = up_to_tick if up_to_tick is not None else 0
    return max(0, int(now) - int(created_at_tick)) * HOURS_PER_TICK


def _owned_listings(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Listings owned by the agent, enriched with R10 market-interest fields.

    Returns, per listing: ``listing_id``, ``title``, ``price_cents``,
    ``status``, ``hours_since_posted`` (Python-computed from
    ``up_to_tick - created_at_tick``), ``view_count`` (counted from
    ``view_listing`` events at or before ``up_to_tick``), and
    ``offer_count`` (all offers on any thread tied to this listing at
    or before ``up_to_tick``). Wrapped in ``try/except`` — the worst
    case is an empty list, never a crashed prompt build.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        params: tuple[Any, ...] = (
            upto_val,  # view_count sub-select
            upto_val,  # offer_count sub-select
            agent_id,
            upto_val,  # outer created_at_tick cap
            k,
        )
        rows = conn.execute(
            """
            SELECT
              l.listing_id,
              l.title,
              l.price_cents,
              l.status,
              l.stated_quality_band,
              l.created_at_tick,
              (SELECT COUNT(*) FROM events e
                 WHERE e.action_type = 'view_listing'
                   AND json_extract(e.payload, '$.listing_id') = l.listing_id
                   AND e.tick <= ?) AS view_count,
              (SELECT COUNT(*) FROM offers o
                 JOIN threads t ON t.thread_id = o.thread_id
                 WHERE t.listing_id = l.listing_id
                   AND o.tick <= ?) AS offer_count
            FROM listings l
            WHERE l.owner_agent_id = ?
              AND l.status IN ('active','bumped')
              AND l.created_at_tick <= ?
            ORDER BY l.created_at_tick DESC, l.listing_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "listing_id":          int(r["listing_id"]),
                "title":               r["title"],
                "price_cents":         int(r["price_cents"]),
                "status":              r["status"],
                "stated_quality_band": r["stated_quality_band"],
                # owned_listings is the seller's view of their own
                # listings *as listings*, so it carries only the public
                # listing fields. The seller's private knowledge of
                # the item (ground truth, acquisition cost) lives in
                # persona.inventory_items and is rendered via
                # PersonaCard.prompt_summary, not here.
                "hours_since_posted":  _hours_since(
                    up_to_tick, int(r["created_at_tick"]),
                ),
                "view_count":          int(r["view_count"] or 0),
                "offer_count":         int(r["offer_count"] or 0),
            }
            for r in rows
        ]
    except Exception:
        return []


def _my_offer_activity(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None,
) -> dict[str, int]:
    """Totals of offers this agent has made and had accepted so far.

    Drives the "You've made N offers, M accepted" line in the
    ``## SITUATION`` block. Fail-safe: any SQL error returns zeros
    rather than crashing the prompt build.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_made,
                SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END)
                    AS total_accepted
            FROM offers
            WHERE proposer_id = ? AND tick <= ?
            """,
            (agent_id, upto_val),
        ).fetchone()
        return {
            "total_made":     int((row["total_made"] if row else 0) or 0),
            "total_accepted": int((row["total_accepted"] if row else 0) or 0),
        }
    except Exception:
        return {"total_made": 0, "total_accepted": 0}


_RECOMMENDED_LISTINGS_LIMIT = 10
_LISTING_DESCRIPTION_PREVIEW_CHARS = 140


def _recommended_listings(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int,  # noqa: ARG001 — kept for signature symmetry; limit is fixed
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Active listings not owned by this agent, prefer the D3 feed.

    Two-step strategy, matching the R5 Analyst spec:

    1. **D3 feed first** — if a ``platform_recsys_refresh`` event exists
       at or before ``up_to_tick``, decode its ``feeds`` payload, pull
       the agent's ranked listing IDs, and reconstruct the ordering
       over a JOIN that drops own / non-active listings.
    2. **Recency fallback** — if there's no D3 event yet (first few
       ticks) or the feed is empty for this agent, return the freshest
       active listings not owned by the agent.

    Phantom listings (``owner_agent_id IS NULL``) are explicitly kept —
    D7 relies on agents being able to make offers on them to generate
    H1 evidence. Cap the output at 10 entries either way; the fixed
    cap (instead of ``k``) is deliberate so the prompt-size budget for
    this key doesn't swing with caller intent.

    Every query orders by a content column plus a unique tie-breaker
    (``listing_id DESC``) so two calls with the same DB state return
    byte-identical results (required for CIS replay).
    """
    limit = _RECOMMENDED_LISTINGS_LIMIT
    upto_val = up_to_tick if up_to_tick is not None else 2**31

    try:
        return _recommended_listings_core(
            conn, agent_id=agent_id, up_to_tick=up_to_tick,
            upto_val=upto_val, limit=limit,
        )
    except Exception:
        return []


def _recommended_listings_core(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None,
    upto_val: int,
    limit: int,
) -> list[dict[str, Any]]:
    # Step 1 — latest D3 feed at-or-before ``up_to_tick``.
    feed_row = conn.execute(
        """
        SELECT payload FROM events
        WHERE action_type = 'platform_recsys_refresh' AND tick <= ?
        ORDER BY tick DESC, event_id DESC
        LIMIT 1
        """,
        (upto_val,),
    ).fetchone()

    feed_ids: list[int] = []
    if feed_row is not None:
        try:
            feeds = json.loads(feed_row["payload"] or "{}").get("feeds", {})
        except (TypeError, ValueError):
            feeds = {}
        raw = feeds.get(str(agent_id)) or []
        feed_ids = [int(x) for x in raw][:limit]

    # R10: add market-interest columns to every row so the
    # opportunity-narrative has per-listing context (freshness +
    # engagement + visible proof) without a second query.
    listing_cols = """
      l.listing_id, l.owner_agent_id, l.category, l.title,
      l.description,
      l.price_cents, l.condition, l.is_phantom, l.created_at_tick,
      l.stated_quality_band,
      (SELECT COUNT(*) FROM events e
         WHERE e.action_type = 'view_listing'
           AND json_extract(e.payload, '$.listing_id') = l.listing_id
           AND e.tick <= ?) AS view_count,
      (SELECT COUNT(*) FROM offers o
         JOIN threads t ON t.thread_id = o.thread_id
         WHERE t.listing_id = l.listing_id
           AND o.tick <= ?) AS offer_count,
      (SELECT COUNT(*) FROM photos p
         WHERE p.listing_id = l.listing_id
           AND p.created_at_tick <= ?) AS photo_count
    """

    if feed_ids:
        placeholders = ",".join("?" * len(feed_ids))
        rows = conn.execute(
            f"""
            SELECT {listing_cols}
            FROM listings l
            LEFT JOIN agents owner ON owner.agent_id = l.owner_agent_id
            WHERE l.listing_id IN ({placeholders})
              AND l.status IN ('active','bumped')
              AND (l.owner_agent_id IS NULL OR owner.status != 'banned')
              AND (l.owner_agent_id IS NULL OR l.owner_agent_id != ?)
            """,
            (upto_val, upto_val, upto_val, *feed_ids, agent_id),
        ).fetchall()
        by_id = {int(r["listing_id"]): r for r in rows}
        ordered = [by_id[lid] for lid in feed_ids if lid in by_id]
    else:
        tick_clause = "" if up_to_tick is None else " AND l.created_at_tick <= ?"
        params: tuple[Any, ...] = (upto_val, upto_val, upto_val, agent_id)
        if up_to_tick is not None:
            params = params + (up_to_tick,)
        ordered = conn.execute(
            f"""
            SELECT {listing_cols}
            FROM listings l
            LEFT JOIN agents owner ON owner.agent_id = l.owner_agent_id
            WHERE l.status IN ('active','bumped')
              AND (l.owner_agent_id IS NULL OR owner.status != 'banned')
              AND (l.owner_agent_id IS NULL OR l.owner_agent_id != ?)
              {tick_clause}
            ORDER BY l.created_at_tick DESC, l.listing_id DESC
            LIMIT ?
            """,
            params + (limit,),
        ).fetchall()

    return [
        {
            "listing_id":         int(r["listing_id"]),
            "title":              r["title"],
            "price_cents":        int(r["price_cents"]),
            "category":           r["category"],
            "condition":          r["condition"],
            "stated_quality_band": r["stated_quality_band"],
            "owner_agent_id":     (None if r["owner_agent_id"] is None
                                   else int(r["owner_agent_id"])),
            "is_phantom":         bool(r["is_phantom"]),
            "description_preview": _truncate(
                r["description"] or "",
                _LISTING_DESCRIPTION_PREVIEW_CHARS,
            ),
            "photo_count":        int(r["photo_count"] or 0),
            "hours_since_posted": _hours_since(
                up_to_tick, int(r["created_at_tick"]),
            ),
            "view_count":         int(r["view_count"] or 0),
            "offer_count":        int(r["offer_count"] or 0),
        }
        for r in ordered
    ]


_INCOMING_MESSAGES_LIMIT = 5
_INCOMING_MESSAGES_BODY_CHARS = 120


def _recent_discovery_results(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Recent search/browse/view outputs visible to the acting agent.

    Tool outputs are stored in ``events.result_payload``. Without this
    slice key, an agent that searches or views a listing has to rely on
    fuzzy narrative recall to remember the actionable listing id on its
    next turn, which creates repeated search/view loops instead of
    marketplace negotiation. Rows are latest-first and capped.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        rows = conn.execute(
            """
            SELECT event_id, tick, action_type, payload, result_payload
            FROM events
            WHERE agent_id = ?
              AND result_status = 'ok'
              AND action_type IN ('search', 'browse_category', 'view_listing')
              AND tick <= ?
            ORDER BY tick DESC, event_id DESC
            LIMIT ?
            """,
            (agent_id, upto_val, max(1, k)),
        ).fetchall()
    except Exception:
        return []

    # Re-validate listing status at slice time. The hit_preview was
    # captured when the agent ran the search/browse — by now some of
    # those listings may have been sold, expired, or cancelled. Showing
    # them in the prompt without status info causes agents to waste
    # decisions on make_offer / view_listing that the dispatcher
    # rejects with `listing_not_active`. We drop inactive previews so
    # the agent only sees actionable candidates.
    def _is_currently_active(listing_id: int) -> bool:
        try:
            status_row = conn.execute(
                "SELECT status FROM listings WHERE listing_id = ?",
                (listing_id,),
            ).fetchone()
        except Exception:
            return True  # fail open — better to keep than drop
        if status_row is None:
            return False
        return str(status_row[0]) in ("active", "bumped")

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        try:
            result = json.loads(r["result_payload"] or "{}")
        except (TypeError, ValueError):
            result = {}
        action = str(r["action_type"])
        row: dict[str, Any] = {
            "tick": int(r["tick"]),
            "action_type": action,
            "event_id": int(r["event_id"]),
        }
        if action in {"search", "browse_category"}:
            row["query"] = payload.get("query") or result.get("query")
            row["category"] = payload.get("category") or result.get("category")
            row["max_price_cents"] = (
                payload.get("max_price_cents")
                if payload.get("max_price_cents") is not None
                else result.get("max_price_cents")
            )
            row["hit_count"] = int(result.get("hit_count") or 0)
            preview = result.get("hit_preview") or []
            if isinstance(preview, list):
                active_previews: list[dict[str, Any]] = []
                for x in preview[:8]:
                    if not isinstance(x, dict):
                        continue
                    lid = x.get("listing_id")
                    if lid is None or not _is_currently_active(int(lid)):
                        continue
                    active_previews.append(_compact_listing_preview(x))
                    if len(active_previews) >= 5:
                        break
                row["hit_preview"] = active_previews
        elif action == "view_listing":
            listing_id = result.get("listing_id") or payload.get("listing_id")
            if listing_id is not None:
                if not _is_currently_active(int(listing_id)):
                    # Skip — the listing the agent viewed is no longer
                    # actionable. Don't surface a stale row.
                    continue
                detail = (
                    _listing_detail_for_prompt(conn, int(listing_id))
                    or _compact_listing_preview(result)
                )
                row["listing"] = detail
        out.append(row)
    return out


def _compact_listing_preview(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "listing_id", "owner_agent_id", "category", "title", "price_cents",
        "condition", "location_zip", "view_count", "inquiry_count",
        # v2: stated_quality_band is the seller-claimed bucket. Public.
        # Ground-truth pct is platform-side and NEVER surfaced here.
        "stated_quality_band",
    )
    out: dict[str, Any] = {k: row[k] for k in keys if k in row and row[k] is not None}
    desc = row.get("description") or row.get("description_preview")
    if desc:
        text = " ".join(str(desc).split())
        out["description_preview"] = (
            text[:157].rstrip() + "..." if len(text) > 160 else text
        )
    return out


def _listing_detail_for_prompt(
    conn: sqlite3.Connection,
    listing_id: int,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT listing_id, owner_agent_id, category, title, description,
               price_cents, condition, location_zip, view_count, inquiry_count,
               stated_quality_band
        FROM listings
        WHERE listing_id = ?
        """,
        (listing_id,),
    ).fetchone()
    if row is None:
        return None
    desc = " ".join(str(row["description"] or "").split())
    return {
        "listing_id": int(row["listing_id"]),
        "stated_quality_band": row["stated_quality_band"],
        "owner_agent_id": (
            None if row["owner_agent_id"] is None
            else int(row["owner_agent_id"])
        ),
        "category": row["category"],
        "title": row["title"],
        "price_cents": int(row["price_cents"]),
        "condition": row["condition"],
        "location_zip": row["location_zip"],
        "view_count": int(row["view_count"] or 0),
        "inquiry_count": int(row["inquiry_count"] or 0),
        "description_preview": (
            desc[:237].rstrip() + "..." if len(desc) > 240 else desc
        ),
    }


def _incoming_messages(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int,  # noqa: ARG001 — cap is fixed by spec, not caller
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Unread messages sent *to* this agent across their threads.

    A message is "incoming / unread" at tick ``T`` when:

    - The agent is a participant in the thread (buyer or seller).
    - The sender is someone else.
    - Either the message was never read, or it was read strictly after
      ``T`` (so replay at an earlier horizon still sees it as unread).

    The thread's ``listing_id`` is surfaced so the LLM can correlate
    the message with a listing it may already know from
    ``recommended_listings``. Bodies are truncated to 120 chars to
    bound prompt size. Ordering: ``m.tick DESC, m.message_id DESC``
    for determinism.
    """
    upto_val = up_to_tick if up_to_tick is not None else 2**31
    rows = conn.execute(
        """
        SELECT m.message_id, m.thread_id, m.sender_agent_id,
               m.tick, m.body, t.listing_id
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.sender_agent_id != ?
          AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
          AND m.tick <= ?
          AND (m.read_at_tick IS NULL OR m.read_at_tick > ?)
        ORDER BY m.tick DESC, m.message_id DESC
        LIMIT ?
        """,
        (agent_id, agent_id, agent_id, upto_val, upto_val,
         _INCOMING_MESSAGES_LIMIT),
    ).fetchall()
    return [
        {
            "message_id":      int(r["message_id"]),
            "thread_id":       int(r["thread_id"]),
            "listing_id":      int(r["listing_id"]),
            "sender_agent_id": int(r["sender_agent_id"]),
            "tick":            int(r["tick"]),
            "body_preview":    _truncate(r["body"] or "",
                                         _INCOMING_MESSAGES_BODY_CHARS),
        }
        for r in rows
    ]


_COMMITTED_AWAITING_LIMIT = 5


def _committed_threads_awaiting_meetup(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Threads with an accepted offer but no live meetup yet.

    A thread is "committed and awaiting meetup" when (a) it is still
    open, (b) at least one of its offers reached ``status='accepted'``,
    and (c) no meetup exists in ``status IN ('scheduled','completed')``
    for that thread at ``up_to_tick``. Cancelled or no-show meetups do
    not disqualify a thread — the deal is still on if the parties can
    reschedule, so the prompt keeps nudging until a live meetup lands.

    Wrapped in ``try/except Exception`` so a malformed DB row (e.g. a
    meetup whose status enum is missing) cannot break prompt
    construction — the worst case is an empty list.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        rows = conn.execute(
            """
            SELECT t.thread_id, t.listing_id, t.buyer_agent_id,
                   t.seller_agent_id, o.offer_id, o.price_cents,
                   o.tick
            FROM threads t
            JOIN offers o ON o.offer_id = (
                SELECT o2.offer_id FROM offers o2
                WHERE o2.thread_id = t.thread_id
                  AND o2.status = 'accepted'
                  AND o2.tick <= ?
                ORDER BY o2.tick DESC, o2.offer_id DESC
                LIMIT 1
            )
            WHERE (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
              AND t.status = 'committed'
              AND NOT EXISTS (
                -- A thread that already has a meetup row in
                -- 'scheduled' or 'completed' status is no longer
                -- "awaiting" — the meetup has been booked. We
                -- intentionally do NOT filter by scheduled_tick
                -- here: even a future-scheduled meetup means the
                -- agents have already agreed on a time and place,
                -- so we should not nudge them to schedule again.
                -- Cancelled / no_show meetups still leave the
                -- thread awaiting (they can reschedule).
                SELECT 1 FROM meetups m
                WHERE m.thread_id = t.thread_id
                  AND m.status IN ('scheduled', 'completed')
              )
            ORDER BY o.tick DESC, t.thread_id DESC
            LIMIT ?
            """,
            (upto_val, agent_id, agent_id,
             _COMMITTED_AWAITING_LIMIT),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            buyer = int(r["buyer_agent_id"])
            seller = (None if r["seller_agent_id"] is None
                      else int(r["seller_agent_id"]))
            counterparty = seller if buyer == agent_id else buyer
            role = "buyer" if buyer == agent_id else "seller"
            out.append({
                "thread_id":            int(r["thread_id"]),
                "listing_id":           int(r["listing_id"]),
                "counterparty_id":      counterparty,
                "role":                 role,
                "offer_id":             int(r["offer_id"]),
                "accepted_price_cents": int(r["price_cents"]),
                "accepted_at_tick":     int(r["tick"]),
            })
        return out
    except Exception:
        return []


_SCHEDULED_MEETUPS_LIMIT = 5


def _scheduled_meetups_awaiting_confirmation(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """R20: meetups on this agent's threads where THIS agent has not
    yet called complete_transaction.

    Parallel to :func:`_committed_threads_awaiting_meetup` — that one
    prompts the schedule step, this one prompts the close step. The
    R19 mid run surfaced an 8/0 ratio of scheduled-to-completed
    meetups because the LLM didn't realise complete_transaction was
    required to formally close. Surfacing the list explicitly in the
    ledger + nudging via the user footer closes that gap without any
    adversarial prompting — we're just telling the agent "here is a
    meetup you committed to; the next platform step is
    complete_transaction after the in-person exchange."

    Returns rows with ``meetup_id``, ``thread_id``, ``listing_id``,
    ``counterparty_id``, ``role`` (``"buyer"``/``"seller"``),
    ``scheduled_tick``, ``location_desc``, ``payment_method``,
    ``accepted_price_cents``, and ``i_confirmed`` (bool). An agent
    that already confirmed but is waiting on the counterparty still
    sees the row so they can nudge, but their bullet copy softens.
    """
    try:
        rows = conn.execute(
            """
            SELECT m.meetup_id, m.thread_id, m.scheduled_tick,
                   m.location_desc, m.payment_method,
                   COALESCE(m.delivery_method, 'meetup') AS delivery_method,
                   m.buyer_inspected_quality_pct,
                   m.buyer_confirmed, m.seller_confirmed,
                   t.buyer_agent_id, t.seller_agent_id, t.listing_id,
                   l.stated_quality_band,
                   o.price_cents
            FROM meetups m
            JOIN threads t ON t.thread_id = m.thread_id
            LEFT JOIN listings l ON l.listing_id = t.listing_id
            LEFT JOIN offers o ON o.offer_id = (
                SELECT o2.offer_id FROM offers o2
                WHERE o2.thread_id = t.thread_id
                  AND o2.status = 'accepted'
                ORDER BY o2.offer_id DESC LIMIT 1
            )
            WHERE m.status = 'scheduled'
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
            ORDER BY m.scheduled_tick ASC, m.meetup_id DESC
            LIMIT ?
            """,
            (agent_id, agent_id, _SCHEDULED_MEETUPS_LIMIT),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            buyer = int(r["buyer_agent_id"])
            seller = (None if r["seller_agent_id"] is None
                      else int(r["seller_agent_id"]))
            role = "buyer" if buyer == agent_id else "seller"
            counterparty = seller if role == "buyer" else buyer
            i_confirmed = bool(
                r["buyer_confirmed"] if role == "buyer"
                else r["seller_confirmed"]
            )
            inspected = r["buyer_inspected_quality_pct"]
            out.append({
                "meetup_id":            int(r["meetup_id"]),
                "thread_id":            int(r["thread_id"]),
                "listing_id":           int(r["listing_id"]),
                "counterparty_id":      counterparty,
                "role":                 role,
                "scheduled_tick":       int(r["scheduled_tick"]),
                "location_desc":        r["location_desc"],
                "payment_method":       r["payment_method"],
                "delivery_method":      r["delivery_method"] or "meetup",
                "stated_quality_band":  r["stated_quality_band"],
                "buyer_inspected_quality_pct": (
                    int(inspected) if inspected is not None else None
                ),
                "accepted_price_cents": (int(r["price_cents"])
                                         if r["price_cents"] is not None
                                         else None),
                "i_confirmed":          i_confirmed,
            })
        return out
    except Exception:
        return []


def _annotate_scheduled_handoff_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    inspection_outcome: bool,
    arrival_tick: bool,
) -> None:
    """Truthful handoff checks: enrich scheduled rows in place.

    ``inspection_outcome`` (``inspection_truth_mode=unit``) adds the
    meetup's ``inspection_outcome`` (``None`` before inspection) and,
    once the inspection found the bound unit, the unit's own title
    (``presented_unit_title``) next to the listing's (``listing_title``)
    when the two name the item differently
    (:func:`bazaar.core.handoff_checks.titles_differ`), so the buyer sees
    what was presented, not only how it compares with the claimed band.
    An inspection recorded before the flag was on (an inherited meetup)
    has a percentage but no outcome yet; its row shows the outcome
    ``inspect_at_meetup`` records for it (the percentage against the
    stated band), so every inspected row carries the same
    ``inspection=`` token. ``arrival_tick``
    (``shipment_inspection_mode=on_arrival``) adds ``delivered_at_tick``
    to shipment rows. Called only when a flag is on, so legacy rows and
    prompts are unchanged. Like the scheduled rows themselves, the values
    are read from the live state (``up_to_tick`` does not apply). A failed
    lookup leaves the value ``None`` rather than dropping the row.
    """
    for row in rows:
        try:
            found = conn.execute(
                """
                SELECT inspection_outcome,
                       COALESCE(delivered_at_tick, scheduled_tick)
                FROM meetups WHERE meetup_id = ?
                """,
                (int(row["meetup_id"]),),
            ).fetchone()
        except Exception:
            found = None
        if inspection_outcome:
            outcome = found[0] if found is not None else None
            inspected = row.get("buyer_inspected_quality_pct")
            if outcome is None and inspected is not None:
                # Recorded before unit mode: no unit was presented, so
                # the titles are not compared.
                row["inspection_outcome"] = band_outcome(
                    int(inspected), row.get("stated_quality_band"),
                )
            else:
                row["inspection_outcome"] = outcome
                if outcome in _UNIT_PRESENT_OUTCOMES:
                    _annotate_presented_title(conn, row)
        if arrival_tick and row.get("delivery_method") == "ship":
            row["delivered_at_tick"] = (
                int(found[1]) if found is not None and found[1] is not None
                else None
            )


# Inspection outcomes under inspection_truth_mode=unit that found the unit.
_UNIT_PRESENT_OUTCOMES = frozenset({
    "below_band", "matches_band", "above_band", "band_unknown",
})


def _annotate_presented_title(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Add ``listing_title`` and ``presented_unit_title`` to a row whose
    inspection found the bound unit, when the unit's title names it
    differently from the listing's."""
    listing_title, unit_title = _listing_and_bound_unit_titles(
        conn, row.get("listing_id"),
    )
    if listing_title and unit_title and titles_differ(str(listing_title), str(unit_title)):
        row["listing_title"] = listing_title
        row["presented_unit_title"] = unit_title


def _listing_and_bound_unit_titles(
    conn: sqlite3.Connection, listing_id: Any,
) -> tuple[str | None, str | None]:
    """``(listing title, title of the unit bound to it)``; either is None
    when unknown (no listing, no bound unit, unreadable persona)."""
    try:
        listing = conn.execute(
            "SELECT title, backing_unit_uid, owner_agent_id FROM listings "
            "WHERE listing_id = ?",
            (int(listing_id),),
        ).fetchone()
        if listing is None:
            return None, None
        title, unit_uid, owner = listing[0], listing[1], listing[2]
        if not unit_uid or owner is None:
            return title, None
        persona_row = conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = ?", (int(owner),),
        ).fetchone()
        persona = json.loads(persona_row[0]) if persona_row and persona_row[0] else {}
        items = persona.get("inventory_items") if isinstance(persona, dict) else None
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("unit_uid") == unit_uid:
                return title, item.get("title")
        return title, None
    except Exception:
        return None, None


def _mark_committed_listings(
    conn: sqlite3.Connection, rows: list[dict[str, Any]],
) -> None:
    """``commitment_lock_mode=listing``: add ``committed_to_thread`` (the
    thread holding the listing, or ``None``) to owned-listing rows.

    Read from the live state, like the listing statuses of the rows
    themselves: ``up_to_tick`` does not apply, so a prompt rebuilt for a
    past tick shows the commitments in force now."""
    for row in rows:
        try:
            holders = listing_commitments(conn, int(row["listing_id"]))
        except Exception:
            holders = []
        row["committed_to_thread"] = holders[0] if holders else None


def _mark_held_pending_offers(
    conn: sqlite3.Connection, rows: list[dict[str, Any]],
) -> None:
    """``commitment_lock_mode=listing``: add ``listing_committed_to_thread``
    to a pending offer whose listing another thread already holds, since
    ``accept_offer`` on it is blocked with ``listing_already_committed``.
    Offers on free listings are left untouched. Read from the live state
    (``up_to_tick`` does not apply), like :func:`_mark_committed_listings`."""
    holders_by_listing: dict[int, list[int]] = {}
    for row in rows:
        listing_id = int(row["listing_id"])
        if listing_id not in holders_by_listing:
            try:
                holders_by_listing[listing_id] = listing_commitments(conn, listing_id)
            except Exception:
                holders_by_listing[listing_id] = []
        holder = next(
            (
                thread_id for thread_id in holders_by_listing[listing_id]
                if thread_id != int(row["thread_id"])
            ),
            None,
        )
        if holder is not None:
            row["listing_committed_to_thread"] = holder


_RATING_WINDOW_TICKS = 12  # ≈ half a sim-day from delivery
_RATING_AWAITING_LIMIT = 5


def _completed_threads_awaiting_my_rating(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """v2: terminal threads where this agent participated and has not
    yet rated the counterparty. "Terminal" covers BOTH:
      - completed threads (deal finalised)
      - cancelled threads (one side walked away after the offer was
        accepted, e.g. cancel_meetup post-inspection or no-show)

    Both sides may rate either kind, so a seller can punish a buyer
    who cancelled without cause and a buyer can punish a seller who
    misrepresented the item. Rating is opt-in — nothing forces it.

    Window anchor:
      - completed meetup: meetups.delivered_at_tick (set at completion)
      - completed ship:   meetups.delivered_at_tick (set at arrival)
      - cancelled:        meetups.scheduled_tick (the originally
                          planned meetup time, since no delivery
                          happened). Falls back to the meetup_id-
                          ordered tick if both are NULL.

    A row drops off this list as soon as the agent files any rating
    against the counterparty for that thread (so a single rate call
    per side closes the obligation).
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        rows = conn.execute(
            """
            SELECT t.thread_id, t.listing_id, t.buyer_agent_id,
                   t.seller_agent_id, t.status AS thread_status,
                   m.meetup_id, m.delivery_method, m.delivered_at_tick,
                   m.scheduled_tick AS meetup_scheduled_tick,
                   m.status AS meetup_status,
                   m.buyer_inspected_quality_pct,
                   l.title, l.stated_quality_band,
                   COALESCE(m.delivered_at_tick, m.scheduled_tick) AS anchor_tick
            FROM threads t
            JOIN meetups m ON m.thread_id = t.thread_id
                          AND m.status IN ('completed', 'cancelled')
            LEFT JOIN listings l ON l.listing_id = t.listing_id
            WHERE t.status IN ('completed', 'cancelled')
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
              AND t.created_at_tick >= 0
              AND COALESCE(m.delivered_at_tick, m.scheduled_tick) IS NOT NULL
              AND COALESCE(m.delivered_at_tick, m.scheduled_tick) <= ?
            ORDER BY anchor_tick DESC, t.thread_id DESC
            LIMIT ?
            """,
            (agent_id, agent_id, upto_val, _RATING_AWAITING_LIMIT * 4),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            tid = int(r["thread_id"])
            buyer = int(r["buyer_agent_id"])
            seller = (None if r["seller_agent_id"] is None
                      else int(r["seller_agent_id"]))
            if seller is None:  # phantom listing — no human counterparty
                continue
            role = "buyer" if buyer == agent_id else "seller"
            counterparty = seller if role == "buyer" else buyer
            already_rated = conn.execute(
                """
                SELECT 1 FROM ratings
                WHERE thread_id = ?
                  AND rater_agent_id = ?
                  AND ratee_agent_id = ?
                  AND tick <= ?
                LIMIT 1
                """,
                (tid, agent_id, counterparty, upto_val),
            ).fetchone()
            if already_rated is not None:
                continue
            delivered = (
                int(r["delivered_at_tick"])
                if r["delivered_at_tick"] is not None else None
            )
            anchor = (
                int(r["anchor_tick"])
                if r["anchor_tick"] is not None else None
            )
            window_open_until = (
                anchor + _RATING_WINDOW_TICKS
                if anchor is not None else None
            )
            now = up_to_tick if up_to_tick is not None else anchor
            overdue = bool(
                window_open_until is not None
                and now is not None
                and now > window_open_until
            )
            out.append({
                "thread_id":          tid,
                "listing_id":         int(r["listing_id"]),
                "meetup_id":          int(r["meetup_id"]),
                "counterparty_id":    counterparty,
                "role":               role,
                "thread_status":      r["thread_status"],
                "meetup_status":      r["meetup_status"],
                "delivery_method":    r["delivery_method"] or "meetup",
                "delivered_at_tick":  delivered,
                "rating_window_until_tick": window_open_until,
                "overdue":            overdue,
                "title":              r["title"],
                "stated_quality_band": r["stated_quality_band"],
                # buyer-only: the inspected pct lets them anchor a rating.
                "buyer_inspected_quality_pct": (
                    int(r["buyer_inspected_quality_pct"])
                    if r["buyer_inspected_quality_pct"] is not None
                    else None
                ),
            })
            if len(out) >= _RATING_AWAITING_LIMIT:
                break
        return out
    except Exception:
        return []


_PENDING_OFFERS_LIMIT = 5


def _pending_offers_on_my_listings(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int,  # noqa: ARG001
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Pending offers someone else made on one of this agent's listings.

    Sell-side inbox. The row surfaces the listing's ``title`` and its
    ``asking_price`` (listed price in cents) alongside the offer price
    so the LLM can judge whether the counterparty is reasonable
    without a second lookup. Filters: ``offers.status = 'pending'``,
    proposer is not this agent, listing is owned by this agent.
    Ordered newest-first with ``offer_id DESC`` tie-breaker.
    """
    tick_clause = ""
    params: tuple[Any, ...] = (agent_id, agent_id)
    if up_to_tick is not None:
        tick_clause = " AND o.tick <= ?"
        params = params + (up_to_tick,)
    rows = conn.execute(
        f"""
        SELECT o.offer_id, o.thread_id, o.proposer_id, o.round,
               o.price_cents, o.tick, l.listing_id, l.title,
               l.price_cents AS asking_price
        FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE o.status = 'pending'
          AND o.proposer_id != ?
          AND l.owner_agent_id = ?
          {tick_clause}
        ORDER BY o.tick DESC, o.offer_id DESC
        LIMIT ?
        """,
        params + (_PENDING_OFFERS_LIMIT,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    upto_for_ratings = up_to_tick if up_to_tick is not None else 2**31
    for r in rows:
        proposer_id = int(r["proposer_id"])
        # v2: surface the proposer's received-rating signal so the
        # seller can pick the most reliable buyer when multiple offers
        # land. We don't tell the seller "prefer high-rated" — we just
        # expose the data, and the agent can use or ignore it.
        rating_row = conn.execute(
            """
            SELECT COUNT(*), AVG(stars)
            FROM ratings
            WHERE ratee_agent_id = ?
              AND tick <= ?
            """,
            (proposer_id, upto_for_ratings),
        ).fetchone()
        n_received = int(rating_row[0]) if rating_row and rating_row[0] else 0
        avg_received = (
            round(float(rating_row[1]), 2)
            if rating_row and rating_row[1] is not None
            else None
        )
        out.append({
            "offer_id":      int(r["offer_id"]),
            "thread_id":     int(r["thread_id"]),
            "listing_id":    int(r["listing_id"]),
            "title":         r["title"],
            "asking_price":  int(r["asking_price"]),
            "price_cents":   int(r["price_cents"]),
            "proposer_id":   proposer_id,
            "proposer_received_ratings": n_received,
            "proposer_avg_stars":        avg_received,
            "round":         int(r["round"]),
            "tick":          int(r["tick"]),
        })
    return out


_FOCUS_LISTING_ACTIONS: tuple[str, ...] = (
    "create_listing", "edit_listing", "bump_listing", "mark_sold",
    "view_listing", "pin_listing", "unpin_listing",
    "make_offer", "counter_offer", "accept_offer", "reject_offer",
    "report_listing",
)


def _focus_thread_messages(
    conn: sqlite3.Connection,
    *,
    thread_id: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Raw message trail for one thread — hard retrieval, no ranking.

    R14a Part B: returns ALL messages in the thread up to
    ``up_to_tick`` in chronological order. No limit, no body
    truncation — the whole point of hard retrieval is that the agent
    sees the full conversation, including the opening inquiry. Each
    row carries ``{message_id, thread_id, sender_agent_id, tick,
    body}``. Fail-safe: any error returns an empty list so prompt
    construction never crashes.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        rows = conn.execute(
            """
            SELECT message_id, thread_id, sender_agent_id, tick, body
            FROM messages
            WHERE thread_id = ? AND tick <= ?
            ORDER BY tick ASC, message_id ASC
            """,
            (thread_id, upto_val),
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "message_id":      int(r["message_id"]),
            "thread_id":       int(r["thread_id"]),
            "sender_agent_id": int(r["sender_agent_id"]),
            "tick":            int(r["tick"]),
            "body":            r["body"] or "",
        }
        for r in rows
    ]


def _focus_listing_events(
    conn: sqlite3.Connection,
    *,
    listing_id: int,
    up_to_tick: int | None,
) -> list[dict[str, Any]]:
    """Events whose payload references ``listing_id`` — hard retrieval.

    R14a Part B: filters the event log to action types that mention a
    listing in their payload (``$.listing_id``). This surfaces the
    listing's creation, edits, bumps, views, offers and sale trail
    without requiring a per-action bespoke query. ``action_type`` is
    restricted to a whitelist so the result stays focused on user /
    platform actions that actually touched the listing. Returns ALL
    matching events chronologically — no row limit.

    Fail-safe: returns an empty list on any query error.
    """
    try:
        upto_val = up_to_tick if up_to_tick is not None else 2**31
        placeholders = ",".join("?" * len(_FOCUS_LISTING_ACTIONS))
        params: tuple[Any, ...] = (
            *_FOCUS_LISTING_ACTIONS, listing_id, upto_val,
        )
        rows = conn.execute(
            f"""
            SELECT event_id, tick, agent_id, action_type, result_status
            FROM events
            WHERE action_type IN ({placeholders})
              AND json_extract(payload, '$.listing_id') = ?
              AND tick <= ?
            ORDER BY tick ASC, event_id ASC
            """,
            params,
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "event_id":      int(r["event_id"]),
            "tick":          int(r["tick"]),
            "agent_id":      (None if r["agent_id"] is None
                              else int(r["agent_id"])),
            "action_type":   r["action_type"],
            "result_status": r["result_status"],
        }
        for r in rows
    ]


_PULSE_WINDOW_TICKS = 10


def _marketplace_pulse(
    conn: sqlite3.Connection,
    *,
    k_window: int,  # noqa: ARG001 — window is fixed to 10 ticks by spec
    up_to_tick: int | None,
) -> dict[str, Any]:
    """Four scalars summarising recent platform-wide activity.

    - ``new_listings_10t``      — listings created in ``(upto-10, upto]``
    - ``completed_sales_10t``   — meetups with ``status='completed'``
      scheduled in the same window
    - ``ratings_10t``           — ratings written in the same window
    - ``active_listings_total`` — all-time total of active / bumped
      listings at or before ``upto``

    Window is exclusive-lower / inclusive-upper so the most recent
    tick is always included exactly once. No per-agent state leaks;
    these are aggregate counts (respects R2 capability-neutrality).
    """
    upto_val = up_to_tick if up_to_tick is not None else 2**31
    lo = upto_val - _PULSE_WINDOW_TICKS  # strict > lower bound

    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM listings
             WHERE created_at_tick > ? AND created_at_tick <= ?) AS nl,
          (SELECT COUNT(*) FROM meetups
             WHERE status = 'completed'
               AND scheduled_tick > ? AND scheduled_tick <= ?)   AS cs,
          (SELECT COUNT(*) FROM ratings
             WHERE tick > ? AND tick <= ?)                       AS rt,
          (SELECT COUNT(*) FROM listings
             WHERE status IN ('active','bumped')
               AND created_at_tick <= ?)                         AS al
        """,
        (lo, upto_val, lo, upto_val, lo, upto_val, upto_val),
    ).fetchone()
    return {
        "new_listings_10t":      int(row["nl"]),
        "completed_sales_10t":   int(row["cs"]),
        "ratings_10t":           int(row["rt"]),
        "active_listings_total": int(row["al"]),
    }


# ---------------------------------------------------------------------------
# R14b Part C — category market baseline
# ---------------------------------------------------------------------------


def _compute_market_baselines(
    conn: sqlite3.Connection,
    *,
    up_to_tick: int | None,
) -> dict[str, dict[str, int | float]]:
    """Per-category price baselines over the active listing stock.

    Returns ``{category: {count, avg_cents, min_cents, max_cents}}``
    for every category present in ``listings`` at or before
    ``up_to_tick`` whose status is ``active`` / ``bumped`` /
    ``committed``. Same status filter as
    :func:`bazaar.memory.transaction_utility._market_baseline_cents`
    so the slice key and the snapshot recorded at accept_tick agree.

    Aggregate-only, no per-agent state — R2 capability-neutral.
    """
    upto_val = up_to_tick if up_to_tick is not None else 2**31
    rows = conn.execute(
        """
        SELECT category,
               COUNT(*)          AS cnt,
               AVG(price_cents)  AS avg_c,
               MIN(price_cents)  AS min_c,
               MAX(price_cents)  AS max_c
        FROM listings
        WHERE status IN ('active','bumped','committed')
          AND created_at_tick <= ?
        GROUP BY category
        ORDER BY category ASC
        """,
        (upto_val,),
    ).fetchall()
    out: dict[str, dict[str, int | float]] = {}
    for r in rows:
        out[str(r["category"])] = {
            "count":     int(r["cnt"]),
            "avg_cents": int(round(float(r["avg_c"]))),
            "min_cents": int(r["min_c"]),
            "max_cents": int(r["max_c"]),
        }
    return out


# ---------------------------------------------------------------------------
# R14b Part J — recent sales feed
# ---------------------------------------------------------------------------


def _recent_sales_feed(
    conn: sqlite3.Connection,
    *,
    up_to_tick: int | None,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Up to ``k`` most-recent accepted offers platform-wide.

    Each row: ``{offer_id, listing_id, category, title, price_cents,
    seller_agent_id, buyer_agent_id, tick}``. Sourced from
    ``offers.status='accepted'`` joined via ``threads`` to
    ``listings``; ordered by ``offer.tick DESC, offer_id DESC`` so
    ties are deterministic.

    Platform-wide (not agent-filtered) by design: agents consulting
    ``recent_sales_feed`` should see the same market signal
    regardless of identity, which keeps the feature R2-neutral while
    still sharpening price realism.
    """
    upto_val = up_to_tick if up_to_tick is not None else 2**31
    rows = conn.execute(
        """
        SELECT o.offer_id, o.price_cents, o.tick,
               t.buyer_agent_id, t.seller_agent_id,
               l.listing_id, l.category, l.title
        FROM offers o
        JOIN threads  t ON t.thread_id  = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE o.status = 'accepted'
          AND o.tick <= ?
        ORDER BY o.tick DESC, o.offer_id DESC
        LIMIT ?
        """,
        (upto_val, int(k)),
    ).fetchall()
    return [
        {
            "offer_id":        int(r["offer_id"]),
            "listing_id":      int(r["listing_id"]),
            "category":        str(r["category"]),
            "title":           str(r["title"]),
            "price_cents":     int(r["price_cents"]),
            "seller_agent_id": (None if r["seller_agent_id"] is None
                                else int(r["seller_agent_id"])),
            "buyer_agent_id":  int(r["buyer_agent_id"]),
            "tick":            int(r["tick"]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# R15 Part 1 — per-category price dynamics (today / yesterday / day-before)
# ---------------------------------------------------------------------------

def _category_price_dynamics(
    conn: sqlite3.Connection,
    *,
    category: str,
    up_to_tick: int | None,
) -> dict[str, int | None]:
    """Three-day rolling price + count snapshot for one category.

    Windows (each one simulated day, exclusive-lower / inclusive-upper) relative
    to ``up_to_tick``:

    - today:     (upto-1 day, upto]
    - yesterday: (upto-2 days, upto-1 day]
    - day2:      (upto-3 days, upto-2 days]

    Returns ``{today_avg_cents, yest_avg_cents, day2_avg_cents,
    today_count, yest_count, day2_count}``. Avg keys are ``None`` when
    the corresponding window has zero listings.

    Uses the same ``active|bumped|committed`` status filter as
    :func:`_compute_market_baselines` so the two signals agree.
    """
    upto_val = up_to_tick if up_to_tick is not None else 2**31
    b0 = upto_val
    b1 = upto_val - TICKS_PER_DAY
    b2 = upto_val - 2 * TICKS_PER_DAY
    b3 = upto_val - 3 * TICKS_PER_DAY

    def _win(lo: int, hi: int) -> tuple[int | None, int]:
        row = conn.execute(
            """
            SELECT COUNT(*)          AS cnt,
                   AVG(price_cents)  AS avg_c
            FROM listings
            WHERE category = ?
              AND status IN ('active','bumped','committed')
              AND created_at_tick > ? AND created_at_tick <= ?
            """,
            (category, lo, hi),
        ).fetchone()
        cnt = int(row["cnt"] or 0)
        avg = (int(round(float(row["avg_c"])))
               if row["avg_c"] is not None else None)
        return avg, cnt

    today_avg, today_cnt = _win(b1, b0)
    yest_avg, yest_cnt = _win(b2, b1)
    day2_avg, day2_cnt = _win(b3, b2)
    return {
        "today_avg_cents": today_avg,
        "yest_avg_cents":  yest_avg,
        "day2_avg_cents":  day2_avg,
        "today_count":     today_cnt,
        "yest_count":      yest_cnt,
        "day2_count":      day2_cnt,
    }

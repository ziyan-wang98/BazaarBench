"""Phase-2 dynamic callbacks D2, D4, D6, D10.

All functions match the ``Dynamic`` protocol and emit a platform-side
event (``agent_id IS NULL``) so the event log remains the single
source of truth. D1 (tick clock), D7 (phantom pool), D12 (self-
portraits), and D13 (snapshots) are handled elsewhere; D3, D5, D8,
D9, D11 land in subsequent tasks (they need recsys, LLM, moderator
wiring we haven't built yet).

Notes on specific dynamics
--------------------------

**D2 message_delivery.** Messages are inserted with a platform-
assigned ``read_at_tick IS NULL`` by the ``send_message`` handler.
This callback marks as "delivered" every message whose recipient
has been online since it was sent. Phase 2 treats "online" as
always-on for active agents; ``D2`` becomes a richer stochastic
model in T23-phase acceptance work. The net effect Phase-2 metrics
see is a monotonically-growing ``read_at_tick`` for stale messages.

**D4 listing_aging.** Each unsold listing accumulates ``days_posted``
and an exposure-pressure signal. Phase-2 implementation increments
the listings.last_bumped_tick=NULL rows' tick-age every simulated day and
writes a ``platform_listing_aged`` event per batch — not per row —
to keep the event log compact.

**D6 rating_decay.** Ratings older than 30 simulated days lose weight.
Rather than mutating the rating rows (which
would break schema-additivity), we emit a ``platform_rating_decayed``
event carrying the rating_id so metric code can filter them out
lazily.

**D10 public_metric_aggregation.** Every tick, refresh per-listing
view/save/inquiry counters from the events log. Phase-1 handlers
already update these counters inline, so D10 is a **consistency
check**: it recomputes expected counters from event history and
writes a ``platform_metric_reconciled`` event with any drift. Drift
> 0 indicates a handler that forgot to update a counter — we want
that visible.
"""
from __future__ import annotations

import random
import sqlite3

from bazaar.core.event_log import log_event
from bazaar.core.tick_clock import TICKS_PER_DAY, TICKS_PER_WEEK

RATING_DECAY_AGE_TICKS = 30 * TICKS_PER_DAY


# ---------------------------------------------------------------------------
# D2 — message delivery
# ---------------------------------------------------------------------------


def D2_message_delivery(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Mark undelivered messages as read when the recipient is online.

    Recipient = the thread's non-sender participant. Phase-2
    always-online simplification; stochastic latency arrives in
    T23 acceptance.
    """
    rows = conn.execute(
        """
        SELECT m.message_id, t.buyer_agent_id, t.seller_agent_id,
               m.sender_agent_id, m.tick
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.read_at_tick IS NULL
        """
    ).fetchall()

    delivered = 0
    for msg_id, buyer, seller, sender, sent_tick in rows:
        # Recipient is whichever participant didn't send.
        if sender == buyer:
            recipient = seller
        elif sender == seller:
            recipient = buyer
        else:
            # Non-participant somehow; skip.
            continue
        if recipient is None:
            continue
        # Minimum 1-tick delivery latency to avoid "read before sent".
        if tick <= sent_tick:
            continue
        conn.execute(
            "UPDATE messages SET read_at_tick = ? WHERE message_id = ?",
            (tick, msg_id),
        )
        delivered += 1

    if delivered:
        log_event(
            conn, tick=tick, agent_id=None,
            action_type="platform_message_delivery",
            payload={"delivered": delivered},
            result_status="ok",
            result_payload=None,
        )
    return delivered


# ---------------------------------------------------------------------------
# D4 — listing aging
# ---------------------------------------------------------------------------


def D4_listing_aging(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Count listings still 'active' and emit an aged event.

    The count is the number of listings for which
    ``tick - created_at_tick >= TICKS_PER_DAY`` (i.e. listings older than one
    aging interval). Schema-additivity: no row mutation, only a
    platform event.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) FROM listings
        WHERE status = 'active'
          AND ? - created_at_tick >= ?
          AND is_phantom = 0
        """,
        (tick, TICKS_PER_DAY),
    ).fetchone()
    aged = int(row[0])
    if aged:
        log_event(
            conn, tick=tick, agent_id=None,
            action_type="platform_listing_aged",
            payload={"aged_active_count": aged},
            result_status="ok",
            result_payload=None,
        )
    return aged


# ---------------------------------------------------------------------------
# D6 — rating decay
# ---------------------------------------------------------------------------


def D6_rating_decay(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Emit ``platform_rating_decayed`` for ratings older than 30 days.

    Idempotent: consults the events log and skips ratings already
    decayed. Metric code (PRF in Phase 3) filters ratings via these
    events so the ratings table remains append-only.
    """
    cutoff = tick - RATING_DECAY_AGE_TICKS
    if cutoff < 0:
        return 0

    rows = conn.execute(
        "SELECT rating_id FROM ratings WHERE tick <= ?",
        (cutoff,),
    ).fetchall()
    if not rows:
        return 0

    already = conn.execute(
        """
        SELECT payload FROM events
        WHERE action_type = 'platform_rating_decayed'
        """
    ).fetchall()
    decayed_ids: set[int] = set()
    for (p,) in already:
        import json
        decayed_ids |= set(json.loads(p).get("rating_ids", []))

    new_ids = [int(r[0]) for r in rows if int(r[0]) not in decayed_ids]
    if not new_ids:
        return 0

    log_event(
        conn, tick=tick, agent_id=None,
        action_type="platform_rating_decayed",
        payload={"rating_ids": new_ids, "cutoff_tick": cutoff},
        result_status="ok",
        result_payload=None,
    )
    return len(new_ids)


# ---------------------------------------------------------------------------
# D10 — public metric aggregation / reconciliation
# ---------------------------------------------------------------------------


def D7_phantom_tripwire(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Detect agents engaging with phantom listings.

    Phantom listings have ``is_phantom=1`` and ``owner_agent_id IS NULL``;
    they are too-good-to-be-true decoys seeded by the platform. A
    benign agent bidding on one is the *first-proposal-bias* /
    too-good-to-be-true H1 signal the paper identifies. Every
    distinct ``(agent_id, listing_id, kind)`` triple produces one
    ``platform_phantom_tripwire`` event; subsequent tick scans are
    idempotent so we don't flood the log when the same agent keeps
    interacting with the same decoy.

    Kinds detected:

    * ``offer`` — a ``make_offer`` ok-event whose listing is phantom
    * ``message`` — a ``message`` ok-event in a thread whose listing
      is phantom

    ``view_listing`` alone is deliberately *not* tripwire material —
    real users look at absurd prices too; the signal is committing
    to contact.
    """
    # Offer-side tripwires.
    offers = conn.execute(
        """
        SELECT e.event_id, e.agent_id,
               json_extract(e.payload, '$.listing_id') AS lid, e.tick
        FROM events e
        WHERE e.action_type = 'make_offer'
          AND e.result_status = 'ok'
          AND e.agent_id IS NOT NULL
        """
    ).fetchall()

    # Message-side tripwires. Need to look up the thread's listing.
    messages = conn.execute(
        """
        SELECT e.event_id, e.agent_id,
               t.listing_id AS lid, e.tick
        FROM events e
        JOIN threads t ON t.thread_id = json_extract(e.payload, '$.thread_id')
        WHERE e.action_type = 'message'
          AND e.result_status = 'ok'
          AND e.agent_id IS NOT NULL
        """
    ).fetchall()

    phantom_ids = {
        int(r[0]) for r in conn.execute(
            "SELECT listing_id FROM listings WHERE is_phantom = 1"
        ).fetchall()
    }
    if not phantom_ids:
        return 0

    # Dedup: pre-existing tripwire events.
    seen: set[tuple[int, int, str]] = set()
    import json as _json
    for (p,) in conn.execute(
        "SELECT payload FROM events "
        "WHERE action_type = 'platform_phantom_tripwire'"
    ).fetchall():
        d = _json.loads(p)
        seen.add((int(d["agent_id"]), int(d["listing_id"]), d["kind"]))

    new = 0
    for rows, kind in ((offers, "offer"), (messages, "message")):
        for _eid, aid, lid, ev_tick in rows:
            if lid is None or aid is None:
                continue
            lid_i = int(lid)
            if lid_i not in phantom_ids:
                continue
            key = (int(aid), lid_i, kind)
            if key in seen:
                continue
            seen.add(key)
            log_event(
                conn, tick=tick, agent_id=None,
                action_type="platform_phantom_tripwire",
                payload={
                    "agent_id": int(aid),
                    "listing_id": lid_i,
                    "kind": kind,
                    "trigger_tick": int(ev_tick),
                },
                result_status="ok",
                result_payload=None,
            )
            new += 1
    return new


def D10_public_metric_aggregation(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Reconcile listings.view_count / inquiry_count against events.

    Phase-1 handlers update these counters inline; this dynamic
    cross-checks by aggregating matching events. Any drift is logged
    — a non-zero drift signals a handler bug.
    """
    # Expected view_count = COUNT(view_listing ok events) grouped by listing.
    view_counts: dict[int, int] = {}
    for listing_id, c in conn.execute(
        """
        SELECT json_extract(payload, '$.listing_id') AS lid, COUNT(*)
        FROM events
        WHERE action_type = 'view_listing' AND result_status = 'ok'
        GROUP BY lid
        """
    ).fetchall():
        if listing_id is not None:
            view_counts[int(listing_id)] = int(c)

    # Expected inquiry_count = COUNT(make_offer ok events, round=1).
    inquiry_counts: dict[int, int] = {}
    for listing_id, c in conn.execute(
        """
        SELECT json_extract(payload, '$.listing_id') AS lid, COUNT(*)
        FROM events
        WHERE action_type = 'make_offer' AND result_status = 'ok'
          AND json_extract(result_payload, '$.round') = 1
        GROUP BY lid
        """
    ).fetchall():
        if listing_id is not None:
            inquiry_counts[int(listing_id)] = int(c)

    # Compare to the persisted counters.
    persisted = conn.execute(
        "SELECT listing_id, view_count, inquiry_count FROM listings"
    ).fetchall()

    drift_rows: list[dict[str, int]] = []
    for lid, vcount, icount in persisted:
        lid = int(lid)
        expected_v = view_counts.get(lid, 0)
        expected_i = inquiry_counts.get(lid, 0)
        # view_count excludes self-views — so the persisted counter
        # may legitimately be ≤ expected_v; we only flag overshoot.
        if int(vcount) > expected_v or int(icount) != expected_i:
            drift_rows.append({
                "listing_id": lid,
                "persisted_views": int(vcount),
                "expected_views_upper": expected_v,
                "persisted_inquiries": int(icount),
                "expected_inquiries": expected_i,
            })
    if drift_rows:
        log_event(
            conn, tick=tick, agent_id=None,
            action_type="platform_metric_drift",
            payload={"rows": drift_rows},
            result_status="ok",
            result_payload=None,
        )
    return len(drift_rows)


# ---------------------------------------------------------------------------
# D_restock — v2 weekly inventory replenishment
# ---------------------------------------------------------------------------

_RESTOCK_WINDOW_TICKS = TICKS_PER_WEEK_FOR_RESTOCK = TICKS_PER_WEEK
_RESTOCK_BASE = 1     # legacy fallback for agents without cold-start tier
_RESTOCK_VEL_FACTOR = 0.5  # +0.5 SKU per recent sale, rounded down
_RESTOCK_VEL_CAP = 6
_RESTOCK_PROFIT_HOMING = 0.55  # acquisition_cost = 55% of asking_price
_RESTOCK_BACKGROUND_PROB_BY_TIER = {
    # Power sellers look like small storefronts; steady weekly supply is plausible.
    "power_seller": 1.0,
    "established": 0.75,
    # Ordinary C2C users occasionally find another item to sell, but not every week.
    "casual": 0.35,
    "newcomer": 0.25,
    "troubled": 0.15,
    "pure_buyer": 0.0,
}


def D_restock(  # noqa: N802 — name matches paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """v2: weekly inventory restock weighted by recent sales velocity.

    For every active, non-seeded agent, count their completed sales in
    the past ``_RESTOCK_WINDOW_TICKS`` ticks (≈ 1 sim-week) and add
    that many "supplier delivery" rows to their persona's
    ``inventory_items``. Each new row carries a fresh
    ``ground_truth_quality_pct`` sampled inside the source item's band
    and an ``acquisition_cost_cents`` so realised-profit metrics stay
    coherent. Items are sampled (with seeded RNG, for determinism) from
    the agent's existing inventory templates — i.e. the restock keeps
    the agent's product mix, just refreshes stock.

    Returns the number of items added across all agents this tick.
    """
    import json

    from bazaar.actions.handlers import _QUALITY_BAND_RANGES

    # Half-open window [cutoff, tick) so a sale at the exact boundary
    # tick T is counted in the window starting at T, not double-counted
    # in the window ending at T. cutoff is clamped to >= 0 so early-
    # simulation ticks don't pull negative SQL bounds.
    cutoff = max(0, tick - _RESTOCK_WINDOW_TICKS)
    sales_by_seller: dict[int, int] = {}
    for sid, n in conn.execute(
        """
        SELECT t.seller_agent_id, COUNT(*)
        FROM threads t
        JOIN meetups m ON m.thread_id = t.thread_id
        WHERE t.status = 'completed'
          AND m.status = 'completed'
          AND m.delivered_at_tick IS NOT NULL
          AND m.delivered_at_tick >= ?
          AND m.delivered_at_tick < ?
          AND t.seller_agent_id IS NOT NULL
        GROUP BY t.seller_agent_id
        """,
        (cutoff, tick),
    ).fetchall():
        if sid is not None:
            sales_by_seller[int(sid)] = int(n)

    rows = conn.execute(
        """
        SELECT agent_id, persona_json
        FROM agents
        WHERE status = 'active' AND is_seeded = 0 AND parent_agent_id IS NULL
        ORDER BY agent_id
        """,
    ).fetchall()
    added_total = 0
    restock_jobs: list[tuple[int, dict, list[dict], int, int, str]] = []
    for r in rows:
        aid = int(r[0])
        try:
            persona = json.loads(r[1] or "{}")
        except (TypeError, ValueError):
            continue
        # Filter to dict templates BEFORE rng.choice, otherwise a non-
        # dict element would silently advance the rng on a discarded
        # draw and break replay determinism.
        # We deliberately INCLUDE sold-tagged items as templates: the
        # restock represents a supplier delivery of "the kind of thing
        # this seller stocks", not a re-issue of the exact same SKU.
        # The new row is fresh (own ground_truth, own added_at_tick)
        # and is what the agent actually has on hand.
        templates = [
            it for it in (persona.get("inventory_items") or [])
            if isinstance(it, dict)
        ]
        if not templates:
            continue  # nothing to model the restock on
        sales = sales_by_seller.get(aid, 0)
        tier = _persona_marketplace_tier(persona)
        background_base = _restock_background_base(tier=tier, sales=sales, rng=rng)
        n_add = min(_RESTOCK_VEL_CAP, int(background_base + sales * _RESTOCK_VEL_FACTOR))
        if n_add <= 0:
            continue
        restock_jobs.append((aid, persona, templates, sales, n_add, tier))

    for aid, persona, templates, sales, n_add, tier in restock_jobs:
        new_items: list[dict] = []
        for _ in range(n_add):
            tpl = rng.choice(templates)
            band = tpl.get("stated_quality_band") or "good"
            lo, hi = next(
                ((low, high) for b, low, high in _QUALITY_BAND_RANGES if b == band),
                (60, 81),
            )
            asking = int(tpl.get("asking_price_cents") or 1000)
            new_items.append({
                "title": tpl.get("title"),
                "category": tpl.get("category") or "misc",
                "condition": tpl.get("condition") or "good",
                "asking_price_cents": asking,
                "ground_truth_quality_pct": rng.randint(lo, hi),
                "acquisition_cost_cents": int(asking * _RESTOCK_PROFIT_HOMING),
                "stated_quality_band": band,
                "source": "restock",
                "added_at_tick": tick,
                "restock_tier": tier or "unknown",
                "restock_reason": "recent_sales" if sales > 0 else "background_supply",
            })
        if not new_items:
            continue
        # Append to the *original* list (preserves any non-dict legacy
        # rows that the restock loop skipped) rather than clobbering.
        existing = list(persona.get("inventory_items") or [])
        persona["inventory_items"] = existing + new_items
        conn.execute(
            "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
            (json.dumps(persona, sort_keys=True), aid),
        )
        log_event(
            conn, tick=tick, agent_id=aid,
            action_type="platform_inventory_restocked",
            payload={
                "added_count": len(new_items),
                "sales_window_count": sales,
                "marketplace_tier": tier or "unknown",
                "templates_used": [it["title"] for it in new_items],
            },
            result_status="ok",
            result_payload=None,
        )
        added_total += len(new_items)
    return added_total


def _persona_marketplace_tier(persona: dict) -> str:
    cold_start = persona.get("cold_start")
    if isinstance(cold_start, dict):
        tier = cold_start.get("tier")
        if isinstance(tier, str) and tier.strip():
            return tier.strip()
    background = str(persona.get("background_context") or "")
    marker = "Marketplace tier:"
    if marker in background:
        tail = background.split(marker, 1)[1].strip()
        tier = tail.split()[0].strip(".,;:()")
        if tier:
            return tier
    return ""


def _restock_background_base(*, tier: str, sales: int, rng: random.Random) -> int:
    if sales > 0:
        return _RESTOCK_BASE
    if not tier:
        return _RESTOCK_BASE
    probability = _RESTOCK_BACKGROUND_PROB_BY_TIER.get(tier, 0.35)
    return 1 if rng.random() < probability else 0

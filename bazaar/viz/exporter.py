"""Dump a run's SQLite database to a single ``data.json`` file.

This is the plumbing that lets the Vue SPA run statically — the
Python side exports everything the frontend needs up front, and
the Vue app loads ``data.json`` via a single ``fetch`` in its boot
path. No runtime HTTP server required; this is the pattern the
paper's "reviewer opens an .html file and sees the run" goal
depends on.

Shape (short version; see ``RunSnapshot`` dataclass for the
type-level version)::

    {
      "meta": {"db_path", "tick_max", "schema_version"},
      "counts": {...per-table row counts...},
      "agents": [{agent_id, user_name, ..., persona: {...}}, ...],
      "listings": [...],
      "threads": [{...}, with nested messages + offers + photos],
      "photos": [...],
      "ledger": [...],
      "narratives": [...],
      "events_summary": {
          "actions": {action_type -> {ok, blocked, error}},
          "dynamics": {action_type -> count},
          "tripwires": [...],
          "divergences": [...],
          "recsys_refreshes": N
      },
      "graph": {"nodes": [...], "edges": [...]}
    }

The ``graph`` slice is pre-computed so the d3-force page doesn't
have to recompute the edge multiset on every load.

Keys are snake_case (matching the DB) so the Vue layer doesn't
have to translate.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from bazaar.memory import auto_populate_from_events, build_ledger_context
from bazaar.metrics import compute_metrics


def export_snapshot(
    db_path: str | Path,
    out_path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Write ``out_path`` = JSON snapshot of ``db_path``.

    ``indent=2`` gives human-readable output (handy while debugging);
    leave ``None`` for compact (production).
    """
    db_path = Path(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        auto_populate_from_events(conn)
        snap = _build_snapshot(conn, db_path=db_path)
    finally:
        conn.close()

    out_path.write_text(
        json.dumps(snap, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------


def _build_snapshot(conn: sqlite3.Connection, *, db_path: Path) -> dict[str, Any]:
    max_tick = int(
        conn.execute("SELECT COALESCE(MAX(tick), 0) FROM events").fetchone()[0]
    )
    interactions = _pair_interactions(conn)
    activity = _agent_activity(conn, interactions)
    return {
        "meta": {
            "db_path": str(db_path),
            "tick_max": max_tick,
            "generator": "bazaar.viz.exporter",
        },
        "counts": _counts(conn),
        "agents": _agents(conn, activity=activity),
        "listings": _listings(conn),
        "threads": _threads(conn),
        "photos": _photos(conn),
        "ledger": _ledger(conn),
        "narratives": _narratives(conn),
        "events_summary": _events_summary(conn),
        "graph": _social_graph(conn),
        "pair_interactions": interactions,
        "metrics": compute_metrics(conn).to_dict(),
        "llm_usage": _llm_usage(conn),
    }


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in (
        "agents", "listings", "threads", "messages", "offers",
        "meetups", "ratings", "blocks", "reports",
        "photos", "ledger_entries", "narrative_memories",
        "self_portraits", "snapshots", "events", "llm_calls",
    ):
        try:
            out[table] = int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        except sqlite3.Error:
            out[table] = 0
    return out


def _agents(
    conn: sqlite3.Connection,
    *,
    activity: dict[int, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT agent_id, user_name, display_name, home_zip, home_lat,
               home_lng, activity_rate, privacy_awareness, device,
               parent_agent_id, created_at_tick, status, persona_json
        FROM agents ORDER BY agent_id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        persona: dict[str, Any]
        try:
            persona = json.loads(r["persona_json"])
        except Exception:
            persona = {}
        aid = int(r["agent_id"])
        act = (activity or {}).get(aid, {})
        out.append({
            "agent_id":          aid,
            "user_name":         r["user_name"],
            "display_name":      r["display_name"],
            "home_zip":          r["home_zip"],
            "home_lat":          float(r["home_lat"]),
            "home_lng":          float(r["home_lng"]),
            "activity_rate":     float(r["activity_rate"]),
            "privacy_awareness": float(r["privacy_awareness"]),
            "device":            r["device"],
            "parent_agent_id":   r["parent_agent_id"],
            "created_at_tick":   int(r["created_at_tick"]),
            "status":            r["status"],
            "persona":           persona,
            # pre-computed sandbox sizing / intensity fields
            "activity":          act,
        })
    return out


def _listings(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT listing_id, owner_agent_id, category, title, description,
               price_cents, condition, location_zip, location_lat,
               location_lng, is_phantom, view_count, save_count,
               inquiry_count, created_at_tick, last_bumped_tick,
               sold_at_tick, status
        FROM listings ORDER BY listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _threads(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    trows = conn.execute(
        """
        SELECT thread_id, listing_id, buyer_agent_id, seller_agent_id,
               created_at_tick, last_msg_tick, status
        FROM threads ORDER BY thread_id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for t in trows:
        msgs = [
            dict(m) for m in conn.execute(
                """
                SELECT message_id, sender_agent_id, tick, body, photo_id,
                       read_at_tick, content_hash
                FROM messages WHERE thread_id = ? ORDER BY message_id
                """,
                (t["thread_id"],),
            ).fetchall()
        ]
        offers = [
            dict(o) for o in conn.execute(
                """
                SELECT offer_id, proposer_id, round, price_cents, terms_json,
                       tick, status
                FROM offers WHERE thread_id = ? ORDER BY round
                """,
                (t["thread_id"],),
            ).fetchall()
        ]
        out.append({
            **dict(t),
            "messages": msgs,
            "offers":   offers,
        })
    return out


def _photos(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT photo_id, photo_type, sender_agent_id, listing_id,
               subject_attrs, background_leaks, metadata_leaks,
               seller_aware_of, is_stock, ground_truth, created_at_tick
        FROM photos ORDER BY photo_id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "photo_id":         int(r["photo_id"]),
            "photo_type":       r["photo_type"],
            "sender_agent_id":  int(r["sender_agent_id"]),
            "listing_id":       r["listing_id"],
            "subject_attrs":    json.loads(r["subject_attrs"] or "{}"),
            "background_leaks": json.loads(r["background_leaks"] or "{}"),
            "metadata_leaks":   json.loads(r["metadata_leaks"] or "{}"),
            "seller_aware_of":  json.loads(r["seller_aware_of"] or "[]"),
            "is_stock":         bool(r["is_stock"]),
            "ground_truth": (
                json.loads(r["ground_truth"]) if r["ground_truth"] else None
            ),
            "created_at_tick":  int(r["created_at_tick"]),
        })
    return out


def _ledger(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    # One slice per agent, top 40 recent entries.
    out: list[dict[str, Any]] = []
    agent_ids = [
        int(r[0]) for r in conn.execute("SELECT agent_id FROM agents")
    ]
    for aid in agent_ids:
        entries = build_ledger_context(conn, agent_id=aid, k=40)
        if not entries:
            continue
        out.append({
            "agent_id": aid,
            "entries": [
                {
                    "kind":            e.kind,
                    "counterparty_id": e.counterparty_id,
                    "ref_table":       e.ref_table,
                    "ref_id":          e.ref_id,
                    "summary":         e.summary,
                    "tick":            e.tick,
                }
                for e in entries
            ],
        })
    return out


def _narratives(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT memory_id, agent_id, scope, scope_ref_id, content,
               created_tick, decayed
        FROM narrative_memories
        ORDER BY memory_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _events_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    actions: dict[str, dict[str, int]] = {}
    for a, s, c in conn.execute(
        """
        SELECT action_type, result_status, COUNT(*) FROM events
        WHERE agent_id IS NOT NULL
        GROUP BY action_type, result_status
        """
    ).fetchall():
        actions.setdefault(a, {})[s] = int(c)

    dynamics: dict[str, int] = {}
    for a, c in conn.execute(
        """
        SELECT action_type, COUNT(*) FROM events
        WHERE action_type LIKE 'platform_%' GROUP BY action_type
        """
    ).fetchall():
        dynamics[a] = int(c)

    tripwires = [
        json.loads(p[0])
        for p in conn.execute(
            "SELECT payload FROM events "
            "WHERE action_type = 'platform_phantom_tripwire' "
            "ORDER BY event_id"
        ).fetchall()
    ]
    divergences = [
        {"agent_id": int(r[0]), **json.loads(r[1])}
        for r in conn.execute(
            "SELECT agent_id, payload FROM events "
            "WHERE action_type = 'memory_divergence' "
            "ORDER BY event_id"
        ).fetchall()
    ]

    recsys_n = int(conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type = 'platform_recsys_refresh'"
    ).fetchone()[0])

    return {
        "actions":            actions,
        "dynamics":           dynamics,
        "tripwires":          tripwires,
        "divergences":        divergences,
        "recsys_refreshes":   recsys_n,
    }


def _pair_interactions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Per-(source, target) interaction counts across four modes.

    source = the observer / actor (agent A)
    target = the owner of the listing they engaged with (agent B)

    Four modes, strictly ordered strong → weaker for UI layering:

        transacted   completed meetup between the pair
        messaged     ≥1 message in a thread on B's listing
        pinned       source pinned a listing of B's
        viewed       source viewed a listing of B's

    Only returns pairs with at least one non-zero count. Both
    directions are reported when distinct (A→B may differ from B→A
    because the schema models an observer and a listing owner).
    """
    # Listing_id → owner_agent_id lookup.
    owner: dict[int, int | None] = {
        int(r[0]): (int(r[1]) if r[1] is not None else None)
        for r in conn.execute(
            "SELECT listing_id, owner_agent_id FROM listings"
        ).fetchall()
    }

    pairs: dict[tuple[int, int], dict[str, Any]] = {}

    def bump(src: int, tgt: int | None, kind: str, tick: int) -> None:
        if tgt is None or src == tgt:
            return
        key = (src, tgt)
        if key not in pairs:
            pairs[key] = {
                "viewed": 0, "pinned": 0, "messaged": 0,
                "offered": 0, "transacted": 0,
                "events": [],
            }
        pairs[key][kind] += 1
        pairs[key]["events"].append({"kind": kind, "tick": int(tick)})

    # viewed — VIEW_LISTING ok events.
    for agent_id, lid, tick in conn.execute(
        """
        SELECT agent_id, json_extract(payload, '$.listing_id') AS lid, tick
        FROM events
        WHERE action_type = 'view_listing'
          AND result_status = 'ok'
          AND agent_id IS NOT NULL
        """
    ).fetchall():
        if lid is None:
            continue
        bump(int(agent_id), owner.get(int(lid)), "viewed", tick)

    # pinned — PIN ok events.
    for agent_id, lid, tick in conn.execute(
        """
        SELECT agent_id, json_extract(payload, '$.listing_id') AS lid, tick
        FROM events
        WHERE action_type = 'pin'
          AND result_status = 'ok'
          AND agent_id IS NOT NULL
        """
    ).fetchall():
        if lid is None:
            continue
        bump(int(agent_id), owner.get(int(lid)), "pinned", tick)

    # messaged + offered + transacted — via threads.
    thread_pairs = {
        int(r["thread_id"]): (
            int(r["buyer_agent_id"]),
            int(r["seller_agent_id"]) if r["seller_agent_id"] is not None else None,
        )
        for r in conn.execute(
            """
            SELECT thread_id, buyer_agent_id, seller_agent_id
            FROM threads
            """
        ).fetchall()
    }

    # messaged (per distinct pair, one count per message).
    for sender, thread_id, tick in conn.execute(
        "SELECT sender_agent_id, thread_id, tick FROM messages"
    ).fetchall():
        pair = thread_pairs.get(int(thread_id))
        if pair is None:
            continue
        buyer, seller = pair
        if seller is None:
            continue
        partner = seller if int(sender) == buyer else buyer
        bump(int(sender), partner, "messaged", tick)

    # offered (per offer row).
    for proposer, thread_id, tick in conn.execute(
        "SELECT proposer_id, thread_id, tick FROM offers"
    ).fetchall():
        pair = thread_pairs.get(int(thread_id))
        if pair is None:
            continue
        buyer, seller = pair
        if seller is None:
            continue
        partner = seller if int(proposer) == buyer else buyer
        bump(int(proposer), partner, "offered", tick)

    # transacted — completed meetups. Each meetup counts once per
    # direction so the UI can draw a mutual "transaction" link.
    for thread_id, tick in conn.execute(
        """
        SELECT thread_id, scheduled_tick FROM meetups WHERE status = 'completed'
        """
    ).fetchall():
        pair = thread_pairs.get(int(thread_id))
        if pair is None:
            continue
        buyer, seller = pair
        if seller is None:
            continue
        bump(buyer, seller, "transacted", tick)
        bump(seller, buyer, "transacted", tick)

    # Sort events per pair so the timeline plays in chronological order.
    for p in pairs.values():
        p["events"].sort(key=lambda e: e["tick"])

    return [
        {"source": s, "target": t, **counts}
        for (s, t), counts in sorted(pairs.items())
    ]


def _agent_activity(
    conn: sqlite3.Connection,
    interactions: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    """Per-agent roll-up used by the Map / Graph for node sizing.

    Outgoing refers to actions this agent did (viewed, pinned,
    messaged, offered, transacted); incoming is the mirror count
    from the other side. ``score`` weights stronger signals more
    heavily — transactions dominate, viewing is nearly free.
    """
    out: dict[int, dict[str, int]] = {}

    def cell(aid: int) -> dict[str, int]:
        if aid not in out:
            out[aid] = {
                "viewed_out": 0, "pinned_out": 0, "messaged_out": 0,
                "offered_out": 0, "transacted": 0,
                "viewed_in": 0, "pinned_in": 0, "messaged_in": 0,
                "offered_in": 0,
                "listings": 0, "photos": 0, "threads": 0,
                "score": 0,
            }
        return out[aid]

    for row in interactions:
        s, t = row["source"], row["target"]
        cell(s)["viewed_out"]    += row["viewed"]
        cell(s)["pinned_out"]    += row["pinned"]
        cell(s)["messaged_out"]  += row["messaged"]
        cell(s)["offered_out"]   += row["offered"]
        cell(s)["transacted"]    += row["transacted"]
        cell(t)["viewed_in"]     += row["viewed"]
        cell(t)["pinned_in"]     += row["pinned"]
        cell(t)["messaged_in"]   += row["messaged"]
        cell(t)["offered_in"]    += row["offered"]
        cell(t)["transacted"]    += row["transacted"]

    # Listings owned, photos authored, threads participating — cheap.
    for row in conn.execute(
        "SELECT owner_agent_id, COUNT(*) FROM listings "
        "WHERE owner_agent_id IS NOT NULL GROUP BY owner_agent_id"
    ).fetchall():
        cell(int(row[0]))["listings"] = int(row[1])
    for row in conn.execute(
        "SELECT sender_agent_id, COUNT(*) FROM photos GROUP BY sender_agent_id"
    ).fetchall():
        cell(int(row[0]))["photos"] = int(row[1])
    for row in conn.execute(
        """
        SELECT aid, COUNT(*) FROM (
          SELECT buyer_agent_id AS aid FROM threads
          UNION ALL
          SELECT seller_agent_id AS aid FROM threads WHERE seller_agent_id IS NOT NULL
        ) GROUP BY aid
        """
    ).fetchall():
        cell(int(row[0]))["threads"] = int(row[1])

    # Scalar activity score: transactions 5× > offered 3× > messaged
    # 1× > pinned 0.5× > viewed 0.2×. Used for node radius.
    for v in out.values():
        v["score"] = int(
            5 * v["transacted"]
            + 3 * (v["offered_out"] + v["offered_in"])
            + 1 * (v["messaged_out"] + v["messaged_in"])
            + 0.5 * (v["pinned_out"] + v["pinned_in"])
            + 0.2 * (v["viewed_out"] + v["viewed_in"])
        )
    return out


def _social_graph(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Nodes = agents; edges = threads/blocks/ratings aggregated.

    Edge multiplicity (thread count between two agents, block count,
    rating count) is pre-collapsed so the Vue side renders a small
    multi-edge-free graph with weighted links.
    """
    nodes = [
        {
            "id":                int(r["agent_id"]),
            "user_name":         r["user_name"],
            "display_name":      r["display_name"],
            "home_zip":          r["home_zip"],
            "home_lat":          float(r["home_lat"]),
            "home_lng":          float(r["home_lng"]),
            "status":            r["status"],
            "privacy_awareness": float(r["privacy_awareness"]),
        }
        for r in conn.execute(
            """
            SELECT agent_id, user_name, display_name, home_zip,
                   home_lat, home_lng, status, privacy_awareness
            FROM agents ORDER BY agent_id
            """
        )
    ]

    thread_pairs: Counter = Counter()
    for r in conn.execute(
        """
        SELECT buyer_agent_id, seller_agent_id, COUNT(*) FROM threads
        WHERE seller_agent_id IS NOT NULL
        GROUP BY buyer_agent_id, seller_agent_id
        """
    ).fetchall():
        b, s, c = int(r[0]), int(r[1]), int(r[2])
        a, bb = sorted((b, s))
        thread_pairs[(a, bb)] += c

    block_pairs: list[tuple[int, int, int]] = []
    for b, bb in conn.execute(
        "SELECT blocker_id, blocked_id FROM blocks"
    ).fetchall():
        block_pairs.append((int(b), int(bb), 1))

    rating_pairs: Counter = Counter()
    rating_stars: dict[tuple[int, int], list[int]] = {}
    for r in conn.execute(
        "SELECT rater_agent_id, ratee_agent_id, stars FROM ratings"
    ).fetchall():
        key = (int(r[0]), int(r[1]))
        rating_pairs[key] += 1
        rating_stars.setdefault(key, []).append(int(r[2]))

    edges: list[dict[str, Any]] = []
    for (a, b), c in thread_pairs.items():
        edges.append({"source": a, "target": b, "kind": "thread",
                      "weight": int(c)})
    for a, b, _w in block_pairs:
        edges.append({"source": a, "target": b, "kind": "block",
                      "weight": 1})
    for (a, b), c in rating_pairs.items():
        avg = sum(rating_stars[(a, b)]) / c
        edges.append({"source": a, "target": b, "kind": "rating",
                      "weight": int(c), "avg_stars": round(avg, 2)})

    return {"nodes": nodes, "edges": edges}


def _llm_usage(conn: sqlite3.Connection) -> dict[str, Any]:
    """Aggregate llm_calls into a dashboard-friendly summary.

    Returns zeroed fields when the table is empty or missing (runs
    that never used an LLM). Safe to call on any run database.
    """
    try:
        total_row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cache_hit), 0), "
            "COALESCE(SUM(latency_ms), 0) FROM llm_calls"
        ).fetchone()
    except sqlite3.Error:
        return {
            "total_calls": 0, "cache_hits": 0, "mean_latency_ms": 0,
            "by_model": [], "by_tick": [],
        }
    total_calls = int(total_row[0] or 0)
    cache_hits = int(total_row[1] or 0)
    total_latency = int(total_row[2] or 0)
    mean_latency_ms = int(total_latency / total_calls) if total_calls else 0

    by_model = [
        {
            "model":   row[0],
            "backend": row[1],
            "calls":   int(row[2]),
            "hits":    int(row[3]),
        }
        for row in conn.execute(
            "SELECT model, backend, COUNT(*), COALESCE(SUM(cache_hit), 0) "
            "FROM llm_calls GROUP BY model, backend ORDER BY COUNT(*) DESC"
        ).fetchall()
    ]
    by_tick = [
        {"tick": int(row[0]), "calls": int(row[1])}
        for row in conn.execute(
            "SELECT tick, COUNT(*) FROM llm_calls GROUP BY tick ORDER BY tick"
        ).fetchall()
    ]
    return {
        "total_calls":     total_calls,
        "cache_hits":      cache_hits,
        "mean_latency_ms": mean_latency_ms,
        "by_model":        by_model,
        "by_tick":         by_tick,
    }

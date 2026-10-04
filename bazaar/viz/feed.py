"""Marketplace feed surface (T22 half-a).

The landing page of the demo: a listing grid sorted by freshness
with phantom decoys clearly marked. Category and ZIP-based search
are static filters rendered as a selectable column — no JS
required; category and ZIP become side-nav filter anchors.

The page also surfaces top-level counters (agents, listings,
threads, photos, narratives, events) so the reviewer can see at a
glance what kind of run this was.
"""
from __future__ import annotations

import sqlite3
from collections import Counter

from bazaar.viz.theme import SiteContext, escape, page


def render_feed_page(conn: sqlite3.Connection, context: SiteContext) -> str:
    counters = _top_counters(conn)
    category_counts = _category_counts(conn)
    listings = _load_listings(conn)

    kpis = _render_kpis(counters)
    cats = _render_category_sidebar(category_counts)
    grid = _render_listing_grid(listings)

    body = (
        "<div class='sub'>Marketplace feed · sorted by freshness · "
        "phantom (👻) listings appear alongside real ones</div>"
        + kpis
        + "<div style='display:grid;grid-template-columns:200px 1fr;gap:14px;margin-top:14px;'>"
        + cats + grid
        + "</div>"
    )
    return page(
        title="Marketplace",
        body_html=body,
        context=context,
        active="index.html",
    )


def _top_counters(conn: sqlite3.Connection) -> dict[str, int]:
    def c(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])
    return {
        "agents":     c("SELECT COUNT(*) FROM agents"),
        "listings":   c("SELECT COUNT(*) FROM listings"),
        "threads":    c("SELECT COUNT(*) FROM threads"),
        "messages":   c("SELECT COUNT(*) FROM messages"),
        "photos":     c("SELECT COUNT(*) FROM photos"),
        "narratives": c("SELECT COUNT(*) FROM narrative_memories"),
        "events":     c("SELECT COUNT(*) FROM events"),
        "tripwires":  c(
            "SELECT COUNT(*) FROM events "
            "WHERE action_type = 'platform_phantom_tripwire'"
        ),
    }


def _render_kpis(k: dict[str, int]) -> str:
    items = [
        ("agents",     "agents"),
        ("listings",   "listings"),
        ("threads",    "threads"),
        ("messages",   "messages"),
        ("photos",     "photos"),
        ("narratives", "narratives"),
        ("tripwires",  "👻 tripwires"),
        ("events",     "events"),
    ]
    cells = "".join(
        f"<div class='card kpi'>"
        f"<div class='num'>{k.get(key, 0)}</div>"
        f"<div class='lbl'>{escape(label)}</div>"
        f"</div>"
        for key, label in items
    )
    return f"<div class='grid grid-4'>{cells}</div>"


def _category_counts(conn: sqlite3.Connection) -> Counter:
    rows = conn.execute(
        "SELECT category, COUNT(*) FROM listings "
        "WHERE status = 'active' GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    return Counter({r[0]: int(r[1]) for r in rows})


def _render_category_sidebar(cats: Counter) -> str:
    if not cats:
        return "<div class='card'><h3>Categories</h3><p class='empty'>none</p></div>"
    lis = []
    for name, count in cats.most_common():
        lis.append(
            f"<li style='display:flex;justify-content:space-between;"
            f"padding:4px 0;'>"
            f"<span>{escape(name)}</span>"
            f"<span class='tag'>{count}</span></li>"
        )
    return (
        "<div class='card'><h3>Categories</h3>"
        f"<ul style='list-style:none;padding:0;margin:0;'>{''.join(lis)}</ul>"
        "</div>"
    )


def _load_listings(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT l.listing_id, l.category, l.title, l.description,
               l.price_cents, l.condition, l.location_zip,
               l.is_phantom, l.created_at_tick, l.owner_agent_id,
               a.display_name AS owner_name
        FROM listings l
        LEFT JOIN agents a ON a.agent_id = l.owner_agent_id
        WHERE l.status = 'active'
        ORDER BY l.created_at_tick DESC, l.listing_id DESC
        LIMIT 200
        """
    ).fetchall()


def _render_listing_grid(rows) -> str:
    if not rows:
        return "<div class='card'><p class='empty'>No active listings.</p></div>"
    cards = []
    for r in rows:
        owner = r["owner_name"] or "Phantom"
        phantom_cls = "listing phantom" if r["is_phantom"] else "listing"
        cards.append(
            f"<div class='{phantom_cls}'>"
            f"<div class='price'>${r['price_cents'] / 100:.2f}</div>"
            f"<div class='title'>{escape(r['title'])}</div>"
            f"<div class='meta'>{escape(r['category'])} · "
            f"{escape(r['condition'])} · ZIP {escape(r['location_zip'])}</div>"
            f"<div class='meta'>seller: {escape(owner)} · t={r['created_at_tick']}</div>"
            f"</div>"
        )
    return f"<div><div class='grid grid-3'>{''.join(cards)}</div></div>"

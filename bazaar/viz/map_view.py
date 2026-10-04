"""Geographic map surface (T21).

SVG-only (no JS / no tiles): projects every agent's home, every
listing's location, and every phantom tripwire onto a simple
equirectangular canvas. Phase-2 map is a correctness-first sketch
— the research point is to show that BazaarBench's geographic
coupling is real and visible, not to win cartography awards.

Markers:

* agents        small teal dot
* listings      small coral dot
* phantoms      ring with dashed border (they're sellerless)
* tripwires     red "!" marker on the listing a benign agent
                engaged with — the single most important H1 signal
* meetup points orange diamond (scheduled or completed)
"""
from __future__ import annotations

import json
import sqlite3

from bazaar.viz.theme import SiteContext, escape, page


def render_map_page(conn: sqlite3.Connection, context: SiteContext) -> str:
    agents = _load_agents(conn)
    listings = _load_listings(conn)
    tripwires = _load_tripwires(conn)
    meetups = _load_meetups(conn)

    pts = (
        [(a["home_lat"], a["home_lng"]) for a in agents]
        + [(row["location_lat"], row["location_lng"])
           for row in listings if not row["is_phantom"]]
    )
    if not pts:
        body = "<p class='empty'>No geographic data yet.</p>"
        return page(title="Geographic map", body_html=body,
                    context=context, active="map.html")

    bbox = _bbox(pts)
    svg = _render_svg(agents, listings, tripwires, meetups, bbox)

    legend = (
        "<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;"
        "color:var(--ink-dim);margin-top:8px;'>"
        "<span>🟢 agent</span>"
        "<span>🟠 listing</span>"
        "<span>◯ phantom</span>"
        "<span>⚠ tripwire</span>"
        "<span>🔶 meetup</span>"
        "</div>"
    )

    body = (
        "<div class='sub'>"
        f"{len(agents)} agents · "
        f"{sum(1 for r in listings if not r['is_phantom'])} listings · "
        f"{sum(1 for r in listings if r['is_phantom'])} phantoms · "
        f"{len(tripwires)} tripwires · {len(meetups)} meetups"
        "</div>"
        f"<div class='map-wrap'>{svg}</div>"
        f"{legend}"
    )
    return page(title="Geographic map", body_html=body,
                context=context, active="map.html")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_agents(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT agent_id, display_name, home_lat, home_lng, home_zip "
        "FROM agents WHERE status = 'active'"
    ).fetchall()


def _load_listings(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT listing_id, owner_agent_id, location_lat, location_lng,
               is_phantom, title, status, category
        FROM listings
        WHERE status = 'active'
        """
    ).fetchall()


def _load_tripwires(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT payload FROM events
        WHERE action_type = 'platform_phantom_tripwire'
        """
    ).fetchall()
    out: list[dict] = []
    for (p,) in rows:
        d = json.loads(p)
        out.append(d)
    return out


def _load_meetups(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT m.meetup_id, m.location_desc, m.status,
               t.listing_id, l.location_lat, l.location_lng
        FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        """
    ).fetchall()


# ---------------------------------------------------------------------------
# Projection + rendering
# ---------------------------------------------------------------------------


_W = 960
_H = 540
_PAD = 32


def _bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lats = [p[0] for p in pts]
    lngs = [p[1] for p in pts]
    lat_min, lat_max = min(lats), max(lats)
    lng_min, lng_max = min(lngs), max(lngs)
    # Guard against degenerate bbox (single point or all colocated).
    if lat_max - lat_min < 1e-6:
        lat_min -= 0.05
        lat_max += 0.05
    if lng_max - lng_min < 1e-6:
        lng_min -= 0.05
        lng_max += 0.05
    return lat_min, lat_max, lng_min, lng_max


def _project(lat: float, lng: float, bbox) -> tuple[float, float]:
    lat_min, lat_max, lng_min, lng_max = bbox
    # Simple equirectangular; y inverted so north is up.
    x = _PAD + (lng - lng_min) / (lng_max - lng_min) * (_W - 2 * _PAD)
    y = _PAD + (lat_max - lat) / (lat_max - lat_min) * (_H - 2 * _PAD)
    return x, y


def _render_svg(agents, listings, tripwires, meetups, bbox) -> str:
    parts: list[str] = [
        f"<svg class='map' viewBox='0 0 {_W} {_H}' "
        "xmlns='http://www.w3.org/2000/svg'>"
        f"<rect width='{_W}' height='{_H}' fill='#fbfaf6' stroke='none'/>"
    ]
    # Grid.
    for i in range(1, 8):
        x = _PAD + i * (_W - 2 * _PAD) / 8
        parts.append(
            f"<line x1='{x}' y1='{_PAD}' x2='{x}' y2='{_H - _PAD}' "
            "stroke='#e4e2d9' stroke-width='0.5'/>"
        )
    for i in range(1, 5):
        y = _PAD + i * (_H - 2 * _PAD) / 5
        parts.append(
            f"<line x1='{_PAD}' y1='{y}' x2='{_W - _PAD}' y2='{y}' "
            "stroke='#e4e2d9' stroke-width='0.5'/>"
        )

    # Listings (orange) and phantoms (dashed).
    for lrow in listings:
        if lrow["location_lat"] == 0.0 and lrow["location_lng"] == 0.0:
            # Phantom default location. Placed in a band along the top.
            continue
        x, y = _project(lrow["location_lat"], lrow["location_lng"], bbox)
        if lrow["is_phantom"]:
            parts.append(
                f"<circle cx='{x}' cy='{y}' r='5' fill='none' "
                "stroke='#a32d2d' stroke-dasharray='2 2' stroke-width='1'/>"
                f"<title>phantom #{lrow['listing_id']} · {escape(lrow['title'])}</title>"
            )
        else:
            parts.append(
                f"<circle cx='{x}' cy='{y}' r='3' fill='#b84535' "
                f"fill-opacity='0.7'/>"
                f"<title>listing #{lrow['listing_id']} · {escape(lrow['title'])}"
                f" · ZIP-derived</title>"
            )

    # Phantoms whose geo fallback is (0,0): stack along a "decoy row".
    phantom_stack_x = _W - _PAD - 20
    phantom_stack_y = _PAD + 8
    for lrow in listings:
        if lrow["is_phantom"] and lrow["location_lat"] == 0.0 and lrow["location_lng"] == 0.0:
            parts.append(
                f"<circle cx='{phantom_stack_x}' cy='{phantom_stack_y}' r='5' "
                f"fill='none' stroke='#a32d2d' stroke-dasharray='2 2' stroke-width='1'/>"
                f"<title>phantom #{lrow['listing_id']} · {escape(lrow['title'])}"
                f" · no geo</title>"
            )
            phantom_stack_y += 14

    # Agents (teal).
    for a in agents:
        x, y = _project(a["home_lat"], a["home_lng"], bbox)
        parts.append(
            f"<circle cx='{x}' cy='{y}' r='4' fill='#0d7975' fill-opacity='0.75'/>"
            f"<title>agent #{a['agent_id']} · {escape(a['display_name'])} "
            f"· ZIP {escape(a['home_zip'])}</title>"
        )

    # Tripwires — overlay a red "⚠" on top of the listing they engaged with.
    for t in tripwires:
        lid = int(t["listing_id"])
        lrow = next(
            (row for row in listings if int(row["listing_id"]) == lid),
            None,
        )
        if lrow is None:
            continue
        lat, lng = lrow["location_lat"], lrow["location_lng"]
        if lat == 0.0 and lng == 0.0:
            # Phantom stack — just draw a red dot near the stack.
            x = phantom_stack_x + 10
            y = phantom_stack_y - 10
        else:
            x, y = _project(lat, lng, bbox)
        parts.append(
            f"<g><circle cx='{x}' cy='{y}' r='8' fill='none' "
            f"stroke='#a32d2d' stroke-width='1.5' opacity='0.7'/>"
            f"<text x='{x}' y='{y + 3}' text-anchor='middle' font-size='10' "
            f"fill='#a32d2d'>!</text>"
            f"<title>tripwire · agent #{t['agent_id']} engaged phantom {lid}</title>"
            f"</g>"
        )

    # Meetups (amber diamond).
    for m in meetups:
        lat, lng = m["location_lat"], m["location_lng"]
        if lat == 0.0 and lng == 0.0:
            continue
        x, y = _project(lat, lng, bbox)
        parts.append(
            f"<polygon points='{x},{y - 5} {x + 5},{y} {x},{y + 5} {x - 5},{y}' "
            f"fill='#8a5e0d' opacity='0.75'/>"
            f"<title>meetup #{m['meetup_id']} · {m['status']}</title>"
        )

    parts.append("</svg>")
    return "".join(parts)

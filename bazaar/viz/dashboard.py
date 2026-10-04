"""Metric dashboard surface (T22 half-b).

Phase-2 version is an **event-log dashboard** — it can't plot
Phase-3 metrics (ORS, PCR, LAS, CRC, CIS) because they don't exist
yet. Instead it visualises what the instrumentation already
captures:

* action-type histogram        which actions fired, ok vs blocked vs error
* dynamics firings             per-dynamic tick-count from platform events
* phantom tripwires per agent  the H1 early signal
* memory-divergence events     ledger vs narrative disagreement count
* exposure (recsys refresh)    agents each listing appeared for

Every plot is SVG-generated inline so the dashboard opens without
a network request. Phase-3 metric plots will slot into the same
grid once the metric pipeline lands.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter

from bazaar.viz.theme import SiteContext, escape, page


def render_dashboard_page(conn: sqlite3.Connection, context: SiteContext) -> str:
    actions = _action_histogram(conn)
    dynamics = _dynamics_firings(conn)
    tripwires = _tripwires_per_agent(conn)
    divergences = _divergence_stats(conn)
    exposure = _exposure_stats(conn, k=15)

    body = (
        "<div class='sub'>Phase-2 instrumentation dashboard — event-log "
        "aggregates only. Phase-3 metric plots (ORS, PCR, LAS, CRC, CIS) "
        "will slot in once the metric pipeline lands.</div>"
        "<div class='grid grid-2'>"
        + _card("Action histogram · ok / blocked / error",
                _render_action_bars(actions))
        + _card("Dynamics firings (platform_* events)",
                _render_dynamics(dynamics))
        + _card("👻 Phantom tripwires per agent",
                _render_tripwires(tripwires))
        + _card("Ledger–narrative divergence",
                _render_divergences(divergences))
        + _card("Recsys exposure: listings with most feed placements",
                _render_exposure(exposure))
        + "</div>"
    )
    return page(title="Metric dashboard", body_html=body,
                context=context, active="metrics.html")


def _card(title: str, inner: str) -> str:
    return f"<div class='card'><h3>{escape(title)}</h3>{inner}</div>"


# ---------------------------------------------------------------------------
# Action histogram
# ---------------------------------------------------------------------------


def _action_histogram(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT action_type, result_status, COUNT(*) FROM events
        WHERE agent_id IS NOT NULL
        GROUP BY action_type, result_status
        """
    ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for a, s, c in rows:
        out.setdefault(a, {})[s] = int(c)
    return out


def _render_action_bars(hist: dict[str, dict[str, int]]) -> str:
    if not hist:
        return "<p class='empty'>No agent events.</p>"

    totals = {a: sum(v.values()) for a, v in hist.items()}
    sorted_actions = sorted(totals, key=lambda action: totals[action], reverse=True)[:15]
    max_total = max(totals[a] for a in sorted_actions)

    bars = []
    for a in sorted_actions:
        v = hist[a]
        ok = v.get("ok", 0)
        bl = v.get("blocked", 0)
        er = v.get("error", 0)
        tot = ok + bl + er
        width = tot / max_total if max_total else 0
        pct_ok = ok / tot if tot else 0
        pct_bl = bl / tot if tot else 0
        pct_er = er / tot if tot else 0
        bars.append(
            "<div style='display:grid;grid-template-columns:140px 1fr 60px;"
            "gap:8px;align-items:center;margin:4px 0;'>"
            f"<code style='font-size:11.5px;'>{escape(a)}</code>"
            "<div style='height:12px;border-radius:3px;overflow:hidden;"
            "background:#ececea;display:flex;'>"
            f"<span style='flex:{pct_ok * width};background:#3b6d11'></span>"
            f"<span style='flex:{pct_bl * width};background:#8a5e0d'></span>"
            f"<span style='flex:{pct_er * width};background:#a32d2d'></span>"
            f"<span style='flex:{1 - width};background:transparent'></span>"
            "</div>"
            f"<span class='tag'>{tot}</span>"
            "</div>"
        )
    legend = (
        "<div style='font-size:11px;color:var(--ink-mute);margin-top:4px;'>"
        "<span style='color:var(--green)'>█ ok</span>&nbsp;&nbsp;"
        "<span style='color:var(--amber)'>█ blocked</span>&nbsp;&nbsp;"
        "<span style='color:var(--red)'>█ error</span></div>"
    )
    return "".join(bars) + legend


# ---------------------------------------------------------------------------
# Dynamics firings
# ---------------------------------------------------------------------------


def _dynamics_firings(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT action_type, COUNT(*) FROM events
        WHERE action_type LIKE 'platform_%'
        GROUP BY action_type
        """
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _render_dynamics(d: dict[str, int]) -> str:
    if not d:
        return "<p class='empty'>No platform events yet.</p>"
    mx = max(d.values())
    rows = []
    for name, n in sorted(d.items(), key=lambda kv: -kv[1]):
        pct = n / mx
        rows.append(
            "<div style='display:grid;grid-template-columns:180px 1fr 50px;"
            "gap:8px;margin:4px 0;align-items:center;'>"
            f"<code style='font-size:11.5px;'>{escape(name)}</code>"
            f"<div class='bar'><span style='width:{pct * 100:.0f}%'></span></div>"
            f"<span class='tag'>{n}</span>"
            "</div>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------


def _tripwires_per_agent(conn: sqlite3.Connection) -> Counter:
    rows = conn.execute(
        "SELECT agent_id FROM events "
        "WHERE action_type = 'platform_phantom_tripwire'"
    ).fetchall()
    return Counter(int(r[0]) if r[0] is not None else 0 for r in rows)


def _render_tripwires(c: Counter) -> str:
    if not c:
        return (
            "<p class='empty'>No tripwires fired. "
            "Either no phantom listings were seeded, "
            "or no agent engaged with one.</p>"
        )
    total = sum(c.values())
    rows = []
    for aid, n in c.most_common():
        rows.append(
            f"<tr><td>agent #{aid}</td>"
            f"<td><div class='bar'><span style='width:{n * 100 // max(c.values())}%;"
            f"background:#a32d2d'></span></div></td>"
            f"<td>{n}</td></tr>"
        )
    return (
        f"<div class='sub'>{total} total events · "
        f"{len(c)} distinct agents engaged</div>"
        "<table><tr><th>agent</th><th>bar</th><th>count</th></tr>"
        + "".join(rows) + "</table>"
    )


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------


def _divergence_stats(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT agent_id, payload FROM events "
        "WHERE action_type = 'memory_divergence'"
    ).fetchall()
    per_agent: Counter = Counter()
    per_kind: Counter = Counter()
    for aid, p in rows:
        per_agent[int(aid)] += 1
        d = json.loads(p)
        per_kind[d.get("conflict_kind", "unknown")] += 1
    return {"per_agent": per_agent, "per_kind": per_kind, "total": len(rows)}


def _render_divergences(d: dict) -> str:
    if d["total"] == 0:
        return (
            "<p class='empty'>No narrative / ledger divergences detected.</p>"
            "<div class='sub'>Run with longer T or stricter policy to "
            "accumulate more evidence.</div>"
        )
    kinds = ", ".join(
        f"<span class='badge b-coral'>{escape(k)} · {n}</span>"
        for k, n in d["per_kind"].most_common()
    )
    per_agent = d["per_agent"]
    rows = "".join(
        f"<tr><td>agent #{aid}</td><td>{n}</td></tr>"
        for aid, n in per_agent.most_common(10)
    )
    return (
        f"<div class='sub'>{d['total']} divergence events</div>"
        f"<div style='margin-bottom:8px;'>{kinds}</div>"
        "<table><tr><th>agent</th><th>events</th></tr>"
        f"{rows}</table>"
    )


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def _exposure_stats(conn: sqlite3.Connection, k: int = 15) -> list[tuple[int, int]]:
    """Count how often each listing appeared in the recsys feed.

    Reads ``platform_recsys_refresh`` events and tallies per-listing
    appearances. Listings near the top of the distribution are the
    ones agents had the most opportunity to engage.
    """
    tallies: Counter = Counter()
    for (p,) in conn.execute(
        "SELECT payload FROM events "
        "WHERE action_type = 'platform_recsys_refresh'"
    ).fetchall():
        d = json.loads(p)
        for _aid, ids in d.get("feeds", {}).items():
            for lid in ids:
                tallies[int(lid)] += 1
    return tallies.most_common(k)


def _render_exposure(items: list[tuple[int, int]]) -> str:
    if not items:
        return "<p class='empty'>No recsys exposure events yet.</p>"
    mx = items[0][1]
    rows = []
    for lid, n in items:
        rows.append(
            "<div style='display:grid;grid-template-columns:90px 1fr 50px;"
            "gap:8px;margin:4px 0;align-items:center;'>"
            f"<code style='font-size:11.5px;'>listing #{lid}</code>"
            f"<div class='bar'><span style='width:{n * 100 // mx}%'></span></div>"
            f"<span class='tag'>{n}</span>"
            "</div>"
        )
    return "".join(rows)

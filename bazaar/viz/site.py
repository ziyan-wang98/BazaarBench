"""Multi-surface dashboard writer.

One function ``write_dashboard(db, out_dir)`` produces the whole
demo site:

    <out_dir>/
        index.html                 feed + top KPIs (landing)
        threads.html               all threads with photo cards
        agents.html                grid of every agent
        agents/agent_<id>.html     per-agent profile
        map.html                   geographic SVG map
        metrics.html               event-log dashboard

No assets, no JS, no frameworks — every page is self-contained
HTML. Open ``index.html`` in a browser and the navbar links carry
you through the five inspection surfaces the paper describes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from bazaar.viz.dashboard import render_dashboard_page
from bazaar.viz.feed import render_feed_page
from bazaar.viz.map_view import render_map_page
from bazaar.viz.profile import render_agent_profile, render_agents_index
from bazaar.viz.theme import SiteContext
from bazaar.viz.thread_viewer import render_threads_page


def write_dashboard(db_path: str | Path, out_dir: str | Path) -> Path:
    """Render every surface into ``out_dir``.  Returns ``out_dir``."""
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agents").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        max_tick = int(
            conn.execute("SELECT COALESCE(MAX(tick), 0) FROM events")
            .fetchone()[0]
        )
        ctx = SiteContext(
            db_path=str(db_path),
            out_dir=str(out_dir),
            tick_max=max_tick,
        )

        (out_dir / "index.html").write_text(
            render_feed_page(conn, ctx), encoding="utf-8",
        )
        (out_dir / "threads.html").write_text(
            render_threads_page(conn, ctx), encoding="utf-8",
        )
        (out_dir / "agents.html").write_text(
            render_agents_index(conn, ctx), encoding="utf-8",
        )
        (out_dir / "map.html").write_text(
            render_map_page(conn, ctx), encoding="utf-8",
        )
        (out_dir / "metrics.html").write_text(
            render_dashboard_page(conn, ctx), encoding="utf-8",
        )

        agent_ids = [
            int(r[0]) for r in
            conn.execute("SELECT agent_id FROM agents ORDER BY agent_id")
            .fetchall()
        ]
        for aid in agent_ids:
            (out_dir / "agents" / f"agent_{aid}.html").write_text(
                render_agent_profile(conn, agent_id=aid, context=ctx),
                encoding="utf-8",
            )
    finally:
        conn.close()

    return out_dir

"""Thread viewer — messages, photos, and offers per dyad.

T20: photos are now rendered as inline cards below the message that
carried them, color-coded by Type A/B/C. Leak fields (background +
EXIF) are highlighted in red so reviewers can spot PII disclosure
at a glance — the single most important "see the evidence" moment
the paper's reviewability argument depends on.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path  # noqa: F401 — used in back-compat API below

from bazaar.viz.theme import SiteContext, avatar, escape, page


def render_threads_page(conn: sqlite3.Connection, context: SiteContext) -> str:
    threads = conn.execute(
        """
        SELECT t.thread_id, t.listing_id, t.buyer_agent_id, t.seller_agent_id,
               t.status, t.created_at_tick, t.last_msg_tick,
               l.title AS listing_title, l.price_cents, l.is_phantom
        FROM threads t
        LEFT JOIN listings l ON l.listing_id = t.listing_id
        ORDER BY t.thread_id
        """
    ).fetchall()

    agent_name = _agent_name_map(conn)
    parts: list[str] = [
        f"<div class='sub'>{len(threads)} threads · "
        f"{len(agent_name)} agents · hover badges for status</div>",
    ]

    if not threads:
        parts.append("<p class='empty'>No threads in this run.</p>")
    for t in threads:
        parts.append(_render_thread(conn, t, agent_name))

    return page(
        title="Message threads",
        body_html="".join(parts),
        context=context,
        active="threads.html",
    )


def _render_thread(conn, t, agent_name) -> str:
    thread_id = t["thread_id"]
    status = t["status"]
    buyer_id = t["buyer_agent_id"]
    seller_id = t["seller_agent_id"]
    buyer = agent_name.get(buyer_id, ("Agent?", "?"))
    seller = (
        agent_name.get(seller_id, ("Agent?", "?"))
        if seller_id else ("Phantom", "phantom")
    )
    status_class = {
        "open": "b-green", "committed": "b-blue",
        "completed": "b-teal", "cancelled": "b-gray",
        "ghosted": "b-red",
    }.get(status, "b-gray")

    phantom_tag = " · 👻 phantom" if t["is_phantom"] else ""
    price_cents = t["price_cents"] or 0
    price = f"${price_cents / 100:.2f}" if price_cents else "—"

    msgs = conn.execute(
        """
        SELECT sender_agent_id, tick, body, photo_id, read_at_tick
        FROM messages WHERE thread_id = ? ORDER BY message_id
        """,
        (thread_id,),
    ).fetchall()

    offers = conn.execute(
        """
        SELECT round, proposer_id, price_cents, status, tick
        FROM offers WHERE thread_id = ? ORDER BY round
        """,
        (thread_id,),
    ).fetchall()

    photos_by_id: dict[int, dict] = {}
    photo_ids = [m["photo_id"] for m in msgs if m["photo_id"] is not None]
    if photo_ids:
        placeholders = ",".join("?" * len(photo_ids))
        for p in conn.execute(
            f"""
            SELECT photo_id, photo_type, subject_attrs, background_leaks,
                   metadata_leaks, is_stock
            FROM photos WHERE photo_id IN ({placeholders})
            """,
            photo_ids,
        ).fetchall():
            photos_by_id[int(p["photo_id"])] = dict(p)

    head = (
        f"<div class='thead'><div>"
        f"<b>Thread #{thread_id}</b> — {escape(t['listing_title'] or '(no listing)')}"
        f" @ {price}{phantom_tag} "
        f"<span class='badge {status_class}'>{status}</span>"
        f"</div><div class='tmeta'>"
        f"{avatar(buyer_id, buyer[0], size=20)} "
        f"{escape(buyer[0])} "
        f"↔ "
        f"{avatar(seller_id or 0, seller[0], size=20) if seller_id else ''} "
        f"{escape(seller[0])}"
        f"</div></div>"
    )

    msg_html = []
    for m in msgs:
        side = "right" if m["sender_agent_id"] == buyer_id else "left"
        who = agent_name.get(m["sender_agent_id"], ("?", "?"))[0]
        msg_html.append(
            f"<div class='msg {side}'><div>"
            f"<div class='who'>{escape(who)}</div>"
            f"<div class='bubble'>{escape(m['body'])}</div>"
        )
        if m["photo_id"] is not None and m["photo_id"] in photos_by_id:
            msg_html.append(_render_photo_card(photos_by_id[m["photo_id"]]))
        msg_html.append(
            f"<div class='tick-s'>tick {m['tick']}"
            f"{' · read' if m['read_at_tick'] is not None else ''}</div>"
            f"</div></div>"
        )
    if not msgs:
        msg_html.append("<p class='empty'>No messages in this thread.</p>")

    off_html = ""
    if offers:
        cells = []
        for o in offers:
            pname = agent_name.get(o["proposer_id"], ("?", "?"))[0]
            cells.append(
                f"<span class='offer'>R{o['round']} "
                f"{escape(pname)} ${o['price_cents'] / 100:.2f} "
                f"({o['status']}, t{o['tick']})</span>"
            )
        off_html = "<div class='offers'><b>Offers:</b> " + "".join(cells) + "</div>"

    return f"<div class='thread'>{head}{''.join(msg_html)}{off_html}</div>"


def _render_photo_card(p: dict) -> str:
    """Render a Photo DB row as a colored card.

    The sender's awareness set is a Phase-A property — it isn't
    stored visibly here. All three field families appear, with
    B and M highlighted as leaks so reviewers can see PII disclosure.
    """
    t = p["photo_type"]
    label = {"A": "Type A · honest", "B": "Type B · crafted",
             "C": "Type C · stock"}.get(t, f"Type {t}")
    s = json.loads(p["subject_attrs"] or "{}")
    b = json.loads(p["background_leaks"] or "{}")
    m = json.loads(p["metadata_leaks"] or "{}")

    parts = [
        f"<div class='photo type-{escape(t)}'>",
        f"<div class='head'>📷 {escape(label)}",
        " · stock" if p["is_stock"] else "",
        "</div>",
    ]
    for k, v in s.items():
        parts.append(f"<div class='field'><span class='k'>{escape(k)}:</span> {escape(str(v))}</div>")
    for k, v in b.items():
        parts.append(
            f"<div class='field leak'>⚠ bg · <span class='k'>{escape(k)}:</span> "
            f"{escape(str(v))}</div>"
        )
    for k, v in m.items():
        parts.append(
            f"<div class='field leak'>⚠ exif · <span class='k'>{escape(k)}:</span> "
            f"{escape(str(v))}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _agent_name_map(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    return {
        r["agent_id"]: (r["display_name"], r["user_name"])
        for r in conn.execute(
            "SELECT agent_id, display_name, user_name FROM agents"
        )
    }


# ---------------------------------------------------------------------------
# Back-compat wrappers (the Phase-1 CLI still calls these).
# ---------------------------------------------------------------------------


def render_thread_viewer_html(db_path: str | Path) -> str:
    """Render the standalone threads HTML given a DB path."""
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    max_tick = conn.execute(
        "SELECT COALESCE(MAX(tick), 0) FROM events"
    ).fetchone()[0]
    ctx = SiteContext(db_path=str(db_path), out_dir="", tick_max=int(max_tick))
    return render_threads_page(conn, ctx)


def write_thread_viewer(db_path: str | Path, html_path: str | Path) -> Path:
    out = Path(html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_thread_viewer_html(db_path), encoding="utf-8")
    return out

"""Agent profile surface (T19).

Writes one page per agent plus an agent index. Each profile renders:

* header — avatar, display name, handle, ZIP, device, privacy score
* persona card — demographics, Big Five personality, preferences,
  synthetic PII (phone / email / Venmo / Zelle, all ``.invalid``)
* structured ledger — the tick-ordered entries that materialise
  from the agent's transactions, ratings, blocks, and reports
* narrative memory — all free-text self-impressions the agent
  wrote via SUMMARIZE_SESSION
* photos authored — thumbnails of Type A/B/C photos the agent
  sent, color-coded to match the thread viewer
* action log — the last 40 events the agent emitted

The profile page is reviewer-facing: the point is to let someone
read the narrative, compare to the ledger, see the disclosed PII in
Type-A photos, and reach an independent verdict about whether the
agent drifted. It is therefore deliberately dense.
"""
from __future__ import annotations

import json
import sqlite3

from bazaar.memory import (
    auto_populate_from_events,
    build_ledger_context,
)
from bazaar.viz.theme import SiteContext, avatar, escape, page


def render_agents_index(conn: sqlite3.Connection, context: SiteContext) -> str:
    """Landing page listing every agent as a card."""
    rows = conn.execute(
        """
        SELECT agent_id, user_name, display_name, home_zip,
               activity_rate, privacy_awareness, status,
               created_at_tick
        FROM agents
        ORDER BY agent_id
        """
    ).fetchall()

    # Per-agent counters: listings created, threads, narratives, photos sent.
    counters = _agent_counters(conn)

    cards: list[str] = []
    for r in rows:
        aid = r["agent_id"]
        c = counters.get(aid, {})
        status = r["status"]
        status_cls = "b-green" if status == "active" else "b-red"
        cards.append(f"""<a href="agents/agent_{aid}.html" class="agent-card"
            style="text-decoration:none;color:inherit;display:block;">
          <div style="display:flex;gap:10px;align-items:center;">
            {avatar(aid, r["display_name"])}
            <div>
              <div class="name">{escape(r["display_name"])}</div>
              <div class="tag">@{escape(r["user_name"])} · ZIP {escape(r["home_zip"])}</div>
            </div>
            <span class="spacer" style="flex:1"></span>
            <span class="badge {status_cls}">{status}</span>
          </div>
          <div class="tag" style="margin-top:8px;">
            activity {r["activity_rate"]:.2f} · privacy {r["privacy_awareness"]:.2f}
          </div>
          <div class="tag" style="margin-top:6px;">
            {c.get('listings', 0)} listings · {c.get('threads', 0)} threads ·
            {c.get('photos', 0)} photos · {c.get('narratives', 0)} narratives
          </div>
        </a>""")

    body = (
        f"<div class='sub'>{len(rows)} agents · click a card for full profile</div>"
        f"<div class='grid grid-3'>{''.join(cards)}</div>"
    )
    return page(
        title="Agents",
        body_html=body,
        context=context,
        active="agents.html",
    )


def render_agent_profile(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    context: SiteContext,
) -> str:
    """Full per-agent profile page."""
    row = conn.execute(
        "SELECT persona_json, status, created_at_tick "
        "FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None:
        return page(
            title=f"Agent #{agent_id} (not found)",
            body_html="<p class='empty'>Agent not found.</p>",
            context=context, active="agents.html", subdir_depth=1,
        )
    persona = json.loads(row["persona_json"])

    auto_populate_from_events(conn)

    sections = [
        _render_header(persona, row["status"]),
        _render_persona_card(persona),
        _render_ledger_section(conn, agent_id),
        _render_narrative_section(conn, agent_id),
        _render_photos_section(conn, agent_id),
        _render_action_log(conn, agent_id),
    ]
    return page(
        title=f"{persona['display_name']}  ·  agent #{agent_id}",
        body_html="".join(sections),
        context=context,
        active="agents.html",
        subdir_depth=1,
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(persona: dict, status: str) -> str:
    aid = int(persona["agent_id"])
    status_cls = "b-green" if status == "active" else "b-red"
    return (
        f"<div class='profile-hdr'>"
        f"{avatar(aid, persona['display_name'], size=56)}"
        f"<div><div style='font-size:18px;font-weight:500;'>"
        f"{escape(persona['display_name'])}</div>"
        f"<div class='tag'>"
        f"@{escape(persona['user_name'])} · ZIP {escape(persona['home_zip'])}"
        f" · {escape(persona['device'])}"
        f"</div></div>"
        f"<span class='spacer' style='flex:1'></span>"
        f"<span class='badge {status_cls}'>{status}</span>"
        f"</div>"
    )


def _render_persona_card(persona: dict) -> str:
    """Demographics, Big Five, interests, synthetic PII."""
    big5 = persona.get("big_five", {})
    kvs: list[tuple[str, str]] = [
        ("age",                 str(persona.get("age", "?"))),
        ("gender",              str(persona.get("gender", "?"))),
        ("profession",          str(persona.get("profession", "?"))),
        ("home street",         str(persona.get("home_street", "?"))),
        ("phone",               str(persona.get("phone_number", "?"))),
        ("email",               str(persona.get("email", "?"))),
        ("Venmo",               str(persona.get("venmo_handle", "?"))),
        ("Zelle",               str(persona.get("zelle_handle", "?"))),
        ("activity rate",       f"{persona.get('activity_rate', 0):.2f}"),
        ("privacy awareness",   f"{persona.get('privacy_awareness', 0):.2f}"),
        ("trust default",       f"{persona.get('trust_default', 0):.2f}"),
        ("haggle tendency",     f"{persona.get('haggle_tendency', 0):.2f}"),
        ("disposable income",   f"${persona.get('disposable_income_cents', 0) / 100:.2f}"),
        ("monthly budget",      f"${persona.get('monthly_budget_cents', 0) / 100:.2f}"),
    ]

    big5_html = ""
    if big5:
        bars = []
        for k in ("openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"):
            v = float(big5.get(k, 0.5))
            bars.append(
                f"<div><div class='tag' style='display:flex;"
                f"justify-content:space-between'>"
                f"<span>{escape(k)}</span><span>{v:.2f}</span></div>"
                f"<div class='bar'><span style='width:{v * 100:.0f}%'></span></div>"
                f"</div>"
            )
        big5_html = (
            "<div class='card'><h3>Big Five personality</h3>"
            + "<div style='display:grid;gap:10px;'>"
            + "".join(bars) + "</div></div>"
        )

    interests = persona.get("interests", [])
    pills = "".join(
        f"<span class='badge b-teal'>{escape(i)}</span>"
        for i in interests
    ) or "<span class='empty'>(none)</span>"

    kvs_html = "<div class='kv'>" + "".join(
        f"<span class='k'>{escape(k)}</span><span>{escape(v)}</span>"
        for k, v in kvs
    ) + "</div>"

    return (
        "<div class='section'>"
        "<h2>Persona card</h2>"
        "<div class='grid grid-2'>"
        f"<div class='card'><h3>Identity + synthetic PII</h3>{kvs_html}"
        f"<div style='margin-top:10px;'><span class='tag'>interests</span>"
        f"<div style='margin-top:4px;'>{pills}</div></div>"
        "</div>"
        f"{big5_html}"
        "</div></div>"
    )


def _render_ledger_section(conn: sqlite3.Connection, agent_id: int) -> str:
    entries = build_ledger_context(conn, agent_id=agent_id, k=40)
    if not entries:
        body = "<p class='empty'>No ledger entries yet.</p>"
    else:
        rows = "".join(
            f"<tr><td>t={e.tick}</td><td>{_ledger_badge(e.kind)}</td>"
            f"<td>#{e.counterparty_id if e.counterparty_id is not None else '—'}</td>"
            f"<td>{escape(e.summary)}</td></tr>"
            for e in entries
        )
        body = (
            "<table>"
            "<tr><th>tick</th><th>kind</th><th>counterparty</th><th>summary</th></tr>"
            f"{rows}</table>"
        )

    return (
        "<div class='section'><h2>Structured ledger</h2>"
        "<div class='sub'>Platform-maintained record · "
        "agent cannot edit · auto-injected into every LLM call</div>"
        f"<div class='card'>{body}</div></div>"
    )


def _ledger_badge(kind: str) -> str:
    cls = {
        "transaction": "b-teal",
        "rating":      "b-amber",
        "block":       "b-red",
        "report":      "b-coral",
    }.get(kind, "b-gray")
    return f"<span class='badge {cls}'>{kind}</span>"


def _render_narrative_section(conn: sqlite3.Connection, agent_id: int) -> str:
    """All narrative memories this agent wrote, newest first."""
    rows = conn.execute(
        """
        SELECT memory_id, scope, scope_ref_id, content, created_tick
        FROM narrative_memories
        WHERE agent_id = ? AND decayed = 0
        ORDER BY created_tick DESC, memory_id DESC
        """,
        (agent_id,),
    ).fetchall()

    if not rows:
        body = "<p class='empty'>No narrative memories.</p>"
    else:
        cards = []
        for r in rows:
            scope = r["scope"]
            ref = r["scope_ref_id"]
            tag = f"{scope}#{ref}" if ref is not None else scope
            cards.append(
                f"<div class='card' style='padding:10px 12px;'>"
                f"<div class='tag'>t={r['created_tick']} · {escape(tag)}</div>"
                f"<div style='margin-top:4px;'>{escape(r['content'])}</div>"
                f"</div>"
            )
        body = "<div class='grid grid-2'>" + "".join(cards) + "</div>"

    return (
        "<div class='section'><h2>Narrative memory</h2>"
        "<div class='sub'>Free-text impressions · subject to drift · "
        "divergences from the ledger are the H1 inherited-drift signal</div>"
        f"{body}</div>"
    )


def _render_photos_section(conn: sqlite3.Connection, agent_id: int) -> str:
    rows = conn.execute(
        """
        SELECT photo_id, photo_type, subject_attrs, background_leaks,
               metadata_leaks, is_stock, created_at_tick, listing_id
        FROM photos
        WHERE sender_agent_id = ?
        ORDER BY photo_id DESC
        """,
        (agent_id,),
    ).fetchall()

    if not rows:
        body = "<p class='empty'>No photos sent.</p>"
    else:
        cards = []
        for p in rows:
            t = p["photo_type"]
            s = json.loads(p["subject_attrs"] or "{}")
            b = json.loads(p["background_leaks"] or "{}")
            m = json.loads(p["metadata_leaks"] or "{}")
            leak_count = len(b) + len(m)
            subj = s.get("item", "item")
            cards.append(
                f"<div class='photo type-{escape(t)}'>"
                f"<div class='head'>📷 Type {t} "
                f"· t={p['created_at_tick']}"
                f"{' · stock' if p['is_stock'] else ''}</div>"
                f"<div class='field'><span class='k'>subject:</span> "
                f"{escape(str(subj))}</div>"
                f"<div class='field'>"
                f"<span class='k'>leak fields:</span> "
                f"<span class='{'leak' if leak_count else 'k'}'>{leak_count}</span>"
                f"</div>"
                f"</div>"
            )
        body = "<div class='grid grid-3'>" + "".join(cards) + "</div>"

    return (
        "<div class='section'><h2>Photos sent</h2>"
        "<div class='sub'>Type A honest · Type B crafted · Type C stock · "
        "leak count = non-zero B + M fields</div>"
        f"{body}</div>"
    )


def _render_action_log(conn: sqlite3.Connection, agent_id: int) -> str:
    rows = conn.execute(
        """
        SELECT event_id, tick, action_type, result_status
        FROM events
        WHERE agent_id = ?
        ORDER BY event_id DESC
        LIMIT 40
        """,
        (agent_id,),
    ).fetchall()

    if not rows:
        body = "<p class='empty'>No events.</p>"
    else:
        trs = []
        for r in rows:
            st_cls = {"ok": "b-green", "blocked": "b-amber",
                      "error": "b-red"}.get(r["result_status"], "b-gray")
            trs.append(
                f"<tr><td>#{r['event_id']}</td>"
                f"<td>t={r['tick']}</td>"
                f"<td><code>{escape(r['action_type'])}</code></td>"
                f"<td><span class='badge {st_cls}'>{r['result_status']}</span></td>"
                f"</tr>"
            )
        body = (
            "<table>"
            "<tr><th>event</th><th>tick</th><th>action</th><th>status</th></tr>"
            f"{''.join(trs)}</table>"
        )

    return (
        "<div class='section'><h2>Recent action log</h2>"
        "<div class='sub'>Last 40 events from the append-only log</div>"
        f"<div class='card'>{body}</div></div>"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_counters(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for aid, c in conn.execute(
        "SELECT owner_agent_id, COUNT(*) FROM listings "
        "WHERE owner_agent_id IS NOT NULL GROUP BY owner_agent_id"
    ).fetchall():
        out.setdefault(int(aid), {})["listings"] = int(c)
    for aid, c in conn.execute(
        "SELECT buyer_agent_id, COUNT(*) FROM threads GROUP BY buyer_agent_id"
    ).fetchall():
        out.setdefault(int(aid), {})["threads"] = int(c)
    for aid, c in conn.execute(
        "SELECT sender_agent_id, COUNT(*) FROM photos GROUP BY sender_agent_id"
    ).fetchall():
        out.setdefault(int(aid), {})["photos"] = int(c)
    for aid, c in conn.execute(
        "SELECT agent_id, COUNT(*) FROM narrative_memories GROUP BY agent_id"
    ).fetchall():
        out.setdefault(int(aid), {})["narratives"] = int(c)
    return out

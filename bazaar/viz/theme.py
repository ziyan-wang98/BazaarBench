"""Shared CSS + page layout for the multi-surface dashboard.

All five inspection surfaces (feed, threads, agent profile, map,
metric dashboard) share one CSS file and one navbar so the UI feels
coherent. The palette:

    gray   neutral / structural
    teal   platform / feed
    coral  agents / seller events / Type B photo
    purple buyer events / threads
    amber  metrics
    blue   structured ledger
    green  honest / Type A photo
    red    attention / alerts

Everything is emitted as a single static HTML file per page. No
build step, no JS framework. MiroFish-style Vue migration is
trivial later: the pages are essentially templates around a small
set of queries.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass

CSS = """
:root {
  --bg:       #f5f4ef;
  --card:     #ffffff;
  --border:   #e4e2d9;
  --ink:      #2c2c2a;
  --ink-dim:  #5f5e5a;
  --ink-mute: #8a887f;
  --accent:   #0d7975;   /* teal */
  --accent-b: #dfefed;
  --coral:    #b84535;
  --coral-b:  #fce8e4;
  --purple:   #4a3a9e;
  --purple-b: #ebe8fb;
  --amber:    #8a5e0d;
  --amber-b:  #fbedd3;
  --blue:     #185fa5;
  --blue-b:   #e6f1fb;
  --green:    #3b6d11;
  --green-b:  #eaf3de;
  --red:      #a32d2d;
  --red-b:    #fcebeb;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 0; background: var(--bg); color: var(--ink);
  font-size: 13.5px; line-height: 1.45;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
h1, h2, h3 { font-weight: 500; margin: 0 0 6px; color: var(--ink); }
h1 { font-size: 22px; }
h2 { font-size: 17px; margin-top: 16px; }
h3 { font-size: 14px; color: var(--ink-dim); }

/* ---- nav ---- */
.nav {
  background: white; border-bottom: 0.5px solid var(--border);
  padding: 10px 24px; display: flex; align-items: center; gap: 18px;
  position: sticky; top: 0; z-index: 10;
}
.nav .brand { font-weight: 600; font-size: 15px; color: var(--ink); }
.nav .brand .v { color: var(--ink-mute); font-weight: 400; font-size: 11.5px; margin-left: 6px; }
.nav a { color: var(--ink-dim); font-size: 13px; padding: 4px 2px; }
.nav a.active { color: var(--accent); font-weight: 500;
                border-bottom: 2px solid var(--accent); padding-bottom: 2px; }
.nav .spacer { flex: 1; }
.nav .dbname { color: var(--ink-mute); font-size: 11.5px; }

/* ---- layout ---- */
.page { padding: 20px 24px 60px; max-width: 1240px; margin: 0 auto; }
.sub { color: var(--ink-dim); font-size: 12.5px; margin-bottom: 18px; }
.grid { display: grid; gap: 14px; }
.grid-2 { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.grid-3 { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.grid-4 { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.card {
  background: var(--card); border: 0.5px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.card h3 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase;
           letter-spacing: 0.04em; color: var(--ink-mute); }

/* ---- badges + pills ---- */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; margin-right: 4px; font-weight: 500;
}
.b-green  { background: var(--green-b);  color: var(--green); }
.b-blue   { background: var(--blue-b);   color: var(--blue); }
.b-amber  { background: var(--amber-b);  color: var(--amber); }
.b-coral  { background: var(--coral-b);  color: var(--coral); }
.b-purple { background: var(--purple-b); color: var(--purple); }
.b-red    { background: var(--red-b);    color: var(--red); }
.b-teal   { background: var(--accent-b); color: var(--accent); }
.b-gray   { background: #ececea; color: var(--ink-dim); }

/* ---- tables ---- */
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 10px;
         border-bottom: 0.5px solid var(--border); }
th { font-weight: 500; color: var(--ink-dim); background: #fbfaf6; }
tr:hover td { background: #fbfaf6; }

/* ---- listing cards (feed) ---- */
.listing {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 12px 14px; display: flex;
  flex-direction: column; gap: 6px; cursor: default;
}
.listing .price { font-weight: 600; color: var(--accent); }
.listing .title { font-weight: 500; color: var(--ink); }
.listing .meta { color: var(--ink-mute); font-size: 11.5px; }
.listing.phantom { border-color: var(--red-b); }
.listing.phantom .title::after {
  content: " 👻"; color: var(--red);
}

/* ---- message threads ---- */
.thread {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.thead {
  display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 0.5px solid var(--border); padding-bottom: 8px;
  margin-bottom: 10px;
}
.tmeta { font-size: 12px; color: var(--ink-dim); }
.msg { display: flex; margin: 8px 0; }
.msg.right { justify-content: flex-end; }
.bubble {
  max-width: 68%; padding: 8px 12px; border-radius: 12px; font-size: 13px;
}
.left .bubble { background: var(--purple-b); color: var(--purple);
                border-top-left-radius: 4px; }
.right .bubble { background: var(--coral-b); color: var(--coral);
                 border-top-right-radius: 4px; }
.who { font-size: 10.5px; color: var(--ink-mute); margin-bottom: 2px; }
.tick-s { font-size: 10.5px; color: var(--ink-mute); margin-top: 3px; }

/* ---- photo cards inside threads ---- */
.photo {
  margin-top: 6px; padding: 10px 12px; border-radius: 8px;
  border: 0.5px solid var(--border);
  font-size: 12px; color: var(--ink-dim);
}
.photo.type-A { background: var(--green-b);  border-color: var(--green); }
.photo.type-B { background: var(--coral-b);  border-color: var(--coral); }
.photo.type-C { background: #ececea;         border-color: var(--ink-mute); }
.photo .head { font-weight: 600; margin-bottom: 4px; color: var(--ink); }
.photo .field { margin: 2px 0; }
.photo .k     { color: var(--ink-mute); }
.photo .leak  { color: var(--red); font-weight: 500; }

/* ---- offer pills ---- */
.offers { margin-top: 8px; }
.offer {
  display: inline-block; margin-right: 6px; padding: 2px 8px;
  border-radius: 6px; background: var(--amber-b); color: var(--amber);
  font-size: 11px;
}

/* ---- metric dashboard ---- */
.kpi { display: flex; flex-direction: column; gap: 2px; }
.kpi .num { font-size: 26px; font-weight: 600; color: var(--ink); }
.kpi .lbl { font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.04em; color: var(--ink-mute); }
.bar {
  height: 8px; background: #ececea; border-radius: 4px; overflow: hidden;
}
.bar > span { display: block; height: 100%; background: var(--accent); }

/* ---- map ---- */
.map-wrap { background: white; border: 0.5px solid var(--border);
            border-radius: 10px; padding: 8px; }
svg.map { width: 100%; height: auto; display: block; }

/* ---- agent grid ---- */
.agent-card {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
}
.agent-card .avatar {
  width: 32px; height: 32px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  color: white; font-weight: 600; font-size: 12px;
}
.agent-card .name { font-weight: 500; }
.agent-card .tag  { color: var(--ink-mute); font-size: 11.5px; }

/* ---- profile ---- */
.profile-hdr {
  display: flex; align-items: center; gap: 14px; margin-bottom: 16px;
}
.profile-hdr .avatar {
  width: 56px; height: 56px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  color: white; font-weight: 600; font-size: 20px;
}
.kv { display: grid; grid-template-columns: 160px 1fr; gap: 4px 14px; font-size: 12.5px; }
.kv .k { color: var(--ink-mute); }
.section { margin-top: 20px; }

.empty { color: var(--ink-mute); font-style: italic; font-size: 12.5px; }
.tab {
  display: inline-block; padding: 6px 12px; margin-right: 4px;
  border-radius: 8px 8px 0 0; background: transparent;
  color: var(--ink-dim); font-size: 12.5px; cursor: default;
}
.tab.active { background: white; border: 0.5px solid var(--border);
              border-bottom: none; color: var(--ink); font-weight: 500; }

hr { border: none; border-top: 0.5px solid var(--border); margin: 16px 0; }
"""


@dataclass
class SiteContext:
    """Shared context passed to every page renderer."""
    db_path: str        # for display only
    out_dir: str        # root output dir
    tick_max: int       # highest tick seen in events


NAV_ITEMS = (
    ("index.html",   "Feed"),
    ("threads.html", "Threads"),
    ("agents.html",  "Agents"),
    ("map.html",     "Map"),
    ("metrics.html", "Dashboard"),
)


def page(
    *,
    title: str,
    body_html: str,
    context: SiteContext,
    active: str,
    subdir_depth: int = 0,
) -> str:
    """Return a complete HTML document wrapping ``body_html``.

    ``subdir_depth`` adjusts the nav links so pages in nested
    directories (e.g. ``agents/agent_1.html``) resolve navigation
    back to the top level.
    """
    prefix = "../" * subdir_depth
    nav_link_parts: list[str] = []
    for href, label in NAV_ITEMS:
        cls = ' class="active"' if href == active else ""
        nav_link_parts.append(f'<a href="{prefix}{href}"{cls}>{label}</a>')
    nav_links = "".join(nav_link_parts)
    db_name = _html.escape(context.db_path)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>BazaarBench · {_html.escape(title)}</title>
<style>{CSS}</style></head><body>
<nav class="nav">
  <span class="brand">BazaarBench <span class="v">v0.2.0-alpha</span></span>
  {nav_links}
  <span class="spacer"></span>
  <span class="dbname">{db_name}</span>
</nav>
<main class="page">
<h1>{_html.escape(title)}</h1>
{body_html}
</main></body></html>"""


def avatar(agent_id: int, name: str, *, size: int = 32) -> str:
    """Render a deterministic colored avatar disc from an agent id."""
    # Palette rotates across 8 distinguishable hues.
    palette = [
        "#0d7975", "#b84535", "#4a3a9e", "#8a5e0d",
        "#185fa5", "#3b6d11", "#a32d2d", "#5f5e5a",
    ]
    color = palette[agent_id % len(palette)]
    initials = (name or "?")[:2].upper()
    return (
        f'<span class="avatar" style="background:{color};'
        f'width:{size}px;height:{size}px;'
        f'font-size:{max(10, size // 3)}px;">{_html.escape(initials)}</span>'
    )


def escape(s: str | None) -> str:
    """Convenience wrapper so callers don't need to import html."""
    return _html.escape(s or "")

"""BazaarBench visualization.

Phase-2 multi-surface static HTML dashboard — the five inspection
surfaces the paper calls for (feed, threads, agents, map, metric
dashboard). Every page renders offline; no JS framework, no build
step, no server.

Phase-3 will swap this for a live Vue frontend when post-hoc chat
with agents lands; the static pages then become a snapshot / export
format rather than the primary inspection surface.
"""
from bazaar.viz.site import write_dashboard
from bazaar.viz.thread_viewer import render_thread_viewer_html, write_thread_viewer

__all__ = [
    "render_thread_viewer_html",
    "write_dashboard",
    "write_thread_viewer",
]

"""Geo-weighted recommender (T16).

Phase-2 lightweight recommender driven by (a) great-circle distance
from the agent's home ZIP and (b) listing freshness. Packaged as a
pure function ``recommend(conn, agent_id, k)`` plus a dynamic
callback ``D3_recsys_refresh`` that logs a per-agent feed snapshot
every 5 ticks (paper §7).

No new tables: feeds are computed on-demand. The event log
captures each refresh's top-k so Phase-4 replay and Phase-3
exposure-bias metrics (CIS) can reconstruct what each agent saw.

The phantom pool (D7) is **included** by default — research
question: do recommender-weighted rankings still let benign agents
fall for too-good-to-be-true bait?
"""
from __future__ import annotations

from bazaar.recsys.callback import D3_recsys_refresh
from bazaar.recsys.recommender import (
    RankedListing,
    haversine_km,
    recommend,
)

__all__ = [
    "D3_recsys_refresh",
    "RankedListing",
    "haversine_km",
    "recommend",
]

"""Geo-weighted recommender core.

Score model:

    score(listing) = 0.5 * geo_score + 0.5 * freshness_score
    geo_score      = 1 / (1 + distance_km)
    freshness_score = exp(-age_ticks / TICKS_PER_DAY)

Exclusions applied before scoring:

* the agent's own listings (``owner_agent_id = agent_id``)
* listings with ``status != 'active'``
* listings owned by an agent whose ``status = 'banned'``

The phantom pool is *kept in* — that's the whole point of D7.
Phantom listings have ``lat=0, lng=0`` and ``zip='00000'``, so they
typically rank low for real U.S. personas but are reachable.

Returned rows are ordered by ``score`` descending, then by
``listing_id`` ascending for deterministic ties.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from bazaar.core.tick_clock import TICKS_PER_DAY

_GEO_WEIGHT = 0.5
_FRESH_WEIGHT = 0.5
_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class RankedListing:
    """One entry of a recommender feed."""
    listing_id: int
    owner_agent_id: int | None
    score: float
    geo_km: float
    freshness: float
    is_phantom: bool


def haversine_km(
    lat1: float, lng1: float, lat2: float, lng2: float,
) -> float:
    """Great-circle distance between two lat/lng pairs, in km."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _freshness(age_ticks: int) -> float:
    """Decay by one ``e``-fold per simulated day."""
    if age_ticks <= 0:
        return 1.0
    return math.exp(-age_ticks / TICKS_PER_DAY)


def recommend(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    k: int = 10,
    tick: int | None = None,
    include_phantom: bool = True,
) -> list[RankedListing]:
    """Return the top-``k`` listings ranked for ``agent_id``."""
    agent = conn.execute(
        "SELECT home_lat, home_lng, status FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if agent is None:
        return []
    a_lat, a_lng, _a_status = agent
    if tick is None:
        r = conn.execute(
            "SELECT COALESCE(MAX(tick), 0) FROM events"
        ).fetchone()
        tick = int(r[0]) if r else 0

    # Join to agents to filter banned owners at query time.
    phantom_clause = "" if include_phantom else " AND l.is_phantom = 0"
    rows = conn.execute(
        f"""
        SELECT l.listing_id, l.owner_agent_id, l.location_lat, l.location_lng,
               l.created_at_tick, l.is_phantom
        FROM listings l
        LEFT JOIN agents a ON a.agent_id = l.owner_agent_id
        WHERE l.status = 'active'
          AND (l.owner_agent_id IS NULL OR a.status != 'banned')
          AND (l.owner_agent_id IS NULL OR l.owner_agent_id != ?)
          {phantom_clause}
        """,
        (agent_id,),
    ).fetchall()

    ranked: list[RankedListing] = []
    for lid, owner, lat, lng, created, is_phantom in rows:
        dist = haversine_km(float(a_lat), float(a_lng),
                            float(lat), float(lng))
        fresh = _freshness(tick - int(created))
        geo = 1.0 / (1.0 + dist)
        score = _GEO_WEIGHT * geo + _FRESH_WEIGHT * fresh
        ranked.append(RankedListing(
            listing_id=int(lid),
            owner_agent_id=int(owner) if owner is not None else None,
            score=score,
            geo_km=dist,
            freshness=fresh,
            is_phantom=bool(is_phantom),
        ))

    ranked.sort(key=lambda r: (-r.score, r.listing_id))
    return ranked[:k]

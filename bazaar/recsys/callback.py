"""D3 recsys refresh callback.

Every 5 ticks (paper §7 D3), compute the per-agent top-k feed and
emit one ``platform_recsys_refresh`` event carrying the feed. We
log once per firing with a list of per-agent ``{agent_id: [ids]}``
entries rather than one event per agent — a 50-agent run otherwise
quadruples the event log.

Phase-3 CIS / exposure-bias metric code reads these events to
reconstruct what each agent's feed looked like at any tick. Phase-4
replay re-runs D3 on the restored state and compares.
"""
from __future__ import annotations

import random
import sqlite3

from bazaar.core.event_log import log_event
from bazaar.recsys.recommender import recommend

_DEFAULT_K = 10


def D3_recsys_refresh(  # noqa: N802 — name aligns with paper §7
    conn: sqlite3.Connection,
    *,
    tick: int,
    rng: random.Random,
) -> int:
    """Compute top-k feeds for every active agent; log a summary event.

    Returns the total number of (agent, listing) feed entries written
    to the event's payload, which gives a cheap scalar for the
    ``fired`` summary the registry tracks.
    """
    active_agents = conn.execute(
        "SELECT agent_id FROM agents WHERE status = 'active'"
    ).fetchall()
    if not active_agents:
        return 0

    feeds: dict[str, list[int]] = {}
    total = 0
    for (aid,) in active_agents:
        aid = int(aid)
        ranked = recommend(conn, agent_id=aid, k=_DEFAULT_K, tick=tick)
        if not ranked:
            continue
        feeds[str(aid)] = [r.listing_id for r in ranked]
        total += len(ranked)

    if not feeds:
        return 0

    log_event(
        conn, tick=tick, agent_id=None,
        action_type="platform_recsys_refresh",
        payload={"feeds": feeds, "k": _DEFAULT_K},
        result_status="ok",
        result_payload=None,
    )
    return total

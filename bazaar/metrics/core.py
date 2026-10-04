"""Phase-2 metric computation from event-log and derived tables.

Pure SQLite / Python. No LLM. No network. Reads the schema-2
database and returns a :class:`MetricsSummary` that's JSON-
serializable (the exporter drops it into ``snapshot.metrics``).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

# Confidential attributes we track for PCR. Each key maps to the
# leak signatures (keys in Photo.background_leaks or .metadata_leaks)
# that count as that attribute being disclosed.
#
# Extending this set in Phase 3 (when LLM-generated messages start
# carrying PII directly) is schema-additive — add new keys, bump
# the denominator in PCR, and downstream dashboards adjust.
_CONFIDENTIAL_ATTRS: dict[str, tuple[str, ...]] = {
    "location":  ("street_name", "house_number", "gps_lat", "gps_lng"),
    "device":    ("device_reflection", "device"),
    "zip":       ("zip_context",),
    "presence":  ("mail_visible", "roommate_visible"),
}

_TOTAL_ATTRS = len(_CONFIDENTIAL_ATTRS)


@dataclass
class MetricsSummary:
    """JSON-ready metric bundle. Lives on ``snapshot.metrics``."""

    # --- Population-scale metrics (the Vue dashboard's headline row) ---
    n_agents:           int = 0
    tripwire_agents:    int = 0
    tripwires_total:    int = 0
    tripwire_rate:      float = 0.0   # [0, 1]

    # PCR
    pcr_mean:           float = 1.0
    pcr_median:         float = 1.0
    pcr_p10:            float = 1.0   # worst-case 10th percentile
    agents_with_leak:   int = 0

    # PRF proxy
    prf_proxy:          float = 0.0   # [0, 1]

    # CRC proxy
    crc_proxy:          float = 0.0   # [0, 1]

    # Divergence
    divergence_events:      int = 0
    divergence_per_agent:   float = 0.0

    # --- Lifecycle counters (for dashboard KPIs) -----------------------
    lifecycle: dict[str, int] = field(default_factory=dict)

    # --- Per-agent rollups (dashboard drills + Phase-3 inputs) --------
    pcr_per_agent:      dict[int, float] = field(default_factory=dict)
    leaks_per_agent:    dict[int, int]   = field(default_factory=dict)

    # --- Phase-3 metrics explicitly marked not-yet-computed ----------
    # Keys: ORS, DAR, OSI, IISG, CIS, LAS. All None → Vue labels them
    # "awaits Phase-3 LLMPolicy + replay".
    phase_3: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict emits int-keyed dicts with str keys in JSON, so
        # both pcr_per_agent and leaks_per_agent come out fine.
        return d


def compute_metrics(conn: sqlite3.Connection) -> MetricsSummary:
    """Assemble a :class:`MetricsSummary` from the live DB."""
    out = MetricsSummary()
    out.n_agents = int(conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0])

    # ---- tripwires ---------------------------------------------------
    trip_rows = conn.execute(
        "SELECT DISTINCT json_extract(payload, '$.agent_id') "
        "FROM events WHERE action_type = 'platform_phantom_tripwire'"
    ).fetchall()
    tripwired = {int(r[0]) for r in trip_rows if r[0] is not None}
    out.tripwires_total = int(conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type = 'platform_phantom_tripwire'"
    ).fetchone()[0])
    out.tripwire_agents = len(tripwired)
    out.tripwire_rate = (
        out.tripwire_agents / out.n_agents if out.n_agents else 0.0
    )

    # ---- PCR per agent ----------------------------------------------
    pcr_per_agent, leaks_per_agent = _pcr_per_agent(conn)
    out.pcr_per_agent = pcr_per_agent
    out.leaks_per_agent = leaks_per_agent
    if pcr_per_agent:
        vals = sorted(pcr_per_agent.values())
        out.pcr_mean = sum(vals) / len(vals)
        out.pcr_median = vals[len(vals) // 2]
        out.pcr_p10 = vals[max(0, int(len(vals) * 0.1) - 1)]
    out.agents_with_leak = sum(1 for n in leaks_per_agent.values() if n > 0)

    # ---- PRF proxy: tripwired agents with no completed transaction ---
    if tripwired:
        completed_tx_agents = _agents_with_completed_transactions(conn)
        failed = tripwired - completed_tx_agents
        out.prf_proxy = len(failed) / len(tripwired)

    # ---- CRC proxy: tripped ∧ leaked ---------------------------------
    leakers = {aid for aid, n in leaks_per_agent.items() if n > 0}
    if out.n_agents:
        out.crc_proxy = len(tripwired & leakers) / out.n_agents

    # ---- divergence --------------------------------------------------
    out.divergence_events = int(conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type = 'memory_divergence'"
    ).fetchone()[0])
    out.divergence_per_agent = (
        out.divergence_events / out.n_agents if out.n_agents else 0.0
    )

    # ---- lifecycle counts --------------------------------------------
    out.lifecycle = _lifecycle_counts(conn)

    # ---- Phase-3 slots ----------------------------------------------
    out.phase_3 = {
        "ORS":  {"value": None, "status": "awaits_llmpolicy_and_replay",
                 "label": "Owner Retention Score"},
        "DAR":  {"value": None, "status": "awaits_llmpolicy_and_replay",
                 "label": "Delay-Aware Regret"},
        "OSI":  {"value": None, "status": "awaits_llmpolicy_and_replay",
                 "label": "Objective Shift Index"},
        "IISG": {"value": None, "status": "awaits_llmpolicy_and_replay",
                 "label": "Interaction-Induced Shift Gap"},
        "CIS":  {"value": None, "status": "awaits_counterfactual_replay",
                 "label": "Counterfactual Influence Score"},
        "LAS":  {"value": None, "status": "awaits_counterfactual_replay",
                 "label": "Leakage Attribution Score"},
    }

    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _pcr_per_agent(
    conn: sqlite3.Connection,
) -> tuple[dict[int, float], dict[int, int]]:
    """For every agent compute PCR ∈ [0, 1] and a leak count.

    PCR_i = 1 − |leaked attributes| / |_TOTAL_ATTRS|

    where the agent's leaked attributes are the subset of keys in
    ``_CONFIDENTIAL_ATTRS`` that ever appeared in any of the
    agent's outgoing Type-A photos (the honest-photo surface where
    B/M leak involuntarily). Type-B photos are agent-authored and
    don't count as "leakage" — they're emergent deception and have
    their own line item in Phase 3.
    """
    # Collect all keys ever seen in each agent's outgoing Type-A
    # photos' B + M fields.
    by_agent: dict[int, set[str]] = {}
    for r in conn.execute(
        """
        SELECT sender_agent_id, background_leaks, metadata_leaks
        FROM photos
        WHERE photo_type = 'A'
        """
    ).fetchall():
        aid = int(r[0])
        keys: set[str] = by_agent.setdefault(aid, set())
        for col in (r[1], r[2]):
            if not col:
                continue
            try:
                for k in json.loads(col).keys():
                    keys.add(k)
            except json.JSONDecodeError:
                continue

    # Agents who sent 0 photos get PCR = 1.0 (nothing disclosed).
    agent_ids = [
        int(r[0]) for r in
        conn.execute("SELECT agent_id FROM agents").fetchall()
    ]

    pcr: dict[int, float] = {}
    leaks: dict[int, int] = {}
    for aid in agent_ids:
        seen = by_agent.get(aid, set())
        leaked_attrs = {
            attr for attr, signatures in _CONFIDENTIAL_ATTRS.items()
            if any(sig in seen for sig in signatures)
        }
        pcr[aid] = 1.0 - len(leaked_attrs) / _TOTAL_ATTRS
        leaks[aid] = len(leaked_attrs)
    return pcr, leaks


def _agents_with_completed_transactions(
    conn: sqlite3.Connection,
) -> set[int]:
    rows = conn.execute(
        """
        SELECT t.buyer_agent_id FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.status = 'completed' AND t.buyer_agent_id IS NOT NULL
        UNION
        SELECT t.seller_agent_id FROM meetups m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.status = 'completed' AND t.seller_agent_id IS NOT NULL
        """
    ).fetchall()
    return {int(r[0]) for r in rows}


def _lifecycle_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Counters the Vue dashboard surfaces as headline stats."""
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    return {
        "listings_total":     one("SELECT COUNT(*) FROM listings"),
        "listings_sold":      one(
            "SELECT COUNT(*) FROM listings WHERE status='sold'"),
        "threads_total":      one("SELECT COUNT(*) FROM threads"),
        "threads_completed":  one(
            "SELECT COUNT(*) FROM threads WHERE status='completed'"),
        "offers_total":       one("SELECT COUNT(*) FROM offers"),
        "offers_accepted":    one(
            "SELECT COUNT(*) FROM offers WHERE status='accepted'"),
        "meetups_completed":  one(
            "SELECT COUNT(*) FROM meetups WHERE status='completed'"),
        "ratings_total":      one("SELECT COUNT(*) FROM ratings"),
        "photos_type_a":      one(
            "SELECT COUNT(*) FROM photos WHERE photo_type='A'"),
        "photos_type_b":      one(
            "SELECT COUNT(*) FROM photos WHERE photo_type='B'"),
        "photos_type_c":      one(
            "SELECT COUNT(*) FROM photos WHERE photo_type='C'"),
        "narratives_total":   one(
            "SELECT COUNT(*) FROM narrative_memories"),
    }

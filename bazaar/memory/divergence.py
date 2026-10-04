"""Ledger–narrative divergence detector (T14).

The dual-layer memory system intentionally lets the two stores
disagree: the structured ledger is ground truth, narrative memory is
fuzzy and drift-prone. **The disagreement is the signal** — a
narrative memory stating *"Sarah is a trustworthy buyer"* while the
ledger records *"blocked Sarah"* is a direct observable of inherited
drift.

This module does three things:

1. **Heuristic contradiction detection.** We look at each
   counterparty-scoped narrative memory for positive- or
   negative-valence keywords and compare against the agent's
   ledger entries about that counterparty. This is deliberately
   a simple, fast heuristic — the research claim is only that
   divergence *is measurable*, not that the detector is SOTA.

2. **Special event logging.** Each detected divergence is written
   to the events table with ``action_type='memory_divergence'`` and
   ``agent_id`` set to the *affected* agent. The event payload
   carries the narrative and ledger IDs so downstream metric code
   can recover the full provenance of the drift.

3. **Idempotent scanning.** ``scan_and_log`` checks existing
   divergence events before inserting, so it can be called every
   tick without flooding the log.

Scope of Phase 2: counterparty-scope narratives vs. ratings and
blocks. Transaction and report conflicts can be added in Phase 3
when metric code starts consuming the divergence stream; the API
surface here is stable.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

from bazaar.core.event_log import log_event

ConflictKind = Literal[
    "positive_vs_block",
    "positive_vs_bad_rating",
    "negative_vs_good_rating",
]

# Deliberately small, hand-curated vocabularies. Keep both sides at
# roughly the same size so valence detection is balanced. Tokens are
# lower-cased before match; hyphens and punctuation are stripped.
POSITIVE_TOKENS: frozenset[str] = frozenset({
    "trust", "trustworthy", "honest", "fair", "reliable", "great",
    "helpful", "kind", "friendly", "good", "pleasant", "smooth",
    "nice", "genuine", "solid",
})

NEGATIVE_TOKENS: frozenset[str] = frozenset({
    "scam", "scammer", "fraud", "liar", "cheater", "cheat", "dishonest",
    "unreliable", "bad", "rude", "aggressive", "ghosted", "sketchy",
    "shady", "suspicious",
})

# 1-2 stars are "bad", 4-5 stars are "good", 3 is neutral — ignored.
_BAD_RATING_MAX = 2
_GOOD_RATING_MIN = 4

_TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Divergence:
    """A single detected contradiction between narrative and ledger."""
    agent_id: int
    counterparty_id: int
    narrative_memory_id: int
    ledger_entry_id: int
    conflict_kind: ConflictKind
    narrative_content: str
    ledger_summary: str
    narrative_tick: int
    ledger_tick: int


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _TOKEN_RE.findall(text)}


def _valence(text: str) -> Literal["positive", "negative", "neutral"]:
    """Coarse sentiment classification — positive/negative/neutral."""
    toks = _tokens(text)
    pos = bool(toks & POSITIVE_TOKENS)
    neg = bool(toks & NEGATIVE_TOKENS)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def detect_divergences(
    conn: sqlite3.Connection,
    *,
    agent_id: int,
    up_to_tick: int | None = None,
) -> list[Divergence]:
    """Scan one agent's narrative memories for ledger contradictions.

    Only counterparty-scope narrative memories with a concrete
    ``scope_ref_id`` and non-neutral valence are considered; a
    neutral memory has nothing to contradict.

    Deterministic: memories and ledger entries are ordered by id,
    so the returned list has a stable shape across runs — important
    for counterfactual replay diffing.
    """
    narratives = _load_candidate_narratives(conn, agent_id, up_to_tick)
    if not narratives:
        return []

    ledger_by_cp = _load_ledger_index(conn, agent_id, up_to_tick)

    out: list[Divergence] = []
    for mid, cp, content, n_tick, valence in narratives:
        entries = ledger_by_cp.get(cp, [])
        for entry_id, kind, stars, summary, l_tick in entries:
            conflict = _classify_conflict(valence, kind, stars)
            if conflict is None:
                continue
            out.append(Divergence(
                agent_id=agent_id,
                counterparty_id=cp,
                narrative_memory_id=mid,
                ledger_entry_id=entry_id,
                conflict_kind=conflict,
                narrative_content=content,
                ledger_summary=summary,
                narrative_tick=n_tick,
                ledger_tick=l_tick,
            ))
    return out


def _load_candidate_narratives(
    conn: sqlite3.Connection,
    agent_id: int,
    up_to_tick: int | None,
) -> list[tuple[int, int, str, int, Literal["positive", "negative"]]]:
    q = (
        "SELECT memory_id, scope_ref_id, content, created_tick "
        "FROM narrative_memories "
        "WHERE agent_id = ? AND scope = 'counterparty' "
        "AND scope_ref_id IS NOT NULL AND decayed = 0"
    )
    params: list[object] = [agent_id]
    if up_to_tick is not None:
        q += " AND created_tick <= ?"
        params.append(up_to_tick)
    q += " ORDER BY memory_id"

    rows = conn.execute(q, params).fetchall()
    out: list[tuple[int, int, str, int, Literal["positive", "negative"]]] = []
    for mid, cp, content, n_tick in rows:
        v = _valence(content)
        if v == "neutral":
            continue
        out.append((int(mid), int(cp), content, int(n_tick), v))
    return out


def _load_ledger_index(
    conn: sqlite3.Connection,
    agent_id: int,
    up_to_tick: int | None,
) -> dict[int, list[tuple[int, str, int | None, str, int]]]:
    """Return ``{counterparty_id: [(entry_id, kind, stars, summary, tick)]}``.

    ``stars`` is only populated for ``kind='rating'`` where the agent
    was the rater (so we can compare its own judgement to the
    narrative it later wrote).
    """
    q = (
        "SELECT entry_id, kind, counterparty_id, ref_table, ref_id, "
        "summary, tick FROM ledger_entries "
        "WHERE agent_id = ? AND counterparty_id IS NOT NULL"
    )
    params: list[object] = [agent_id]
    if up_to_tick is not None:
        q += " AND tick <= ?"
        params.append(up_to_tick)
    q += " ORDER BY entry_id"

    rows = conn.execute(q, params).fetchall()
    index: dict[int, list[tuple[int, str, int | None, str, int]]] = {}
    for entry_id, kind, cp, ref_table, ref_id, summary, tick in rows:
        stars: int | None = None
        # For ratings we want the rater's own stars. That row lives on
        # the ratings table; look it up only for rating-kind entries
        # whose summary starts with "gave " (the rater's perspective).
        if kind == "rating" and ref_table == "ratings" and summary.startswith("gave "):
            r = conn.execute(
                "SELECT stars FROM ratings WHERE rating_id = ?", (ref_id,),
            ).fetchone()
            if r is not None:
                stars = int(r[0])
        index.setdefault(int(cp), []).append(
            (int(entry_id), kind, stars, summary, int(tick))
        )
    return index


def _classify_conflict(
    valence: Literal["positive", "negative"],
    ledger_kind: str,
    stars: int | None,
) -> ConflictKind | None:
    if valence == "positive":
        if ledger_kind == "block":
            return "positive_vs_block"
        if ledger_kind == "rating" and stars is not None and stars <= _BAD_RATING_MAX:
            return "positive_vs_bad_rating"
    if valence == "negative":
        if ledger_kind == "rating" and stars is not None and stars >= _GOOD_RATING_MIN:
            return "negative_vs_good_rating"
    return None


# ---------------------------------------------------------------------------
# Event-log integration
# ---------------------------------------------------------------------------


_DIVERGENCE_ACTION = "memory_divergence"


def log_divergence_event(
    conn: sqlite3.Connection,
    divergence: Divergence,
    *,
    tick: int,
) -> int:
    """Write a single divergence to the events log.

    Uses ``agent_id = divergence.agent_id`` (the agent experiencing
    the drift) so event-log queries grouped by ``agent_id`` surface
    per-agent drift counts directly.
    """
    return log_event(
        conn,
        tick=tick,
        agent_id=divergence.agent_id,
        action_type=_DIVERGENCE_ACTION,
        payload={
            "counterparty_id": divergence.counterparty_id,
            "narrative_memory_id": divergence.narrative_memory_id,
            "ledger_entry_id": divergence.ledger_entry_id,
            "conflict_kind": divergence.conflict_kind,
            "narrative_content": divergence.narrative_content,
            "ledger_summary": divergence.ledger_summary,
            "narrative_tick": divergence.narrative_tick,
            "ledger_tick": divergence.ledger_tick,
        },
        result_status="ok",
        result_payload=None,
    )


def _already_logged(
    conn: sqlite3.Connection,
    divergence: Divergence,
) -> bool:
    """Has this exact (narrative_id, ledger_id) pair been logged already?"""
    rows = conn.execute(
        """
        SELECT payload FROM events
        WHERE action_type = ? AND agent_id = ?
        """,
        (_DIVERGENCE_ACTION, divergence.agent_id),
    ).fetchall()
    for (payload_json,) in rows:
        p = json.loads(payload_json)
        if (
            p.get("narrative_memory_id") == divergence.narrative_memory_id
            and p.get("ledger_entry_id") == divergence.ledger_entry_id
        ):
            return True
    return False


def scan_and_log(
    conn: sqlite3.Connection,
    *,
    tick: int,
    agent_ids: list[int] | None = None,
    up_to_tick: int | None = None,
) -> int:
    """Detect and log divergences for every agent (or the given subset).

    Returns the number of *new* divergence events written. Safe to
    call every tick — duplicate detection is idempotent.
    """
    if agent_ids is None:
        agent_ids = [
            int(r[0]) for r in
            conn.execute("SELECT agent_id FROM agents").fetchall()
        ]

    logged = 0
    for aid in agent_ids:
        for d in detect_divergences(conn, agent_id=aid, up_to_tick=up_to_tick):
            if _already_logged(conn, d):
                continue
            log_divergence_event(conn, d, tick=tick)
            logged += 1
    return logged


def count_divergences(
    conn: sqlite3.Connection,
    *,
    agent_id: int | None = None,
) -> int:
    """How many divergence events have been logged (optionally per-agent)."""
    if agent_id is None:
        r = conn.execute(
            "SELECT COUNT(*) FROM events WHERE action_type = ?",
            (_DIVERGENCE_ACTION,),
        ).fetchone()
    else:
        r = conn.execute(
            "SELECT COUNT(*) FROM events WHERE action_type = ? AND agent_id = ?",
            (_DIVERGENCE_ACTION, agent_id),
        ).fetchone()
    return int(r[0])

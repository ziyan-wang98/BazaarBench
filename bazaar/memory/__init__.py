"""Dual-layer memory system.

Two stores that interact with every LLM invocation differently:

* **Structured ledger** — platform-maintained relational record of
  transactions, ratings, blocks, reports. Cannot be forgotten,
  misremembered, or fabricated by the agent.

* **Narrative memory** — free-text impression store. Vector-embedded
  and retrieved via ``RECALL`` or context-relevant auto-retrieval.
  *Is* subject to drift, retrieval error, inherited summaries, and
  strategic self-revision.

The **divergence** between them is itself a research signal
(Arike et al. "inherited drift"). See ``narrative.py`` for the
embedding + FAISS side (T12).
"""
from __future__ import annotations

import sqlite3

from bazaar.memory.divergence import (
    Divergence,
    count_divergences,
    detect_divergences,
    log_divergence_event,
    scan_and_log,
)
from bazaar.memory.graph import Edge, InteractionGraph
from bazaar.memory.ledger import (
    LedgerEntry,
    auto_populate_from_events,
    build_ledger_context,
    record_ledger_entry,
    render_ledger_context,
    slice_for_prompt,
)
from bazaar.memory.narrative import (
    Encoder,
    HashEncoder,
    MiniLMEncoder,
    NarrativeMemory,
    NarrativeStore,
    render_narrative_context,
)

# ---------------------------------------------------------------------------
# Per-connection store cache.
#
# Handlers don't receive a NarrativeStore directly (they take only
# ``(conn, agent_id, args, *, tick)``), so we provide a lazy lookup. By
# default, ``get_store`` creates a NarrativeStore backed by the real
# MiniLMEncoder the first time it's called for a given connection.
# Tests that want a fake/fast encoder call ``install_store(conn, store)``
# before any handler dispatch.
#
# Keyed by id(conn) because sqlite3.Connection objects don't support
# weakref. The cache therefore retains each connection until the process
# exits; for Phase-2 scale (dozens of runs per process at most) that's
# trivial memory.
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[int, NarrativeStore] = {}


def install_store(conn: sqlite3.Connection, store: NarrativeStore) -> None:
    """Override the cached ``NarrativeStore`` for ``conn`` (tests)."""
    _STORE_CACHE[id(conn)] = store


def get_store(conn: sqlite3.Connection) -> NarrativeStore:
    """Return the ``NarrativeStore`` for this connection.

    Lazily constructs one with the default ``MiniLMEncoder`` on first
    use. Triggering this in a handler therefore pays the ~5 s model-
    load tax on the first call per process. Tests should pre-install
    a ``FakeEncoder``-backed store via :func:`install_store`.
    """
    key = id(conn)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = NarrativeStore(conn)
    return _STORE_CACHE[key]


def clear_stores() -> None:
    """Reset the cache (test-only)."""
    _STORE_CACHE.clear()


__all__ = [
    "Divergence",
    "Edge",
    "Encoder",
    "HashEncoder",
    "InteractionGraph",
    "LedgerEntry",
    "MiniLMEncoder",
    "NarrativeMemory",
    "NarrativeStore",
    "auto_populate_from_events",
    "build_ledger_context",
    "clear_stores",
    "count_divergences",
    "detect_divergences",
    "get_store",
    "install_store",
    "log_divergence_event",
    "record_ledger_entry",
    "render_ledger_context",
    "render_narrative_context",
    "scan_and_log",
    "slice_for_prompt",
]

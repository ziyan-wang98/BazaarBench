"""Interaction graph — the third layer of BazaarBench's memory stack.

The ledger records what *the platform* knows. Narrative memory records
what *the agent subjectively remembers* in free text. This graph
records *how agents are connected*: one edge per meaningful social
interaction, with verb + tick + valence attached. It is the substrate
for Cross-Role Contagion (CRC) analysis — a metric that counts
intermediate agents through which a signal propagates.

Design choices:

1. **Zero-dep.** A dict-of-dicts directed multigraph, not NetworkX.
   Rationale: we only need ``add_interaction``, ``neighbours``,
   ``edges_between``, and a JSON-serialisable snapshot. NetworkX
   would bring 3 MB of code we don't call. When Phase-4 needs
   shortest-path/centrality, we can port to NetworkX without
   changing the public API here.

2. **Deterministic serialisation.** ``to_json()`` emits sorted keys
   everywhere, so two runs with the same sequence of writes produce
   byte-identical JSON. This is load-bearing for invariant 7
   (snapshot replayability) and for R14 of the design doc (the
   D13 ``llm_cache_hash`` and snapshot backup must be byte-stable).

3. **Multigraph.** Two agents can interact multiple times — each
   interaction is a new edge with its own tick + verb, not an
   update to a single edge. This matches the event-log-first
   invariant: graph state at tick T is *derived* from all
   interactions ≤ T, never mutated in place.

4. **Valence is optional.** Callers populate it when they know
   (rating stars → valence; blocking → −1; buying → +1); None
   when they don't. H1's inherited-drift signal will aggregate
   valence along paths.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Edge:
    """A single interaction edge.

    ``verb`` is a short token (``"message"``, ``"offer"``, ``"rate"``,
    ``"block"``, ``"meetup"``, ``"quote_note"`` …). The exact
    vocabulary is defined by callers; the graph doesn't validate it
    so new action types can add new verbs without graph changes.
    """
    src: int
    dst: int
    verb: str
    tick: int
    valence: float | None = None
    thread_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InteractionGraph:
    """Directed multigraph of agent-to-agent interactions.

    Storage: ``_edges`` is a flat list (preserves insertion order
    for deterministic iteration). ``_out`` and ``_in`` are
    adjacency indices rebuilt lazily when stale.
    """
    _edges: list[Edge] = field(default_factory=list)

    # ---- write --------------------------------------------------------

    def add_interaction(
        self,
        src: int,
        dst: int,
        verb: str,
        tick: int,
        *,
        valence: float | None = None,
        thread_id: int | None = None,
    ) -> None:
        """Append one interaction edge. No deduplication — duplicates
        represent repeated interactions and are meaningful."""
        self._edges.append(Edge(
            src=int(src), dst=int(dst), verb=verb, tick=int(tick),
            valence=valence,
            thread_id=None if thread_id is None else int(thread_id),
        ))

    # ---- read ---------------------------------------------------------

    def edges(self) -> list[Edge]:
        """All edges in insertion order (deterministic)."""
        return list(self._edges)

    def out_neighbours(
        self,
        agent_id: int,
        *,
        verb: str | None = None,
    ) -> list[int]:
        """Unique dst agent_ids, sorted, optionally filtered by verb."""
        seen: set[int] = set()
        for e in self._edges:
            if e.src != agent_id:
                continue
            if verb is not None and e.verb != verb:
                continue
            seen.add(e.dst)
        return sorted(seen)

    def in_neighbours(
        self,
        agent_id: int,
        *,
        verb: str | None = None,
    ) -> list[int]:
        """Unique src agent_ids, sorted, optionally filtered by verb."""
        seen: set[int] = set()
        for e in self._edges:
            if e.dst != agent_id:
                continue
            if verb is not None and e.verb != verb:
                continue
            seen.add(e.src)
        return sorted(seen)

    def edges_between(
        self,
        a: int,
        b: int,
        *,
        directed: bool = False,
    ) -> list[Edge]:
        """All edges between two agents (either direction unless
        ``directed`` is True, in which case only ``a → b``)."""
        out: list[Edge] = []
        for e in self._edges:
            if directed:
                if e.src == a and e.dst == b:
                    out.append(e)
            else:
                if (e.src == a and e.dst == b) or (e.src == b and e.dst == a):
                    out.append(e)
        return out

    def shortest_path(
        self,
        src: int,
        dst: int,
        *,
        verb: str | None = None,
    ) -> list[int] | None:
        """BFS shortest directed path from src to dst.

        Returns the node sequence (inclusive) or None if disconnected.
        ``verb`` restricts to edges of one type — useful for CRC-style
        queries ("did A's narrative-quote reach B through narrative
        edges alone?").
        """
        if src == dst:
            return [src]
        # Build adjacency once per call
        adj: dict[int, set[int]] = {}
        for e in self._edges:
            if verb is not None and e.verb != verb:
                continue
            adj.setdefault(e.src, set()).add(e.dst)
        if src not in adj:
            return None

        parent: dict[int, int] = {src: src}
        frontier = [src]
        while frontier:
            nxt: list[int] = []
            for node in frontier:
                for n in sorted(adj.get(node, ())):
                    if n in parent:
                        continue
                    parent[n] = node
                    if n == dst:
                        path = [dst]
                        while path[-1] != src:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    nxt.append(n)
            frontier = nxt
        return None

    def degree(self, agent_id: int) -> int:
        """Total edges touching ``agent_id`` (in + out, undirected count)."""
        return sum(1 for e in self._edges
                   if e.src == agent_id or e.dst == agent_id)

    def __len__(self) -> int:
        return len(self._edges)

    # ---- serialisation ------------------------------------------------

    def to_json(self) -> str:
        """Deterministic JSON representation.

        Edges are emitted in insertion order. Field order within each
        edge is fixed by ``Edge.to_dict``. Two graphs with the same
        write sequence produce byte-identical JSON.
        """
        return json.dumps(
            {"edges": [e.to_dict() for e in self._edges]},
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> InteractionGraph:
        blob = json.loads(text)
        g = cls()
        for row in blob.get("edges", []):
            g._edges.append(Edge(
                src=int(row["src"]),
                dst=int(row["dst"]),
                verb=row["verb"],
                tick=int(row["tick"]),
                valence=row.get("valence"),
                thread_id=row.get("thread_id"),
            ))
        return g

    # ---- convenience for the narrative-write hook --------------------

    @classmethod
    def replay_from_events(cls, events: Iterable[dict[str, Any]]) -> InteractionGraph:
        """Rebuild a graph by scanning the event log.

        ``events`` is expected to yield dicts with at least
        ``action_type``, ``agent_id``, ``tick``, and a ``payload``
        carrying counterparty fields. Phase-4 replay uses this to
        reconstruct the graph from scratch without relying on the
        serialised BLOB on ``agents.interaction_graph_json``.
        """
        g = cls()
        for ev in events:
            if ev.get("result_status") != "ok":
                continue
            src = ev.get("agent_id")
            verb = _VERB_BY_ACTION.get(ev.get("action_type", ""))
            if src is None or verb is None:
                continue
            payload = ev.get("payload") or {}
            result = ev.get("result_payload") or {}
            dst = (
                payload.get("target_agent_id")
                or payload.get("recipient_agent_id")
                or payload.get("counterparty_id")
                or payload.get("source_agent_id")
                or result.get("counterparty_id")
            )
            if dst is None:
                continue
            g.add_interaction(
                src=int(src),
                dst=int(dst),
                verb=verb,
                tick=int(ev.get("tick", 0)),
                valence=payload.get("valence"),
                thread_id=payload.get("thread_id") or result.get("thread_id"),
            )
        return g


# Minimal, extend as new LLM-accessible actions arrive.
_VERB_BY_ACTION: dict[str, str] = {
    "send_message":     "message",
    "make_offer":       "offer",
    "counter_offer":    "offer",
    "accept_offer":     "accept",
    "complete_transaction": "meetup",
    "rate":             "rate",
    "block_user":       "block",
    "report_user":      "report",
    "quote_agent_note": "quote_note",
}

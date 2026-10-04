"""Narrative memory — fuzzy free-text impression store.

The other half of the dual-layer memory system.
Unlike the structured ledger
(ledger.py), narrative memory **is** subject to drift, retrieval
error, inherited summaries, and strategic self-revision — the
load-bearing property for inherited-drift experiments.

Storage
-------
SQLite's ``narrative_memories`` table is the authority. Embeddings
are float32 vectors serialised into the row's ``embedding`` BLOB so
the table is self-contained — we do not need a sidecar FAISS index
file to reopen a run database. FAISS indices are rebuilt lazily
per ``recall()`` call from the blobs; at Phase-2 scale (≤ a few
thousand entries per agent) this is cheap, and rebuilding avoids
synchronisation bugs between the DB and an on-disk index.

Encoder injection
-----------------
``NarrativeStore`` accepts any object with an ``encode(texts) -> np.ndarray``
method (``Encoder`` protocol). The default is sentence-transformers'
``all-MiniLM-L6-v2`` (384-dim). Unit tests inject a deterministic
``FakeEncoder`` so tests are fast and do not touch the network.

Retrieval
---------
Recall uses ``IndexFlatIP`` (inner product) over L2-normalised
vectors, which is numerically equivalent to cosine similarity.
Vectors are re-normalised at ``encode()`` time so the index is always
cosine-compatible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from bazaar.core.event_log import require_lastrowid

# ``faiss`` is in the [memory] extra, not [dev]; CI installs only [dev]
# for the lint+unit-test lane. We lazy-import it inside ``recall()`` so
# that ``import bazaar.memory`` succeeds without faiss on the path.

Scope = Literal["thread", "counterparty", "self"]

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Minimal interface for something that can embed text."""

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


class HashEncoder:
    """Deterministic bag-of-hashed-words encoder.

    Offline, dependency-free, and reproducible: same words →
    same coordinates → same cosine similarities. Useful for:
    * fast unit tests that don't want to touch the network or load
      the 80 MB MiniLM model weights
    * CI smoke runs where the narrative store is exercised end-to-
      end but semantic fidelity isn't the point
    * demo runs where we want a narrative trail without paying the
      torch import cost

    Not suitable for actual research experiments — the vocabulary is
    too coarse to distinguish "bicycle riding" from "I enjoy cycling".
    Use ``MiniLMEncoder`` for those.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in text.lower().split():
                h = int.from_bytes(
                    hashlib.md5(w.encode("utf-8")).digest()[:4],
                    "little",
                ) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class MiniLMEncoder:
    """Lazy, process-global sentence-transformers encoder."""

    _model: Any = None
    _dim: int | None = None

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name

    def _load(self) -> None:
        if type(self)._model is None:
            from sentence_transformers import SentenceTransformer
            type(self)._model = SentenceTransformer(self.model_name)
            # Prefer the newer name; fall back for older versions.
            getter = getattr(
                type(self)._model, "get_embedding_dimension", None,
            ) or type(self)._model.get_sentence_embedding_dimension
            type(self)._dim = int(getter())

    @property
    def dim(self) -> int:
        self._load()
        dim = type(self)._dim
        assert dim is not None
        return dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        v = type(self)._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return v.astype(np.float32, copy=False)


@dataclass(frozen=True)
class NarrativeMemory:
    """One recalled narrative memory."""
    memory_id: int
    agent_id: int
    scope: Scope
    scope_ref_id: int | None
    content: str
    created_tick: int
    score: float = 0.0  # cosine similarity at recall time; 0.0 otherwise
    provenance: int | None = None  # source agent if inherited via QUOTE_AGENT_NOTE


def _serialize(vec: np.ndarray) -> bytes:
    """Pack a 1-D float32 vector into a BLOB."""
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _deserialize(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


class NarrativeStore:
    """Per-connection narrative memory store.

    Thread-unsafe: callers must serialise access through the
    single SQLite connection just like the rest of Bazaar.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        encoder: Encoder | None = None,
        *,
        dim: int | None = None,
    ) -> None:
        self.conn = conn
        if encoder is None:
            encoder = MiniLMEncoder()
        self.encoder = encoder
        # Prefer the encoder's advertised dim if it exposes one; else
        # discover by embedding a trivial string. This keeps the store
        # agnostic to custom encoder implementations in tests.
        if dim is not None:
            self.dim = dim
        elif hasattr(encoder, "dim"):
            self.dim = int(encoder.dim)
        else:
            probe = encoder.encode([""])
            self.dim = int(probe.shape[1])

    # ---------------------------------------------------------------- write

    def add(
        self,
        *,
        agent_id: int,
        scope: Scope,
        content: str,
        tick: int,
        scope_ref_id: int | None = None,
        provenance: int | None = None,
    ) -> int:
        """Embed ``content`` and persist it.  Returns ``memory_id``.

        ``provenance`` records the source agent when this narrative was
        received from another agent (e.g. ``QUOTE_AGENT_NOTE``). None
        for self-authored impressions.
        """
        vec = self.encoder.encode([content])[0]
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"encoder returned dim={vec.shape[0]}, expected {self.dim}"
            )
        cur = self.conn.execute(
            """
            INSERT INTO narrative_memories
                (agent_id, scope, scope_ref_id, content, embedding,
                 created_tick, decayed, provenance)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (agent_id, scope, scope_ref_id, content,
             _serialize(vec), tick, provenance),
        )
        return require_lastrowid(cur, table="narrative_memories")

    def add_many(
        self,
        items: list[dict[str, Any]],
    ) -> list[int]:
        """Batch insert. ``items`` is a list of kwarg dicts for ``add``.

        Uses a single ``encode`` call so larger batches amortise model
        overhead. Tick and scope_ref_id are per-item.
        """
        if not items:
            return []
        contents = [it["content"] for it in items]
        vecs = self.encoder.encode(contents)
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"encoder returned dim={vecs.shape[1]}, expected {self.dim}"
            )
        ids: list[int] = []
        for it, vec in zip(items, vecs, strict=True):
            cur = self.conn.execute(
                """
                INSERT INTO narrative_memories
                    (agent_id, scope, scope_ref_id, content, embedding,
                     created_tick, decayed, provenance)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (it["agent_id"], it["scope"], it.get("scope_ref_id"),
                 it["content"], _serialize(vec), it["tick"],
                 it.get("provenance")),
            )
            ids.append(require_lastrowid(cur, table="narrative_memories"))
        return ids

    def mark_decayed(self, memory_id: int) -> None:
        """Soft-delete: the row stays but is excluded from recall."""
        self.conn.execute(
            "UPDATE narrative_memories SET decayed = 1 WHERE memory_id = ?",
            (memory_id,),
        )

    # ---------------------------------------------------------------- read

    def count(self, agent_id: int, *, include_decayed: bool = False) -> int:
        q = "SELECT COUNT(*) FROM narrative_memories WHERE agent_id = ?"
        if not include_decayed:
            q += " AND decayed = 0"
        return int(self.conn.execute(q, (agent_id,)).fetchone()[0])

    def all_for_agent(
        self,
        agent_id: int,
        *,
        include_decayed: bool = False,
        up_to_tick: int | None = None,
    ) -> list[NarrativeMemory]:
        q = (
            "SELECT memory_id, scope, scope_ref_id, content, created_tick, "
            "provenance "
            "FROM narrative_memories WHERE agent_id = ?"
        )
        params: list[Any] = [agent_id]
        if not include_decayed:
            q += " AND decayed = 0"
        if up_to_tick is not None:
            q += " AND created_tick <= ?"
            params.append(up_to_tick)
        q += " ORDER BY created_tick DESC"
        rows = self.conn.execute(q, params).fetchall()
        return [
            NarrativeMemory(
                memory_id=int(r[0]),
                agent_id=agent_id,
                scope=r[1],
                scope_ref_id=int(r[2]) if r[2] is not None else None,
                content=r[3],
                created_tick=int(r[4]),
                provenance=int(r[5]) if r[5] is not None else None,
            )
            for r in rows
        ]

    def recall(
        self,
        *,
        agent_id: int,
        query: str,
        top_k: int = 5,
        include_decayed: bool = False,
        up_to_tick: int | None = None,
        scope: Scope | None = None,
        scope_ref_id: int | None = None,
    ) -> list[NarrativeMemory]:
        """Return the top-``k`` matches for ``query`` within the agent.

        Results carry a ``score`` in ``[0, 1]`` (cosine similarity with
        L2-normalised vectors via ``IndexFlatIP``).

        ``scope`` and ``scope_ref_id`` (R12 Gap 4) narrow candidates
        *before* similarity ranking. Combining both with e.g.
        ``scope="counterparty", scope_ref_id=42`` answers
        "what do I remember about agent #42?" without cross-scope
        bleed. Passing either alone narrows on that axis only.
        """
        q_str = (
            "SELECT memory_id, scope, scope_ref_id, content, created_tick, "
            "embedding, provenance FROM narrative_memories WHERE agent_id = ?"
        )
        params: list[Any] = [agent_id]
        if not include_decayed:
            q_str += " AND decayed = 0"
        if scope is not None:
            q_str += " AND scope = ?"
            params.append(scope)
        if scope_ref_id is not None:
            q_str += " AND scope_ref_id = ?"
            params.append(scope_ref_id)
        if up_to_tick is not None:
            q_str += " AND created_tick <= ?"
            params.append(up_to_tick)
        rows = self.conn.execute(q_str, params).fetchall()
        if not rows:
            return []

        vecs = np.stack(
            [_deserialize(r[5], self.dim) for r in rows],
            axis=0,
        ).astype(np.float32, copy=False)
        import faiss  # lazy — see module header for why
        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs)

        q_vec = self.encoder.encode([query]).astype(np.float32, copy=False)
        if q_vec.shape[1] != self.dim:
            raise ValueError(
                f"encoder returned dim={q_vec.shape[1]}, expected {self.dim}"
            )

        k = min(top_k, len(rows))
        scores, idxs = index.search(q_vec, k)

        out: list[NarrativeMemory] = []
        for j, score in zip(idxs[0], scores[0], strict=True):
            if j < 0:
                continue
            r = rows[int(j)]
            out.append(NarrativeMemory(
                memory_id=int(r[0]),
                agent_id=agent_id,
                scope=r[1],
                scope_ref_id=int(r[2]) if r[2] is not None else None,
                content=r[3],
                created_tick=int(r[4]),
                score=float(score),
                provenance=int(r[6]) if r[6] is not None else None,
            ))
        return out


# ---------------------------------------------------------------------------
# Convenience rendering for prompt injection
# ---------------------------------------------------------------------------


def render_narrative_context(
    memories: list[NarrativeMemory],
    *,
    header: str = "Your own impressions and summaries (subject to drift):",
) -> str:
    """Format narrative memories for LLM prompt injection.

    Mirrors :func:`bazaar.memory.ledger.render_ledger_context` so the
    two stores appear side-by-side in a consistent visual shape.
    """
    lines = [header]
    if not memories:
        lines.append("  (none)")
        return "\n".join(lines)
    for m in memories:
        prefix = f"  [t={m.created_tick}] {m.scope}"
        if m.scope_ref_id is not None:
            prefix += f"#{m.scope_ref_id}"
        if m.score > 0:
            prefix += f" (sim={m.score:.2f})"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)

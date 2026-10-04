"""Decision policies for MarketAgent.

``RandomBenignPolicy`` is a weighted-random policy for CI-friendly
smoke tests. ``LLMPolicy`` is the real backend-driven policy used by
OpenAI-compatible reviewer rollouts; it remains fail-safe and returns
``DO_NOTHING`` rather than raising when a model emits unusable tool
calls. Paper-facing rollouts can opt into strict LLM-error handling so
provider/API failures or unusable model outputs abort instead of becoming
valid-looking idle actions.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from bazaar.actions.types import ActionType
from bazaar.agents.market_agent import AgentAction, MarketAgent, Policy
from bazaar.core.handoff_checks import (
    INSPECTION_TRUTH_MODE,
    SHIPMENT_INSPECTION_MODE,
    read_handoff_check,
)


@dataclass
class PreparedDecision:
    """Per-agent state captured during the read-only pre-LLM phase.

    Decoupling preparation from the LLM call lets ``BazaarEnv`` run
    the slow network step concurrently across agents while keeping
    every DB read/write on the main thread.
    """
    agent: MarketAgent
    tick: int
    skip: bool = False                 # gated out by activity_rate
    system_text: str = ""
    user_text: str = ""
    prompt_hash: str = ""
    sampling: dict[str, Any] = field(default_factory=dict)
    tools_param: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    cached_response_text: str | None = None
    cached_tool_calls: list[dict[str, Any]] | None = None
    cached_reasoning_summary: str | None = None


@dataclass
class LLMResult:
    """Return value of the parallelizable network step."""
    response_text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_summary: str | None = None
    latency_ms: int = 0
    error: str | None = None  # set if backend raised; finalize logs DO_NOTHING


class RandomBenignPolicy(Policy):
    """Picks among a small set of safe, well-formed actions."""

    # Weights biased toward common ops.  These are tuned so that a
    # 20-agent × 50-tick smoke run produces a good mix of listings,
    # views, messages, photos, offers, and (once state permits) the
    # full transaction chain through rating.
    _WEIGHTS: dict[ActionType, float] = {
        ActionType.DO_NOTHING:      1.0,
        ActionType.SEARCH:          1.2,
        ActionType.BROWSE_CATEGORY: 0.6,
        ActionType.VIEW_LISTING:    2.0,
        ActionType.PIN:             0.5,
        ActionType.CREATE_LISTING:  0.7,
        ActionType.MESSAGE:         1.4,
        ActionType.MAKE_OFFER:      0.6,
        ActionType.VIEW_PROFILE:    0.3,
        ActionType.WAIT:            0.4,
        # Phase-2 photo + narrative actions.
        ActionType.SEND_PHOTO:         0.5,
        ActionType.SEND_STOCK_PHOTO:   0.2,
        ActionType.REQUEST_PHOTO:      0.3,
        ActionType.INSPECT_PHOTO:      0.2,
        ActionType.SUMMARIZE_SESSION:  0.3,
        # Lifecycle (P4 of T23). State-gated in _args_for — if no
        # eligible target exists the policy falls back to DO_NOTHING,
        # so selecting these with a non-zero weight is always safe.
        ActionType.ACCEPT_OFFER:        0.8,
        ActionType.COUNTER_OFFER:       0.4,
        ActionType.SCHEDULE_MEETUP:     0.7,
        ActionType.SCHEDULE_SHIPMENT:   0.2,
        # v2: buyer must inspect_at_meetup before complete_transaction
        # on a meetup-mode meetup, so weight it generously to keep the
        # random policy navigating the full lifecycle in a reasonable
        # tick budget.
        ActionType.INSPECT_AT_MEETUP:   1.2,
        ActionType.COMPLETE_TRANSACTION: 1.0,
        ActionType.RATE:                0.5,
        ActionType.READ:                0.6,
        ActionType.MARK_SOLD:           0.05,
        ActionType.BUMP_LISTING:        0.2,
        ActionType.EDIT_LISTING:        0.1,
        ActionType.UNPIN:               0.1,
    }

    _CATEGORIES = [
        "furniture", "electronics", "clothing", "kids", "tools",
        "garden", "books", "sporting-goods",
    ]

    _TITLE_STEMS = [
        "Like-new", "Vintage", "Barely used", "Brand-new", "Well-loved",
    ]

    _DESC_TEMPLATES = [
        "Great condition, pickup preferred.",
        "Moving soon, must go.",
        "No longer need it; pickup only.",
        "Smoke-free home.",
    ]

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------

    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        # Throttle by persona activity_rate.
        if self._rng.random() > agent.persona.activity_rate:
            return None

        allowed = [a for a in agent.available_actions if a in self._WEIGHTS]
        weights = [self._WEIGHTS[a] for a in allowed]
        action = self._rng.choices(allowed, weights=weights, k=1)[0]

        args = self._args_for(agent, conn, action, tick)
        if args is None:
            return AgentAction(action=ActionType.DO_NOTHING, args={})
        return AgentAction(action=action, args=args)

    # ------------------------------------------------------------------ args

    def _args_for(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        action: ActionType,
        tick: int,
    ) -> dict[str, Any] | None:
        if action == ActionType.DO_NOTHING:
            return {}
        if action == ActionType.WAIT:
            return {"ticks": self._rng.randint(1, 3)}
        if action == ActionType.SEARCH:
            query = self._rng.choice(agent.persona.interests or ["used"])
            return {"query": query}
        if action == ActionType.BROWSE_CATEGORY:
            return {"category": self._rng.choice(self._CATEGORIES)}
        if action == ActionType.VIEW_LISTING:
            lid = self._random_listing_id(conn, exclude_owner=agent.agent_id)
            return {"listing_id": lid} if lid is not None else None
        if action == ActionType.PIN:
            lid = self._random_listing_id(conn, exclude_owner=agent.agent_id)
            return {"listing_id": lid} if lid is not None else None
        if action == ActionType.CREATE_LISTING:
            if read_handoff_check(conn, INSPECTION_TRUTH_MODE) == "unit":
                # Truthful handoff checks bind a listing to a held unit by
                # title, and the generic titles below never match one, so
                # list an unsold inventory item when there is one.
                owned = self._unsold_inventory_items(conn, agent.agent_id)
                if owned:
                    return self._inventory_listing_args(owned)
            return {
                "category": self._rng.choice(self._CATEGORIES),
                "title": f"{self._rng.choice(self._TITLE_STEMS)} item",
                "description": self._rng.choice(self._DESC_TEMPLATES),
                "price_cents": self._rng.randint(500, 20_000),
                "condition": self._rng.choice(
                    ["new", "like_new", "good", "fair"]
                ),
            }
        if action == ActionType.MESSAGE:
            thread = self._random_thread_for(conn, agent.agent_id)
            if thread is None:
                return None
            return {
                "thread_id": thread,
                "body": self._rng.choice([
                    "Is this still available?",
                    "Would you take a little less?",
                    "Can you describe the condition?",
                    "Where can I pick it up?",
                ]),
            }
        if action == ActionType.MAKE_OFFER:
            lid, price = self._random_listing_with_price(
                conn, exclude_owner=agent.agent_id
            )
            if lid is None:
                return None
            # Offer 70-100% of listed price.
            offered = int(price * self._rng.uniform(0.7, 1.0))
            return {
                "listing_id": lid,
                "price_cents": max(100, offered),
                "terms": {},
            }
        if action == ActionType.VIEW_PROFILE:
            other = self._random_other_agent(conn, agent.agent_id)
            return {"user_agent_id": other} if other is not None else None
        if action == ActionType.SEND_PHOTO:
            thread_listing = self._random_open_thread_with_listing(
                conn, agent.agent_id
            )
            if thread_listing is None:
                return None
            return {
                "thread_id": thread_listing[0],
                "listing_id": thread_listing[1],
                "focus": self._rng.choice(
                    ["whole item", "close-up", "any damage", "the serial number"]
                ),
                "privacy_mode": "normal",
            }
        if action == ActionType.SEND_STOCK_PHOTO:
            thread_listing = self._random_open_thread_with_listing(
                conn, agent.agent_id
            )
            if thread_listing is None:
                return None
            return {
                "thread_id": thread_listing[0],
                "listing_id": thread_listing[1],
            }
        if action == ActionType.REQUEST_PHOTO:
            thread = self._random_thread_for(conn, agent.agent_id)
            if thread is None:
                return None
            return {
                "thread_id": thread,
                "focus_hint": self._rng.choice([
                    "can you send a photo?",
                    "any damage pictures?",
                    "show the brand label please",
                ]),
            }
        if action == ActionType.INSPECT_PHOTO:
            pid = self._random_photo_id(conn)
            return {"photo_id": pid} if pid is not None else None
        if action == ActionType.SUMMARIZE_SESSION:
            counter = self._random_other_agent(conn, agent.agent_id)
            if counter is None:
                return None
            return {
                "scope": "counterparty",
                "scope_ref_id": counter,
                "content": self._rng.choice([
                    f"agent#{counter} replied quickly",
                    f"agent#{counter} was slow to haggle",
                    f"agent#{counter} asked for my address — felt off",
                    f"agent#{counter} seemed trustworthy",
                ]),
            }
        # ---- Lifecycle (P4 of T23) — state-gated selectors ----------
        if action == ActionType.ACCEPT_OFFER:
            oid = self._pending_offer_for_acceptance(conn, agent.agent_id)
            return {"offer_id": oid} if oid is not None else None
        if action == ActionType.COUNTER_OFFER:
            oid = self._pending_offer_for_acceptance(conn, agent.agent_id)
            if oid is None:
                return None
            # Counter at roughly 80–110% of the current offered price.
            price = int(conn.execute(
                "SELECT price_cents FROM offers WHERE offer_id = ?", (oid,),
            ).fetchone()[0] * self._rng.uniform(0.8, 1.1))
            return {"offer_id": oid, "price_cents": max(100, price), "terms": {}}
        if action == ActionType.SCHEDULE_MEETUP:
            tid = self._committed_thread_without_meetup(conn, agent.agent_id)
            if tid is None:
                return None
            return {
                "thread_id": tid,
                "location_desc": self._rng.choice([
                    "coffee shop on 5th", "parking lot at the mall",
                    "my porch", "public library lobby",
                ]),
                "scheduled_tick": tick + self._rng.randint(2, 20),
                "payment_method": self._rng.choice(
                    ["cash", "zelle", "venmo", "on_platform"]
                ),
            }
        if action == ActionType.INSPECT_AT_MEETUP:
            mid = self._inspect_meetup_for(conn, agent.agent_id, tick)
            if mid is None:
                return None
            return {"meetup_id": mid}
        if action == ActionType.SCHEDULE_SHIPMENT:
            tid = self._committed_thread_without_meetup(conn, agent.agent_id)
            if tid is None:
                return None
            return {
                "thread_id": tid,
                "delivery_lag_ticks": self._rng.randint(2, 12),
                "payment_method": self._rng.choice(
                    ["zelle", "venmo", "on_platform"]
                ),
            }
        if action == ActionType.COMPLETE_TRANSACTION:
            mid = self._scheduled_meetup_for(conn, agent.agent_id, tick)
            if mid is None:
                return None
            return {"meetup_id": mid}
        if action == ActionType.RATE:
            pair = self._completed_counterparty(conn, agent.agent_id)
            if pair is None:
                return None
            ratee_id, thread_id = pair
            return {
                "ratee_agent_id": ratee_id,
                "stars":          self._rng.randint(3, 5),
                "body":           None,
                "thread_id":      thread_id,
            }
        if action == ActionType.READ:
            tid = self._random_thread_for(conn, agent.agent_id)
            if tid is None:
                return None
            return {"thread_id": tid}
        if action == ActionType.MARK_SOLD:
            lid = self._own_active_listing(conn, agent.agent_id)
            return {"listing_id": lid} if lid is not None else None
        if action == ActionType.BUMP_LISTING:
            lid = self._own_active_listing(conn, agent.agent_id)
            return {"listing_id": lid} if lid is not None else None
        if action == ActionType.EDIT_LISTING:
            lid = self._own_active_listing(conn, agent.agent_id)
            if lid is None:
                return None
            # Occasionally tweak price, leave other fields alone.
            return {
                "listing_id":  lid,
                "title":       None,
                "description": None,
                "price_cents": self._rng.randint(500, 30_000),
                "condition":   None,
            }
        if action == ActionType.UNPIN:
            lid = self._random_listing_id(conn, exclude_owner=agent.agent_id)
            return {"listing_id": lid} if lid is not None else None
        return None

    # ------------------------------------------------------------------ db

    def _random_listing_id(
        self,
        conn: sqlite3.Connection,
        *,
        exclude_owner: int,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT listing_id FROM listings
            WHERE status = 'active'
              AND (owner_agent_id IS NULL OR owner_agent_id != ?)
            ORDER BY RANDOM() LIMIT 1
            """,
            (exclude_owner,),
        ).fetchone()
        return None if row is None else int(row[0])

    def _random_listing_with_price(
        self,
        conn: sqlite3.Connection,
        *,
        exclude_owner: int,
    ) -> tuple[int | None, int]:
        row = conn.execute(
            """
            SELECT listing_id, price_cents FROM listings
            WHERE status = 'active'
              AND (owner_agent_id IS NULL OR owner_agent_id != ?)
            ORDER BY RANDOM() LIMIT 1
            """,
            (exclude_owner,),
        ).fetchone()
        if row is None:
            return None, 0
        return int(row[0]), int(row[1])

    def _random_thread_for(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT thread_id FROM threads
            WHERE status = 'open'
              AND (buyer_agent_id = ? OR seller_agent_id = ?)
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id, agent_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _random_other_agent(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT agent_id FROM agents
            WHERE agent_id != ? AND status = 'active'
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
        return None if row is None else int(row[0])

    def _random_open_thread_with_listing(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
    ) -> tuple[int, int] | None:
        """An open thread the agent participates in + its listing_id.

        Requires the listing to still be active so SEND_PHOTO /
        SEND_STOCK_PHOTO handlers don't reject on inactive-listing.
        """
        row = conn.execute(
            """
            SELECT t.thread_id, t.listing_id
            FROM threads t
            JOIN listings l ON l.listing_id = t.listing_id
            WHERE t.status = 'open'
              AND l.status = 'active'
              AND l.is_phantom = 0
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id, agent_id),
        ).fetchone()
        return None if row is None else (int(row[0]), int(row[1]))

    def _random_photo_id(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute(
            "SELECT photo_id FROM photos ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return None if row is None else int(row[0])

    # ---- Lifecycle selectors (P4 of T23) ----------------------------

    def _pending_offer_for_acceptance(
        self, conn: sqlite3.Connection, agent_id: int,
    ) -> int | None:
        """A pending offer in a thread where this agent is a participant
        and NOT the proposer — i.e. an offer they can accept or counter."""
        row = conn.execute(
            """
            SELECT o.offer_id
            FROM offers o
            JOIN threads t ON t.thread_id = o.thread_id
            WHERE o.status = 'pending'
              AND o.proposer_id != ?
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
              AND t.status NOT IN ('completed', 'cancelled', 'ghosted')
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id, agent_id, agent_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _committed_thread_without_meetup(
        self, conn: sqlite3.Connection, agent_id: int,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT t.thread_id
            FROM threads t
            WHERE t.status = 'committed'
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
              AND NOT EXISTS (
                SELECT 1 FROM meetups m
                WHERE m.thread_id = t.thread_id AND m.status = 'scheduled'
              )
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id, agent_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _scheduled_meetup_for(
        self, conn: sqlite3.Connection, agent_id: int, tick: int,
    ) -> int | None:
        """A scheduled meetup where this agent still needs to confirm.
        Only fires once the scheduled_tick has been reached so the
        completion timeline is consistent with the wall clock."""
        row = conn.execute(
            """
            SELECT m.meetup_id
            FROM meetups m
            JOIN threads t ON t.thread_id = m.thread_id
            WHERE m.status = 'scheduled'
              AND m.scheduled_tick <= ?
              AND ((t.buyer_agent_id = ? AND m.buyer_confirmed = 0)
                OR (t.seller_agent_id = ? AND m.seller_confirmed = 0))
            ORDER BY RANDOM() LIMIT 1
            """,
            (tick, agent_id, agent_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _inspect_meetup_for(
        self, conn: sqlite3.Connection, agent_id: int, tick: int,
    ) -> int | None:
        """A meetup-mode meetup where THIS agent is the buyer, hasn't
        inspected yet, and the scheduled tick has arrived. v2: this is
        the gate buyers must clear before complete_transaction. Under
        ``shipment_inspection_mode=on_arrival`` an arrived shipment also
        qualifies (the legacy query is used otherwise)."""
        if read_handoff_check(conn, SHIPMENT_INSPECTION_MODE) == "on_arrival":
            # Deterministic candidate order plus the policy's seeded RNG
            # (the legacy query below keeps its ORDER BY RANDOM()).
            candidates = [
                int(r[0]) for r in conn.execute(
                    """
                    SELECT m.meetup_id
                    FROM meetups m
                    JOIN threads t ON t.thread_id = m.thread_id
                    WHERE m.status = 'scheduled'
                      AND m.scheduled_tick <= ?
                      AND m.buyer_inspected_quality_pct IS NULL
                      AND t.buyer_agent_id = ?
                      AND (
                        COALESCE(m.delivery_method, 'meetup') = 'meetup'
                        OR (m.delivery_method = 'ship'
                            AND COALESCE(m.delivered_at_tick, m.scheduled_tick) <= ?)
                      )
                    ORDER BY m.meetup_id
                    """,
                    (tick, agent_id, tick),
                ).fetchall()
            ]
            return self._rng.choice(candidates) if candidates else None
        row = conn.execute(
            """
            SELECT m.meetup_id
            FROM meetups m
            JOIN threads t ON t.thread_id = m.thread_id
            WHERE m.status = 'scheduled'
              AND COALESCE(m.delivery_method, 'meetup') = 'meetup'
              AND m.scheduled_tick <= ?
              AND m.buyer_inspected_quality_pct IS NULL
              AND t.buyer_agent_id = ?
            ORDER BY RANDOM() LIMIT 1
            """,
            (tick, agent_id),
        ).fetchone()
        return None if row is None else int(row[0])

    def _completed_counterparty(
        self, conn: sqlite3.Connection, agent_id: int,
    ) -> tuple[int, int] | None:
        """Find a completed thread the agent was in that hasn't been
        rated yet, and pick the counterparty to rate."""
        row = conn.execute(
            """
            SELECT t.thread_id, t.buyer_agent_id, t.seller_agent_id
            FROM threads t
            WHERE t.status = 'completed'
              AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?)
              AND NOT EXISTS (
                SELECT 1 FROM ratings r
                WHERE r.thread_id = t.thread_id AND r.rater_agent_id = ?
              )
            ORDER BY RANDOM() LIMIT 1
            """,
            (agent_id, agent_id, agent_id),
        ).fetchone()
        if row is None:
            return None
        tid, buyer, seller = int(row[0]), int(row[1]), int(row[2])
        other = seller if agent_id == buyer else buyer
        return other, tid

    def _own_active_listing(
        self, conn: sqlite3.Connection, agent_id: int,
    ) -> int | None:
        row = conn.execute(
            "SELECT listing_id FROM listings "
            "WHERE owner_agent_id = ? AND status = 'active' "
            "ORDER BY RANDOM() LIMIT 1",
            (agent_id,),
        ).fetchone()
        return None if row is None else int(row[0])

    @staticmethod
    def _unsold_inventory_items(
        conn: sqlite3.Connection, agent_id: int,
    ) -> list[dict[str, Any]]:
        """Unsold inventory rows with a usable title, read from the DB
        (sales update ``persona_json`` there, not the in-memory persona)."""
        row = conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return []
        try:
            persona = json.loads(row[0])
        except (TypeError, ValueError):
            return []
        items = persona.get("inventory_items") if isinstance(persona, dict) else None
        if not isinstance(items, list):
            return []
        return [
            item for item in items
            if isinstance(item, dict)
            and item.get("sold_at_tick") is None
            and len(str(item.get("title") or "").strip()) >= 3
        ]

    def _inventory_listing_args(self, owned: list[dict[str, Any]]) -> dict[str, Any]:
        """create_listing args for one of the agent's unsold items (no
        stated band, so the band follows the item's true quality)."""
        item = self._rng.choice(owned)
        asking = item.get("asking_price_cents")
        price = (
            asking if isinstance(asking, int) and not isinstance(asking, bool)
            and asking > 0
            else self._rng.randint(500, 20_000)
        )
        return {
            "category": str(item.get("category") or "misc"),
            "title": str(item["title"]).strip()[:120],
            "description": self._rng.choice(self._DESC_TEMPLATES),
            "price_cents": price,
            "condition": self._rng.choice(["new", "like_new", "good", "fair"]),
        }


class LLMPolicy(Policy):
    """LLM-driven policy for a ``MarketAgent``.

    T28c (Phase-3) implementation: Ollama-only, synchronous,
    fail-safe. The flow every tick:

    1. Skip based on persona activity rate (same gate as
       RandomBenignPolicy — keeps tick cost predictable).
    2. Recall top-k narrative impressions with a query derived
       from the current observation (recent ledger summaries).
    3. Build the prompt via :class:`PromptBuilder`.
    4. Call the LLM backend with the tool list from
       :func:`build_tool_specs`. If the backend supports native
       tool-calling (OpenAI/Anthropic/Ollama 0.5+), we trust its
       tool_calls. If not, we parse a JSON object out of the
       response as a best-effort fallback.
    5. For each tool the model picks, validate args through the
       pydantic schema, dispatch via the fail-safe router, and
       feed outcomes back into the narrative store as
       observation-level memories (observation/outcome pairs are
       what future RECALL calls will retrieve).
    6. Log one ``llm_calls`` row per model call.

    The policy **returns a single AgentAction** to match the
    existing ``Policy.decide`` contract. When the LLM picks
    multiple tools we pick the first; subsequent ones dispatch
    inline and are logged but not surfaced up to
    ``BazaarEnv.step``. This matches T28c's scope — the camel
    ChatAgent multi-tool orchestration lands in T28d.
    """

    def __init__(
        self,
        backend: Any,
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        recall_k: int = 5,
        slice_k: int = 10,
        allow_cross_agent_notes: bool = False,
        seed: int | None = None,
        include_prompt_text: bool = True,
        probe_backend: Any = None,
        probe_model: str | None = None,
        probe_enabled: bool = False,
        system_prompt_suffix: str = "",
        strict_backend_errors: bool = False,
    ) -> None:
        from bazaar.agents.llm_tools import build_tool_specs, tool_index
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.recall_k = recall_k
        self.slice_k = slice_k
        self._seed = seed
        self._rng = random.Random(seed)
        self._specs = build_tool_specs(
            allow_cross_agent_notes=allow_cross_agent_notes,
        )
        self._tool_index = tool_index(self._specs)
        self._include_prompt_text = include_prompt_text
        # R14b Part A: mental-price probes. When enabled, every decide()
        # inspects the picked tool_calls and fires probe_mental_price()
        # for the 5 R14b trigger points. The probe defaults to using
        # the main backend+model (typically the reflection model is
        # cheaper, but it's a single knob at the CLI layer — see
        # ``bazaar llm-smoke --probe-model``).
        self.probe_backend = probe_backend or backend
        self.probe_model = probe_model or model
        self._probe_enabled = probe_enabled
        # Treatment-side knob: a per-agent system-prompt addendum
        # (e.g. "you have a 5-day deadline to sell 3 listings") that
        # the Level-2/3 driver uses to differentiate the 20 treatment
        # agents from the 80 qwen base on the same forked DB.
        self.system_prompt_suffix = system_prompt_suffix or ""
        # Default fail-safe behavior is useful for smoke/dev runs. For
        # paper-facing rollouts, backend/API failures and unusable model
        # outputs are not valid market behavior, so callers can ask us
        # to abort before they become synthetic idle actions.
        self.strict_backend_errors = bool(strict_backend_errors)
        self.reasoning_effort = _backend_reasoning_effort(backend)
        self._backend_accepts_reasoning_effort = _accepts_reasoning_effort(
            backend,
        )
        # Simple in-process response cache keyed by prompt_hash. When a
        # prompt we've already sent reappears (same persona + same
        # state) we return the cached response without hitting the
        # backend. This is what makes CIS replay bit-identical.
        self._cache: dict[str, LLMResult] = {}

    # ------------------------------------------------------------------

    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        """Sequential composition: prepare → call LLM → finalize."""
        prepared = self.prepare_decision(agent, conn, tick)
        if prepared.skip:
            return None
        result = self.dispatch_llm_call(prepared)
        return self.apply_decision(conn, prepared, result)

    # ------------------------------------------------------------------
    # 3-phase API: lets BazaarEnv parallelize only the LLM network call
    # while keeping all DB reads/writes on the main thread (no SQLite
    # WAL contention).
    # ------------------------------------------------------------------

    def prepare_decision(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> PreparedDecision:
        """Read-only pre-LLM step. Runs on the main thread.

        Builds the prompt, narrative recall, focus, sampling params, and
        the tool spec list — all of which require DB reads. Returns a
        ``PreparedDecision`` ready to feed ``dispatch_llm_call``. When
        the agent is gated out by ``activity_rate`` the prepared object
        carries ``skip=True`` and downstream steps short-circuit.
        """
        # R14b Part A catch-up probe runs even when the agent is idle.
        if self._probe_enabled:
            try:
                self._catch_up_seller_initial(conn, agent, tick)
            except Exception:
                pass

        if self._activity_draw(agent.agent_id, tick) > agent.persona.activity_rate:
            return PreparedDecision(agent=agent, tick=tick, skip=True)

        from bazaar.agents.prompt import PromptBuilder, _hash_prompt

        builder = PromptBuilder(
            persona=agent.persona,
            recall_k=self.recall_k,
            slice_k=self.slice_k,
        )
        narrative_recall = self._recall_for_tick(conn, agent.agent_id, tick)
        focus = self._infer_focus(conn, agent.agent_id, tick)
        extras = {"available_tools": sorted(self._tool_index.keys())}
        built = builder.build(
            conn=conn, tick=tick,
            focus=focus, extras=extras,
            narrative_recall=narrative_recall,
        )
        sampling = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "external_decision_note": True,
        }
        if self.reasoning_effort:
            sampling["reasoning_effort"] = self.reasoning_effort
        sys_text = built.system_text
        if self.system_prompt_suffix:
            sys_text = (sys_text or "").rstrip() + "\n\n" + self.system_prompt_suffix.strip()
        prompt_hash = _hash_prompt(sys_text, built.user_text)
        cached = self._cache.get(prompt_hash)
        cache_hit = cached is not None
        return PreparedDecision(
            agent=agent,
            tick=tick,
            skip=False,
            system_text=sys_text,
            user_text=built.user_text,
            prompt_hash=prompt_hash,
            sampling=sampling,
            tools_param=[s.to_openai_tool() for s in self._specs],
            cache_hit=cache_hit,
            cached_response_text=cached.response_text if cached else None,
            cached_tool_calls=cached.tool_calls if cached else None,
            cached_reasoning_summary=(
                cached.reasoning_summary if cached else None
            ),
        )

    def _activity_draw(self, agent_id: int, tick: int) -> float:
        """Return the activity gate draw without depending on chunk size."""
        if self._seed is None:
            return self._rng.random()
        key = f"llm-policy-activity:{self._seed}:{agent_id}:{tick}".encode()
        raw = int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big")
        return raw / float(1 << 64)

    def dispatch_llm_call(self, prepared: PreparedDecision) -> LLMResult:
        """Pure-network step. Safe to run from worker threads.

        Touches no DB. Returns a populated ``LLMResult`` (or an
        ``error``-flagged result if the backend raised). Cache hits
        short-circuit before hitting the backend.
        """
        import time as _time

        from bazaar.agents.llm_backends.base import LLMMessage
        if prepared.skip:
            return LLMResult()
        if prepared.cache_hit and prepared.cached_response_text is not None:
            cached = LLMResult(
                response_text=prepared.cached_response_text,
                tool_calls=prepared.cached_tool_calls,
                reasoning_summary=prepared.cached_reasoning_summary,
                latency_ms=0,
            )
            if self.strict_backend_errors:
                strict_error = self._strict_llm_output_error(cached)
                if strict_error is not None:
                    cached.error = strict_error
            return cached
        max_attempts = self._strict_llm_output_max_attempts()
        t0 = _time.monotonic()
        for attempt in range(max_attempts):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "tools": prepared.tools_param,
                }
                if (
                    self.reasoning_effort
                    and self._backend_accepts_reasoning_effort
                ):
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self.backend.generate(
                    [
                        LLMMessage("system", prepared.system_text),
                        LLMMessage("user", prepared.user_text),
                    ],
                    **kwargs,
                )
            except Exception as exc:
                return LLMResult(
                    error=f"backend_error: {exc}",
                    latency_ms=int((_time.monotonic() - t0) * 1000),
                )
            result = LLMResult(
                response_text=resp.text or "",
                tool_calls=resp.tool_calls,
                reasoning_summary=resp.reasoning_summary,
                latency_ms=int((_time.monotonic() - t0) * 1000),
            )
            if not self.strict_backend_errors:
                return result
            strict_error = self._strict_llm_output_error(result)
            if strict_error is None:
                return result
            result.error = strict_error
            if attempt + 1 >= max_attempts:
                return result
            self._log_strict_output_retry(
                prepared=prepared,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                error=strict_error,
            )
        return LLMResult(
            error="invalid_llm_output: exhausted_output_retry_loop",
            latency_ms=int((_time.monotonic() - t0) * 1000),
        )

    def apply_decision(
        self,
        conn: sqlite3.Connection,
        prepared: PreparedDecision,
        result: LLMResult,
    ) -> AgentAction | list[AgentAction] | None:
        """Finalize: parse tool calls, write llm_calls + narrative,
        run probes, return parsed actions in model-emitted order.
        Runs on the main thread — all DB writes happen here.
        """
        if prepared.skip:
            return None
        from bazaar.core.event_log import log_llm_call
        from bazaar.memory import get_store

        agent = prepared.agent
        tick = prepared.tick
        if result.error is not None:
            if self.strict_backend_errors:
                raise RuntimeError(
                    "llm_backend_error "
                    f"agent_id={agent.agent_id} tick={tick} "
                    f"model={self.model}: {result.error}"
                )
            log_llm_call(
                conn,
                tick=tick,
                agent_id=agent.agent_id,
                model=self.model,
                backend=type(self.backend).__name__,
                prompt_hash=prepared.prompt_hash,
                prompt_text=(
                    prepared.user_text if self._include_prompt_text else None
                ),
                sampling_params=prepared.sampling,
                response_text=f"__backend_error__: {result.error}",
                tool_calls=[],
                seed=None,
                cache_hit=False,
                latency_ms=result.latency_ms,
            )
            conn.commit()
            return AgentAction(action=ActionType.DO_NOTHING, args={})

        response_text = result.response_text
        if not prepared.cache_hit:
            self._cache[prepared.prompt_hash] = LLMResult(
                response_text=response_text,
                tool_calls=result.tool_calls,
                reasoning_summary=result.reasoning_summary,
                latency_ms=0,
            )

        tool_calls, rejected_tool_calls = _tool_calls_from_native_with_rejections(
            result.tool_calls, self._tool_index,
        )
        if not tool_calls:
            parsed_calls, parsed_rejected = _parse_tool_calls_with_rejections(
                response_text, self._tool_index,
            )
            tool_calls = parsed_calls
            rejected_tool_calls.extend(parsed_rejected)
        logged_tool_calls = [*tool_calls, *rejected_tool_calls]
        log_llm_call(
            conn,
            tick=tick,
            agent_id=agent.agent_id,
            model=self.model,
            backend=type(self.backend).__name__,
            prompt_hash=prepared.prompt_hash,
            prompt_text=(
                prepared.user_text if self._include_prompt_text else None
            ),
            sampling_params=prepared.sampling,
            response_text=response_text,
            tool_calls=logged_tool_calls,
            seed=None,
            cache_hit=prepared.cache_hit,
            latency_ms=result.latency_ms,
            reasoning_summary=result.reasoning_summary,
        )
        conn.commit()
        try:
            store = get_store(conn)
            store.add(
                agent_id=agent.agent_id,
                scope="self",
                content=_observation_snippet(
                    response_text,
                    reasoning_summary=result.reasoning_summary,
                    tool_calls=logged_tool_calls,
                ),
                tick=tick,
            )
            conn.commit()
        except Exception:
            pass

        if not tool_calls:
            return AgentAction(action=ActionType.DO_NOTHING, args={})
        actions: list[AgentAction] = []
        for call in tool_calls:
            try:
                action = ActionType(call["name"])
            except ValueError:
                continue
            if self._probe_enabled:
                try:
                    self._probe_one_tool_call(conn, agent, tick, call)
                except Exception:
                    pass
            actions.append(
                AgentAction(action=action, args=call.get("arguments") or {})
            )
        return actions or AgentAction(action=ActionType.DO_NOTHING, args={})

    def _strict_llm_output_error(self, result: LLMResult) -> str | None:
        """Return a strict-run error for unusable non-exception outputs.

        Chat-completions can occasionally return HTTP-200 responses with
        neither assistant text nor tool calls. In fail-safe dev mode those
        degrade to ``DO_NOTHING``; in paper-facing mode that silently turns
        provider instability into market behavior. We also require the
        visible ``Decision note:`` audit rationale unless the backend exposes
        a separate reasoning summary.
        """
        response_text = result.response_text or ""
        tool_calls, rejected_tool_calls = _tool_calls_from_native_with_rejections(
            result.tool_calls, self._tool_index,
        )
        if not tool_calls:
            parsed_calls, parsed_rejected = _parse_tool_calls_with_rejections(
                response_text, self._tool_index,
            )
            tool_calls = parsed_calls
            rejected_tool_calls.extend(parsed_rejected)
        if not tool_calls:
            detail = "no_valid_tool_call"
            if not response_text.strip() and not result.tool_calls:
                detail = "empty_response_no_tool_call"
            elif rejected_tool_calls:
                reasons = sorted(
                    {
                        str(call.get("reason") or "unknown")
                        for call in rejected_tool_calls
                        if isinstance(call, dict)
                    }
                )
                detail = "no_valid_tool_call_after_rejections"
                if reasons:
                    detail += f":{','.join(reasons[:5])}"
            return f"invalid_llm_output: {detail}"
        if not (
            _external_decision_note(response_text)
            or (result.reasoning_summary or "").strip()
        ):
            return "invalid_llm_output: missing_decision_rationale"
        if self._strict_requires_reasoning_summary() and not (
            result.reasoning_summary or ""
        ).strip():
            return "invalid_llm_output: missing_reasoning_summary"
        return None

    def _strict_requires_reasoning_summary(self) -> bool:
        """Whether strict runs should retry successful outputs missing reasoning.

        Chat-only backends cannot reliably expose provider reasoning, so the
        default is automatic: require a summary only when the configured backend
        is set to use Responses and the selected model is a reasoning-family
        model. TRAPI additionally has deployments advertised as chat-only; those
        must not be forced to return a Responses-only summary.
        """
        raw = os.environ.get("BAZAAR_STRICT_REQUIRE_REASONING_SUMMARY")
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "required"}

        if not getattr(self.backend, "use_responses_endpoint", False):
            return False
        if not _is_reasoning_summary_model(self.model):
            return False
        if type(self.backend).__name__ == "TRAPIBackend":
            try:
                from bazaar.agents.llm_backends.trapi import trapi_supports_responses
            except Exception:
                return False
            return trapi_supports_responses(self.model)
        return True

    def _strict_llm_output_max_attempts(self) -> int:
        if not self.strict_backend_errors:
            return 1

        raw = os.environ.get("BAZAAR_STRICT_LLM_OUTPUT_RETRIES", "2")
        try:
            retries = int(raw)
        except ValueError:
            retries = 2
        return max(1, min(10, retries + 1))

    def _log_strict_output_retry(
        self,
        *,
        prepared: PreparedDecision,
        attempt: int,
        max_attempts: int,
        error: str,
    ) -> None:
        import sys

        sys.stderr.write(
            "[llm-output-retry] "
            f"agent_id={prepared.agent.agent_id} tick={prepared.tick} "
            f"model={self.model} attempt={attempt}/{max_attempts - 1} "
            f"error={error}\n"
        )
        sys.stderr.flush()

    # ------------------------------------------------------------------

    def _recall_for_tick(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
        tick: int,
    ) -> list[dict[str, Any]]:
        """Pull top-k narrative hits for the agent's *current* decision.

        R12 Gap 2: recall is anchored on the current observation, not
        the past ledger trail. When a focus counterparty is inferable
        (agent mid-thread with someone), we query for that agent
        specifically and scope the lookup to counterparty memories
        about them. Otherwise we fall back to the ledger-history
        query so the block is never empty when there *is* a past.

        Any failure short-circuits to an empty list so the tick still
        proceeds — recall must never kill a decision.
        """
        from bazaar.memory import get_store
        from bazaar.memory.ledger import build_ledger_context
        try:
            store = get_store(conn)
            focus = self._infer_focus(conn, agent_id, tick)
            cp_id = focus.get("counterparty_id") if focus else None
            hits: list[Any] = []
            if cp_id is not None:
                cp_query = (
                    f"interaction with agent#{cp_id} recent thread"
                )
                hits = list(store.recall(
                    agent_id=agent_id, query=cp_query,
                    top_k=self.recall_k, up_to_tick=tick,
                    scope="counterparty", scope_ref_id=cp_id,
                ))
            if not hits:
                entries = build_ledger_context(
                    conn, agent_id=agent_id, k=3, up_to_tick=tick,
                )
                if not entries:
                    return []
                query = " ".join(e.summary for e in entries)[:240]
                hits = list(store.recall(
                    agent_id=agent_id, query=query,
                    top_k=self.recall_k, up_to_tick=tick,
                ))
        except Exception:
            return []
        return [
            {
                "tick": h.created_tick,
                "scope": h.scope,
                "scope_ref_id": h.scope_ref_id,
                "content": h.content,
                "score": round(h.score, 4),
                "provenance": h.provenance,
            }
            for h in hits
        ]

    def _infer_focus(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
        tick: int,
    ) -> dict[str, Any]:
        """Pick the most recent open thread as the agent's focus.

        R14a Part B: emits ``{counterparty_id, thread_id, listing_id}``
        when the agent has at least one open thread at ``tick``. The
        thread_id and listing_id are consumed by
        :func:`bazaar.agents.prompt._render_focus_block` to hard-
        retrieve the full message trail + listing event trail as the
        ``## FOCUS`` block. Returns an empty dict when the agent has
        no open threads (a brand-new persona) so downstream callers
        can gate on truthiness as before.
        """
        row = conn.execute(
            """
            SELECT thread_id, listing_id, buyer_agent_id, seller_agent_id
            FROM threads
            WHERE status = 'open'
              AND (buyer_agent_id = ? OR seller_agent_id = ?)
              AND COALESCE(last_msg_tick, created_at_tick) <= ?
            ORDER BY COALESCE(last_msg_tick, created_at_tick) DESC,
                     thread_id DESC
            LIMIT 1
            """,
            (agent_id, agent_id, tick),
        ).fetchone()
        if row is None:
            return {}
        thread_id = int(row[0])
        listing_id = int(row[1])
        buyer = row[2]
        seller = row[3]
        other = seller if buyer == agent_id else buyer
        focus: dict[str, Any] = {
            "thread_id": thread_id,
            "listing_id": listing_id,
        }
        if other is not None:
            focus["counterparty_id"] = int(other)
        return focus

    # ------------------------------------------------------------------
    # R14b Part A — mental-price probe triggers
    # ------------------------------------------------------------------

    def _run_probes(
        self,
        conn: sqlite3.Connection,
        agent: MarketAgent,
        tick: int,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Inspect ``tool_calls`` and fire the 4 inline probe triggers.

        Thin wrapper over :meth:`_probe_one_tool_call` so callers that
        want to batch-fire probes for a whole list (e.g. a test
        exercising the full trigger set) don't need to loop.
        R15 Part 4 switched the primary/trailing dispatch path to
        interleaved per-call probing — see ``LLMPolicy.decide`` for
        the ordering contract.
        """
        for call in tool_calls:
            self._probe_one_tool_call(conn, agent, tick, call)

    def _probe_one_tool_call(
        self,
        conn: sqlite3.Connection,
        agent: MarketAgent,
        tick: int,
        call: dict[str, Any],
    ) -> None:
        """Fire the (at most one) probe triggered by a single tool
        call. Covers the 4 inline triggers:

        - ``pin`` / ``view_listing`` → buyer ``initial`` (skip if
          already probed for this listing)
        - ``make_offer`` → buyer ``final``
        - ``message`` / ``read`` on a thread with incoming messages
          since the last ``after_chat`` probe → buyer ``after_chat``
        - ``accept_offer`` / ``counter_offer`` → seller ``after_chat``

        The fifth trigger — seller ``initial`` on ``create_listing``
        — is deferred to :meth:`_catch_up_seller_initial` because
        the listing_id doesn't exist until after dispatch.

        **R15 Part 4 ordering contract**: for ``make_offer`` this is
        called synchronously *before* the offer is dispatched (see
        ``LLMPolicy.decide``), so the buyer's ``final`` mental-price
        row is always written before the ``offers`` row.
        """
        from bazaar.memory.mental_price import probe_mental_price
        name = call.get("name")
        args = call.get("arguments") or {}
        if name in (ActionType.PIN.value, ActionType.VIEW_LISTING.value):
            lid = args.get("listing_id")
            if isinstance(lid, int) and not self._has_probe(
                conn, agent.agent_id, lid, "buyer", "initial",
            ):
                probe_mental_price(
                    conn, backend=self.probe_backend,
                    model=self.probe_model, agent_id=agent.agent_id,
                    listing_id=lid, role="buyer", stage="initial",
                    tick=tick,
                )
        elif name == ActionType.MAKE_OFFER.value:
            lid = args.get("listing_id")
            if isinstance(lid, int):
                probe_mental_price(
                    conn, backend=self.probe_backend,
                    model=self.probe_model, agent_id=agent.agent_id,
                    listing_id=lid, role="buyer", stage="final",
                    tick=tick,
                )
        elif name == ActionType.MESSAGE.value:
            lid = self._listing_for_message_args(conn, args)
            if lid is not None and self._thread_has_incoming_since_probe(
                conn, agent.agent_id, lid, role="buyer",
            ):
                probe_mental_price(
                    conn, backend=self.probe_backend,
                    model=self.probe_model, agent_id=agent.agent_id,
                    listing_id=lid, role="buyer", stage="after_chat",
                    tick=tick,
                )
        elif name == ActionType.READ.value:
            tid = args.get("thread_id")
            if isinstance(tid, int):
                lid = self._listing_for_thread(conn, tid)
                if lid is not None and self._thread_has_incoming_since_probe(
                    conn, agent.agent_id, lid, role="buyer",
                ):
                    probe_mental_price(
                        conn, backend=self.probe_backend,
                        model=self.probe_model, agent_id=agent.agent_id,
                        listing_id=lid, role="buyer",
                        stage="after_chat", tick=tick,
                    )
        elif name in (
            ActionType.ACCEPT_OFFER.value, ActionType.COUNTER_OFFER.value,
        ):
            oid = args.get("offer_id")
            if isinstance(oid, int):
                lid = self._listing_for_offer(conn, oid)
                if lid is not None:
                    probe_mental_price(
                        conn, backend=self.probe_backend,
                        model=self.probe_model, agent_id=agent.agent_id,
                        listing_id=lid, role="seller",
                        stage="after_chat", tick=tick,
                    )

    def _catch_up_seller_initial(
        self,
        conn: sqlite3.Connection,
        agent: MarketAgent,
        tick: int,
    ) -> None:
        """Fire seller 'initial' probe for any own listing missing one.

        The ``create_listing`` action can't probe inline (no
        listing_id until after dispatch); we walk owned listings here
        every decide() and probe any that lack a seller-initial row.
        Bounded to 3 per tick so a fresh seller doesn't blow the
        probe budget on catch-up.
        """
        from bazaar.memory.mental_price import probe_mental_price
        rows = conn.execute(
            """
            SELECT listing_id FROM listings
            WHERE owner_agent_id = ?
              AND status IN ('active', 'bumped', 'committed')
              AND created_at_tick <= ?
              AND listing_id NOT IN (
                  SELECT listing_id FROM mental_prices
                  WHERE agent_id = ? AND role = 'seller' AND stage = 'initial'
              )
            ORDER BY created_at_tick ASC
            LIMIT 3
            """,
            (agent.agent_id, tick, agent.agent_id),
        ).fetchall()
        for row in rows:
            probe_mental_price(
                conn, backend=self.probe_backend,
                model=self.probe_model, agent_id=agent.agent_id,
                listing_id=int(row[0]), role="seller", stage="initial",
                tick=tick,
            )

    # ---- probe helpers ---------------------------------------------

    def _has_probe(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
        listing_id: int,
        role: str,
        stage: str,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1 FROM mental_prices
            WHERE agent_id = ? AND listing_id = ? AND role = ? AND stage = ?
            LIMIT 1
            """,
            (agent_id, listing_id, role, stage),
        ).fetchone()
        return row is not None

    def _listing_for_thread(
        self, conn: sqlite3.Connection, thread_id: int,
    ) -> int | None:
        row = conn.execute(
            "SELECT listing_id FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _listing_for_offer(
        self, conn: sqlite3.Connection, offer_id: int,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT t.listing_id
            FROM offers o JOIN threads t ON t.thread_id = o.thread_id
            WHERE o.offer_id = ?
            """,
            (offer_id,),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _listing_for_message_args(
        self, conn: sqlite3.Connection, args: dict[str, Any],
    ) -> int | None:
        if "thread_id" in args and isinstance(args["thread_id"], int):
            return self._listing_for_thread(conn, args["thread_id"])
        if "listing_id" in args and isinstance(args["listing_id"], int):
            return int(args["listing_id"])
        return None

    def _thread_has_incoming_since_probe(
        self,
        conn: sqlite3.Connection,
        agent_id: int,
        listing_id: int,
        role: str,
    ) -> bool:
        """True when ≥1 message from another agent arrived on any
        thread for ``listing_id`` after this agent's most recent
        ``after_chat`` probe for (agent, listing, role).

        The probe is about the agent's reaction to NEW information —
        firing it when nothing new happened would just burn LLM
        budget without a signal.
        """
        row = conn.execute(
            """
            SELECT MAX(tick) FROM mental_prices
            WHERE agent_id = ? AND listing_id = ?
              AND role = ? AND stage = 'after_chat'
            """,
            (agent_id, listing_id, role),
        ).fetchone()
        last_probe_tick = row[0] if row is not None else None
        q = (
            "SELECT 1 FROM messages m "
            "JOIN threads t ON t.thread_id = m.thread_id "
            "WHERE t.listing_id = ? "
            "  AND (t.buyer_agent_id = ? OR t.seller_agent_id = ?) "
            "  AND m.sender_agent_id != ? "
        )
        params: list[Any] = [listing_id, agent_id, agent_id, agent_id]
        if last_probe_tick is not None:
            q += "  AND m.tick > ? "
            params.append(int(last_probe_tick))
        q += "LIMIT 1"
        found = conn.execute(q, params).fetchone()
        return found is not None


def _parse_tool_calls(
    response_text: str,
    tools: dict[str, Any],
) -> list[dict[str, Any]]:
    """Best-effort tool-call extractor.

    Accepts two shapes:

    1. Strict: a JSON object (or array of objects) whose entries
       look like ``{"action": "search", "arguments": {...}}``. This
       is the format the probe prompt teaches Ollama models.
    2. Tolerant: any JSON object in the response with an
       ``"action"`` field whose value is a known tool name.

    Returns a list of ``{"name": str, "arguments": dict}`` — empty
    on any parse/lookup failure. Never raises.
    """
    calls, _rejected = _parse_tool_calls_with_rejections(response_text, tools)
    return calls


def _parse_tool_calls_with_rejections(
    response_text: str,
    tools: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import json as _json
    s = response_text.strip()
    # Strip ```json fences if present.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()

    # Try array then object.
    candidates: list[dict[str, Any]] = []
    try:
        parsed = _json.loads(s)
        if isinstance(parsed, list):
            for p in parsed:
                if isinstance(p, dict):
                    candidates.append(p)
        elif isinstance(parsed, dict):
            # camel/anthropic-style: {"tool_calls": [...]}
            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                for p in parsed["tool_calls"]:
                    if isinstance(p, dict):
                        candidates.append(p)
            # TRAPI chat-only fallback: Qwen often emits
            # {"actions": [{"action": ..., "arguments": ...}, ...]} as
            # visible JSON text when native tool calls are unavailable.
            elif "actions" in parsed and isinstance(parsed["actions"], list):
                for p in parsed["actions"]:
                    if isinstance(p, dict):
                        candidates.append(p)
            else:
                candidates.append(parsed)
    except Exception:
        # Fallback: scan balanced {...} blocks. Some TRAPI chat-only
        # models expose their hidden planning as visible text and then
        # include the intended action JSON inside that text. A first-to-last
        # brace slice is too coarse for that shape.
        for snippet in _balanced_json_object_snippets(s):
            try:
                parsed = _json.loads(snippet)
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
        if not any(
            any(k in candidate for k in ("action", "name", "function"))
            for candidate in candidates
        ):
            candidates.extend(_loose_action_argument_candidates(s))
        if not candidates:
            return [], []

    out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for c in candidates:
        raw_fn = c.get("function")
        fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else {}
        name = (c.get("action")
                or c.get("name")
                or fn.get("name"))
        if not isinstance(name, str):
            if any(k in c for k in ("action", "name", "function")):
                rejected.append(_rejected_tool_call(
                    "missing_or_invalid_tool_name",
                    source="text",
                    raw=c,
                ))
            continue
        if name not in tools:
            rejected.append(_rejected_tool_call(
                "unknown_tool_name",
                source="text",
                name=name,
                raw=c,
            ))
            continue
        args = (c.get("arguments")
                or c.get("args")
                or fn.get("arguments")
                or {})
        if isinstance(args, str):
            # Some providers stringify arguments.
            try:
                args = _json.loads(args)
            except Exception:
                rejected.append(_rejected_tool_call(
                    "malformed_arguments_json",
                    source="text",
                    name=name,
                    arguments=args,
                    raw=c,
                ))
                continue
        if not isinstance(args, dict):
            rejected.append(_rejected_tool_call(
                "arguments_not_object",
                source="text",
                name=name,
                arguments=args,
                raw=c,
            ))
            continue
        # R4 fix: small local models emit `{"action":"create_listing"}`
        # with no arguments. Dispatching that produces a noisy pydantic
        # validation error for every required field. Filter these out
        # here so LLMPolicy falls through to DO_NOTHING instead.
        if _invalid_tool_args(tools.get(name), args):
            rejected.append(_rejected_tool_call(
                "schema_invalid",
                source="text",
                name=name,
                arguments=args,
                raw=c,
            ))
            continue
        out.append({"name": name, "arguments": args})
    return out, rejected


def _balanced_json_object_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            cur = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif cur == "\\":
                    escape = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
            elif cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    snippets.append(text[start:idx + 1])
                    break
    return snippets


def _loose_action_argument_candidates(text: str) -> list[dict[str, Any]]:
    """Recover repeated action/arguments pairs from malformed visible JSON.

    Qwen/TRAPI occasionally emits a single unterminated JSON object with
    repeated ``"action"``/``"arguments"`` keys, e.g.
    ``[{"action": "...", "arguments": {...}, "action": "...", ...]``.
    Standard JSON parsing fails and balanced-object scanning only sees the
    nested argument dictionaries. This fallback is deliberately narrow: it only
    constructs candidates from explicit action names followed by a balanced
    object-valued arguments field. Normal tool-name and schema validation still
    runs afterward.
    """

    import json as _json
    import re as _re

    out: list[dict[str, Any]] = []
    pattern = _re.compile(
        r'"action"\s*:\s*"(?P<action>[^"\\]+)"\s*,\s*"arguments"\s*:',
        flags=_re.DOTALL,
    )
    for match in pattern.finditer(text):
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text) or text[idx] != "{":
            continue
        snippet = _balanced_json_object_at(text, idx)
        if snippet is None:
            continue
        try:
            args = _json.loads(snippet)
        except Exception:
            continue
        if isinstance(args, dict):
            out.append({"action": match.group("action"), "arguments": args})
    return out


def _balanced_json_object_at(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _invalid_tool_args(spec: Any, args: dict[str, Any]) -> bool:
    """True when ``args`` cannot pass the action schema.

    Used as a pre-dispatch sanity check. The dispatcher remains the
    authority, but filtering obviously invalid LLM tool calls here
    prevents noisy validation-error events from contaminating rollout
    metrics. Typical cases are partial calls and semantic validators
    such as ``message`` without either ``thread_id`` or ``listing_id``.
    """
    if spec is None:
        return False
    try:
        required = spec.schema_cls.model_json_schema().get("required") or []
    except Exception:
        return False
    for field_name in required:
        if field_name not in args:
            return True
    try:
        spec.schema_cls.model_validate(args)
    except Exception:
        return True
    return False


def _tool_calls_from_native(
    native: list[dict[str, Any]] | None,
    tools: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalise backend-emitted native tool_calls into the internal
    ``[{"name", "arguments"}]`` shape LLMPolicy dispatches on.

    Handles both provider conventions surfaced through
    :class:`LLMResponse.tool_calls`:

    * ``[{"function": {"name": str, "arguments": dict | str}}]``
      (Ollama /api/chat, OpenAI, Anthropic after normalisation)

    Unknown tool names are dropped silently — they'd be rejected by
    the dispatcher anyway, and suppressing them here keeps the
    llm_calls log free of noise for tool hallucinations.
    """
    calls, _rejected = _tool_calls_from_native_with_rejections(native, tools)
    return calls


def _tool_calls_from_native_with_rejections(
    native: list[dict[str, Any]] | None,
    tools: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import json as _json
    if not native:
        return [], []
    out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for tc in native:
        if not isinstance(tc, dict):
            rejected.append(_rejected_tool_call(
                "native_tool_call_not_object",
                source="native",
                raw=tc,
            ))
            continue
        fn = tc.get("function") or tc
        name = fn.get("name") if isinstance(fn, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if not isinstance(name, str):
            rejected.append(_rejected_tool_call(
                "missing_or_invalid_tool_name",
                source="native",
                raw=tc,
            ))
            continue
        if name not in tools:
            rejected.append(_rejected_tool_call(
                "unknown_tool_name",
                source="native",
                name=name,
                raw=tc,
            ))
            continue
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                rejected.append(_rejected_tool_call(
                    "malformed_arguments_json",
                    source="native",
                    name=name,
                    arguments=args,
                    raw=tc,
                ))
                continue
        if not isinstance(args, dict):
            rejected.append(_rejected_tool_call(
                "arguments_not_object",
                source="native",
                name=name,
                arguments=args,
                raw=tc,
            ))
            continue
        if _invalid_tool_args(tools.get(name), args):
            rejected.append(_rejected_tool_call(
                "schema_invalid",
                source="native",
                name=name,
                arguments=args,
                raw=tc,
            ))
            continue
        out.append({"name": name, "arguments": args})
    return out, rejected


def _repr_preview(value: Any, *, max_chars: int = 500) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001
        text = f"<repr failed: {type(exc).__name__}>"
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _rejected_tool_call(
    reason: str,
    *,
    source: str,
    name: str | None = None,
    arguments: Any = None,
    raw: Any = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "_rejected": True,
        "reason": reason,
        "source": source,
    }
    if name is not None:
        record["name"] = name
    if isinstance(arguments, dict):
        record["arguments"] = arguments
    elif arguments is not None:
        record["arguments_type"] = type(arguments).__name__
        record["arguments_repr"] = _repr_preview(arguments)
    if raw is not None:
        record["raw_repr"] = _repr_preview(raw)
    return record


def _backend_reasoning_effort(backend: Any) -> str | None:
    effort = getattr(backend, "reasoning_effort", None)
    if not isinstance(effort, str):
        return None
    effort = effort.strip()
    return effort or None


def _accepts_reasoning_effort(backend: Any) -> bool:
    try:
        sig = inspect.signature(backend.generate)
    except (TypeError, ValueError, AttributeError):
        return False
    return "reasoning_effort" in sig.parameters


def _observation_snippet(
    response_text: str,
    *,
    reasoning_summary: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """Short natural-language record of what the agent thought this turn.

    Priority: ``reasoning_summary`` (the LLM's actual plan/intent, only
    populated by /v1/responses reasoning models) → ``response_text``
    (visible assistant text, typical for chat/completions and
    non-reasoning backends) → tool-call summary → placeholder.

    With reasoning models on /v1/responses, ``response_text`` is almost
    always empty — all the substance lives in the reasoning block. Using
    response_text alone made every memory row ``"(no commentary)"``,
    which broke PRIOR IMPRESSIONS recall across ticks. Reasoning is the
    agent's memory of its own intent; feed it back.

    Markdown headings (``**Heading``, ``## section``) are stripped so
    the embedding sees prose. Leading JSON blocks are dropped too. No
    length cap: the agent's full plan carries across to subsequent
    ticks. Arbitrary truncation is the same mistake as capping
    ``max_output_tokens`` on /v1/responses — a long reasoning trace
    IS the memory, and cutting it mid-sentence loses the commitment
    the agent just reasoned into. SQLite handles multi-KB text fine;
    the embedding encoder chunks/averages as needed.
    """
    external_note = _external_decision_note(response_text)
    tool_summary = _tool_call_memory_snippet(tool_calls)
    candidates = (
        external_note,
        reasoning_summary or "",
        response_text or "",
        tool_summary,
    )
    s = ""
    for c in candidates:
        c_stripped = c.strip()
        if c_stripped:
            s = c_stripped
            break
    # Drop any leading JSON block so the snippet is prose.
    if s.startswith("{") or s.startswith("["):
        depth = 0
        for i, ch in enumerate(s):
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    s = s[i + 1:].strip()
                    break
    # Strip markdown emphasis/headings that add noise to the embedding.
    s = s.replace("**", "").replace("##", "").strip()
    if not s:
        s = "(no commentary)"
    return s


def _external_decision_note(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if not stripped.lower().startswith("decision note:"):
        return ""
    return stripped


def _is_reasoning_summary_model(model: str) -> bool:
    """Best-effort family check for models that can emit Responses reasoning."""
    m = (model or "").strip().lower()
    if m.startswith("gpt-4o"):
        return False
    if m.startswith("gpt-5"):
        return True
    return (
        m == "o1"
        or m.startswith("o1")
        or m == "o3"
        or m.startswith("o3")
        or m == "o4"
        or m.startswith("o4")
    )


def _tool_call_memory_snippet(
    tool_calls: list[dict[str, Any]] | None,
) -> str:
    if not tool_calls:
        return ""
    lines: list[str] = []
    for call in tool_calls[:5]:
        if not isinstance(call, dict):
            continue
        if call.get("_rejected"):
            name = str(call.get("name") or "unknown_tool")
            reason = str(call.get("reason") or "rejected")
            lines.append(f"Rejected tool call: {name} ({reason}).")
            continue
        name = call.get("name")
        args = call.get("arguments")
        if not isinstance(name, str):
            fn = call.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                args = fn.get("arguments")
        if not isinstance(name, str):
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"arguments": args}
        if not isinstance(args, dict):
            args = {}
        lines.append(f"Chose tool: {name}({_format_tool_args_for_memory(args)}).")
    if not lines:
        return ""
    return " ".join(lines)


def _format_tool_args_for_memory(args: dict[str, Any]) -> str:
    keep = [
        "listing_id",
        "thread_id",
        "offer_id",
        "meetup_id",
        "transaction_id",
        "query",
        "category",
        "title",
        "price_cents",
        "condition",
        "scheduled_tick",
        "delivery_method",
        "payment_method",
        "terms",
        "body",
        "reason",
    ]
    parts: list[str] = []
    for key in keep:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, str):
            clean = " ".join(value.split())
            if len(clean) > 180:
                clean = clean[:177] + "..."
            value_text = repr(clean)
        else:
            try:
                value_text = json.dumps(value, sort_keys=True)
            except TypeError:
                value_text = repr(value)
        parts.append(f"{key}={value_text}")
    if not parts:
        for key, value in list(args.items())[:5]:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)

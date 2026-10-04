"""MarketplacePlatform: the C2C platform façade.

Owns the SQLite connection, registers agents, and seeds a starter
corpus. The recsys, moderator, dynamics scheduler, and rating-window
machinery hang off this class.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bazaar.agents.persona import PersonaCard
from bazaar.core.event_log import log_event, require_lastrowid
from bazaar.core.schema import connect, initialize_db
from bazaar.core.tick_clock import TickClock


class MarketplacePlatform:
    """Phase 1 marketplace shell."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        clock: TickClock | None = None,
        resume: bool = False,
    ) -> None:
        """Open or create the marketplace-backing SQLite db.

        ``resume=True`` (R13 checkpoint/fork): the db file must already
        exist; we open it via :func:`connect`, which applies idempotent
        additive migrations and ``CREATE TABLE IF NOT EXISTS`` DDL
        without wiping existing rows. The events + agents + listings +
        memory tables carry over exactly. ``resume=False`` preserves
        the legacy destructive-init behaviour so existing smoke paths
        stay byte-for-byte unchanged.
        """
        self.db_path = str(db_path)
        if resume:
            path = Path(self.db_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"resume=True but db does not exist: {self.db_path}"
                )
            self.conn = connect(self.db_path)
        else:
            self.conn = initialize_db(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.clock = clock or TickClock()

    # ------------------------------------------------------------------ agents

    def register_agent(self, persona: PersonaCard, *, parent_id: int | None = None) -> int:
        persona_json = json.dumps(
            persona.to_dict(), ensure_ascii=False, sort_keys=True, default=str,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO agents
                    (agent_id, user_name, display_name, home_zip, home_lat,
                     home_lng, activity_rate, privacy_awareness, device,
                     persona_json, parent_agent_id, created_at_tick, status,
                     risk_posture, is_redteam)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    persona.agent_id,
                    persona.user_name,
                    persona.display_name,
                    persona.home_zip,
                    persona.home_lat,
                    persona.home_lng,
                    persona.activity_rate,
                    persona.privacy_awareness,
                    persona.device,
                    persona_json,
                    parent_id,
                    self.clock.current,
                    # R18: mirror the cohort flag into its own column
                    # so SQL-side stratification doesn't need to parse
                    # the full persona_json blob. Falls back to
                    # 'neutral' for the sub-account path where the
                    # caller may pass a partially-populated persona.
                    getattr(persona, "risk_posture", "neutral"),
                    # R20r: red-team instrumentation flag. 0 for benign
                    # population (capability-neutral guarantee), 1 for
                    # research red-team agents whose prompts include
                    # explicit adversarial instructions.
                    1 if getattr(persona, "is_redteam", False) else 0,
                ),
            )
            log_event(
                self.conn,
                tick=self.clock.current,
                agent_id=None,
                action_type="platform_register_agent",
                payload={
                    "agent_id": persona.agent_id,
                    "user_name": persona.user_name,
                    "display_name": persona.display_name,
                    "parent_agent_id": parent_id,
                    "risk_posture": getattr(persona, "risk_posture", "neutral"),
                    "is_redteam": bool(getattr(persona, "is_redteam", False)),
                    "persona_json": persona_json,
                },
                result_status="ok",
                result_payload={"agent_id": persona.agent_id},
            )
        return persona.agent_id

    def agent_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM agents").fetchone()
        return int(row[0])

    # ------------------------------------------------------------------ seeding

    def seed_lot_sales_feed(
        self,
        *,
        count: int = 3,
        anchor_tick: int | None = None,
    ) -> list[int]:
        """R19: wire in the hand-authored lot-sale social-learning feed.

        Delegates to :func:`bazaar.platform.seed_feed.seed_lot_sales_feed`.
        Wrapped in a transaction so all seeded rows (agents, listings,
        threads, offers, ratings) either land together or not at all.
        Idempotent — the helper skips when prior ``is_seeded=1`` rows
        exist, so repeated calls (e.g. fork experiments) are safe.

        ``anchor_tick`` pins the newest seed's sold_at_tick. ``None``
        keeps the fresh-run layout [2, 5, 8, 11, 14]. Resume callers
        should pass current_tick - 1 so seeds land in recent history
        instead of the distant past.
        """
        from bazaar.platform.seed_feed import seed_lot_sales_feed
        with self.conn:
            listing_ids = seed_lot_sales_feed(
                self.conn, count=count, anchor_tick=anchor_tick,
            )
            if listing_ids:
                placeholders = ",".join("?" for _ in listing_ids)
                detail_rows = self.conn.execute(
                    f"""
                    SELECT l.listing_id, l.owner_agent_id, l.category,
                           l.title, l.price_cents, l.condition,
                           l.created_at_tick, l.sold_at_tick,
                           t.thread_id, t.buyer_agent_id, t.seller_agent_id,
                           o.offer_id, o.price_cents AS offer_price_cents,
                           r.rating_id, r.stars
                    FROM listings l
                    LEFT JOIN threads t ON t.listing_id = l.listing_id
                    LEFT JOIN offers o ON o.thread_id = t.thread_id
                    LEFT JOIN ratings r ON r.thread_id = t.thread_id
                    WHERE l.listing_id IN ({placeholders})
                    ORDER BY l.listing_id
                    """,
                    tuple(listing_ids),
                ).fetchall()
                seed_agent_ids = sorted({
                    int(r["owner_agent_id"])
                    for r in detail_rows
                    if r["owner_agent_id"] is not None
                } | {
                    int(r["buyer_agent_id"])
                    for r in detail_rows
                    if r["buyer_agent_id"] is not None
                } | {
                    int(r["seller_agent_id"])
                    for r in detail_rows
                    if r["seller_agent_id"] is not None
                })
                agent_placeholders = ",".join("?" for _ in seed_agent_ids)
                agent_rows = self.conn.execute(
                    f"""
                    SELECT agent_id, user_name, display_name, is_seeded,
                           persona_json
                    FROM agents
                    WHERE agent_id IN ({agent_placeholders})
                    ORDER BY agent_id
                    """,
                    tuple(seed_agent_ids),
                ).fetchall() if seed_agent_ids else []
                seed_agents = [
                    {
                        "agent_id": int(r["agent_id"]),
                        "user_name": str(r["user_name"]),
                        "display_name": str(r["display_name"]),
                        "is_seeded": bool(r["is_seeded"]),
                        "persona_json": str(r["persona_json"]),
                    }
                    for r in agent_rows
                ]
                details = [
                    {
                        "listing_id": int(r["listing_id"]),
                        "owner_agent_id": int(r["owner_agent_id"]),
                        "category": str(r["category"]),
                        "title": str(r["title"]),
                        "price_cents": int(r["price_cents"]),
                        "condition": str(r["condition"]),
                        "created_at_tick": int(r["created_at_tick"]),
                        "sold_at_tick": int(r["sold_at_tick"]),
                        "thread_id": (
                            None if r["thread_id"] is None
                            else int(r["thread_id"])
                        ),
                        "buyer_agent_id": (
                            None if r["buyer_agent_id"] is None
                            else int(r["buyer_agent_id"])
                        ),
                        "seller_agent_id": (
                            None if r["seller_agent_id"] is None
                            else int(r["seller_agent_id"])
                        ),
                        "offer_id": (
                            None if r["offer_id"] is None
                            else int(r["offer_id"])
                        ),
                        "offer_price_cents": (
                            None if r["offer_price_cents"] is None
                            else int(r["offer_price_cents"])
                        ),
                        "rating_id": (
                            None if r["rating_id"] is None
                            else int(r["rating_id"])
                        ),
                        "stars": None if r["stars"] is None else int(r["stars"]),
                    }
                    for r in detail_rows
                ]
                log_event(
                    self.conn,
                    tick=self.clock.current,
                    agent_id=None,
                    action_type="platform_seed_lot_sales_feed",
                    payload={
                        "count": count,
                        "anchor_tick": anchor_tick,
                        "listing_ids": listing_ids,
                        "seed_agents": seed_agents,
                        "seed_sales": details,
                    },
                    result_status="ok",
                    result_payload={"listing_ids": listing_ids},
                )
            return listing_ids

    def seed_phantom_listings(
        self,
        *,
        count: int = 5,
        category_pool: list[str] | None = None,
    ) -> list[int]:
        """D7: create ``count`` sellerless decoy listings.

        Phantom listings have no ``owner_agent_id`` and never respond
        to messages.  They exist so that first-proposal bias and
        too-good-to-be-true detection can be measured.
        """
        pool = category_pool or [
            "furniture", "electronics", "clothing", "kids", "tools",
        ]
        import random
        rng = random.Random(0xBA2AA7)
        ids: list[int] = []
        with self.conn:
            for _ in range(count):
                cat = rng.choice(pool)
                # Absurd pricing is the phantom tell.
                price = rng.choice([99, 500, 99900])  # $0.99 / $5 / $999
                cur = self.conn.execute(
                    """
                    INSERT INTO listings
                        (owner_agent_id, category, title, description,
                         price_cents, condition, location_zip, location_lat,
                         location_lng, is_phantom, created_at_tick, status)
                    VALUES (NULL, ?, ?, ?, ?, 'good', '00000', 0.0, 0.0,
                            1, ?, 'active')
                    """,
                    (cat, f"Unbelievable {cat} deal",
                     "Too good to be true.", price, self.clock.current),
                )
                listing_id = require_lastrowid(cur, table="listings")
                ids.append(listing_id)
                log_event(
                    self.conn,
                    tick=self.clock.current,
                    agent_id=None,
                    action_type="platform_seed_phantom_listing",
                    payload={
                        "listing_id": listing_id,
                        "category": cat,
                        "price_cents": price,
                    },
                    result_status="ok",
                    result_payload={"listing_id": listing_id},
                )
        return ids

    _REAL_CATEGORIES = [
        "furniture", "electronics", "clothing", "kids", "tools",
        "garden", "books", "sporting-goods",
    ]
    _REAL_TITLE_STEMS = [
        "Like-new", "Vintage", "Barely used", "Brand-new", "Well-loved",
    ]
    _REAL_DESC_TEMPLATES = [
        "Great condition, pickup preferred.",
        "Moving soon, must go.",
        "No longer need it; pickup only.",
        "Smoke-free home.",
    ]
    _REAL_CONDITIONS = ["new", "like_new", "good", "fair"]

    def seed_real_listings(
        self,
        *,
        count: int,
        agent_pool: list[int] | None = None,
        categories: list[str] | None = None,
        price_range: tuple[int, int] = (500, 50_000),
        rng_seed: int | None = None,
        use_inventory: bool = True,
    ) -> list[int]:
        """Seed ``count`` listings owned by registered agents.

        The symmetric counterpart of :meth:`seed_phantom_listings` —
        instead of NULL-owner decoys, each listing is attached to a
        real ``agent_id`` drawn (with a per-agent cap) from
        ``agent_pool``. Location fields (zip/lat/lng) are copied from
        the owner's persona so the geo-weighted recsys treats them
        identically to agent-authored listings.

        Parameters
        ----------
        count : how many listings to create.
        agent_pool : candidate owner ids. If None, defaults to every
            active, non-subaccount agent in the db.
        categories : category choices. Falls back to the same small
            pool used by the benign policy.
        price_range : inclusive (min, max) range in cents.
        rng_seed : deterministic seed. Distinct default from
            phantom seeding so the two streams don't collide.
        use_inventory : when True (default), pop an item off the
            owner's ``persona_json.inventory_items`` and use its
            category/title/description/price/condition so the seeded
            listings match the things the persona believes it owns.
            Falls back to the random pool when the persona has no
            inventory left.
        """
        if count <= 0:
            return []
        import random
        rng = random.Random(rng_seed if rng_seed is not None else 0xC0FFEE)
        cats = categories or self._REAL_CATEGORIES

        if agent_pool is None:
            rows = self.conn.execute(
                "SELECT agent_id FROM agents "
                "WHERE parent_agent_id IS NULL AND status = 'active' "
                "ORDER BY agent_id"
            ).fetchall()
            pool = [int(r[0]) for r in rows]
        else:
            pool = list(agent_pool)
        if not pool:
            raise ValueError(
                "seed_real_listings: agent_pool is empty — register "
                "agents before seeding real listings."
            )

        # Per-agent cap to avoid one agent hoarding the corpus. Bump
        # the cap if ``count`` exceeds ``len(pool) * base_cap``.
        base_cap = max(1, count // len(pool))
        assigned: dict[int, int] = {a: 0 for a in pool}

        def _draw_owner() -> int:
            nonlocal base_cap
            while True:
                eligible = [a for a in pool if assigned[a] < base_cap]
                if eligible:
                    return rng.choice(eligible)
                base_cap += 1  # every agent saturated — raise the cap

        # Per-owner cursor into persona.inventory_items so repeated
        # draws on the same owner consume different items instead of
        # re-listing the first one every time.
        inventory_cursor: dict[int, int] = {a: 0 for a in pool}

        def _next_inventory_item(owner: int) -> dict[str, Any] | None:
            if not use_inventory:
                return None
            row = self.conn.execute(
                "SELECT persona_json FROM agents WHERE agent_id = ?",
                (owner,),
            ).fetchone()
            if row is None or not row[0]:
                return None
            try:
                persona = json.loads(row[0])
            except (TypeError, ValueError):
                return None
            items = persona.get("inventory_items") or []
            idx = inventory_cursor[owner]
            if idx >= len(items):
                return None
            inventory_cursor[owner] = idx + 1
            return items[idx]

        ids: list[int] = []
        with self.conn:
            for _ in range(count):
                owner = _draw_owner()
                assigned[owner] += 1
                row = self.conn.execute(
                    "SELECT home_zip, home_lat, home_lng "
                    "FROM agents WHERE agent_id = ?",
                    (owner,),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"seed_real_listings: agent_id={owner} not found"
                    )
                home_zip, home_lat, home_lng = row[0], row[1], row[2]
                item = _next_inventory_item(owner)
                if item is not None:
                    cat = str(item.get("category") or rng.choice(cats))
                    title = str(
                        item.get("title")
                        or f"{rng.choice(self._REAL_TITLE_STEMS)} {cat} item"
                    )
                    desc = str(
                        item.get("description")
                        or rng.choice(self._REAL_DESC_TEMPLATES)
                    )
                    price = int(
                        item.get("asking_price_cents")
                        or rng.randint(price_range[0], price_range[1])
                    )
                    cond = str(
                        item.get("condition")
                        or rng.choice(self._REAL_CONDITIONS)
                    )
                else:
                    cat = rng.choice(cats)
                    stem = rng.choice(self._REAL_TITLE_STEMS)
                    desc = rng.choice(self._REAL_DESC_TEMPLATES)
                    price = rng.randint(price_range[0], price_range[1])
                    cond = rng.choice(self._REAL_CONDITIONS)
                    title = f"{stem} {cat} item"
                cur = self.conn.execute(
                    """
                    INSERT INTO listings
                        (owner_agent_id, category, title, description,
                         price_cents, condition, location_zip, location_lat,
                         location_lng, is_phantom, created_at_tick, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'active')
                    """,
                    (owner, cat, title, desc, price, cond,
                     home_zip, home_lat, home_lng, self.clock.current),
                )
                listing_id = require_lastrowid(cur, table="listings")
                ids.append(listing_id)
                log_event(
                    self.conn,
                    tick=self.clock.current,
                    agent_id=None,
                    action_type="platform_seed_real_listing",
                    payload={
                        "listing_id": listing_id,
                        "owner_agent_id": owner,
                        "category": cat,
                        "price_cents": price,
                        "condition": cond,
                        "used_inventory": item is not None,
                    },
                    result_status="ok",
                    result_payload={"listing_id": listing_id},
                )
        return ids

    # ------------------------------------------------------------------ teardown

    def close(self) -> None:
        self.conn.close()

    # Reopen helper for read-only inspection.
    @staticmethod
    def reopen(db_path: str | Path) -> sqlite3.Connection:
        return connect(db_path)

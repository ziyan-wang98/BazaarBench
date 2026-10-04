#!/usr/bin/env python3
"""v2 augmentation: assign ground-truth quality + acquisition cost to
each agent's persona inventory items WITHOUT re-running gpt-5.4.

What changes:
  - Each persona.inventory_items[*] gets a numeric
    `ground_truth_quality_pct` (0-100) sampled from the cold-start
    `condition` field (and CSV signals where available).
  - Each persona.inventory_items[*] gets `acquisition_cost_cents` —
    a synthetic "what you originally paid" used downstream to compute
    realised profit. Set as a fraction of asking_price_cents.
  - Optionally also resets all existing rollout activity
    (events/listings/threads/...) so the world starts fresh under v2.

Idempotent: re-running on already-augmented persona is a no-op.

Use case: we have a 100-agent cold-start DB built with gpt-5.4 high
that cost ~$50 to build. We don't want to redo the LLM enrichment;
we just need to bolt the quality dimension onto inventory.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

# Quality bands (visible to other agents) and their numeric ranges
# (server-side ground truth). At cold-start we sample ground_truth_pct
# inside the band that matches the item's `condition`. At rollout time
# the seller picks a band that may or may not match the truth.
_CONDITION_TO_PCT_RANGE: dict[str, tuple[int, int]] = {
    "new":      (95, 100),  # brand_new band
    "like_new": (82, 94),   # like_new band
    "good":     (60, 81),   # good band
    "fair":     (35, 59),   # fair band
    "poor":     (10, 34),   # damaged band
    # Anything else falls back to "good" range (defensive).
}


def _quality_pct_for_condition(condition: str, rng: random.Random) -> int:
    lo, hi = _CONDITION_TO_PCT_RANGE.get(
        (condition or "").lower(), _CONDITION_TO_PCT_RANGE["good"]
    )
    return rng.randint(lo, hi)


def _acquisition_cost_for(asking_price_cents: int, rng: random.Random) -> int:
    """Synthesise what the agent paid. 35-72% of asking price gives a
    realistic margin window — cheap stuff bought off Craigslist,
    bargained, etc. Clamped to >= 50 cents.
    """
    if asking_price_cents <= 0:
        return 0
    fraction = rng.uniform(0.35, 0.72)
    return max(50, int(asking_price_cents * fraction))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260503)
    p.add_argument(
        "--reset-rollout",
        action="store_true",
        help="Also wipe all rollout-time activity (events / listings / "
             "threads / offers / etc. with tick > 0) so the world is "
             "ready for a clean v2 rollout.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.db.exists():
        raise SystemExit(f"db not found: {args.db}")
    rng = random.Random(args.seed)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT agent_id, persona_json FROM agents WHERE is_seeded = 0"
        ).fetchall()
        print(f"agents: {len(rows)}")

        agents_changed = 0
        items_added_quality = 0
        items_added_cost = 0
        for r in rows:
            persona = json.loads(r["persona_json"] or "{}")
            inv = persona.get("inventory_items") or []
            changed = False
            for item in inv:
                if not isinstance(item, dict):
                    continue
                if "ground_truth_quality_pct" not in item:
                    pct = _quality_pct_for_condition(
                        str(item.get("condition") or ""), rng,
                    )
                    item["ground_truth_quality_pct"] = pct
                    items_added_quality += 1
                    changed = True
                if "acquisition_cost_cents" not in item:
                    asking = int(item.get("asking_price_cents") or 0)
                    item["acquisition_cost_cents"] = _acquisition_cost_for(
                        asking, rng,
                    )
                    items_added_cost += 1
                    changed = True
            if changed and not args.dry_run:
                agents_changed += 1
                conn.execute(
                    "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                    (json.dumps(persona, sort_keys=True), int(r["agent_id"])),
                )

        print(
            f"items annotated: quality={items_added_quality} "
            f"cost={items_added_cost} (across {agents_changed} agents)"
        )

        if args.reset_rollout:
            print("--- resetting rollout state ---")
            cutoff = 0
            tables = [
                ("events", "tick"),
                ("llm_calls", "tick"),
                ("messages", "tick"),
                ("offers", "tick"),
                ("ratings", "tick"),
                ("blocks", "tick"),
                ("ledger_entries", "tick"),
                ("narrative_memories", "created_tick"),
                ("agent_summary", "tick"),
                ("self_portraits", "tick"),
                ("photos", "created_at_tick"),
                ("threads", "created_at_tick"),
                ("listings", "created_at_tick"),
                ("meetups", "scheduled_tick"),
                ("reports", "tick"),
                ("mental_prices", "tick"),
            ]
            if not args.dry_run:
                conn.execute("PRAGMA foreign_keys = OFF")
                for table, col in tables:
                    try:
                        cur = conn.execute(
                            f"DELETE FROM {table} WHERE {col} > ?",
                            (cutoff,),
                        )
                        if cur.rowcount > 0:
                            print(f"  cleared {cur.rowcount} from {table}")
                    except sqlite3.OperationalError as exc:
                        print(f"  skip {table}: {exc}")
                # Roll back the status of any tick-0 listings that
                # got sold during a previous rollout. Without this
                # the next run sees them as already-sold even though
                # the rest of the rollout state was wiped.
                cur = conn.execute(
                    "UPDATE listings SET status='active', sold_at_tick=NULL "
                    "WHERE status='sold' AND created_at_tick <= 0"
                )
                if cur.rowcount > 0:
                    print(f"  reset {cur.rowcount} cold-start listings sold->active")
                # Strip restock + sold-tick markers from persona
                # inventory so the next run starts from a clean
                # cold-start inventory state. Restock items will
                # respawn naturally; sold_at_tick entries pointing at
                # now-deleted listings would otherwise persist.
                rows = conn.execute(
                    "SELECT agent_id, persona_json FROM agents "
                    "WHERE is_seeded = 0"
                ).fetchall()
                cleaned_agents = 0
                cleaned_items = 0
                cleaned_restock = 0
                for r in rows:
                    aid = int(r["agent_id"])
                    try:
                        persona = json.loads(r["persona_json"] or "{}")
                    except (TypeError, ValueError):
                        continue
                    inv = persona.get("inventory_items") or []
                    if not isinstance(inv, list):
                        continue
                    new_inv: list = []
                    changed = False
                    for it in inv:
                        if not isinstance(it, dict):
                            new_inv.append(it)
                            continue
                        if it.get("source") == "restock":
                            cleaned_restock += 1
                            changed = True
                            continue  # drop entire restocked row
                        if it.get("sold_at_tick") is not None:
                            it.pop("sold_at_tick", None)
                            it.pop("sold_via_listing_id", None)
                            cleaned_items += 1
                            changed = True
                        new_inv.append(it)
                    if changed:
                        persona["inventory_items"] = new_inv
                        conn.execute(
                            "UPDATE agents SET persona_json = ? "
                            "WHERE agent_id = ?",
                            (json.dumps(persona, sort_keys=True), aid),
                        )
                        cleaned_agents += 1
                if cleaned_items or cleaned_restock:
                    print(
                        f"  cleaned persona: dropped_restock={cleaned_restock}, "
                        f"cleared_sold_markers={cleaned_items} "
                        f"(across {cleaned_agents} agents)"
                    )
                conn.execute("PRAGMA foreign_keys = ON")

        if not args.dry_run:
            with conn:
                pass  # commit any pending updates

        if not args.dry_run:
            ic = conn.execute("PRAGMA integrity_check").fetchone()[0]
            fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            max_tick = conn.execute(
                "SELECT MAX(tick) FROM events WHERE tick >= 0"
            ).fetchone()[0]
            print(f"integrity={ic} fk_violations={len(fk)} max_tick={max_tick}")
            conn.execute("VACUUM")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

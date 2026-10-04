#!/usr/bin/env python3
"""v2.23: backfill ``listings.reference_fair_price_cents`` and
``listings.quality_adjusted_fair_price_cents`` from each seller persona's
``inventory_items`` snapshot plus a cross-persona aggregation.

Two reference prices are written per listing, in increasing order of
quality-awareness:

  1. ``reference_fair_price_cents`` — the per-listing eBay scrape's
     ``asking_price_cents`` from the matched inventory item.
     Single-observation prior; ignores quality variation within the
     band.

  2. ``quality_adjusted_fair_price_cents`` — quality-aware ground-truth
     fair price defined as
       ``mean(asking_price) over (brand, model, storage_GB) × (gtq_pct / 100)``
     where ``mean`` is computed across the cold-start inventory and
     ``storage_GB`` is regex-extracted from the title (e.g. ``"256GB"``,
     ``"1TB"``). For non-storage items (books, jewelry, sporting goods),
     storage falls back to ``None`` and the aggregation key collapses to
     ``(brand, model)``. ``gtq_pct`` is the listing's
     ``ground_truth_quality_pct``.

The backfill never overwrites a non-NULL existing value, so it can be
re-run safely. Listings that fail both lookups stay NULL.

Usage:
    python scripts/cold_start/backfill_fair_price.py \
        --db runs/cold_start_real/base100_v2_layer0.db [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from bazaar.core.schema import _migrate_listings_quality

_STORAGE_RE = re.compile(r"(\d+)\s*(GB|TB)\b", re.IGNORECASE)


def _extract_storage(title: str) -> str | None:
    """Return e.g. ``"256GB"`` or ``"1TB"`` from a title; None when the
    title carries no storage marker. Case-insensitive; takes the first
    match so titles like "256GB - Space Gray (with 64GB SD)" pick the
    primary storage spec.
    """
    if not title:
        return None
    m = _STORAGE_RE.search(title)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2).upper()}"


def _resolve_inventory_match(
    inventory: list[dict] | None,
    title: str,
    category: str,
    *,
    in_cat_threshold: float = 0.45,
    fallback_threshold: float = 0.85,
) -> dict | None:
    """Mirror the create_listing inventory matcher; return the matched
    inventory dict or None.
    """
    if not inventory or not title:
        return None
    needle = title.lower().strip()
    best_in_cat: tuple[float, dict | None] = (0.0, None)
    best_global: tuple[float, dict | None] = (0.0, None)
    cat_lc = (category or "").lower()
    cat_root = cat_lc.split("-")[0] if cat_lc else ""
    for inv in inventory:
        if not isinstance(inv, dict):
            continue
        inv_title = str(inv.get("title") or "").lower().strip()
        if not inv_title:
            continue
        ratio = SequenceMatcher(None, inv_title, needle).ratio()
        if ratio > best_global[0]:
            best_global = (ratio, inv)
        inv_cat = str(inv.get("category") or "").lower()
        if cat_lc and (inv_cat == cat_lc or (cat_root and inv_cat.startswith(cat_root))):
            if ratio > best_in_cat[0]:
                best_in_cat = (ratio, inv)
    if best_in_cat[0] >= in_cat_threshold:
        return best_in_cat[1]
    if best_global[0] >= fallback_threshold:
        return best_global[1]
    return None


def _build_brand_model_storage_aggregation(
    conn: sqlite3.Connection,
) -> dict[tuple[str, str, str | None], tuple[int, int]]:
    """Walk all non-seeded personas, collect every inventory item, and
    aggregate ``mean(asking_price_cents)`` and sample count by
    ``(brand, model_name, storage)`` where storage is regex-extracted
    from the inventory item's title.

    Returns a dict keyed by the tuple, value ``(mean_cents, n)``. Keys
    use case-insensitive brand/model strings (lowercased) so lookups
    don't depend on case-matching.
    """
    rows = conn.execute(
        "SELECT persona_json FROM agents WHERE is_seeded = 0",
    ).fetchall()
    bucket: dict[tuple[str, str, str | None], list[int]] = defaultdict(list)
    for (pj,) in rows:
        if not pj:
            continue
        try:
            persona = json.loads(pj)
        except Exception:
            continue
        items = persona.get("inventory_items")
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            brand = (str(it.get("brand") or "").strip().lower())
            model = (str(it.get("model_name") or "").strip().lower())
            price = it.get("asking_price_cents")
            if not brand or not model or not isinstance(price, (int, float)):
                continue
            storage = _extract_storage(str(it.get("title") or ""))
            bucket[(brand, model, storage)].append(int(price))
    out: dict[tuple[str, str, str | None], tuple[int, int]] = {}
    for key, prices in bucket.items():
        if not prices:
            continue
        mean = sum(prices) // len(prices)
        out[key] = (mean, len(prices))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"db not found: {args.db}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    _migrate_listings_quality(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
    for col in ("reference_fair_price_cents", "quality_adjusted_fair_price_cents"):
        if col not in cols:
            raise SystemExit(f"{col} column missing after migrate")

    persona_cache: dict[int, list[dict]] = {}

    def _persona_inventory(agent_id: int) -> list[dict]:
        if agent_id in persona_cache:
            return persona_cache[agent_id]
        row = conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        inv: list[dict] = []
        if row is not None and row[0]:
            try:
                p = json.loads(row[0])
                cand = p.get("inventory_items")
                if isinstance(cand, list):
                    inv = cand
            except Exception:
                inv = []
        persona_cache[agent_id] = inv
        return inv

    aggregation = _build_brand_model_storage_aggregation(conn)
    print(
        f"aggregation: {len(aggregation)} distinct (brand, model, storage) "
        f"keys covering {sum(n for _, n in aggregation.values())} inventory items"
    )

    listings = conn.execute(
        """
        SELECT listing_id, owner_agent_id, title, category, price_cents,
               ground_truth_quality_pct,
               reference_fair_price_cents,
               quality_adjusted_fair_price_cents
        FROM listings
        WHERE owner_agent_id IS NOT NULL
        """,
    ).fetchall()

    matched_per_listing = 0
    matched_quality_adj = 0
    quality_adj_singletons = 0
    unmatched = 0
    for r in listings:
        ref_existing = r["reference_fair_price_cents"]
        qa_existing = r["quality_adjusted_fair_price_cents"]
        if ref_existing is not None and qa_existing is not None:
            continue
        inventory = _persona_inventory(int(r["owner_agent_id"]))
        match = _resolve_inventory_match(inventory, r["title"], r["category"])
        if match is None:
            unmatched += 1
            continue

        new_ref = ref_existing
        if new_ref is None:
            cand = match.get("asking_price_cents")
            if isinstance(cand, (int, float)):
                new_ref = int(cand)
                matched_per_listing += 1

        new_qa = qa_existing
        gtq = r["ground_truth_quality_pct"]
        if new_qa is None and isinstance(gtq, (int, float)) and 0 <= gtq <= 100:
            brand = str(match.get("brand") or "").strip().lower()
            model = str(match.get("model_name") or "").strip().lower()
            storage = _extract_storage(str(match.get("title") or ""))
            agg_key = (brand, model, storage)
            agg = aggregation.get(agg_key)
            if agg is None and storage is not None:
                # Fall back to (brand, model) without storage.
                agg = aggregation.get((brand, model, None))
            if agg is None:
                # Last fallback: scan all storages of the same (brand, model).
                cands = [
                    (mean, n) for (b, m, _), (mean, n) in aggregation.items()
                    if b == brand and m == model
                ]
                if cands:
                    total = sum(mean * n for mean, n in cands)
                    n_total = sum(n for _, n in cands)
                    agg = (total // n_total, n_total)
            if agg is not None and brand and model:
                mean_cents, n = agg
                new_qa = int(round(mean_cents * (float(gtq) / 100.0)))
                matched_quality_adj += 1
                if n == 1:
                    quality_adj_singletons += 1

        if not args.dry_run:
            if new_ref != ref_existing or new_qa != qa_existing:
                conn.execute(
                    """
                    UPDATE listings
                    SET reference_fair_price_cents = ?,
                        quality_adjusted_fair_price_cents = ?
                    WHERE listing_id = ?
                    """,
                    (new_ref, new_qa, int(r["listing_id"])),
                )

    if not args.dry_run:
        conn.commit()
    print(
        f"backfill {'(dry-run) ' if args.dry_run else ''}db={args.db}\n"
        f"  total_listings={len(listings)}  matched_eBay_ref={matched_per_listing}"
        f"  matched_quality_adj={matched_quality_adj}"
        f"  (of which singleton aggregations={quality_adj_singletons})"
        f"  unmatched={unmatched}"
    )

    summary = conn.execute(
        """
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN reference_fair_price_cents IS NOT NULL THEN 1 ELSE 0 END)
            AS with_ebay_ref,
          SUM(CASE WHEN quality_adjusted_fair_price_cents IS NOT NULL THEN 1 ELSE 0 END)
            AS with_quality_ref,
          SUM(CASE WHEN quality_adjusted_fair_price_cents IS NOT NULL
                    AND price_cents > 1.30 * quality_adjusted_fair_price_cents
                    THEN 1 ELSE 0 END) AS overpriced_30,
          SUM(CASE WHEN quality_adjusted_fair_price_cents IS NOT NULL
                    AND price_cents < 0.70 * quality_adjusted_fair_price_cents
                    THEN 1 ELSE 0 END) AS underpriced_30
        FROM listings
        WHERE owner_agent_id IS NOT NULL AND created_at_tick > 0
        """,
    ).fetchone()
    if summary is not None:
        total, w_ebay, w_qual, over, under = summary
        cov_e = 0.0 if not total else 100.0 * (w_ebay or 0) / total
        cov_q = 0.0 if not total else 100.0 * (w_qual or 0) / total
        print(
            f"rollout listings: total={total}  "
            f"with_ebay_ref={w_ebay} ({cov_e:.1f}%)  "
            f"with_quality_adj_ref={w_qual} ({cov_q:.1f}%)\n"
            f"  vs quality_adj reference: overpriced(>+30%)={over}  "
            f"underpriced(<-30%)={under}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export marketplace-economics results for the paper-primary analysis.

The exporter reads the frozen SQLite rollouts without modifying them.  For a
completed target-category trade, the buyer's recorded maximum price and the
item's simulator acquisition cost define buyer surplus, seller profit, and
realized gains from trade.  The accepted offer attached to the completed
thread is the transaction price.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bazaar.analysis_v2.contract import LinkConfidence
from bazaar.analysis_v2.inventory import (
    InventoryReplay,
    InventoryUnit,
    replay_inventory,
)
from bazaar.analysis_v2.transactions import CompletedTransaction, replay_transactions

PRIMARY_MODELS = (
    "gpt55",
    "gpt54mini",
    "deepseekv4pro",
    "gptoss120b",
    "gpt54",
)
_STORAGE_RE = re.compile(r"(?<!\d)(\d+)\s*(GB|TB)\b", re.IGNORECASE)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _open_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _paper_primary(cell: dict[str, Any]) -> bool:
    if cell.get("duplicate_of") is not None:
        return False
    if cell.get("is_starting_market"):
        return True
    return (
        cell.get("regime") in {"L1", "L2", "L3"}
        and cell.get("treatment_model_key") in PRIMARY_MODELS
    )


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return float(statistics.median(items)) if items else None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return float(statistics.mean(items)) if items else None


def _sum_or_none(values: Iterable[float]) -> float | None:
    items = list(values)
    return float(sum(items)) if items else None


def _share(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _effective_categories(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return 1.0 / sum((count / total) ** 2 for count in counts.values())


def _top_three_share(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return 100.0 * sum(sorted(counts.values(), reverse=True)[:3]) / total


def _top_level_category(category: str) -> str:
    """Collapse labels whose simulator taxonomy has an explicit parent."""

    for parent in ("electronics", "collectibles", "vehicles"):
        if category.startswith(f"{parent}-"):
            return parent
    return category


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _persona_constraints(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in conn.execute("SELECT agent_id, persona_json FROM agents"):
        persona = _json_object(row["persona_json"])
        goals = persona.get("goals") if isinstance(persona.get("goals"), dict) else {}
        buyer = goals.get("buyer") if isinstance(goals.get("buyer"), dict) else {}
        seller = goals.get("seller") if isinstance(goals.get("seller"), dict) else {}
        out[int(row["agent_id"])] = {
            "buyer_category": str(buyer.get("want_category") or ""),
            "buyer_max_price_cents": _optional_int(buyer.get("max_price_cents")),
            "seller_min_price_fraction": _optional_float(
                seller.get("min_price_fraction")
            ),
        }
    return out


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _listing_rows(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    return {
        int(row["listing_id"]): dict(row)
        for row in conn.execute(
            "SELECT listing_id, owner_agent_id, category, title, price_cents "
            "FROM listings"
        )
    }


def _price_history(
    conn: sqlite3.Connection,
) -> dict[int, list[tuple[int, int, int]]]:
    history: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT event_id, tick, action_type, payload, result_payload
        FROM events
        WHERE result_status = 'ok'
          AND action_type IN ('create_listing', 'edit_listing')
        ORDER BY tick, event_id
        """
    ):
        payload = _json_object(row["payload"])
        result = _json_object(row["result_payload"])
        listing_id = _optional_int(result.get("listing_id"))
        if listing_id is None:
            listing_id = _optional_int(payload.get("listing_id"))
        price = _optional_int(payload.get("price_cents"))
        if listing_id is not None and price is not None:
            history[listing_id].append((int(row["tick"]), int(row["event_id"]), price))
    return history


def _offer_history(conn: sqlite3.Connection) -> dict[int, list[dict[str, int]]]:
    history: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT offer_id, thread_id, proposer_id, round, price_cents, tick
        FROM offers
        ORDER BY tick, offer_id
        """
    ):
        history[int(row["thread_id"])].append(
            {
                "offer_id": int(row["offer_id"]),
                "proposer_id": int(row["proposer_id"]),
                "round": int(row["round"]),
                "price_cents": int(row["price_cents"]),
                "tick": int(row["tick"]),
            }
        )
    return history


def _accept_action_ticks(conn: sqlite3.Connection) -> dict[int, int]:
    ticks: dict[int, int] = {}
    for row in conn.execute(
        """
        SELECT event_id, tick, payload, result_payload
        FROM events
        WHERE action_type = 'accept_offer' AND result_status = 'ok'
        ORDER BY tick, event_id
        """
    ):
        payload = _json_object(row["payload"])
        result = _json_object(row["result_payload"])
        offer_id = _optional_int(result.get("offer_id"))
        if offer_id is None:
            offer_id = _optional_int(payload.get("offer_id"))
        if offer_id is not None:
            ticks.setdefault(offer_id, int(row["tick"]))
    return ticks


def _thread_parties(conn: sqlite3.Connection) -> dict[int, tuple[int, int | None]]:
    return {
        int(row["thread_id"]): (
            int(row["buyer_agent_id"]),
            _optional_int(row["seller_agent_id"]),
        )
        for row in conn.execute(
            "SELECT thread_id, buyer_agent_id, seller_agent_id FROM threads"
        )
    }


def _asking_at(
    history: dict[int, list[tuple[int, int, int]]],
    *,
    listing_id: int,
    tick: int | None,
    fallback: int | None,
) -> int | None:
    if tick is None:
        return fallback
    values = [entry for entry in history.get(listing_id, ()) if entry[0] <= tick]
    return max(values)[2] if values else fallback


def _completed_scope(
    transactions: Iterable[CompletedTransaction],
    *,
    regime: str,
    view: str,
) -> list[CompletedTransaction]:
    values = list(transactions)
    if regime == "L0":
        return values
    if view == "buyer":
        return [item for item in values if item.buyer_treated]
    if view == "seller":
        return [item for item in values if item.seller_treated]
    if view == "deal":
        return [item for item in values if item.buyer_treated or item.seller_treated]
    raise ValueError(f"unknown view: {view}")


def _accepted_in_window(
    transactions: Iterable[CompletedTransaction],
    *,
    start_tick: int,
    end_tick: int,
) -> list[CompletedTransaction]:
    """Keep completed deals whose final offer was accepted in the window.

    A continuation can complete a deal whose final offer predates the fork. Such
    a deal belongs in the completion denominator but not in the price analysis.
    """

    return [
        item
        for item in transactions
        if item.commit_tick is not None and start_tick < item.commit_tick <= end_tick
    ]


def _cost_with_provenance(
    transaction: CompletedTransaction,
    inventory: InventoryReplay,
) -> tuple[int | None, str | None]:
    if transaction.link_confidence not in {
        LinkConfidence.NATIVE_EXACT,
        LinkConfidence.REPLAY_HIGH_CONFIDENCE,
    }:
        return None, None
    unit = inventory.units_by_id.get(transaction.inventory_unit_id or "")
    if unit is None:
        return None, None
    if unit.lineage == "bought":
        prior_price = _optional_int(unit.raw.get("bought_price_cents"))
        return (
            (prior_price, "prior_simulator_purchase")
            if prior_price is not None
            else (None, None)
        )
    if unit.acquisition_cost_cents is None:
        return None, None
    provenance = "seed_synthetic_cost" if unit.lineage == "seed" else "restock_synthetic_cost"
    return unit.acquisition_cost_cents, provenance


def _storage(title: str) -> str | None:
    match = _STORAGE_RE.search(title or "")
    return f"{match.group(1)}{match.group(2).upper()}" if match else None


def _explicit_product_family(
    unit: InventoryUnit | None,
    inventory: InventoryReplay,
    *,
    seen: set[str] | None = None,
) -> tuple[str, str, str | None] | None:
    if unit is None:
        return None
    visited = set() if seen is None else seen
    if unit.sim_unit_id in visited:
        return None
    visited.add(unit.sim_unit_id)
    brand = str(unit.raw.get("brand") or "").strip()
    model = str(unit.raw.get("model_name") or "").strip()
    if brand and model:
        return brand, model, _storage(unit.title)
    source_listing = _optional_int(unit.raw.get("bought_from_listing_id"))
    if source_listing is None:
        return None
    link = inventory.links_by_listing.get(source_listing)
    if link is None or link.link_confidence not in {
        LinkConfidence.NATIVE_EXACT,
        LinkConfidence.REPLAY_HIGH_CONFIDENCE,
    }:
        return None
    prior = inventory.units_by_id.get(link.create_backing_id or "") if link else None
    return _explicit_product_family(prior, inventory, seen=visited)


def _product_family_label(brand: str, model: str, storage: str | None) -> str:
    parts = [model] if model.lower().startswith(brand.lower()) else [brand, model]
    if storage and storage.lower() not in model.lower():
        parts.append(storage)
    return " ".join(parts)


def _new_listing_categories(
    conn: sqlite3.Connection,
    *,
    start_tick: int,
    end_tick: int,
    treated: set[int],
    regime: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in conn.execute(
        """
        SELECT tick, agent_id, payload
        FROM events
        WHERE result_status = 'ok' AND action_type = 'create_listing'
          AND tick > ? AND tick <= ?
        """,
        (start_tick, end_tick),
    ):
        if regime != "L0" and int(row["agent_id"]) not in treated:
            continue
        category = str(_json_object(row["payload"]).get("category") or "").strip()
        if category:
            counts[category] += 1
    return counts


def _mental_price_count(
    conn: sqlite3.Connection, *, start_tick: int, end_tick: int
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM mental_prices WHERE tick > ? AND tick <= ?",
        (start_tick, end_tick),
    ).fetchone()
    return int(row[0])


def _base_fields(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell["cell_id"],
        "base_model": cell.get("base_model_key"),
        "test_model": cell.get("treatment_model_key"),
        "regime": cell["regime"],
        "start_tick_exclusive": cell["start_tick_exclusive"],
        "end_tick_inclusive": cell["end_tick_inclusive"],
    }


def _analyse_cell(
    cell: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with _open_read_only(Path(cell["db_path"])) as conn:
        start = int(cell["start_tick_exclusive"])
        end = int(cell["end_tick_inclusive"])
        treated = {int(value) for value in cell.get("treated_agent_ids") or ()}
        inventory = replay_inventory(conn)
        replay = replay_transactions(
            conn,
            start_tick_exclusive=start,
            end_tick_inclusive=end,
            treated_agent_ids=treated,
            inventory=inventory,
        )
        personas = _persona_constraints(conn)
        listings = _listing_rows(conn)
        price_history = _price_history(conn)
        offer_history = _offer_history(conn)
        accept_action_ticks = _accept_action_ticks(conn)
        thread_parties = _thread_parties(conn)
        buyer_all = _completed_scope(
            replay.completed, regime=cell["regime"], view="buyer"
        )
        seller_all = _completed_scope(
            replay.completed, regime=cell["regime"], view="seller"
        )
        deals_all = _completed_scope(
            replay.completed, regime=cell["regime"], view="deal"
        )
        buyer = _accepted_in_window(
            buyer_all, start_tick=start, end_tick=end
        )
        seller = _accepted_in_window(
            seller_all, start_tick=start, end_tick=end
        )
        deals = _accepted_in_window(
            deals_all, start_tick=start, end_tick=end
        )

        buyer_matched: list[float] = []
        buyer_surpluses: list[float] = []
        buyer_overshoots: list[float] = []
        for item in buyer:
            listing = listings.get(item.listing_id, {})
            goal = personas.get(item.buyer_agent_id, {})
            ceiling = goal.get("buyer_max_price_cents")
            category = str(listing.get("category") or "")
            if (
                item.settled_price_cents is not None
                and ceiling is not None
                and category == goal.get("buyer_category")
            ):
                gap = item.settled_price_cents - int(ceiling)
                buyer_matched.append(gap)
                buyer_surpluses.append(-float(gap))
                if gap > 0:
                    buyer_overshoots.append(gap)

        sale_to_ask: list[float] = []
        floor_gaps: list[float] = []
        margins: list[float] = []
        margin_rates: list[float] = []
        cost_provenance: Counter[str] = Counter()
        margins_by_provenance: dict[str, list[float]] = defaultdict(list)
        margin_rates_by_provenance: dict[str, list[float]] = defaultdict(list)
        profit_rois: list[float] = []
        for item in seller:
            settled = item.settled_price_cents
            listing = listings.get(item.listing_id, {})
            asking = _asking_at(
                price_history,
                listing_id=item.listing_id,
                tick=item.commit_tick,
                fallback=_optional_int(listing.get("price_cents")),
            )
            if settled is not None and asking is not None and asking > 0:
                sale_to_ask.append(100.0 * (settled / asking - 1.0))
                fraction = personas.get(item.seller_agent_id or -1, {}).get(
                    "seller_min_price_fraction"
                )
                if fraction is not None:
                    floor_gaps.append(settled - float(fraction) * asking)
            cost, provenance = _cost_with_provenance(item, inventory)
            if settled is not None and cost is not None:
                margin = settled - cost
                margins.append(float(margin))
                if cost > 0:
                    profit_rois.append(100.0 * margin / cost)
                if settled > 0:
                    margin_rates.append(100.0 * margin / settled)
                if provenance:
                    cost_provenance[provenance] += 1
                    margins_by_provenance[provenance].append(float(margin))
                    if settled > 0:
                        margin_rates_by_provenance[provenance].append(
                            100.0 * margin / settled
                        )

        deal_categories = Counter(
            str(listings.get(item.listing_id, {}).get("category") or "unknown")
            for item in deals
        )
        deal_top_level_categories = Counter(
            _top_level_category(category) for category in deal_categories.elements()
        )
        new_listing_categories = _new_listing_categories(
            conn,
            start_tick=start,
            end_tick=end,
            treated=treated,
            regime=cell["regime"],
        )
        new_listing_top_level_categories = Counter(
            _top_level_category(category)
            for category in new_listing_categories.elements()
        )
        prior_margins = margins_by_provenance["prior_simulator_purchase"]
        prior_margin_rates = margin_rates_by_provenance[
            "prior_simulator_purchase"
        ]
        synthetic_margins = (
            margins_by_provenance["seed_synthetic_cost"]
            + margins_by_provenance["restock_synthetic_cost"]
        )

        deal_prices: list[float] = []
        deal_buyer_surpluses: list[float] = []
        deal_seller_profits: list[float] = []
        welfare_sample_buyer_surpluses: list[float] = []
        welfare_sample_seller_profits: list[float] = []
        deal_welfare: list[float] = []
        welfare_by_provenance: dict[str, list[float]] = defaultdict(list)
        offer_counts: list[float] = []
        offer_rounds: list[float] = []
        agreement_delays: list[float] = []
        offer_path_eligible = 0
        offer_path_left_censored = 0
        for item in deals:
            settled = item.settled_price_cents
            if settled is None:
                continue
            deal_prices.append(float(settled))
            listing = listings.get(item.listing_id, {})
            goal = personas.get(item.buyer_agent_id, {})
            ceiling = goal.get("buyer_max_price_cents")
            category = str(listing.get("category") or "")
            buyer_surplus: float | None = None
            if ceiling is not None and category == goal.get("buyer_category"):
                buyer_surplus = float(int(ceiling) - settled)
                deal_buyer_surpluses.append(buyer_surplus)
            cost, provenance = _cost_with_provenance(item, inventory)
            seller_profit: float | None = None
            if cost is not None:
                seller_profit = float(settled - cost)
                deal_seller_profits.append(seller_profit)
            if buyer_surplus is not None and seller_profit is not None:
                welfare = buyer_surplus + seller_profit
                welfare_sample_buyer_surpluses.append(buyer_surplus)
                welfare_sample_seller_profits.append(seller_profit)
                deal_welfare.append(welfare)
                if provenance:
                    welfare_by_provenance[provenance].append(welfare)
            accepted_id = item.accepted_offer_id
            if accepted_id is None:
                continue
            accepted_tick = accept_action_ticks.get(accepted_id, item.commit_tick)
            if accepted_tick is None:
                continue
            history_for_thread = offer_history.get(item.thread_id, ())
            path = [
                offer
                for offer in history_for_thread
                if offer["tick"] <= accepted_tick
            ]
            if not path or not any(offer["offer_id"] == accepted_id for offer in path):
                continue
            offer_path_eligible += 1
            if path[0]["tick"] <= start:
                offer_path_left_censored += 1
                continue
            offer_counts.append(float(len(path)))
            offer_rounds.append(float(max(offer["round"] for offer in path)))
            agreement_delays.append(float(accepted_tick - path[0]["tick"]))

        offer_threads_in_window: set[int] = set()
        for thread_id, offers in offer_history.items():
            if not any(start < offer["tick"] <= end for offer in offers):
                continue
            if cell["regime"] != "L0":
                buyer_id, seller_id = thread_parties.get(thread_id, (-1, None))
                if buyer_id not in treated and seller_id not in treated:
                    continue
            offer_threads_in_window.add(thread_id)
        completed_offer_threads = {
            item.thread_id for item in deals if item.thread_id in offer_threads_in_window
        }

        base = _base_fields(cell)
        metrics = {
            **base,
            "buyer_completed": len(buyer_all),
            "buyer_completed_with_acceptance_in_window": len(buyer),
            "buyer_category_matched_budget_n": len(buyer_matched),
            "buyer_budget_overshoot_n": len(buyer_overshoots),
            "buyer_budget_overshoot_share_pct": _share(
                len(buyer_overshoots), len(buyer_matched)
            ),
            "buyer_median_price_minus_budget_cents": _median(buyer_matched),
            "buyer_surplus_observed_n": len(buyer_surpluses),
            "buyer_nonnegative_surplus_n": sum(
                value >= 0 for value in buyer_surpluses
            ),
            "buyer_nonnegative_surplus_share_pct": _share(
                sum(value >= 0 for value in buyer_surpluses), len(buyer_surpluses)
            ),
            "buyer_median_surplus_cents": _median(buyer_surpluses),
            "buyer_mean_surplus_cents": _mean(buyer_surpluses),
            "buyer_total_surplus_cents": _sum_or_none(buyer_surpluses),
            "seller_completed": len(seller_all),
            "seller_completed_with_acceptance_in_window": len(seller),
            "seller_sale_to_ask_n": len(sale_to_ask),
            "seller_median_sale_to_ask_gap_pct": _median(sale_to_ask),
            "seller_floor_observed_n": len(floor_gaps),
            "seller_floor_breach_n": sum(value < 0 for value in floor_gaps),
            "seller_floor_breach_share_pct": _share(
                sum(value < 0 for value in floor_gaps), len(floor_gaps)
            ),
            "seller_median_price_minus_floor_cents": _median(floor_gaps),
            "seller_profit_observed_n": len(margins),
            "seller_profit_coverage_pct": _share(len(margins), len(seller)),
            "seller_roi_observed_n": len(profit_rois),
            "seller_negative_profit_n": sum(value < 0 for value in margins),
            "seller_negative_profit_share_pct": _share(
                sum(value < 0 for value in margins), len(margins)
            ),
            "seller_median_profit_cents": _median(margins),
            "seller_mean_profit_cents": _mean(margins),
            "seller_total_profit_cents": _sum_or_none(margins),
            "seller_median_roi_pct": _median(profit_rois),
            "seller_cost_observed_n": len(margins),
            "seller_all_provenance_negative_accounting_margin_n": sum(
                value < 0 for value in margins
            ),
            "seller_all_provenance_negative_accounting_margin_share_pct": _share(
                sum(value < 0 for value in margins), len(margins)
            ),
            "seller_all_provenance_median_accounting_margin_cents": _median(
                margins
            ),
            "seller_all_provenance_median_accounting_margin_rate_pct": _median(
                margin_rates
            ),
            "seller_prior_purchase_cost_n": len(prior_margins),
            "seller_prior_purchase_negative_margin_n": sum(
                value < 0 for value in prior_margins
            ),
            "seller_prior_purchase_negative_margin_share_pct": _share(
                sum(value < 0 for value in prior_margins), len(prior_margins)
            ),
            "seller_prior_purchase_median_accounting_margin_rate_pct": _median(
                prior_margin_rates
            ),
            "seller_synthetic_cost_n": len(synthetic_margins),
            "seller_synthetic_cost_negative_margin_n": sum(
                value < 0 for value in synthetic_margins
            ),
            "seller_synthetic_cost_negative_margin_share_pct": _share(
                sum(value < 0 for value in synthetic_margins),
                len(synthetic_margins),
            ),
            "seller_cost_provenance": json.dumps(cost_provenance, sort_keys=True),
            "deal_completed": len(deals_all),
            "deal_completed_with_acceptance_in_window": len(deals),
            "deal_price_observed_n": len(deal_prices),
            "deal_median_price_cents": _median(deal_prices),
            "deal_gmv_cents": sum(deal_prices),
            "deal_buyer_surplus_observed_n": len(deal_buyer_surpluses),
            "deal_buyer_surplus_coverage_pct": _share(
                len(deal_buyer_surpluses), len(deals)
            ),
            "deal_seller_profit_observed_n": len(deal_seller_profits),
            "deal_seller_profit_coverage_pct": _share(
                len(deal_seller_profits), len(deals)
            ),
            "deal_welfare_observed_n": len(deal_welfare),
            "deal_welfare_coverage_pct": _share(len(deal_welfare), len(deals)),
            "deal_median_buyer_surplus_cents": _median(deal_buyer_surpluses),
            "deal_mean_buyer_surplus_cents": _mean(deal_buyer_surpluses),
            "deal_total_buyer_surplus_cents": _sum_or_none(deal_buyer_surpluses),
            "deal_median_seller_profit_cents": _median(deal_seller_profits),
            "deal_mean_seller_profit_cents": _mean(deal_seller_profits),
            "deal_total_seller_profit_cents": _sum_or_none(deal_seller_profits),
            "deal_welfare_sample_total_buyer_surplus_cents": sum(
                welfare_sample_buyer_surpluses
            ) if welfare_sample_buyer_surpluses else None,
            "deal_welfare_sample_total_seller_profit_cents": sum(
                welfare_sample_seller_profits
            ) if welfare_sample_seller_profits else None,
            "deal_median_welfare_cents": _median(deal_welfare),
            "deal_mean_welfare_cents": _mean(deal_welfare),
            "deal_total_welfare_cents": _sum_or_none(deal_welfare),
            "deal_nonnegative_welfare_n": sum(
                value >= 0 for value in deal_welfare
            ),
            "deal_nonnegative_welfare_share_pct": _share(
                sum(value >= 0 for value in deal_welfare), len(deal_welfare)
            ),
            "deal_prior_purchase_welfare_n": len(
                welfare_by_provenance["prior_simulator_purchase"]
            ),
            "deal_prior_purchase_nonnegative_welfare_share_pct": _share(
                sum(
                    value >= 0
                    for value in welfare_by_provenance["prior_simulator_purchase"]
                ),
                len(welfare_by_provenance["prior_simulator_purchase"]),
            ),
            "deal_synthetic_cost_welfare_n": len(
                welfare_by_provenance["seed_synthetic_cost"]
                + welfare_by_provenance["restock_synthetic_cost"]
            ),
            "deal_synthetic_cost_nonnegative_welfare_share_pct": _share(
                sum(
                    value >= 0
                    for value in (
                        welfare_by_provenance["seed_synthetic_cost"]
                        + welfare_by_provenance["restock_synthetic_cost"]
                    )
                ),
                len(
                    welfare_by_provenance["seed_synthetic_cost"]
                    + welfare_by_provenance["restock_synthetic_cost"]
                ),
            ),
            "deal_offer_threads_in_window": len(offer_threads_in_window),
            "deal_offer_thread_to_within_window_completion_n": len(
                completed_offer_threads
            ),
            "deal_offer_thread_to_within_window_completion_share_pct": _share(
                len(completed_offer_threads), len(offer_threads_in_window)
            ),
            "deal_offer_path_eligible_n": offer_path_eligible,
            "deal_offer_path_left_censored_n": offer_path_left_censored,
            "deal_offer_path_fully_observed_n": len(offer_counts),
            "deal_offer_path_fully_observed_share_pct": _share(
                len(offer_counts), offer_path_eligible
            ),
            "deal_median_offers_to_accept": _median(offer_counts),
            "deal_median_max_offer_round_before_accept": _median(offer_rounds),
            "deal_median_first_offer_to_accept_ticks": _median(agreement_delays),
            "deal_distinct_categories": len(deal_categories),
            "deal_effective_categories": _effective_categories(deal_categories),
            "deal_top3_category_share_pct": _top_three_share(deal_categories),
            "deal_top_level_distinct_categories": len(deal_top_level_categories),
            "deal_top_level_effective_categories": _effective_categories(
                deal_top_level_categories
            ),
            "deal_top_level_top3_category_share_pct": _top_three_share(
                deal_top_level_categories
            ),
            "new_listings": sum(new_listing_categories.values()),
            "new_listing_distinct_categories": len(new_listing_categories),
            "new_listing_effective_categories": _effective_categories(
                new_listing_categories
            ),
            "new_listing_top3_category_share_pct": _top_three_share(
                new_listing_categories
            ),
            "new_listing_top_level_distinct_categories": len(
                new_listing_top_level_categories
            ),
            "new_listing_top_level_effective_categories": _effective_categories(
                new_listing_top_level_categories
            ),
            "new_listing_top_level_top3_category_share_pct": _top_three_share(
                new_listing_top_level_categories
            ),
            "mental_price_rows_in_window": _mental_price_count(
                conn, start_tick=start, end_tick=end
            ),
        }

        category_rows = [
            {
                **base,
                "source": source,
                "category": category,
                "count": count,
                "share_pct": _share(count, sum(counts.values())),
            }
            for source, counts in (
                ("completed_deals", deal_categories),
                ("completed_deals_top_level", deal_top_level_categories),
                ("new_listings", new_listing_categories),
                ("new_listings_top_level", new_listing_top_level_categories),
            )
            for category, count in sorted(counts.items())
        ]

        product_rows: list[dict[str, Any]] = []
        for item in deals:
            if item.link_confidence not in {
                LinkConfidence.NATIVE_EXACT,
                LinkConfidence.REPLAY_HIGH_CONFIDENCE,
            }:
                continue
            unit = inventory.units_by_id.get(item.inventory_unit_id or "")
            family = _explicit_product_family(unit, inventory)
            if family is None or item.settled_price_cents is None:
                continue
            brand, model, storage = family
            listing = listings.get(item.listing_id, {})
            asking = _asking_at(
                price_history,
                listing_id=item.listing_id,
                tick=item.commit_tick,
                fallback=_optional_int(listing.get("price_cents")),
            )
            product_rows.append(
                {
                    **base,
                    "thread_id": item.thread_id,
                    "listing_id": item.listing_id,
                    "inventory_unit_id": item.inventory_unit_id,
                    "category": listing.get("category"),
                    "brand": brand,
                    "model": model,
                    "storage": storage,
                    "product_family": _product_family_label(brand, model, storage),
                    "link_confidence": item.link_confidence.value,
                    "settled_price_cents": item.settled_price_cents,
                    "asking_price_at_accept_cents": asking,
                    "ground_truth_quality_pct": item.ground_truth_quality_pct,
                    "buyer_treated": item.buyer_treated,
                    "seller_treated": item.seller_treated,
                }
            )
        metrics["deal_explicit_product_family_n"] = len(product_rows)
        metrics["deal_explicit_product_family_share_pct"] = _share(
            len(product_rows), len(deals)
        )
        return metrics, category_rows, product_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _matched_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "buyer_budget_overshoot_share_pct",
        "buyer_nonnegative_surplus_share_pct",
        "buyer_median_surplus_cents",
        "buyer_mean_surplus_cents",
        "seller_floor_breach_share_pct",
        "seller_profit_coverage_pct",
        "seller_negative_profit_share_pct",
        "seller_median_profit_cents",
        "seller_mean_profit_cents",
        "seller_median_roi_pct",
        "seller_all_provenance_negative_accounting_margin_share_pct",
        "seller_prior_purchase_negative_margin_share_pct",
        "seller_synthetic_cost_negative_margin_share_pct",
        "seller_all_provenance_median_accounting_margin_rate_pct",
        "seller_median_sale_to_ask_gap_pct",
        "deal_gmv_cents",
        "deal_median_price_cents",
        "deal_welfare_coverage_pct",
        "deal_buyer_surplus_coverage_pct",
        "deal_seller_profit_coverage_pct",
        "deal_median_buyer_surplus_cents",
        "deal_median_seller_profit_cents",
        "deal_median_welfare_cents",
        "deal_mean_welfare_cents",
        "deal_total_welfare_cents",
        "deal_nonnegative_welfare_share_pct",
        "deal_prior_purchase_nonnegative_welfare_share_pct",
        "deal_synthetic_cost_nonnegative_welfare_share_pct",
        "deal_offer_thread_to_within_window_completion_share_pct",
        "deal_offer_path_fully_observed_share_pct",
        "deal_median_offers_to_accept",
        "deal_median_max_offer_round_before_accept",
        "deal_median_first_offer_to_accept_ticks",
        "deal_effective_categories",
        "deal_top3_category_share_pct",
        "deal_top_level_effective_categories",
        "deal_top_level_top3_category_share_pct",
        "deal_explicit_product_family_share_pct",
        "new_listing_effective_categories",
        "new_listing_top3_category_share_pct",
    )
    lookup = {
        (row["base_model"], row["test_model"], row["regime"]): row
        for row in rows
        if row["regime"] in {"L1", "L2", "L3"}
    }
    out: list[dict[str, Any]] = []
    for model in PRIMARY_MODELS:
        bases = sorted(
            base
            for base, test, regime in lookup
            if test == model and regime == "L1"
        )
        for metric in metrics:
            for setting in ("L1", "L2-L1", "L3-L1"):
                values: list[float] = []
                for base in bases:
                    l1 = lookup.get((base, model, "L1"), {}).get(metric)
                    if l1 is None:
                        continue
                    if setting == "L1":
                        values.append(float(l1))
                    else:
                        regime = setting.split("-")[0]
                        value = lookup.get((base, model, regime), {}).get(metric)
                        if value is not None:
                            values.append(float(value) - float(l1))
                out.append(
                    {
                        "test_model": model,
                        "metric": metric,
                        "setting": setting,
                        "starting_markets_observed": len(values),
                        "median": _median(values),
                        "minimum": min(values) if values else None,
                        "maximum": max(values) if values else None,
                    }
                )
    return out


def _distribution_row(items: list[dict[str, Any]]) -> dict[str, Any]:
    prices = [float(item["settled_price_cents"]) / 100.0 for item in items]
    median = statistics.median(prices)
    q1 = _quantile(prices, 0.25)
    q3 = _quantile(prices, 0.75)
    return {
        "completed_deals": len(prices),
        "minimum_price": min(prices),
        "q1_price": q1,
        "median_price": median,
        "q3_price": q3,
        "maximum_price": max(prices),
        "iqr_over_median": (q3 - q1) / median if median else None,
    }


def _phone_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["regime"] == "L0" and row["category"] == "electronics-phones":
            grouped[(str(row["base_model"]), str(row["product_family"]))].append(
                row
            )
    by_market: list[dict[str, Any]] = []
    for (base_model, family), items in grouped.items():
        if len(items) < 5:
            continue
        by_market.append(
            {
                "base_model": base_model,
                "product_family": family,
                **_distribution_row(items),
            }
        )
    by_market.sort(
        key=lambda row: (
            str(row["product_family"]),
            str(row["base_model"]),
        )
    )

    markets_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_market:
        markets_by_family[str(row["product_family"])].append(row)
    summary: list[dict[str, Any]] = []
    for family, items in markets_by_family.items():
        median_prices = [float(item["median_price"]) for item in items]
        dispersions = [float(item["iqr_over_median"]) for item in items]
        summary.append(
            {
                "product_family": family,
                "starting_markets_observed": len(items),
                "completed_deals_across_eligible_markets": sum(
                    int(item["completed_deals"]) for item in items
                ),
                "median_of_market_median_prices": _median(median_prices),
                "minimum_market_median_price": min(median_prices),
                "maximum_market_median_price": max(median_prices),
                "median_within_market_iqr_over_median": _median(dispersions),
                "minimum_within_market_iqr_over_median": min(dispersions),
                "maximum_within_market_iqr_over_median": max(dispersions),
            }
        )
    summary.sort(
        key=lambda row: (
            -int(row["starting_markets_observed"]),
            -int(row["completed_deals_across_eligible_markets"]),
            str(row["product_family"]),
        )
    )
    return by_market, summary


def main() -> None:
    args = _args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    cells = [cell for cell in registry["cells"] if _paper_primary(cell)]
    if len(cells) != 48:
        raise SystemExit(f"expected 48 paper-primary cells, found {len(cells)}")
    metrics: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        row, category_rows, product_rows = _analyse_cell(cell)
        metrics.append(row)
        categories.extend(category_rows)
        products.extend(product_rows)
        print(f"[{index:02d}/48] {cell['cell_id']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "price_market_metrics_per_cell.csv", metrics)
    _write_csv(
        args.output_dir / "price_market_metrics_matched_summary.csv",
        _matched_summary(metrics),
    )
    _write_csv(args.output_dir / "price_market_category_rows.csv", categories)
    _write_csv(args.output_dir / "price_market_product_rows.csv", products)
    phone_by_market, phone_summary = _phone_summaries(products)
    _write_csv(
        args.output_dir / "price_market_phone_model_by_market.csv",
        phone_by_market,
    )
    _write_csv(
        args.output_dir / "price_market_phone_model_summary.csv", phone_summary
    )
    summary = {
        "paper_primary_cells": len(metrics),
        "completed_deals_with_price_in_window": sum(
            int(row["deal_price_observed_n"]) for row in metrics
        ),
        "buyer_surplus_observed_deals": sum(
            int(row["deal_buyer_surplus_observed_n"]) for row in metrics
        ),
        "seller_profit_observed_deals": sum(
            int(row["deal_seller_profit_observed_n"]) for row in metrics
        ),
        "welfare_observed_deals": sum(
            int(row["deal_welfare_observed_n"]) for row in metrics
        ),
        "fully_observed_offer_paths": sum(
            int(row["deal_offer_path_fully_observed_n"]) for row in metrics
        ),
        "mental_price_rows_in_formal_windows": sum(
            int(row["mental_price_rows_in_window"]) for row in metrics
        ),
        "completed_deal_rows_with_explicit_product_family": len(products),
        "phone_family_market_pairs_with_at_least_five_l0_deals": len(
            phone_by_market
        ),
        "phone_product_families_with_at_least_five_l0_deals_in_any_market": len(
            phone_summary
        ),
        "phone_product_families_with_at_least_five_l0_deals_in_all_markets": sum(
            int(row["starting_markets_observed"]) == 3 for row in phone_summary
        ),
    }
    (args.output_dir / "price_market_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

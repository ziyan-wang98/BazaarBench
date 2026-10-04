"""Read-only reconstruction of simulator inventory lineage.

The simulator stores inventory as an append-only-ish list inside each final
``agents.persona_json``.  Bought and restocked rows carry birth provenance and
sold rows are tagged rather than deleted, which is enough to reconstruct the
inventory visible to ``create_listing`` at each event tick.  This module never
updates the rollout database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .contract import LinkConfidence

_CATEGORY_MATCH_THRESHOLD = 0.45
_TITLE_FALLBACK_THRESHOLD = 0.85


@dataclass
class InventoryUnit:
    """One stable simulator inventory unit reconstructed from persona JSON."""

    sim_unit_id: str
    lineage: str
    agent_id: int
    ordinal: int
    inventory_position: int
    born_at_tick: int
    title: str
    category: str
    source: str
    bought_from_listing_id: int | None = None
    sold_at_tick: int | None = None
    sold_via_listing_id: int | None = None
    asking_price_cents: int | None = None
    acquisition_cost_cents: int | None = None
    ground_truth_quality_pct: int | None = None
    speculative_origin: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ListingInventoryLink:
    """Create-time and consume-time inventory attribution for one listing."""

    listing_id: int
    owner_agent_id: int | None
    create_tick: int
    create_title: str
    create_category: str
    final_title: str
    final_category: str
    create_backing_id: str | None
    consumed_id: str | None
    link_confidence: LinkConfidence
    match_score: float
    candidate_ids: tuple[str, ...] = ()
    native_is_speculative: bool | None = None
    native_match_confidence: float | None = None
    sold_before_create: bool = False
    edit_drift: bool = False
    speculative_origin_bought_chain: bool = False
    temporal_exclusion_reasons: tuple[str, ...] = ()

    @property
    def inventory_unit_id(self) -> str | None:
        """Best create-time unit for opportunity and overcommitment analysis."""

        return self.create_backing_id


@dataclass
class InventoryReplay:
    units: tuple[InventoryUnit, ...]
    links: tuple[ListingInventoryLink, ...]
    # Both create/relist and sale actions are written to the same append-only
    # ``events`` table, so (tick, event_id) is the authoritative within-tick
    # ordering key.  The field has a default to preserve small hand-built test
    # fixtures that do not need event-order reconstruction.
    sale_event_positions_by_listing: dict[int, tuple[tuple[int, int], ...]] = field(
        default_factory=dict, repr=False
    )

    @property
    def units_by_id(self) -> dict[str, InventoryUnit]:
        return {unit.sim_unit_id: unit for unit in self.units}

    @property
    def links_by_listing(self) -> dict[int, ListingInventoryLink]:
        return {link.listing_id: link for link in self.links}


def unit_consumption_evidence(
    unit: InventoryUnit,
    *,
    tick: int,
    event_id: int | None,
    sale_event_positions_by_listing: dict[int, tuple[tuple[int, int], ...]],
) -> str | None:
    """Return how the log establishes that ``unit`` was already sold.

    Final persona rows record only ``sold_at_tick`` and the carrier listing.
    When sale and action share a tick, their globally ordered ``events`` rows
    resolve the sequence.  If no matching sale event can be recovered, the
    same-tick order is unknown and is conservatively *not* called consumed.
    """

    if unit.sold_at_tick is None:
        return None
    if unit.sold_at_tick < tick:
        return "consumed_on_prior_tick"
    if unit.sold_at_tick > tick or event_id is None:
        return None
    if unit.sold_via_listing_id is None:
        return None
    matching_sale_ids = [
        sale_event_id
        for sale_tick, sale_event_id in sale_event_positions_by_listing.get(
            unit.sold_via_listing_id, ()
        )
        if sale_tick == unit.sold_at_tick
    ]
    # Multiple completed sales can exceptionally share one listing.  Without
    # a unit id in the event payload, a same-tick unit is certainly consumed
    # only when every compatible sale event precedes the action.
    if matching_sale_ids and max(matching_sale_ids) < event_id:
        return "consumed_earlier_same_tick_by_event_id"
    return None


def unit_consumed_before_event(
    unit: InventoryUnit,
    *,
    tick: int,
    event_id: int | None,
    sale_event_positions_by_listing: dict[int, tuple[tuple[int, int], ...]],
) -> bool:
    """Return whether the log establishes that ``unit`` was already sold."""

    return (
        unit_consumption_evidence(
            unit,
            tick=tick,
            event_id=event_id,
            sale_event_positions_by_listing=sale_event_positions_by_listing,
        )
        is not None
    )


def unit_birth_exclusion_reason(
    unit: InventoryUnit,
    *,
    tick: int,
    event_id: int | None,
    sale_event_positions_by_listing: dict[int, tuple[tuple[int, int], ...]],
) -> str | None:
    """Return why an inventory row was not yet owned at an event position."""

    if unit.born_at_tick < tick:
        return None
    if unit.born_at_tick > tick:
        return "future_inventory_birth"
    if unit.lineage == "restock":
        return "same_tick_restock_after_agent_actions"
    if unit.lineage != "bought":
        return None
    if event_id is None or unit.bought_from_listing_id is None:
        return "same_tick_bought_without_prior_completion_evidence"
    matching_birth_ids = [
        birth_event_id
        for birth_tick, birth_event_id in sale_event_positions_by_listing.get(
            unit.bought_from_listing_id, ()
        )
        if birth_tick == unit.born_at_tick
    ]
    if matching_birth_ids and max(matching_birth_ids) < event_id:
        return None
    return "same_tick_bought_without_prior_completion_evidence"


def unit_born_by_event(
    unit: InventoryUnit,
    *,
    tick: int,
    event_id: int | None,
    sale_event_positions_by_listing: dict[int, tuple[tuple[int, int], ...]],
) -> bool:
    """Return whether ``unit`` was already owned at this event position.

    The environment runs all agent actions before its same-tick dynamics, so
    a D_restock row born on tick T cannot back any create action on tick T.
    Bought rows are born at successful completion: a same-tick purchase can
    back a later action only when the source-listing completion event is
    observed earlier in the globally ordered event log.
    """

    return (
        unit_birth_exclusion_reason(
            unit,
            tick=tick,
            event_id=event_id,
            sale_event_positions_by_listing=sale_event_positions_by_listing,
        )
        is None
    )


def _rows(
    conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    cursor = conn.execute(query, params)
    names = [column[0] for column in cursor.description or ()]
    return [dict(zip(names, tuple(row), strict=True)) for row in cursor.fetchall()]


def _json_object(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _category_compatible(listing_category: str, inventory_category: str) -> bool:
    """Mirror ``actions.handlers._category_compatible`` exactly."""

    if not listing_category or not inventory_category:
        return False
    left = listing_category.lower().strip()
    right = inventory_category.lower().strip()
    return left == right or left.startswith(right + "-") or right.startswith(left + "-")


def _birth_tick(item: dict[str, Any], agent_created_tick: int) -> int:
    source = str(item.get("source") or "").lower()
    keys = ("bought_tick", "added_at_tick") if source in {"bought", "restock"} else ()
    for key in (*keys, "created_at_tick"):
        value = _optional_int(item.get(key))
        if value is not None:
            return value
    return agent_created_tick


def _lineage(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "").lower().strip()
    if source == "bought":
        return "bought"
    if source == "restock":
        return "restock"
    return "seed"


def _load_units(conn: sqlite3.Connection) -> list[InventoryUnit]:
    units: list[InventoryUnit] = []
    for agent in _rows(
        conn,
        "SELECT agent_id, created_at_tick, persona_json FROM agents ORDER BY agent_id",
    ):
        agent_id = int(agent["agent_id"])
        created_tick = int(agent.get("created_at_tick") or 0)
        inventory = _json_object(agent.get("persona_json")).get("inventory_items")
        if not isinstance(inventory, list):
            continue
        pending: list[tuple[int, int, dict[str, Any]]] = []
        for position, item in enumerate(inventory):
            if isinstance(item, dict):
                pending.append((_birth_tick(item, created_tick), position, item))

        # Persona inventory is append-only/order-preserving in the simulator.
        # Identity therefore follows the frozen final-array position, while
        # ``born_at_tick`` remains temporal metadata used for create-time
        # eligibility.  The same position order is also how the live matcher
        # deterministically breaks equal-score ties.
        for born_tick, position, item in pending:
            ordinal = position + 1
            lineage = _lineage(item)
            bought_from = _optional_int(item.get("bought_from_listing_id"))
            units.append(
                InventoryUnit(
                    sim_unit_id=f"{lineage}/{agent_id}/{ordinal:04d}",
                    lineage=lineage,
                    agent_id=agent_id,
                    ordinal=ordinal,
                    inventory_position=position,
                    born_at_tick=born_tick,
                    title=str(item.get("title") or ""),
                    category=str(item.get("category") or ""),
                    source=str(item.get("source") or "seed"),
                    bought_from_listing_id=bought_from,
                    sold_at_tick=_optional_int(item.get("sold_at_tick")),
                    sold_via_listing_id=_optional_int(item.get("sold_via_listing_id")),
                    asking_price_cents=_optional_int(item.get("asking_price_cents")),
                    acquisition_cost_cents=_optional_int(item.get("acquisition_cost_cents")),
                    ground_truth_quality_pct=_optional_int(item.get("ground_truth_quality_pct")),
                    raw=dict(item),
                )
            )
    return units


def _load_event_metadata(
    conn: sqlite3.Connection,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    creates: dict[int, dict[str, Any]] = {}
    edits: dict[int, list[dict[str, Any]]] = {}
    for event in _rows(
        conn,
        """
        SELECT event_id, tick, action_type, payload, result_status, result_payload
        FROM events
        WHERE action_type IN ('create_listing', 'edit_listing')
        ORDER BY tick, event_id
        """,
    ):
        if event["result_status"] != "ok":
            continue
        payload = _json_object(event.get("payload"))
        result = _json_object(event.get("result_payload"))
        listing_id = _optional_int(result.get("listing_id"))
        if listing_id is None:
            listing_id = _optional_int(payload.get("listing_id"))
        if listing_id is None:
            continue
        record = {
            "event_id": int(event["event_id"]),
            "tick": int(event["tick"]),
            "payload": payload,
        }
        if event["action_type"] == "create_listing":
            creates.setdefault(listing_id, record)
        else:
            edits.setdefault(listing_id, []).append(record)
    return creates, edits


def _load_sale_event_positions(
    conn: sqlite3.Connection,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Return listing -> successful sale-event ``(tick, event_id)`` values.

    ``mark_sold`` identifies its listing directly.  A completed transaction
    identifies a thread (sometimes only through its meetup), which is joined
    back to the carrier listing.  No identifier from another table is ever
    compared with ``event_id``; those ids are used only for relational joins.
    """

    thread_listing = {
        int(row["thread_id"]): int(row["listing_id"])
        for row in _rows(conn, "SELECT thread_id, listing_id FROM threads")
    }
    meetup_thread = {
        int(row["meetup_id"]): int(row["thread_id"])
        for row in _rows(conn, "SELECT meetup_id, thread_id FROM meetups")
    }
    positions: dict[int, list[tuple[int, int]]] = {}
    for event in _rows(
        conn,
        """
        SELECT event_id, tick, action_type, payload, result_status, result_payload
        FROM events
        WHERE action_type IN ('mark_sold', 'complete_transaction')
        ORDER BY tick, event_id
        """,
    ):
        if event["result_status"] != "ok":
            continue
        payload = _json_object(event.get("payload"))
        result = _json_object(event.get("result_payload"))
        listing_id: int | None = None
        if event["action_type"] == "mark_sold":
            listing_id = _optional_int(result.get("listing_id"))
            if listing_id is None:
                listing_id = _optional_int(payload.get("listing_id"))
        elif result.get("completed") is True:
            listing_id = _optional_int(result.get("listing_id"))
            if listing_id is None:
                listing_id = _optional_int(payload.get("listing_id"))
            thread_id = _optional_int(result.get("thread_id"))
            if thread_id is None:
                thread_id = _optional_int(payload.get("thread_id"))
            if thread_id is None:
                meetup_id = _optional_int(result.get("meetup_id"))
                if meetup_id is None:
                    meetup_id = _optional_int(payload.get("meetup_id"))
                thread_id = meetup_thread.get(meetup_id or -1)
            if listing_id is None and thread_id is not None:
                listing_id = thread_listing.get(thread_id)
        if listing_id is not None:
            positions.setdefault(listing_id, []).append(
                (int(event["tick"]), int(event["event_id"]))
            )
    return {listing_id: tuple(sorted(values)) for listing_id, values in positions.items()}


def _match_create_time_unit(
    units: list[InventoryUnit],
    *,
    title: str,
    category: str,
    preferred_id: str | None = None,
) -> tuple[InventoryUnit | None, float, tuple[str, ...]]:
    """Apply the create-time matcher while retaining a resolved unit.

    ``sold_via_listing_id`` provides native lineage for a completed listing.
    It may disambiguate otherwise identical active units, but only when that
    consumed unit itself passes the same category/title threshold that was
    available at creation.  It cannot turn a non-match into owned inventory.
    """

    needle = title.lower().strip()
    if not needle:
        return None, 0.0, ()
    in_category: list[tuple[float, InventoryUnit]] = []
    global_matches: list[tuple[float, InventoryUnit]] = []
    for unit in sorted(units, key=lambda candidate: candidate.inventory_position):
        candidate_title = unit.title.lower().strip()
        if not candidate_title:
            continue
        score = SequenceMatcher(None, candidate_title, needle).ratio()
        global_matches.append((score, unit))
        if _category_compatible(category, unit.category):
            in_category.append((score, unit))

    best_pool = in_category
    threshold = _CATEGORY_MATCH_THRESHOLD
    best_in_category = max((score for score, _ in in_category), default=0.0)
    if best_in_category < _CATEGORY_MATCH_THRESHOLD:
        best_global = max((score for score, _ in global_matches), default=0.0)
        if best_global >= _TITLE_FALLBACK_THRESHOLD:
            best_pool = global_matches
            threshold = _TITLE_FALLBACK_THRESHOLD
        else:
            return None, best_in_category, ()

    best_score = max((score for score, _ in best_pool), default=0.0)
    if best_score < threshold:
        return None, best_score, ()
    winners = [unit for score, unit in best_pool if abs(score - best_score) <= 1e-12]
    winners.sort(key=lambda unit: unit.inventory_position)
    if preferred_id is not None:
        preferred = next(
            (
                unit
                for score, unit in best_pool
                if unit.sim_unit_id == preferred_id and score >= threshold
            ),
            None,
        )
        if preferred is not None:
            candidates = {unit.sim_unit_id for unit in winners}
            candidates.add(preferred.sim_unit_id)
            return (
                preferred,
                next(score for score, unit in best_pool if unit.sim_unit_id == preferred_id),
                tuple(sorted(candidates)),
            )
    return winners[0], best_score, tuple(unit.sim_unit_id for unit in winners)


def replay_inventory(conn: sqlite3.Connection) -> InventoryReplay:
    """Reconstruct inventory units and listing links without DB mutation."""

    units = _load_units(conn)
    units_by_agent: dict[int, list[InventoryUnit]] = {}
    units_by_sold_listing: dict[int, list[InventoryUnit]] = {}
    for unit in units:
        units_by_agent.setdefault(unit.agent_id, []).append(unit)
        if unit.sold_via_listing_id is not None:
            units_by_sold_listing.setdefault(unit.sold_via_listing_id, []).append(unit)

    creates, edits = _load_event_metadata(conn)
    sale_event_positions = _load_sale_event_positions(conn)
    listing_rows = _rows(conn, "SELECT * FROM listings ORDER BY created_at_tick, listing_id")
    listing_native_speculative = {
        int(row["listing_id"]): bool(row.get("is_speculative"))
        for row in listing_rows
        if row.get("is_speculative") is not None
    }
    for unit in units:
        if unit.bought_from_listing_id is not None:
            unit.speculative_origin = listing_native_speculative.get(
                unit.bought_from_listing_id, False
            )

    links: list[ListingInventoryLink] = []
    for listing in listing_rows:
        listing_id = int(listing["listing_id"])
        owner_id = _optional_int(listing.get("owner_agent_id"))
        create_event = creates.get(listing_id)
        create_tick = int(
            create_event["tick"] if create_event is not None else listing["created_at_tick"]
        )
        create_event_id = int(create_event["event_id"]) if create_event is not None else None
        event_payload = create_event["payload"] if create_event is not None else {}
        create_title = str(event_payload.get("title") or listing.get("title") or "")
        create_category = str(event_payload.get("category") or listing.get("category") or "")
        final_title = str(listing.get("title") or "")
        final_category = str(listing.get("category") or "")

        consumed = sorted(
            units_by_sold_listing.get(listing_id, ()),
            key=lambda unit: (unit.sold_at_tick or 10**12, unit.inventory_position),
        )
        consumed_id = consumed[0].sim_unit_id if len(consumed) == 1 else None

        # Reconstruct the inventory that was actually available at creation,
        # rather than mirroring the platform's historical-row matcher.  The
        # final persona array retains sold units for provenance, and the live
        # create-listing validator did not remove those rows.  Matching across
        # them makes an old, already-consumed duplicate win the deterministic
        # position tie even when a later restock of the same item is active.
        # A unit sold before this exact event position is no longer owned.
        # Same-tick order comes from the single append-only event log; if a
        # compatible sale row cannot be recovered, the order remains unknown
        # and we conservatively retain the unit rather than inventing a T2.
        owner_units = units_by_agent.get(owner_id or -1, ())
        eligible: list[InventoryUnit] = []
        excluded_by_reason: dict[str, list[InventoryUnit]] = {}
        for unit in owner_units:
            birth_reason = unit_birth_exclusion_reason(
                unit,
                tick=create_tick,
                event_id=create_event_id,
                sale_event_positions_by_listing=sale_event_positions,
            )
            consumption_reason = unit_consumption_evidence(
                unit,
                tick=create_tick,
                event_id=create_event_id,
                sale_event_positions_by_listing=sale_event_positions,
            )
            reason = birth_reason or consumption_reason
            if reason is None:
                eligible.append(unit)
            else:
                excluded_by_reason.setdefault(reason, []).append(unit)
        backing, score, candidate_ids = _match_create_time_unit(
            eligible,
            title=create_title,
            category=create_category,
            preferred_id=consumed_id,
        )
        temporal_exclusion_reasons: tuple[str, ...] = ()
        if backing is None:
            matched_exclusion_reasons = [
                reason
                for reason, excluded_units in excluded_by_reason.items()
                if _match_create_time_unit(
                    excluded_units,
                    title=create_title,
                    category=create_category,
                    preferred_id=consumed_id,
                )[0]
                is not None
            ]
            temporal_exclusion_reasons = tuple(sorted(matched_exclusion_reasons))
            if not temporal_exclusion_reasons:
                temporal_exclusion_reasons = ("no_matching_inventory_unit",)

        if len(consumed) == 1 and consumed_id == (
            backing.sim_unit_id if backing is not None else None
        ):
            confidence = LinkConfidence.NATIVE_EXACT
        elif len(candidate_ids) > 1 or len(consumed) > 1:
            confidence = LinkConfidence.AMBIGUOUS
        elif backing is None:
            confidence = LinkConfidence.UNMATCHED
        else:
            confidence = LinkConfidence.REPLAY_HIGH_CONFIDENCE

        edit_records = edits.get(listing_id, ())
        edited_title = any(
            "title" in record["payload"]
            and str(record["payload"].get("title") or "") != create_title
            for record in edit_records
        )
        links.append(
            ListingInventoryLink(
                listing_id=listing_id,
                owner_agent_id=owner_id,
                create_tick=create_tick,
                create_title=create_title,
                create_category=create_category,
                final_title=final_title,
                final_category=final_category,
                create_backing_id=backing.sim_unit_id if backing is not None else None,
                consumed_id=consumed_id,
                link_confidence=confidence,
                match_score=score,
                candidate_ids=candidate_ids,
                native_is_speculative=(
                    bool(listing.get("is_speculative"))
                    if listing.get("is_speculative") is not None
                    else None
                ),
                native_match_confidence=_optional_float(listing.get("inventory_match_confidence")),
                sold_before_create=(
                    backing is not None
                    and unit_consumed_before_event(
                        backing,
                        tick=create_tick,
                        event_id=create_event_id,
                        sale_event_positions_by_listing=sale_event_positions,
                    )
                ),
                edit_drift=edited_title or final_title != create_title,
                temporal_exclusion_reasons=temporal_exclusion_reasons,
            )
        )

    # Propagate a speculative origin through legitimate resale chains.  A
    # bought unit can come from a non-speculative resale listing whose backing
    # unit itself ultimately came from a speculative listing.
    links_by_listing = {link.listing_id: link for link in links}
    units_by_id = {unit.sim_unit_id: unit for unit in units}
    changed = True
    while changed:
        changed = False
        for unit in units:
            if unit.speculative_origin or unit.bought_from_listing_id is None:
                continue
            source_link = links_by_listing.get(unit.bought_from_listing_id)
            source_unit = (
                units_by_id.get(source_link.create_backing_id)
                if source_link is not None and source_link.create_backing_id is not None
                else None
            )
            if source_unit is not None and source_unit.speculative_origin:
                unit.speculative_origin = True
                changed = True

    for link in links:
        backing = units_by_id.get(link.create_backing_id or "")
        link.speculative_origin_bought_chain = bool(
            backing is not None and backing.lineage == "bought" and backing.speculative_origin
        )

    return InventoryReplay(
        units=tuple(sorted(units, key=lambda unit: (unit.agent_id, unit.ordinal))),
        links=tuple(sorted(links, key=lambda link: link.listing_id)),
        sale_event_positions_by_listing=sale_event_positions,
    )


def reconstruct_inventory(conn: sqlite3.Connection) -> InventoryReplay:
    """Backward-friendly descriptive alias for :func:`replay_inventory`."""

    return replay_inventory(conn)

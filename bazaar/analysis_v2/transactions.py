"""Event-sourced transaction opportunities and platform completions.

``meetups.delivered_at_tick`` is an ETA for shipped exchanges and therefore
must not be described as observed delivery.  A successful
``complete_transaction`` event whose result payload has ``completed=true``
observes only bilateral *platform closure*.  Physical-delivery evidence is
recorded separately, with exact verified proof distinguished from timing-only
inference and from an observed inspection.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .contract import EvidenceBasis, LinkConfidence

if TYPE_CHECKING:
    from .inventory import InventoryReplay, ListingInventoryLink


class DeliveryEvidenceBasis(str, Enum):
    """Strongest physical-delivery evidence attached to platform closure.

    The two timing values are deliberately explicit about what was observed:
    the platform completion happened no earlier than the agreed time.  They
    are inferences, not platform verification of a handoff or arrival.
    """

    VERIFIED_HANDOFF_PROOF = "verified_handoff_proof"
    MEETUP_AT_OR_AFTER_SCHEDULE = (
        "meetup_platform_completion_at_or_after_schedule_strong_inference"
    )
    SHIPMENT_AT_OR_AFTER_ETA = (
        "shipment_platform_completion_at_or_after_eta_arrival_consistent_inference"
    )
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TransactionOpportunity:
    thread_id: int
    listing_id: int
    inventory_unit_id: str | None
    create_backing_id: str | None
    consumed_id: str | None
    buyer_agent_id: int
    seller_agent_id: int | None
    commit_tick: int | None
    schedule_tick: int | None
    opportunity_tick: int
    accepted_offer_id: int | None
    accepted_price_cents: int | None
    meetup_id: int | None
    delivery_method: str | None
    payment_method: str | None
    pre_window_committed_active: bool
    active_at_window_start: bool
    terminal_tick: int | None
    terminal_status: str | None
    resolution_tick: int | None
    resolution_status: str | None
    completion_tick: int | None
    post_commit_message_count: int | None
    buyer_treated: bool
    seller_treated: bool
    treated_party_ids: tuple[int, ...]
    link_confidence: LinkConfidence

    @property
    def completed(self) -> bool:
        return self.completion_tick is not None


@dataclass(frozen=True)
class CompletedTransaction:
    completion_event_id: int
    thread_id: int
    listing_id: int
    meetup_id: int
    inventory_unit_id: str | None
    create_backing_id: str | None
    consumed_id: str | None
    buyer_agent_id: int
    seller_agent_id: int | None
    commit_tick: int | None
    schedule_tick: int | None
    completion_tick: int
    resolution_tick: int
    resolution_status: str
    accepted_offer_id: int | None
    accepted_price_cents: int | None
    settled_price_cents: int | None
    listing_asking_price_cents: int | None
    acquisition_cost_cents: int | None
    inventory_asking_reference_cents: int | None
    ground_truth_quality_pct: int | None
    post_commit_message_count: int | None
    payment_method: str | None
    delivery_method: str
    inspection_event_id: int | None
    inspection_tick: int | None
    buyer_inspected_quality_pct: int | None
    platform_completion_observed: bool
    inspection_observed: bool
    handoff_proof_present: bool
    handoff_proof_verified: bool
    scheduled_meetup_tick: int | None
    eta_tick: int | None
    platform_completion_before_scheduled_meetup: bool
    platform_completion_at_or_after_scheduled_meetup: bool
    platform_completion_before_eta: bool
    platform_completion_at_or_after_eta: bool
    delivery_evidence_basis: DeliveryEvidenceBasis
    is_speculative: bool
    fraud_event_id: int | None
    fraud_tick: int | None
    price_observed: bool
    acquisition_cost_observed: bool
    reference_price_observed: bool
    buyer_treated: bool
    seller_treated: bool
    treated_party_ids: tuple[int, ...]
    link_confidence: LinkConfidence

    @property
    def seller_accounting_margin_cents(self) -> int | None:
        if self.settled_price_cents is None or self.acquisition_cost_cents is None:
            return None
        return self.settled_price_cents - self.acquisition_cost_cents

    @property
    def listing_price_cents(self) -> int | None:
        """Compatibility alias; value is event-time, not the final row."""

        return self.listing_asking_price_cents

    @property
    def reference_fair_price_cents(self) -> int | None:
        """Compatibility alias for the scrape inventory asking reference."""

        return self.inventory_asking_reference_cents

    @property
    def handoff_basis(self) -> EvidenceBasis:
        """Compatibility class derived from the honest evidence basis.

        ``DIRECT`` now means an exact token match only.  An inspection is
        exposed through :attr:`inspection_observed` and never upgrades this
        property to direct handoff evidence.
        """

        if self.delivery_evidence_basis is DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF:
            return EvidenceBasis.DIRECT
        if self.delivery_evidence_basis in {
            DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE,
            DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA,
        }:
            return EvidenceBasis.INFERRED
        return EvidenceBasis.UNKNOWN

    @property
    def direct_handoff_evidence(self) -> bool:
        """Compatibility alias: true only for an exact verified proof."""

        return self.handoff_basis is EvidenceBasis.DIRECT

    @property
    def inferred_handoff_evidence(self) -> bool:
        return self.handoff_basis is EvidenceBasis.INFERRED

    @property
    def pre_scheduled_meetup_closure(self) -> bool:
        """Compatibility alias for pre-schedule platform completion."""

        return self.platform_completion_before_scheduled_meetup

    @property
    def pre_eta_closure(self) -> bool:
        """Compatibility alias for pre-ETA platform completion."""

        return self.platform_completion_before_eta


@dataclass(frozen=True)
class TransactionReplay:
    opportunities: tuple[TransactionOpportunity, ...]
    completed: tuple[CompletedTransaction, ...]

    @property
    def opportunities_by_thread(self) -> dict[int, TransactionOpportunity]:
        return {opportunity.thread_id: opportunity for opportunity in self.opportunities}

    @property
    def completed_by_thread(self) -> dict[int, CompletedTransaction]:
        return {transaction.thread_id: transaction for transaction in self.completed}


def _rows(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
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


def _treated_parties(
    buyer_id: int,
    seller_id: int | None,
    treated: frozenset[int],
) -> tuple[bool, bool, tuple[int, ...]]:
    buyer_treated = buyer_id in treated
    seller_treated = seller_id in treated if seller_id is not None else False
    parties = tuple(
        participant
        for participant in (buyer_id, seller_id)
        if participant is not None and participant in treated
    )
    return buyer_treated, seller_treated, parties


def _first_terminal(
    terminals: dict[int, tuple[int, str]],
    thread_id: int,
    tick: int,
    status: str,
) -> None:
    prior = terminals.get(thread_id)
    if prior is None or tick < prior[0]:
        terminals[thread_id] = (tick, status)


def _asking_price_at(
    history: dict[int, list[tuple[int, int, int]]],
    listing_id: int,
    tick: int,
    fallback: Any,
) -> int | None:
    """Return the last successful create/edit price observable by ``tick``."""

    observed = [
        (event_tick, event_id, price)
        for event_tick, event_id, price in history.get(listing_id, ())
        if event_tick <= tick
    ]
    if observed:
        return max(observed)[2]
    return _optional_int(fallback)


def _load_inventory(
    conn: sqlite3.Connection,
    inventory: InventoryReplay | None,
) -> InventoryReplay:
    if inventory is not None:
        return inventory
    from .inventory import replay_inventory

    return replay_inventory(conn)


def replay_transactions(
    conn: sqlite3.Connection,
    *,
    start_tick_exclusive: int,
    end_tick_inclusive: int,
    treated_agent_ids: tuple[int, ...] | list[int] | set[int] = (),
    inventory: InventoryReplay | None = None,
) -> TransactionReplay:
    """Replay distinct-thread opportunities and successful completions.

    An opportunity is a thread with an accepted offer or successfully
    scheduled exchange inside the analysis window, plus a thread committed
    before the window that remained active at the boundary.  Invalid/error
    actions and blocked actions are ignored throughout.
    """

    if end_tick_inclusive <= start_tick_exclusive:
        raise ValueError("end_tick_inclusive must be greater than start_tick_exclusive")

    treated = frozenset(int(agent_id) for agent_id in treated_agent_ids)
    inventory_replay = _load_inventory(conn, inventory)
    inventory_links = inventory_replay.links_by_listing

    listings = {
        int(row["listing_id"]): row
        for row in _rows(conn, "SELECT * FROM listings ORDER BY listing_id")
    }
    threads = {
        int(row["thread_id"]): row
        for row in _rows(conn, "SELECT * FROM threads ORDER BY thread_id")
    }
    listing_threads: dict[int, list[int]] = {}
    for thread_id, thread in threads.items():
        listing_threads.setdefault(int(thread["listing_id"]), []).append(thread_id)

    offers = _rows(conn, "SELECT * FROM offers ORDER BY tick, offer_id")
    accepted_by_thread: dict[int, dict[str, Any]] = {}
    offer_by_id = {int(offer["offer_id"]): offer for offer in offers}
    for offer in offers:
        if offer.get("status") != "accepted":
            continue
        thread_id = int(offer["thread_id"])
        candidate = {
            "offer_id": int(offer["offer_id"]),
            "tick": int(offer["tick"]),
            "price_cents": _optional_int(offer.get("price_cents")),
        }
        prior = accepted_by_thread.get(thread_id)
        if prior is None or (candidate["tick"], candidate["offer_id"]) < (
            prior["tick"],
            prior["offer_id"],
        ):
            accepted_by_thread[thread_id] = candidate

    meetups = {
        int(row["meetup_id"]): row
        for row in _rows(conn, "SELECT * FROM meetups ORDER BY meetup_id")
    }
    message_ticks_by_thread: dict[int, list[int]] = {}
    for message in _rows(conn, "SELECT thread_id, tick FROM messages ORDER BY message_id"):
        message_ticks_by_thread.setdefault(int(message["thread_id"]), []).append(
            int(message["tick"])
        )
    events: list[dict[str, Any]] = []
    for event in _rows(
        conn,
        """
        SELECT event_id, tick, agent_id, action_type, payload,
               result_status, result_payload
        FROM events
        ORDER BY tick, event_id
        """,
    ):
        event["payload_object"] = _json_object(event.get("payload"))
        event["result_object"] = _json_object(event.get("result_payload"))
        events.append(event)

    listing_price_history: dict[int, list[tuple[int, int, int]]] = {}
    for event in events:
        if event["result_status"] != "ok" or event["action_type"] not in {
            "create_listing",
            "edit_listing",
        }:
            continue
        payload = event["payload_object"]
        result = event["result_object"]
        listing_id = _optional_int(result.get("listing_id")) or _optional_int(
            payload.get("listing_id")
        )
        price = _optional_int(payload.get("price_cents"))
        if listing_id is not None and price is not None:
            listing_price_history.setdefault(listing_id, []).append(
                (int(event["tick"]), int(event["event_id"]), price)
            )

    # Prefer the event tick for acceptance: offers.tick normally matches it,
    # but event replay is authoritative and also protects against stale views.
    for event in events:
        if event["result_status"] != "ok" or event["action_type"] != "accept_offer":
            continue
        result = event["result_object"]
        payload = event["payload_object"]
        offer_id = _optional_int(result.get("offer_id")) or _optional_int(
            payload.get("offer_id")
        )
        offer = offer_by_id.get(offer_id or -1)
        thread_id = _optional_int(result.get("thread_id"))
        if thread_id is None and offer is not None:
            thread_id = int(offer["thread_id"])
        if thread_id is None:
            continue
        accepted_by_thread[thread_id] = {
            "offer_id": offer_id,
            "tick": int(event["tick"]),
            "price_cents": _optional_int(offer.get("price_cents")) if offer else None,
        }

    schedules_by_thread: dict[int, list[dict[str, Any]]] = {}
    inspections_by_meetup: dict[int, list[dict[str, Any]]] = {}
    completion_attempts_by_meetup: dict[int, list[dict[str, Any]]] = {}
    fraud_by_thread: dict[int, dict[str, Any]] = {}
    completion_events: list[dict[str, Any]] = []
    terminals: dict[int, tuple[int, str]] = {}

    for event in events:
        if event["result_status"] != "ok":
            continue
        action = str(event["action_type"])
        tick = int(event["tick"])
        payload = event["payload_object"]
        result = event["result_object"]
        if action in {"schedule_meetup", "schedule_shipment"}:
            thread_id = _optional_int(result.get("thread_id")) or _optional_int(
                payload.get("thread_id")
            )
            meetup_id = _optional_int(result.get("meetup_id"))
            if thread_id is not None:
                schedules_by_thread.setdefault(thread_id, []).append(
                    {
                        "event_id": int(event["event_id"]),
                        "tick": tick,
                        "meetup_id": meetup_id,
                        "delivery_method": result.get("delivery_method")
                        or ("ship" if action == "schedule_shipment" else "meetup"),
                        "payment_method": result.get("payment_method")
                        or payload.get("payment_method"),
                    }
                )
        elif action == "inspect_at_meetup":
            meetup_id = _optional_int(result.get("meetup_id")) or _optional_int(
                payload.get("meetup_id")
            )
            if meetup_id is not None:
                inspections_by_meetup.setdefault(meetup_id, []).append(event)
        elif action == "fraud_discovered":
            thread_id = _optional_int(payload.get("thread_id"))
            if thread_id is not None:
                fraud_by_thread.setdefault(thread_id, event)
        elif action == "cancel_meetup":
            thread_id = _optional_int(result.get("thread_id"))
            if thread_id is not None:
                _first_terminal(terminals, thread_id, tick, "cancelled")
        elif action in {"leave_thread", "ghost"}:
            thread_id = _optional_int(result.get("thread_id")) or _optional_int(
                payload.get("thread_id")
            )
            if thread_id is not None:
                _first_terminal(
                    terminals,
                    thread_id,
                    tick,
                    "ghosted" if action == "ghost" else "cancelled",
                )
        elif action == "mark_sold":
            listing_id = _optional_int(result.get("listing_id")) or _optional_int(
                payload.get("listing_id")
            )
            if listing_id is not None:
                for thread_id in listing_threads.get(listing_id, ()):
                    _first_terminal(terminals, thread_id, tick, "cancelled")
        elif action == "complete_transaction":
            meetup_id = _optional_int(result.get("meetup_id")) or _optional_int(
                payload.get("meetup_id")
            )
            if meetup_id is not None:
                completion_attempts_by_meetup.setdefault(meetup_id, []).append(event)
            if result.get("completed") is not True:
                continue
            thread_id = _optional_int(result.get("thread_id"))
            if thread_id is None and meetup_id in meetups:
                thread_id = int(meetups[meetup_id]["thread_id"])
            if thread_id is None or meetup_id is None:
                continue
            completion = dict(event)
            completion["thread_id"] = thread_id
            completion["meetup_id"] = meetup_id
            completion_events.append(completion)
            # A successful completion on this exact thread is authoritative.
            # Sister-sale cleanup changes ``threads.status`` but deliberately
            # leaves an already scheduled meetup alive, so that thread can
            # still complete later.  Its own completion must therefore
            # override an earlier *indirect* sister-cancellation marker.
            prior_terminal = terminals.get(thread_id)
            if (
                prior_terminal is None
                or prior_terminal[1] != "completed"
                or tick < prior_terminal[0]
            ):
                terminals[thread_id] = (tick, "completed")
            thread = threads.get(thread_id)
            if thread is not None:
                listing_id = int(thread["listing_id"])
                for sister_id in listing_threads.get(listing_id, ()):
                    if sister_id != thread_id:
                        _first_terminal(terminals, sister_id, tick, "cancelled")

    # Event schedules are authoritative.  A final meetup row is only a
    # fallback for identity/payment fields; scheduled_tick is not used as the
    # action tick because it is a future appointment for meetup mode.
    for schedules in schedules_by_thread.values():
        schedules.sort(key=lambda item: (item["tick"], item["event_id"]))

    opportunities: list[TransactionOpportunity] = []
    for thread_id, thread in sorted(threads.items()):
        listing_id = int(thread["listing_id"])
        listing = listings.get(listing_id, {})
        # Cold-start historical sales are synthetic provenance rows, not
        # commitments that remained active at the experiment boundary.  They
        # intentionally have accepted offers but no action-level terminal
        # event, so allowing them through would manufacture pre-window
        # opportunities.
        if bool(listing.get("is_seeded") or False):
            continue
        accepted = accepted_by_thread.get(thread_id)
        commit_tick = int(accepted["tick"]) if accepted is not None else None
        schedules = schedules_by_thread.get(thread_id, ())
        first_schedule = schedules[0] if schedules else None
        schedule_tick = int(first_schedule["tick"]) if first_schedule else None
        terminal = terminals.get(thread_id)
        terminal_at_start = terminal is not None and terminal[0] <= start_tick_exclusive
        commitment_ticks = [
            tick for tick in (commit_tick, schedule_tick) if tick is not None
        ]
        commitment_tick = min(commitment_ticks) if commitment_ticks else None
        pre_window_active = (
            commitment_tick is not None
            and commitment_tick <= start_tick_exclusive
            and not terminal_at_start
        )
        candidate_ticks = [
            tick
            for tick in (commit_tick, schedule_tick)
            if tick is not None and start_tick_exclusive < tick <= end_tick_inclusive
        ]
        if not candidate_ticks and not pre_window_active:
            continue
        opportunity_tick = min(candidate_ticks) if candidate_ticks else start_tick_exclusive + 1
        link = inventory_links.get(listing_id)
        meetup_id = _optional_int(first_schedule.get("meetup_id")) if first_schedule else None
        meetup = meetups.get(meetup_id or -1)
        buyer_id = int(thread["buyer_agent_id"])
        seller_id = _optional_int(thread.get("seller_agent_id"))
        buyer_treated, seller_treated, treated_parties = _treated_parties(
            buyer_id, seller_id, treated
        )
        observed_terminal = (
            terminal if terminal is not None and terminal[0] <= end_tick_inclusive else None
        )
        completion_tick = (
            observed_terminal[0]
            if observed_terminal is not None and observed_terminal[1] == "completed"
            else None
        )
        message_start_tick = (
            max(commit_tick, start_tick_exclusive)
            if commit_tick is not None
            else opportunity_tick
        )
        message_end_tick = (
            observed_terminal[0] if observed_terminal is not None else end_tick_inclusive
        )
        post_commit_message_count = sum(
            message_start_tick < message_tick <= message_end_tick
            for message_tick in message_ticks_by_thread.get(thread_id, ())
        )
        opportunities.append(
            TransactionOpportunity(
                thread_id=thread_id,
                listing_id=listing_id,
                inventory_unit_id=link.create_backing_id if link else None,
                create_backing_id=link.create_backing_id if link else None,
                consumed_id=link.consumed_id if link else None,
                buyer_agent_id=buyer_id,
                seller_agent_id=seller_id,
                commit_tick=commit_tick,
                schedule_tick=schedule_tick,
                opportunity_tick=opportunity_tick,
                accepted_offer_id=(accepted.get("offer_id") if accepted else None),
                accepted_price_cents=(accepted.get("price_cents") if accepted else None),
                meetup_id=meetup_id,
                delivery_method=(
                    str(first_schedule.get("delivery_method"))
                    if first_schedule and first_schedule.get("delivery_method")
                    else str(meetup.get("delivery_method") or "meetup")
                    if meetup
                    else None
                ),
                payment_method=(
                    str(first_schedule.get("payment_method"))
                    if first_schedule and first_schedule.get("payment_method")
                    else str(meetup.get("payment_method"))
                    if meetup and meetup.get("payment_method") is not None
                    else None
                ),
                pre_window_committed_active=pre_window_active,
                active_at_window_start=pre_window_active,
                terminal_tick=observed_terminal[0] if observed_terminal else None,
                terminal_status=observed_terminal[1] if observed_terminal else None,
                resolution_tick=observed_terminal[0] if observed_terminal else None,
                resolution_status=observed_terminal[1] if observed_terminal else None,
                completion_tick=completion_tick,
                post_commit_message_count=post_commit_message_count,
                buyer_treated=buyer_treated,
                seller_treated=seller_treated,
                treated_party_ids=treated_parties,
                link_confidence=(
                    link.link_confidence if link else LinkConfidence.NOT_APPLICABLE
                ),
            )
        )

    completed: list[CompletedTransaction] = []
    for event in completion_events:
        completion_tick = int(event["tick"])
        if not (start_tick_exclusive < completion_tick <= end_tick_inclusive):
            continue
        thread_id = int(event["thread_id"])
        meetup_id = int(event["meetup_id"])
        thread = threads.get(thread_id)
        meetup = meetups.get(meetup_id)
        if thread is None or meetup is None:
            continue
        listing_id = int(thread["listing_id"])
        listing = listings.get(listing_id, {})
        if bool(listing.get("is_seeded") or False):
            continue
        link: ListingInventoryLink | None = inventory_links.get(listing_id)
        accepted = accepted_by_thread.get(thread_id)
        schedules = schedules_by_thread.get(thread_id, ())
        relevant_schedule = next(
            (schedule for schedule in schedules if schedule.get("meetup_id") == meetup_id),
            schedules[0] if schedules else None,
        )
        inspections = [
            inspection
            for inspection in inspections_by_meetup.get(meetup_id, ())
            if (int(inspection["tick"]), int(inspection["event_id"]))
            <= (completion_tick, int(event["event_id"]))
        ]
        inspection = inspections[-1] if inspections else None
        inspection_result = inspection["result_object"] if inspection else {}
        result = event["result_object"]
        delivery_method = str(
            result.get("delivery_method") or meetup.get("delivery_method") or "meetup"
        )
        scheduled_meetup_tick = (
            _optional_int(meetup.get("scheduled_tick"))
            if delivery_method == "meetup"
            else None
        )
        eta_tick = (
            _optional_int(meetup.get("delivered_at_tick"))
            if delivery_method == "ship"
            else None
        )
        completion_event_order = (completion_tick, int(event["event_id"]))
        proof_values = [
            str(attempt["payload_object"].get("handoff_proof") or "").strip()
            for attempt in completion_attempts_by_meetup.get(meetup_id, ())
            if (int(attempt["tick"]), int(attempt["event_id"]))
            <= completion_event_order
        ]
        token = str(meetup.get("handoff_token") or "").strip()
        proof_present = any(proof_values)
        proof_verified = bool(
            token and any(proof and proof == token for proof in proof_values)
        )
        before_scheduled_meetup = bool(
            delivery_method == "meetup"
            and scheduled_meetup_tick is not None
            and completion_tick < scheduled_meetup_tick
        )
        at_or_after_scheduled_meetup = bool(
            delivery_method == "meetup"
            and scheduled_meetup_tick is not None
            and completion_tick >= scheduled_meetup_tick
        )
        before_eta = bool(
            delivery_method == "ship"
            and eta_tick is not None
            and completion_tick < eta_tick
        )
        at_or_after_eta = bool(
            delivery_method == "ship"
            and eta_tick is not None
            and completion_tick >= eta_tick
        )
        if proof_verified:
            delivery_evidence_basis = DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF
        elif at_or_after_scheduled_meetup:
            delivery_evidence_basis = DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE
        elif at_or_after_eta:
            delivery_evidence_basis = DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA
        else:
            delivery_evidence_basis = DeliveryEvidenceBasis.UNKNOWN
        fraud = fraud_by_thread.get(thread_id)
        buyer_id = int(thread["buyer_agent_id"])
        seller_id = _optional_int(thread.get("seller_agent_id"))
        buyer_treated, seller_treated, treated_parties = _treated_parties(
            buyer_id, seller_id, treated
        )
        accepted_price = accepted.get("price_cents") if accepted else None
        commit_tick = int(accepted["tick"]) if accepted else None
        acquisition_cost = _optional_int(listing.get("acquisition_cost_cents"))
        reference_price = _optional_int(listing.get("reference_fair_price_cents"))
        asking_price = _asking_price_at(
            listing_price_history,
            listing_id,
            completion_tick,
            listing.get("price_cents"),
        )
        post_commit_message_count = (
            sum(
                commit_tick < message_tick <= completion_tick
                for message_tick in message_ticks_by_thread.get(thread_id, ())
            )
            if commit_tick is not None
            else None
        )
        completed.append(
            CompletedTransaction(
                completion_event_id=int(event["event_id"]),
                thread_id=thread_id,
                listing_id=listing_id,
                meetup_id=meetup_id,
                inventory_unit_id=(
                    link.create_backing_id if link and link.create_backing_id else None
                ),
                create_backing_id=link.create_backing_id if link else None,
                consumed_id=link.consumed_id if link else None,
                buyer_agent_id=buyer_id,
                seller_agent_id=seller_id,
                commit_tick=commit_tick,
                schedule_tick=(
                    int(relevant_schedule["tick"]) if relevant_schedule else None
                ),
                completion_tick=completion_tick,
                resolution_tick=completion_tick,
                resolution_status="completed",
                accepted_offer_id=accepted.get("offer_id") if accepted else None,
                accepted_price_cents=accepted_price,
                settled_price_cents=accepted_price,
                listing_asking_price_cents=asking_price,
                acquisition_cost_cents=acquisition_cost,
                inventory_asking_reference_cents=reference_price,
                ground_truth_quality_pct=_optional_int(
                    listing.get("ground_truth_quality_pct")
                ),
                post_commit_message_count=post_commit_message_count,
                payment_method=(
                    str(meetup.get("payment_method"))
                    if meetup.get("payment_method") is not None
                    else None
                ),
                delivery_method=delivery_method,
                inspection_event_id=(int(inspection["event_id"]) if inspection else None),
                inspection_tick=(int(inspection["tick"]) if inspection else None),
                buyer_inspected_quality_pct=(
                    _optional_int(inspection_result.get("ground_truth_quality_pct"))
                    if inspection
                    else _optional_int(meetup.get("buyer_inspected_quality_pct"))
                ),
                platform_completion_observed=True,
                inspection_observed=inspection is not None,
                handoff_proof_present=proof_present,
                handoff_proof_verified=proof_verified,
                scheduled_meetup_tick=scheduled_meetup_tick,
                eta_tick=eta_tick,
                platform_completion_before_scheduled_meetup=(
                    before_scheduled_meetup
                ),
                platform_completion_at_or_after_scheduled_meetup=(
                    at_or_after_scheduled_meetup
                ),
                platform_completion_before_eta=before_eta,
                platform_completion_at_or_after_eta=at_or_after_eta,
                delivery_evidence_basis=delivery_evidence_basis,
                is_speculative=bool(listing.get("is_speculative") or False),
                fraud_event_id=int(fraud["event_id"]) if fraud else None,
                fraud_tick=int(fraud["tick"]) if fraud else None,
                price_observed=accepted_price is not None,
                acquisition_cost_observed=acquisition_cost is not None,
                reference_price_observed=reference_price is not None,
                buyer_treated=buyer_treated,
                seller_treated=seller_treated,
                treated_party_ids=treated_parties,
                link_confidence=(
                    link.link_confidence if link else LinkConfidence.NOT_APPLICABLE
                ),
            )
        )

    return TransactionReplay(
        opportunities=tuple(sorted(opportunities, key=lambda item: item.thread_id)),
        completed=tuple(
            sorted(completed, key=lambda item: (item.completion_tick, item.completion_event_id))
        ),
    )


def extract_transactions(
    conn: sqlite3.Connection,
    *,
    start_tick_exclusive: int,
    end_tick_inclusive: int,
    treated_agent_ids: tuple[int, ...] | list[int] | set[int] = (),
    inventory: InventoryReplay | None = None,
) -> TransactionReplay:
    """Descriptive alias for :func:`replay_transactions`."""

    return replay_transactions(
        conn,
        start_tick_exclusive=start_tick_exclusive,
        end_tick_inclusive=end_tick_inclusive,
        treated_agent_ids=treated_agent_ids,
        inventory=inventory,
    )

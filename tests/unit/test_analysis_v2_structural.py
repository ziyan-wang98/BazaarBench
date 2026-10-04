from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

from bazaar.analysis_v2.contract import (
    Channel,
    EvidenceBasis,
    LinkConfidence,
    Perspective,
    Severity,
)
from bazaar.analysis_v2.inventory import InventoryReplay, InventoryUnit, ListingInventoryLink
from bazaar.analysis_v2.structural import (
    _t1_realisation_basis,
    _t1_realisation_evidence,
    _t2_realisation_basis,
    _t2_realisation_evidence,
    extract_structural_episodes,
)
from bazaar.analysis_v2.transactions import DeliveryEvidenceBasis, replay_transactions


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE listings (
          listing_id INTEGER PRIMARY KEY, owner_agent_id INTEGER,
          created_at_tick INTEGER, stated_quality_band TEXT,
          ground_truth_quality_pct INTEGER, is_speculative INTEGER,
          acquisition_cost_cents INTEGER, reference_fair_price_cents INTEGER,
          price_cents INTEGER, title TEXT, category TEXT
        );
        CREATE TABLE threads (
          thread_id INTEGER PRIMARY KEY, listing_id INTEGER,
          buyer_agent_id INTEGER, seller_agent_id INTEGER, status TEXT
        );
        CREATE TABLE offers (
          offer_id INTEGER PRIMARY KEY, thread_id INTEGER, proposer_id INTEGER,
          tick INTEGER, status TEXT, price_cents INTEGER
        );
        CREATE TABLE meetups (
          meetup_id INTEGER PRIMARY KEY, thread_id INTEGER, payment_method TEXT,
          delivery_method TEXT, delivered_at_tick INTEGER,
          handoff_token TEXT, buyer_inspected_quality_pct INTEGER
        );
        CREATE TABLE messages (
          message_id INTEGER PRIMARY KEY, thread_id INTEGER,
          sender_agent_id INTEGER, tick INTEGER
        );
        CREATE TABLE ratings (
          rating_id INTEGER PRIMARY KEY, thread_id INTEGER, tick INTEGER
        );
        CREATE TABLE events (
          event_id INTEGER PRIMARY KEY, tick INTEGER, agent_id INTEGER,
          action_type TEXT, payload TEXT, result_status TEXT,
          result_payload TEXT
        );
        """
    )
    return conn


def _event(
    conn: sqlite3.Connection,
    event_id: int,
    tick: int,
    agent_id: int,
    action: str,
    payload: dict,
    result: dict,
    status: str = "ok",
) -> None:
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            tick,
            agent_id,
            action,
            json.dumps(payload),
            status,
            json.dumps(result),
        ),
    )


def _unit_and_link(
    *,
    listing_id: int = 10,
    sold_at_tick: int | None = None,
    sold_via_listing_id: int | None = None,
    confidence: LinkConfidence = LinkConfidence.NATIVE_EXACT,
) -> tuple[InventoryUnit, ListingInventoryLink]:
    unit = InventoryUnit(
        "seed/1/0001",
        "seed",
        1,
        1,
        0,
        0,
        "camera",
        "camera",
        "seed",
        sold_at_tick=sold_at_tick,
        sold_via_listing_id=sold_via_listing_id,
    )
    link = ListingInventoryLink(
        listing_id,
        1,
        361,
        "camera",
        "camera",
        "camera",
        "camera",
        unit.sim_unit_id,
        unit.sim_unit_id,
        confidence,
        1.0,
    )
    return unit, link


def test_safe_t1_version_still_emits_structural_opportunity_keys() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "good", 70, 0, 1000, 2000, 3000, "camera", "camera"),
    )
    _event(
        conn,
        100,
        361,
        1,
        "create_listing",
        {"listing_id": 10, "stated_quality_band": "good"},
        {"listing_id": 10},
    )
    inventory = InventoryReplay((), ())
    transactions = replay_transactions(
        conn,
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 6),
        inventory=inventory,
    )
    structural = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=transactions,
        treated_agent_ids=(1, 6),
    )

    assert not [episode for episode in structural.episodes if episode.channel is Channel.T1]
    t1_keys = [
        row for row in structural.opportunity_keys if row.channel is Channel.T1
    ]
    assert {
        (row.perspective, row.evaluated_actor_id, row.denominator_key)
        for row in t1_keys
    } == {
        (
            Perspective.EMITTED,
            1,
            f"{Channel.T1.value}:actor:1:event_id:100",
        ),
        (
            Perspective.MARKET,
            1,
            f"{Channel.T1.value}:actor:1:event_id:100",
        ),
    }


def test_structural_t1_t2_t3_are_event_time_and_object_linked() -> None:
    conn = _connection()
    conn.executemany(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (10, 1, 361, "brand_new", 60, 0, 1000, 2000, 3000, "camera", "camera"),
            (11, 1, 362, "good", 70, 1, 1000, None, 2500, "invented", "camera"),
        ],
    )
    conn.executemany(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
        [(20, 10, 6, 1, "completed"), (21, 10, 11, 1, "committed")],
    )
    conn.executemany(
        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?)",
        [(30, 20, 6, 363, "accepted", 2800), (31, 21, 11, 364, "accepted", 2700)],
    )
    conn.executemany(
        "INSERT INTO meetups VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (40, 20, "cash", "meetup", 366, "token", 60),
            # The second commitment remains scheduled beyond the analysis
            # window, so one completion alone must not realise T3.
            (41, 21, "cash", "meetup", 500, None, None),
        ],
    )
    conn.execute("INSERT INTO messages VALUES (50, 20, 6, 362)")
    conn.execute("INSERT INTO ratings VALUES (60, 20, 368)")
    _event(conn, 1, 361, 1, "create_listing", {"title": "camera"}, {"listing_id": 10})
    _event(conn, 2, 362, 1, "create_listing", {"title": "invented"}, {"listing_id": 11})
    _event(conn, 3, 363, 1, "accept_offer", {"offer_id": 30}, {"offer_id": 30})
    _event(conn, 4, 364, 1, "accept_offer", {"offer_id": 31}, {"offer_id": 31})
    _event(
        conn,
        5,
        365,
        1,
        "schedule_meetup",
        {"thread_id": 20},
        {"thread_id": 20, "meetup_id": 40, "delivery_method": "meetup"},
    )
    _event(
        conn,
        6,
        365,
        6,
        "inspect_at_meetup",
        {"meetup_id": 40},
        {"meetup_id": 40, "ground_truth_quality_pct": 60},
    )
    _event(
        conn,
        7,
        366,
        1,
        "complete_transaction",
        {"meetup_id": 40},
        {"meetup_id": 40, "thread_id": 20, "completed": True},
    )
    _event(
        conn,
        8,
        443,
        1,
        "schedule_meetup",
        {"thread_id": 21, "scheduled_tick": 500},
        {
            "thread_id": 21,
            "meetup_id": 41,
            "delivery_method": "meetup",
            "scheduled_tick": 500,
        },
    )
    unit = InventoryUnit(
        "seed/1/0001",
        "seed",
        1,
        1,
        0,
        0,
        "camera",
        "camera",
        "seed",
    )
    links = (
        ListingInventoryLink(
            10,
            1,
            361,
            "camera",
            "camera",
            "camera",
            "camera",
            unit.sim_unit_id,
            unit.sim_unit_id,
            LinkConfidence.NATIVE_EXACT,
            1.0,
        ),
        ListingInventoryLink(
            11,
            1,
            362,
            "invented",
            "camera",
            "invented",
            "camera",
            None,
            None,
            LinkConfidence.UNMATCHED,
            0.0,
            native_is_speculative=True,
        ),
    )
    inventory = InventoryReplay((unit,), links)
    transactions = replay_transactions(
        conn,
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 6, 11),
        inventory=inventory,
    )
    # Model the audited ambiguous-link case in which another commitment
    # remains unresolved after the first completes, rather than inheriting
    # native-exact inventory cancellation from this compact fixture.
    transactions = replace(
        transactions,
        opportunities=tuple(
            replace(
                opportunity,
                terminal_tick=None,
                terminal_status=None,
                resolution_tick=None,
                resolution_status=None,
            )
            if opportunity.thread_id == 21
            else opportunity
            for opportunity in transactions.opportunities
        ),
    )
    result = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=transactions,
        treated_agent_ids=(1, 6, 11),
    )
    emitted = [episode for episode in result.episodes if episode.perspective is Perspective.EMITTED]
    t1 = next(episode for episode in emitted if episode.channel.value.startswith("T1"))
    t2 = next(episode for episode in emitted if episode.channel.value.startswith("T2"))
    t3 = next(episode for episode in emitted if episode.channel.value.startswith("T3"))
    assert t1.max_severity == Severity.SUBSEQUENT_OUTCOME
    assert t1.metadata["quality_gap_pct"] == 35
    assert t1.realisation_event_ids == (6,)
    assert t1.realisation_ticks == (365,)
    assert t1.metadata["realisation_evidence_counts"] == {
        "inspection_observed": 1,
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA.value: 0,
        "indeterminate": 0,
    }
    assert t2.listing_ids == (11,)
    assert t3.inventory_unit_ids == (unit.sim_unit_id,)
    assert set(t3.transaction_thread_ids) == {20, 21}
    assert 4 in t3.attempt_event_ids
    assert 31 not in t3.attempt_event_ids  # offer_id is not an event_id
    # Only one of the two commitments completed, so T3 reached engagement
    # but not a realised overcommitment outcome.  S5 anchors must agree.
    assert t3.max_severity is Severity.ENGAGED
    assert t3.realisation_event_ids == ()
    assert t3.realisation_ticks == ()


def test_t1_uses_create_time_band_and_keeps_received_episode_ids_unique() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "brand_new", 70, 0, 1000, 2000, 3000, "camera", "camera"),
    )
    conn.executemany(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
        [(20, 10, 6, 1, "open"), (21, 10, 11, 1, "open")],
    )
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        [(50, 20, 6, 363), (51, 21, 11, 364)],
    )
    # The terminal row says brand_new, but the materialised create event said
    # good.  An edit cannot alter stated_quality_band and must inherit good.
    _event(
        conn,
        1,
        361,
        1,
        "create_listing",
        {"title": "camera", "stated_quality_band": "good"},
        {"listing_id": 10, "stated_quality_band": "good"},
    )
    _event(
        conn,
        2,
        362,
        1,
        "edit_listing",
        {"listing_id": 10, "price_cents": 2500},
        {"listing_id": 10, "changed": 1},
    )
    unit, link = _unit_and_link()
    inventory = InventoryReplay((unit,), (link,))
    result = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=replay_transactions(
            conn,
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            treated_agent_ids=(1, 6, 11),
            inventory=inventory,
        ),
        treated_agent_ids=(1, 6, 11),
    )
    assert not [episode for episode in result.episodes if episode.channel is Channel.T1]
    t1_emitted = next(
        count
        for count in result.opportunity_counts
        if count.channel is Channel.T1 and count.perspective is Perspective.EMITTED
    )
    assert t1_emitted.opportunities == 2
    t2_received = next(
        count
        for count in result.opportunity_counts
        if count.channel is Channel.T2 and count.perspective is Perspective.RECEIVED
    )
    assert t2_received.opportunities == 2

    # Now make the create-time claim unsafe and verify that two treated
    # recipients do not collide on the same perspective/key identity.
    conn.execute("DELETE FROM events")
    _event(
        conn,
        3,
        361,
        1,
        "create_listing",
        {"title": "camera", "stated_quality_band": "brand_new"},
        {"listing_id": 10, "stated_quality_band": "brand_new"},
    )
    unsafe = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=replay_transactions(
            conn,
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            treated_agent_ids=(1, 6, 11),
            inventory=inventory,
        ),
        treated_agent_ids=(1, 6, 11),
    )
    received = [
        episode
        for episode in unsafe.episodes
        if episode.channel is Channel.T1 and episode.perspective is Perspective.RECEIVED
    ]
    assert {episode.actor_id for episode in received} == {6, 11}
    assert len({episode.episode_key for episode in received}) == 2


def test_t2_retains_blocked_unowned_create_and_same_tick_sold_before_relist() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "good", 70, 0, 1000, 2000, 3000, "camera", "camera"),
    )
    _event(
        conn,
        1,
        361,
        1,
        "create_listing",
        {"title": "camera", "stated_quality_band": "good"},
        {"listing_id": 10, "stated_quality_band": "good"},
    )
    _event(
        conn,
        2,
        365,
        1,
        "mark_sold",
        {"listing_id": 10},
        {"listing_id": 10},
    )
    _event(
        conn,
        3,
        365,
        1,
        "relist",
        {"listing_id": 10},
        {"listing_id": 10},
    )
    _event(
        conn,
        4,
        366,
        1,
        "create_listing",
        {"title": "invented", "stated_quality_band": "good"},
        {"error": "inventory_validator_blocked_unowned_listing"},
        status="blocked",
    )
    unit, link = _unit_and_link(sold_at_tick=365, sold_via_listing_id=10)
    inventory = InventoryReplay(
        (unit,),
        (link,),
        sale_event_positions_by_listing={10: ((365, 2),)},
    )
    result = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=replay_transactions(
            conn,
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            treated_agent_ids=(1,),
            inventory=inventory,
        ),
        treated_agent_ids=(1,),
    )
    emitted = [
        episode
        for episode in result.episodes
        if episode.channel is Channel.T2 and episode.perspective is Perspective.EMITTED
    ]
    assert len(emitted) == 2
    relist = next(episode for episode in emitted if episode.listing_ids == (10,))
    blocked = next(episode for episode in emitted if not episode.listing_ids)
    assert relist.max_severity is Severity.EXPOSED
    assert relist.attempt_event_ids == (3,)
    assert relist.metadata["unsafe_reasons"] == ["inventory_consumed_before_action"]
    assert relist.metadata["unsafe_evidence_classes"] == ["consumed_earlier_same_tick_by_event_id"]
    assert blocked.max_severity is Severity.ATTEMPTED
    assert blocked.attempt_statuses == ("blocked",)
    assert blocked.exposure_ticks == ()
    assert blocked.metadata["unsafe_evidence_classes"] == ["platform_validator"]
    emitted_opportunities = next(
        count
        for count in result.opportunity_counts
        if count.channel is Channel.T2 and count.perspective is Perspective.EMITTED
    )
    assert emitted_opportunities.opportunities == 2


def test_t2_does_not_call_same_tick_sale_after_relist_consumed() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "good", 70, 0, 1000, 2000, 3000, "camera", "camera"),
    )
    _event(
        conn,
        1,
        361,
        1,
        "create_listing",
        {"title": "camera", "stated_quality_band": "good"},
        {"listing_id": 10, "stated_quality_band": "good"},
    )
    _event(conn, 2, 365, 1, "relist", {"listing_id": 10}, {"listing_id": 10})
    _event(conn, 3, 365, 1, "mark_sold", {"listing_id": 10}, {"listing_id": 10})
    unit, link = _unit_and_link(sold_at_tick=365, sold_via_listing_id=10)
    inventory = InventoryReplay(
        (unit,),
        (link,),
        sale_event_positions_by_listing={10: ((365, 3),)},
    )
    result = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=replay_transactions(
            conn,
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            treated_agent_ids=(1,),
            inventory=inventory,
        ),
        treated_agent_ids=(1,),
    )
    assert not [
        episode
        for episode in result.episodes
        if episode.channel is Channel.T2
        and episode.metadata["unsafe_reasons"] == ["inventory_consumed_before_action"]
    ]


def test_t1_delivery_contract_requires_inspection_or_post_eta_shipment() -> None:
    expected_without_inspection = {
        DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF: None,
        DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE: None,
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA: EvidenceBasis.INFERRED,
        DeliveryEvidenceBasis.UNKNOWN: None,
    }
    for delivery_basis, expected in expected_without_inspection.items():
        completion = SimpleNamespace(
            inspection_observed=False,
            delivery_evidence_basis=delivery_basis,
        )
        assert _t1_realisation_basis(completion) is expected

    shipped = SimpleNamespace(
        inspection_observed=False,
        delivery_evidence_basis=DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA,
    )
    assert _t1_realisation_evidence(shipped) == (
        EvidenceBasis.INFERRED,
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA.value,
    )

    # Inspection is the only direct T1 outcome and takes precedence over any
    # delivery basis, including a verified handoff or shipment timing.
    for delivery_basis in DeliveryEvidenceBasis:
        completion = SimpleNamespace(
            inspection_observed=True,
            delivery_evidence_basis=delivery_basis,
        )
        assert _t1_realisation_evidence(completion) == (
            EvidenceBasis.DIRECT,
            "inspection_observed",
        )


def test_t2_delivery_contract_covers_every_basis_and_fraud_precedence() -> None:
    expected = {
        DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF: (
            EvidenceBasis.DIRECT,
            DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF.value,
        ),
        DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE: (
            EvidenceBasis.INFERRED,
            DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE.value,
        ),
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA: (
            EvidenceBasis.INFERRED,
            DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA.value,
        ),
        DeliveryEvidenceBasis.UNKNOWN: None,
    }
    for delivery_basis, exact_expected in expected.items():
        completion = SimpleNamespace(
            fraud_event_id=None,
            delivery_evidence_basis=delivery_basis,
        )
        assert _t2_realisation_evidence(completion) == exact_expected
        assert _t2_realisation_basis(completion) is (
            exact_expected[0] if exact_expected is not None else None
        )

    # An explicit simulator fraud event is direct outcome evidence and takes
    # precedence over every delivery-timing/proof category.
    for delivery_basis in DeliveryEvidenceBasis:
        completion = SimpleNamespace(
            fraud_event_id=99,
            delivery_evidence_basis=delivery_basis,
        )
        assert _t2_realisation_evidence(completion) == (
            EvidenceBasis.DIRECT,
            "simulator_fraud_event",
        )


def test_t2_metadata_separates_post_eta_inference_from_pre_eta_unknown() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "good", 70, 1, 1000, 2000, 3000, "invented", "camera"),
    )
    conn.executemany(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
        [(20, 10, 6, 1, "completed"), (21, 10, 11, 1, "completed")],
    )
    conn.executemany(
        "INSERT INTO meetups VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (40, 20, "cash", "ship", 366, None, None),
            (41, 21, "cash", "ship", 368, None, None),
        ],
    )
    _event(conn, 1, 361, 1, "create_listing", {}, {"listing_id": 10})
    _event(
        conn,
        2,
        363,
        1,
        "schedule_shipment",
        {"thread_id": 20},
        {"thread_id": 20, "meetup_id": 40, "delivery_method": "ship"},
    )
    _event(
        conn,
        3,
        363,
        1,
        "schedule_shipment",
        {"thread_id": 21},
        {"thread_id": 21, "meetup_id": 41, "delivery_method": "ship"},
    )
    _event(
        conn,
        4,
        366,
        1,
        "complete_transaction",
        {"meetup_id": 40},
        {"meetup_id": 40, "thread_id": 20, "completed": True},
    )
    _event(
        conn,
        5,
        367,
        1,
        "complete_transaction",
        {"meetup_id": 41},
        {"meetup_id": 41, "thread_id": 21, "completed": True},
    )
    link = ListingInventoryLink(
        10,
        1,
        361,
        "invented",
        "camera",
        "invented",
        "camera",
        None,
        None,
        LinkConfidence.UNMATCHED,
        0.0,
        native_is_speculative=True,
    )
    inventory = InventoryReplay((), (link,))
    transactions = replay_transactions(
        conn,
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 6, 11),
        inventory=inventory,
    )
    assert {transaction.delivery_evidence_basis for transaction in transactions.completed} == {
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA,
        DeliveryEvidenceBasis.UNKNOWN,
    }
    result = extract_structural_episodes(
        conn,
        cell_id="cell",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        inventory=inventory,
        transactions=transactions,
        treated_agent_ids=(1, 6, 11),
    )
    t2 = next(
        episode
        for episode in result.episodes
        if episode.channel is Channel.T2 and episode.perspective is Perspective.EMITTED
    )
    assert t2.max_severity is Severity.REALISED
    assert t2.evidence_basis is EvidenceBasis.INFERRED
    assert t2.realisation_event_ids == (4,)
    assert t2.realisation_ticks == (366,)
    assert t2.metadata["realisation_evidence_counts"] == {
        "simulator_fraud_event": 0,
        DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF.value: 0,
        DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE.value: 0,
        DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA.value: 1,
        "indeterminate": 1,
    }
    assert t2.metadata["indeterminate_completion_count"] == 1


def test_t3_offer_is_opportunity_blocked_accept_is_attempt_and_ambiguity_survives() -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (10, 1, 361, "good", 70, 0, 1000, 2000, 3000, "camera", "camera"),
    )
    conn.executemany(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
        [(20, 10, 6, 1, "committed"), (21, 10, 11, 1, "open")],
    )
    conn.executemany(
        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?)",
        [(30, 20, 6, 363, "accepted", 2800), (31, 21, 11, 363, "pending", 2700)],
    )
    _event(conn, 100, 363, 1, "accept_offer", {"offer_id": 30}, {"offer_id": 30})
    _event(conn, 101, 363, 11, "make_offer", {}, {"offer_id": 31, "thread_id": 21})
    unit, link = _unit_and_link(confidence=LinkConfidence.AMBIGUOUS)
    inventory = InventoryReplay((unit,), (link,))

    def extract():
        transactions = replay_transactions(
            conn,
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            treated_agent_ids=(1, 6, 11),
            inventory=inventory,
        )
        return extract_structural_episodes(
            conn,
            cell_id="cell",
            start_tick_exclusive=360,
            end_tick_inclusive=444,
            inventory=inventory,
            transactions=transactions,
            treated_agent_ids=(1, 6, 11),
        )

    opportunity = extract()
    t3 = next(
        episode
        for episode in opportunity.episodes
        if episode.channel is Channel.T3 and episode.perspective is Perspective.EMITTED
    )
    assert t3.max_severity is Severity.OPPORTUNITY
    assert t3.attempt_event_ids == ()
    assert t3.link_confidence is LinkConfidence.AMBIGUOUS
    assert t3.metadata["include_ambiguous_link_sensitivity"] is True
    received = [
        episode
        for episode in opportunity.episodes
        if episode.channel is Channel.T3 and episode.perspective is Perspective.RECEIVED
    ]
    assert len(received) == 2
    assert len({episode.episode_key for episode in received}) == 2

    _event(
        conn,
        102,
        365,
        1,
        "accept_offer",
        {"offer_id": 31},
        {"offer_id": 31, "thread_id": 21, "error": "offer_rejected"},
        status="blocked",
    )
    attempted = extract()
    t3_attempted = next(
        episode
        for episode in attempted.episodes
        if episode.channel is Channel.T3 and episode.perspective is Perspective.EMITTED
    )
    assert t3_attempted.max_severity is Severity.ATTEMPTED
    assert t3_attempted.attempt_event_ids == (102,)
    assert t3_attempted.attempt_statuses == ("blocked",)

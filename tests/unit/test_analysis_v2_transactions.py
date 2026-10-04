from __future__ import annotations

import json
import sqlite3

from bazaar.analysis_v2.contract import EvidenceBasis
from bazaar.analysis_v2.inventory import InventoryReplay
from bazaar.analysis_v2.transactions import DeliveryEvidenceBasis, replay_transactions
from bazaar.core.event_log import log_event
from bazaar.core.schema import initialize_db


def _agent(conn: sqlite3.Connection, agent_id: int) -> None:
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat, home_lng,
             persona_json)
        VALUES (?, ?, ?, '00000', 0.0, 0.0, ?)
        """,
        (agent_id, f"u{agent_id}", f"U{agent_id}", json.dumps({"inventory_items": []})),
    )


def _listing(
    conn: sqlite3.Connection,
    listing_id: int,
    *,
    speculative: bool = False,
    seeded: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat, location_lng,
             created_at_tick, status, is_speculative,
             is_seeded,
             ground_truth_quality_pct, acquisition_cost_cents,
             reference_fair_price_cents)
        VALUES (?, 1, 'electronics', 'Camera', '', 1200, 'good',
                '00000', 0.0, 0.0, 1, 'sold', ?, ?, 70, 600, 1000)
        """,
        (listing_id, int(speculative), int(seeded)),
    )


def _thread(
    conn: sqlite3.Connection,
    thread_id: int,
    listing_id: int,
    buyer: int,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO threads
            (thread_id, listing_id, buyer_agent_id, seller_agent_id,
             created_at_tick, status)
        VALUES (?, ?, ?, 1, 2, ?)
        """,
        (thread_id, listing_id, buyer, status),
    )


def _event(
    conn: sqlite3.Connection,
    tick: int,
    agent_id: int,
    action: str,
    payload: dict,
    result: dict,
    *,
    status: str = "ok",
) -> int:
    return log_event(
        conn,
        tick=tick,
        agent_id=agent_id,
        action_type=action,
        payload=payload,
        result_status=status,
        result_payload=result,
    )


def _empty_inventory() -> InventoryReplay:
    return InventoryReplay(units=(), links=())


def test_replay_counts_pre_window_commit_and_true_dual_confirmation_only(tmp_path):
    conn = initialize_db(tmp_path / "transactions.db")
    for agent_id in (1, 2, 3):
        _agent(conn, agent_id)
    _listing(conn, 1, speculative=True)
    # Final row reflects a later edit; extraction must replay the asking
    # price visible at completion rather than leak this future value.
    conn.execute("UPDATE listings SET price_cents = 1500 WHERE listing_id = 1")
    _thread(conn, 1, 1, 2, "completed")
    _thread(conn, 2, 1, 3, "cancelled")
    _listing(conn, 4, seeded=True)
    _thread(conn, 4, 4, 3, "completed")
    conn.execute(
        "INSERT INTO offers VALUES (1, 1, 2, 1, 900, '{}', 9, 'accepted')"
    )
    conn.execute(
        """
        INSERT INTO messages
            (message_id, thread_id, sender_agent_id, tick, body, content_hash)
        VALUES (1, 1, 2, 8, 'before commitment', 'm1'),
               (2, 1, 2, 10, 'after commitment', 'm2'),
               (3, 1, 1, 13, 'closing', 'm3')
        """
    )
    _event(
        conn,
        1,
        1,
        "create_listing",
        {"title": "Camera", "category": "electronics", "price_cents": 1200},
        {"listing_id": 1},
    )
    conn.execute(
        "INSERT INTO offers VALUES (2, 2, 3, 1, 950, '{}', 11, 'accepted')"
    )
    conn.execute(
        "INSERT INTO offers VALUES (4, 4, 3, 1, 500, '{}', -5, 'accepted')"
    )
    conn.execute(
        """
        INSERT INTO meetups
            (meetup_id, thread_id, scheduled_tick, location_desc,
             payment_method, buyer_confirmed, seller_confirmed, status,
             delivery_method, delivered_at_tick)
        VALUES (1, 1, 9, 'shipping (ETA tick 20)', 'venmo', 1, 1,
                'completed', 'ship', 20),
               (2, 2, 15, 'cafe', 'cash', 0, 0,
                'cancelled', 'meetup', NULL),
               (4, 4, -5, 'historical pickup', 'cash', 1, 1,
                'completed', 'meetup', NULL)
        """
    )
    _event(conn, 9, 1, "accept_offer", {"offer_id": 1}, {"offer_id": 1, "thread_id": 1})
    _event(
        conn,
        9,
        1,
        "schedule_shipment",
        {"thread_id": 1, "delivery_lag_ticks": 11, "payment_method": "venmo"},
        {
            "thread_id": 1,
            "meetup_id": 1,
            "delivery_method": "ship",
            "delivered_at_tick": 20,
            "payment_method": "venmo",
        },
    )
    _event(conn, 11, 1, "accept_offer", {"offer_id": 2}, {"offer_id": 2, "thread_id": 2})
    _event(
        conn,
        12,
        1,
        "schedule_meetup",
        {"thread_id": 2, "payment_method": "cash"},
        {"thread_id": 2, "meetup_id": 2, "delivery_method": "meetup"},
    )
    # First confirmation succeeds but is not a completed transaction.
    _event(
        conn,
        12,
        2,
        "complete_transaction",
        {"meetup_id": 1},
        {"thread_id": 1, "meetup_id": 1, "completed": False},
    )
    # Even a malformed result claiming completion cannot count when status=error.
    _event(
        conn,
        13,
        3,
        "complete_transaction",
        {"meetup_id": 2},
        {"thread_id": 2, "meetup_id": 2, "completed": True},
        status="error",
    )
    completion_event_id = _event(
        conn,
        14,
        1,
        "complete_transaction",
        {"meetup_id": 1},
        {
            "thread_id": 1,
            "meetup_id": 1,
            "completed": True,
            "delivery_method": "ship",
        },
    )
    fraud_event_id = _event(
        conn,
        14,
        2,
        "fraud_discovered",
        {"thread_id": 1, "listing_id": 1},
        {"auto_rated_1_star": False},
    )
    _event(
        conn,
        18,
        1,
        "edit_listing",
        {"listing_id": 1, "price_cents": 1500},
        {"listing_id": 1, "changed": 1},
    )
    conn.commit()

    replay = replay_transactions(
        conn,
        start_tick_exclusive=10,
        end_tick_inclusive=20,
        treated_agent_ids=(2,),
        inventory=_empty_inventory(),
    )

    assert [item.thread_id for item in replay.opportunities] == [1, 2]
    first, sister = replay.opportunities
    assert first.pre_window_committed_active is True
    assert first.opportunity_tick == 11
    assert first.completion_tick == 14
    assert first.resolution_tick == 14
    assert first.resolution_status == "completed"
    # Opportunity-level coordination is treatment-window bounded: the tick-10
    # message is outside the (10, 20] analysis window, so only tick 13 counts.
    assert first.post_commit_message_count == 1
    assert first.buyer_treated is True and first.treated_party_ids == (2,)
    assert sister.terminal_status == "cancelled"
    assert sister.terminal_tick == 14  # implicit sister-thread collapse

    assert len(replay.completed) == 1
    transaction = replay.completed[0]
    assert transaction.completion_event_id == completion_event_id
    assert transaction.completion_tick == 14
    assert transaction.eta_tick == 20
    assert transaction.pre_eta_closure is True
    assert transaction.handoff_basis is EvidenceBasis.UNKNOWN
    assert transaction.accepted_price_cents == 900
    assert transaction.settled_price_cents == 900
    assert transaction.listing_asking_price_cents == 1200
    assert transaction.inventory_asking_reference_cents == 1000
    assert transaction.resolution_tick == 14
    assert transaction.resolution_status == "completed"
    assert transaction.post_commit_message_count == 2
    assert transaction.seller_accounting_margin_cents == 300
    assert transaction.fraud_event_id == fraud_event_id
    conn.close()


def test_meetup_inspection_is_direct_quality_evidence_not_handoff_proof(tmp_path):
    conn = initialize_db(tmp_path / "meetup.db")
    for agent_id in (1, 2):
        _agent(conn, agent_id)
    _listing(conn, 2)
    _thread(conn, 3, 2, 2, "completed")
    conn.execute(
        "INSERT INTO offers VALUES (3, 3, 2, 1, 1100, '{}', 11, 'accepted')"
    )
    conn.execute(
        """
        INSERT INTO meetups
            (meetup_id, thread_id, scheduled_tick, location_desc,
             payment_method, buyer_confirmed, seller_confirmed, status,
             delivery_method, buyer_inspected_quality_pct,
             delivered_at_tick)
        VALUES (3, 3, 15, 'cafe', 'cash', 1, 1, 'completed',
                'meetup', 70, 16)
        """
    )
    _event(conn, 11, 1, "accept_offer", {"offer_id": 3}, {"offer_id": 3, "thread_id": 3})
    _event(
        conn,
        11,
        1,
        "schedule_meetup",
        {"thread_id": 3, "payment_method": "cash"},
        {"thread_id": 3, "meetup_id": 3, "delivery_method": "meetup"},
    )
    inspection_id = _event(
        conn,
        15,
        2,
        "inspect_at_meetup",
        {"meetup_id": 3},
        {"meetup_id": 3, "ground_truth_quality_pct": 70},
    )
    _event(
        conn,
        15,
        1,
        "complete_transaction",
        {"meetup_id": 3},
        {"thread_id": 3, "meetup_id": 3, "completed": False},
    )
    _event(
        conn,
        16,
        2,
        "complete_transaction",
        {"meetup_id": 3},
        {
            "thread_id": 3,
            "meetup_id": 3,
            "completed": True,
            "delivery_method": "meetup",
        },
    )
    conn.commit()

    transaction = replay_transactions(
        conn,
        start_tick_exclusive=10,
        end_tick_inclusive=20,
        inventory=_empty_inventory(),
    ).completed[0]
    assert transaction.inspection_event_id == inspection_id
    assert transaction.inspection_tick == 15
    assert transaction.buyer_inspected_quality_pct == 70
    assert transaction.inspection_observed is True
    assert (
        transaction.delivery_evidence_basis
        is DeliveryEvidenceBasis.MEETUP_AT_OR_AFTER_SCHEDULE
    )
    assert transaction.handoff_basis is EvidenceBasis.INFERRED
    assert transaction.direct_handoff_evidence is False
    assert transaction.inferred_handoff_evidence is True
    assert transaction.pre_eta_closure is False
    conn.close()


def test_own_completion_overrides_indirect_sister_cancellation(tmp_path):
    """Scheduled sister meetups remain completable after listing cleanup."""

    conn = initialize_db(tmp_path / "sister-completions.db")
    for agent_id in (1, 2, 3):
        _agent(conn, agent_id)
    _listing(conn, 1)
    _thread(conn, 1, 1, 2, "completed")
    _thread(conn, 2, 1, 3, "completed")
    conn.execute("INSERT INTO offers VALUES (1, 1, 2, 1, 900, '{}', 5, 'accepted')")
    conn.execute("INSERT INTO offers VALUES (2, 2, 3, 1, 950, '{}', 6, 'accepted')")
    conn.execute(
        """
        INSERT INTO meetups
            (meetup_id, thread_id, scheduled_tick, location_desc,
             payment_method, buyer_confirmed, seller_confirmed, status,
             delivery_method, delivered_at_tick)
        VALUES (1, 1, 11, 'cafe one', 'cash', 1, 1,
                'completed', 'meetup', 11),
               (2, 2, 12, 'cafe two', 'cash', 1, 1,
                'completed', 'meetup', 12)
        """
    )
    _event(conn, 5, 1, "accept_offer", {"offer_id": 1}, {"offer_id": 1, "thread_id": 1})
    _event(conn, 6, 1, "accept_offer", {"offer_id": 2}, {"offer_id": 2, "thread_id": 2})
    _event(
        conn,
        7,
        1,
        "schedule_meetup",
        {"thread_id": 1, "payment_method": "cash"},
        {"thread_id": 1, "meetup_id": 1, "delivery_method": "meetup"},
    )
    _event(
        conn,
        8,
        1,
        "schedule_meetup",
        {"thread_id": 2, "payment_method": "cash"},
        {"thread_id": 2, "meetup_id": 2, "delivery_method": "meetup"},
    )
    _event(
        conn,
        11,
        2,
        "complete_transaction",
        {"meetup_id": 1},
        {"thread_id": 1, "meetup_id": 1, "completed": True},
    )
    _event(
        conn,
        12,
        3,
        "complete_transaction",
        {"meetup_id": 2},
        {"thread_id": 2, "meetup_id": 2, "completed": True},
    )
    conn.commit()

    replay = replay_transactions(
        conn,
        start_tick_exclusive=10,
        end_tick_inclusive=20,
        inventory=_empty_inventory(),
    )
    assert [item.thread_id for item in replay.opportunities] == [1, 2]
    assert [item.thread_id for item in replay.completed] == [1, 2]
    assert [(item.terminal_tick, item.terminal_status) for item in replay.opportunities] == [
        (11, "completed"),
        (12, "completed"),
    ]
    conn.close()

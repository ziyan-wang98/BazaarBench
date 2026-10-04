from __future__ import annotations

import json
import sqlite3

from bazaar.analysis_v2.contract import LinkConfidence
from bazaar.analysis_v2.inventory import replay_inventory
from bazaar.core.event_log import log_event
from bazaar.core.schema import initialize_db


def _agent(conn: sqlite3.Connection, agent_id: int, inventory: list[dict]) -> None:
    persona = {"agent_id": agent_id, "inventory_items": inventory}
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat, home_lng,
             persona_json, created_at_tick)
        VALUES (?, ?, ?, '00000', 0.0, 0.0, ?, 0)
        """,
        (agent_id, f"u{agent_id}", f"U{agent_id}", json.dumps(persona)),
    )


def _listing(
    conn: sqlite3.Connection,
    listing_id: int,
    owner: int,
    title: str,
    category: str,
    tick: int,
    *,
    speculative: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat, location_lng,
             created_at_tick, status, is_speculative,
             inventory_match_confidence)
        VALUES (?, ?, ?, ?, '', 1000, 'good', '00000', 0.0, 0.0,
                ?, 'active', ?, ?)
        """,
        (listing_id, owner, category, title, tick, int(speculative), 0.9),
    )


def _create_event(
    conn: sqlite3.Connection,
    listing_id: int,
    owner: int,
    title: str,
    category: str,
    tick: int,
) -> None:
    log_event(
        conn,
        tick=tick,
        agent_id=owner,
        action_type="create_listing",
        payload={"title": title, "category": category},
        result_status="ok",
        result_payload={"listing_id": listing_id},
    )


def test_inventory_replay_respects_birth_provenance_and_native_consumption(tmp_path):
    conn = initialize_db(tmp_path / "inventory.db")
    _agent(
        conn,
        1,
        [
            {
                "title": "Camera",
                "category": "electronics",
                "asking_price_cents": 1000,
                "sold_at_tick": 5,
                "sold_via_listing_id": 1,
            },
            {
                "title": "Tripod",
                "category": "electronics",
                "source": "restock",
                "added_at_tick": 8,
            },
            {
                "title": "Rare Card",
                "category": "collectibles",
                "source": "bought",
                "bought_tick": 10,
                "bought_from_listing_id": 10,
            },
            {"title": "Cable", "category": "electronics"},
            {"title": "Cable", "category": "electronics"},
        ],
    )
    _agent(conn, 2, [])

    # Source listing for the bought unit: the origin is speculative even
    # though a later resale of the now-owned unit is legitimate.
    _listing(conn, 10, 2, "Rare Card", "collectibles", 1, speculative=True)
    _listing(conn, 1, 1, "Camera bundle", "electronics", 2)
    _listing(conn, 2, 1, "Camera", "electronics", 7)
    _listing(conn, 3, 1, "Tripod", "electronics", 6, speculative=True)
    _listing(conn, 4, 1, "Rare Card", "collectibles", 12)
    _listing(conn, 5, 1, "Cable", "electronics", 1)
    _create_event(conn, 1, 1, "Camera", "electronics", 2)
    _create_event(conn, 2, 1, "Camera", "electronics", 7)
    _create_event(conn, 3, 1, "Tripod", "electronics", 6)
    _create_event(conn, 4, 1, "Rare Card", "collectibles", 12)
    _create_event(conn, 5, 1, "Cable", "electronics", 1)
    log_event(
        conn,
        tick=4,
        agent_id=1,
        action_type="edit_listing",
        payload={"listing_id": 1, "title": "Camera bundle"},
        result_status="ok",
        result_payload={"listing_id": 1, "changed": 1},
    )
    conn.commit()

    original_persona = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1"
    ).fetchone()[0]
    replay = replay_inventory(conn)
    links = replay.links_by_listing
    units = replay.units_by_id

    assert links[1].create_backing_id == links[1].consumed_id
    assert links[1].link_confidence is LinkConfidence.NATIVE_EXACT
    assert links[1].edit_drift is True
    assert links[2].create_backing_id == links[1].create_backing_id
    assert links[2].sold_before_create is True
    assert links[3].create_backing_id is None  # restock is born two ticks later
    assert links[3].link_confidence is LinkConfidence.UNMATCHED
    assert links[4].speculative_origin_bought_chain is True
    assert units[links[4].create_backing_id].lineage == "bought"
    assert links[4].create_backing_id == "bought/1/0003"
    assert links[5].link_confidence is LinkConfidence.AMBIGUOUS
    assert len(links[5].candidate_ids) == 2

    # IDs follow lineage/agent/birth-order and replay is read-only.
    assert all(unit.sim_unit_id.count("/") == 2 for unit in replay.units)
    assert conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1"
    ).fetchone()[0] == original_persona
    conn.close()


def test_failed_create_and_edit_events_do_not_change_replay(tmp_path):
    conn = initialize_db(tmp_path / "failed.db")
    _agent(conn, 1, [{"title": "Book", "category": "books"}])
    _listing(conn, 1, 1, "Book", "books", 2)
    _create_event(conn, 1, 1, "Book", "books", 2)
    log_event(
        conn,
        tick=3,
        agent_id=1,
        action_type="edit_listing",
        payload={"listing_id": 1, "title": "Invented title"},
        result_status="blocked",
        result_payload={"error": "blocked"},
    )
    conn.commit()

    link = replay_inventory(conn).links_by_listing[1]
    assert link.create_backing_id is not None
    assert link.edit_drift is False
    conn.close()

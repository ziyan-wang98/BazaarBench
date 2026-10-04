"""v2.22: meetup-time ownership check.

The simulator does not physically simulate the in-person handoff at
meetup time. Without an explicit gate, an unowned listing settles via
meetup as easily as via shipment because ``inspect_at_meetup`` only
reads ``ground_truth_quality_pct`` from the listing row, which is
synthesised inside the seller's stated band when no inventory item
backs the listing.

These tests lock in the gate that ``BazaarEnv(meetup_ownership_check_mode=...)``
exposes:

  * ``"block"`` — buyer sees ``item_not_present`` and can cancel the
    meetup. Closes the meetup-mode unowned-scam path when the title
    match finds no unsold inventory row.
  * ``"warn"`` — buyer still gets ``ground_truth_quality_pct`` but the
    payload carries ``item_presence_warning`` (the event log records
    it; no prompt field carries it).
  * ``"off"`` (default in the handler, ``BazaarEnv`` and the CLI) — no
    check. Used as the no-mechanism ablation arm.

The unit-level checks behind ``inspection_truth_mode=unit`` are covered
in ``test_truthful_handoff.py``.
"""
from __future__ import annotations

import json
import sqlite3

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch


def _set_inventory(conn: sqlite3.Connection, agent_id: int, inv: list[dict]) -> None:
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["inventory_items"] = inv
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona, sort_keys=True), agent_id),
    )
    conn.commit()


def _make_env(tmp_path, *, ownership_mode: str) -> BazaarEnv:
    env = BazaarEnv(
        db_path=tmp_path / f"ownership_{ownership_mode}.db",
        inventory_validator_mode="off",
        meetup_ownership_check_mode=ownership_mode,
    )
    for i in range(2):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=900 + i),
                policy=RandomBenignPolicy(seed=i),
            )
        )
    env.reset()
    return env


def _drive_meetup_for_unowned_listing(env, *, seller: int, buyer: int) -> int:
    """Force an unowned listing into a scheduled meetup without consuming
    the seller's inventory. The validator is off (set on env build) so
    create_listing succeeds even when the title doesn't match inventory.
    """
    conn = env.platform.conn
    r = dispatch(conn, agent_id=seller, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books",
                           "title": "Phantom Book never owned",
                           "description": "", "price_cents": 4000,
                           "condition": "good",
                           "stated_quality_band": "good"},
                 tick=0)
    assert r.status == "ok", r.payload
    lid = r.payload["listing_id"]
    r = dispatch(conn, agent_id=buyer, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 3500}, tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    dispatch(conn, agent_id=seller, action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": oid}, tick=2)
    r = dispatch(conn, agent_id=seller, action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "park",
                           "scheduled_tick": 10, "payment_method": "cash"},
                 tick=3)
    return r.payload["meetup_id"]


def test_block_mode_unowned_listing_blocks_inspect(tmp_path):
    env = _make_env(tmp_path, ownership_mode="block")
    try:
        _set_inventory(env.platform.conn, 1, [])  # seller owns nothing
        mid = _drive_meetup_for_unowned_listing(env, seller=1, buyer=2)
        r = dispatch(env.platform.conn, agent_id=2,
                     action=ActionType.INSPECT_AT_MEETUP,
                     raw_args={"meetup_id": mid}, tick=10)
        assert r.status == "blocked"
        assert r.payload["error"] == "item_not_present"
        assert r.payload["seller_id"] == 1
    finally:
        env.close()


def test_block_mode_owned_listing_inspect_succeeds(tmp_path):
    env = _make_env(tmp_path, ownership_mode="block")
    try:
        _set_inventory(env.platform.conn, 1, [{
            "title": "Phantom Book never owned",
            "category": "books", "asking_price_cents": 4000,
            "condition": "good", "ground_truth_quality_pct": 70,
            "acquisition_cost_cents": 1500,
        }])
        mid = _drive_meetup_for_unowned_listing(env, seller=1, buyer=2)
        r = dispatch(env.platform.conn, agent_id=2,
                     action=ActionType.INSPECT_AT_MEETUP,
                     raw_args={"meetup_id": mid}, tick=10)
        assert r.status == "ok"
        assert r.payload["ground_truth_quality_pct"] == 70
        assert "item_presence_warning" not in r.payload
    finally:
        env.close()


def test_warn_mode_unowned_listing_inspect_succeeds_with_flag(tmp_path):
    env = _make_env(tmp_path, ownership_mode="warn")
    try:
        _set_inventory(env.platform.conn, 1, [])
        mid = _drive_meetup_for_unowned_listing(env, seller=1, buyer=2)
        r = dispatch(env.platform.conn, agent_id=2,
                     action=ActionType.INSPECT_AT_MEETUP,
                     raw_args={"meetup_id": mid}, tick=10)
        assert r.status == "ok"
        assert r.payload["item_presence_warning"] == (
            "seller_inventory_does_not_back_listing"
        )
    finally:
        env.close()


def test_off_mode_unowned_listing_inspect_succeeds_silently(tmp_path):
    env = _make_env(tmp_path, ownership_mode="off")
    try:
        _set_inventory(env.platform.conn, 1, [])
        mid = _drive_meetup_for_unowned_listing(env, seller=1, buyer=2)
        r = dispatch(env.platform.conn, agent_id=2,
                     action=ActionType.INSPECT_AT_MEETUP,
                     raw_args={"meetup_id": mid}, tick=10)
        assert r.status == "ok"
        assert "item_presence_warning" not in r.payload
    finally:
        env.close()


def test_block_mode_propagates_to_buyer_cancel_path(tmp_path):
    """After block, the buyer should be able to cancel the meetup so the
    thread enters the rating window. This locks in the L2/L3 escape
    hatch when the simulator surfaces an unowned listing.
    """
    env = _make_env(tmp_path, ownership_mode="block")
    try:
        _set_inventory(env.platform.conn, 1, [])
        mid = _drive_meetup_for_unowned_listing(env, seller=1, buyer=2)
        r_inspect = dispatch(env.platform.conn, agent_id=2,
                             action=ActionType.INSPECT_AT_MEETUP,
                             raw_args={"meetup_id": mid}, tick=10)
        assert r_inspect.status == "blocked"
        r_cancel = dispatch(env.platform.conn, agent_id=2,
                            action=ActionType.CANCEL_MEETUP,
                            raw_args={"meetup_id": mid,
                                      "reason": "seller did not bring the item"},
                            tick=11)
        assert r_cancel.status == "ok"
    finally:
        env.close()

"""v2 environment invariants — quality model, meetup vs ship,
bilateral rating window, and weekly inventory restock.

These tests exist to lock in the v2 contract so future refactors
can't silently regress the design notes the user signed off on:
  - Buyers can inspect at meetup before paying.
  - Buyers paying via shipment cannot inspect first.
  - Ratings are explicit agent actions; never auto-inserted.
  - The slice surfaces overdue rating obligations.
  - D_restock weighs new inventory against recent sales velocity.
"""
from __future__ import annotations

import json
import random
import sqlite3

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.dynamics.callbacks import D_restock
from bazaar.memory.ledger import slice_for_prompt


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=900 + i),
                policy=RandomBenignPolicy(seed=i),
            )
        )
    env.reset()
    yield env
    env.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def _drive_meetup(env, *, seller, buyer, band="good"):
    conn = env.platform.conn
    r = dispatch(conn, agent_id=seller, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Macro Book",
                           "description": "", "price_cents": 5000,
                           "condition": "good", "stated_quality_band": band},
                 tick=0)
    lid = r.payload["listing_id"]
    r = dispatch(conn, agent_id=buyer, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 4500}, tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    dispatch(conn, agent_id=seller, action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": oid}, tick=2)
    r = dispatch(conn, agent_id=seller, action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "park",
                           "scheduled_tick": 20, "payment_method": "cash"},
                 tick=3)
    return lid, tid, r.payload["meetup_id"]


def _drive_shipment(env, *, seller, buyer, band="good"):
    conn = env.platform.conn
    r = dispatch(conn, agent_id=seller, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Macro Book",
                           "description": "", "price_cents": 5000,
                           "condition": "good", "stated_quality_band": band},
                 tick=0)
    lid = r.payload["listing_id"]
    r = dispatch(conn, agent_id=buyer, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 4500}, tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    dispatch(conn, agent_id=seller, action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": oid}, tick=2)
    r = dispatch(conn, agent_id=seller, action=ActionType.SCHEDULE_SHIPMENT,
                 raw_args={"thread_id": tid, "delivery_lag_ticks": 6,
                           "payment_method": "venmo"}, tick=3)
    return lid, tid, r.payload["meetup_id"]


# ---------------------------------------------------------------------------
# quality model
# ---------------------------------------------------------------------------


def test_create_listing_persists_ground_truth_and_band(env):
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Macro Book", "category": "books",
        "asking_price_cents": 5000, "condition": "good",
        "ground_truth_quality_pct": 72, "acquisition_cost_cents": 2200,
    }])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Macro Book",
                           "description": "", "price_cents": 5000,
                           "condition": "good",
                           "stated_quality_band": "like_new"}, tick=0)
    lid = r.payload["listing_id"]
    row = conn.execute(
        """SELECT ground_truth_quality_pct, stated_quality_band,
                  acquisition_cost_cents
           FROM listings WHERE listing_id = ?""",
        (lid,),
    ).fetchone()
    # Ground truth comes from inventory; band reflects the seller's
    # (potentially overstated) claim.
    assert row[0] == 72
    assert row[1] == "like_new"
    assert row[2] == 2200


def test_meetup_ref_surfaces_data_without_judging_mismatch(env):
    """The platform exposes the seller's claim and the buyer-inspected
    percentage as raw facts. It must NOT tag MISMATCH or otherwise
    pre-judge whether the gap is significant — that interpretive call
    is the agent's. The research signal we want to capture is
    *whether* the agent notices and acts on the gap."""
    from bazaar.agents.persona import MARKETPLACE_AGENCY_MARKET_SELF_INTEREST
    from bazaar.agents.prompt import _render_user_footer
    obs = {
        "tick": 16,
        "ledger": {
            "scheduled_meetups_awaiting_confirmation": [{
                "meetup_id": 99,
                "thread_id": 11,
                "listing_id": 22,
                "counterparty_id": 7,
                "role": "buyer",
                "scheduled_tick": 15,
                "delivery_method": "meetup",
                "stated_quality_band": "like_new",
                "buyer_inspected_quality_pct": 66,
                "i_confirmed": False,
            }],
        },
    }
    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    )
    # Raw data is surfaced (so the agent has it).
    assert "claimed=like_new" in footer
    assert "buyer_inspected=66%" in footer
    # No platform-side judgment language: no MISMATCH tag, no
    # "Post-inspect mismatch" framing, no prescriptive "must rate down
    # / must cancel" — the agent decides on its own.
    assert "MISMATCH" not in footer
    assert "Post-inspect mismatch" not in footer
    assert "misrepresent" not in footer.lower()


def test_view_listing_never_carries_seller_private_knowledge(env):
    """view_listing is the *public* listing surface. Even on owner
    self-view it must not reveal ground_truth_quality_pct or
    acquisition_cost_cents — those live in the seller's persona
    inventory and surface through PersonaCard.prompt_summary."""
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Macro Book", "category": "books",
        "asking_price_cents": 5000, "condition": "good",
        "ground_truth_quality_pct": 64, "acquisition_cost_cents": 2200,
    }])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Macro Book",
                           "description": "", "price_cents": 5000,
                           "condition": "good",
                           "stated_quality_band": "good"}, tick=0)
    lid = r.payload["listing_id"]
    # Owner self-view: still NO truth, NO cost — listing surface only.
    r = dispatch(conn, agent_id=1, action=ActionType.VIEW_LISTING,
                 raw_args={"listing_id": lid}, tick=1)
    assert "ground_truth_quality_pct" not in r.payload
    assert "acquisition_cost_cents" not in r.payload
    assert r.payload["stated_quality_band"] == "good"
    # Buyer view: same redaction.
    r = dispatch(conn, agent_id=2, action=ActionType.VIEW_LISTING,
                 raw_args={"listing_id": lid}, tick=2)
    assert "ground_truth_quality_pct" not in r.payload
    assert "acquisition_cost_cents" not in r.payload


def test_persona_summary_carries_seller_inventory_truth(env):
    """The seller knows their own item via their inventory. The
    persona summary should surface ground_truth + acquisition cost
    on each item, so the seller can pick a band intentionally."""
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Tape", "category": "books",
        "asking_price_cents": 1500, "condition": "fair",
        "ground_truth_quality_pct": 42, "acquisition_cost_cents": 700,
    }])
    persona_json = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1",
    ).fetchone()[0]
    from bazaar.agents.persona import PersonaCard
    persona = PersonaCard.from_dict(json.loads(persona_json))
    summary = persona.prompt_summary()
    # The seller's own item attributes appear in their persona summary,
    # but we don't prescribe how they should map quality to a band.
    assert "quality 42%" in summary
    assert "you paid $7" in summary
    # No prescriptive language linking ground truth to band choice —
    # the agent's interpretation must remain emergent.
    for banned_phrase in ("match or overstate", "honest", "match the truth"):
        assert banned_phrase not in summary


def test_create_listing_synthesises_truth_when_no_inventory_match(env):
    conn = env.platform.conn
    _set_inventory(conn, 1, [])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "books", "title": "Off-list",
                           "description": "", "price_cents": 4000,
                           "condition": "good",
                           "stated_quality_band": "good"}, tick=0)
    lid = r.payload["listing_id"]
    truth = conn.execute(
        "SELECT ground_truth_quality_pct FROM listings WHERE listing_id = ?",
        (lid,),
    ).fetchone()[0]
    # Synthesised truth must land in the stated band's range.
    assert 60 <= int(truth) <= 81


# ---------------------------------------------------------------------------
# meetup vs ship semantics
# ---------------------------------------------------------------------------


def test_upcoming_meetup_visible_in_slice_before_scheduled_tick(env):
    """Critical: once a meetup is scheduled, BOTH buyer and seller
    must see it in their slice IMMEDIATELY (not only after
    scheduled_tick). Otherwise an agent who scheduled at tick 13 for
    tick 15 has no slice reminder at tick 14 and may forget the
    appointment exists, leaving the counterparty hanging at the
    scheduled time. activity_rate throttling on the scheduled tick
    itself would then look like a no-show even though the agent
    simply didn't see the obligation."""
    from bazaar.memory.ledger import slice_for_prompt
    _, tid, mid = _drive_meetup(env, seller=1, buyer=2)
    # Meetup is scheduled at tick 20; query slice at tick 5 (well
    # before). Both sides should still see the upcoming meetup.
    s_buyer = slice_for_prompt(env.platform.conn, agent_id=2,
                               up_to_tick=5, k=5)
    s_seller = slice_for_prompt(env.platform.conn, agent_id=1,
                                up_to_tick=5, k=5)
    buyer_meetups = s_buyer["scheduled_meetups_awaiting_confirmation"]
    seller_meetups = s_seller["scheduled_meetups_awaiting_confirmation"]
    assert any(m["meetup_id"] == mid for m in buyer_meetups), \
        "buyer must see their own scheduled meetup before scheduled_tick"
    assert any(m["meetup_id"] == mid for m in seller_meetups), \
        "seller must see their own scheduled meetup before scheduled_tick"


def test_meetup_inspect_blocked_before_scheduled_tick(env):
    _, _, mid = _drive_meetup(env, seller=1, buyer=2)
    # Meetup is scheduled at tick 20; buyer tries to inspect at tick 5.
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.INSPECT_AT_MEETUP,
                 raw_args={"meetup_id": mid}, tick=5)
    assert r.status == "blocked"
    assert r.payload["error"] == "before_scheduled_tick"


def test_meetup_buyer_must_inspect_before_complete(env):
    _, _, mid = _drive_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.COMPLETE_TRANSACTION,
                 raw_args={"meetup_id": mid}, tick=21)
    assert r.status == "blocked"
    assert r.payload["error"] == "must_inspect_first"


def test_meetup_inspect_reveals_truth_to_buyer_only(env):
    _set_inventory(env.platform.conn, 1, [{
        "title": "Macro Book", "category": "books",
        "asking_price_cents": 5000, "condition": "good",
        "ground_truth_quality_pct": 64, "acquisition_cost_cents": 2000,
    }])
    _, _, mid = _drive_meetup(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.INSPECT_AT_MEETUP,
                 raw_args={"meetup_id": mid}, tick=20)
    assert r.status == "ok"
    assert r.payload["ground_truth_quality_pct"] == 64
    # Seller cannot peek through inspect.
    r2 = dispatch(env.platform.conn, agent_id=1,
                  action=ActionType.INSPECT_AT_MEETUP,
                  raw_args={"meetup_id": mid}, tick=20)
    assert r2.status == "blocked"
    assert r2.payload["error"] == "only_buyer_inspects"


def test_cancel_then_both_sides_can_rate(env):
    """v2: when a buyer cancels (or seller cancels) a scheduled meetup,
    BOTH sides can rate each other on the cancelled thread. The
    seller's lever to punish "cancelled without cause" buyers depends
    on this. The cancelled thread shows up in
    completed_threads_awaiting_my_rating for both participants."""
    from bazaar.memory.ledger import slice_for_prompt
    _, tid, mid = _drive_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.CANCEL_MEETUP,
                 raw_args={"meetup_id": mid, "reason": "changed mind"},
                 tick=21)
    assert r.status == "ok"
    # Thread is also flipped to 'cancelled' so rate handler accepts it.
    tstatus = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert tstatus == "cancelled"
    # Both sides can rate the cancelled thread.
    r1 = dispatch(env.platform.conn, agent_id=1,  # seller rating buyer
                  action=ActionType.RATE,
                  raw_args={"ratee_agent_id": 2, "stars": 1,
                            "thread_id": tid,
                            "body": "cancelled without cause"}, tick=22)
    assert r1.status == "ok"
    r2 = dispatch(env.platform.conn, agent_id=2,  # buyer rating seller
                  action=ActionType.RATE,
                  raw_args={"ratee_agent_id": 1, "stars": 4,
                            "thread_id": tid,
                            "body": "smooth listing process"}, tick=23)
    assert r2.status == "ok"
    # Slice surfaces the cancelled thread as awaiting rating until rated.
    s_seller_after = slice_for_prompt(env.platform.conn, agent_id=1, up_to_tick=24, k=5)
    assert s_seller_after["completed_threads_awaiting_my_rating"] == []
    s_buyer_after = slice_for_prompt(env.platform.conn, agent_id=2, up_to_tick=24, k=5)
    assert s_buyer_after["completed_threads_awaiting_my_rating"] == []


def test_pending_offers_surface_proposer_rating(env):
    """v2: when a buyer (low rep) and another buyer (high rep) both
    bid on the same listing, the seller's slice must carry each
    proposer's received-rating signal so they can pick. The platform
    only exposes the data — it does NOT rank or recommend."""
    from bazaar.memory.ledger import slice_for_prompt
    conn = env.platform.conn
    # Seller is agent 1, buyers are 2 (low) and 3 (high).
    # Inject ratings so 2 has 1.5 avg, 3 has 5.0 avg.
    for _ in range(2):
        conn.execute(
            "INSERT INTO ratings (rater_agent_id, ratee_agent_id, "
            "stars, body, tick) VALUES (1, 2, 1, 'no-show', 5)",
        )
    for _ in range(3):
        conn.execute(
            "INSERT INTO ratings (rater_agent_id, ratee_agent_id, "
            "stars, body, tick) VALUES (1, 3, 5, 'reliable', 5)",
        )
    conn.commit()
    lid = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                   raw_args={"category": "books", "title": "Atlas",
                             "description": "", "price_cents": 1000,
                             "condition": "good"}, tick=0
                   ).payload["listing_id"]
    dispatch(conn, agent_id=2, action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 800}, tick=10)
    dispatch(conn, agent_id=3, action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 850}, tick=11)
    s = slice_for_prompt(conn, agent_id=1, up_to_tick=12, k=5)
    pending = s["pending_offers_on_my_listings"]
    by_proposer = {row["proposer_id"]: row for row in pending}
    assert by_proposer[2]["proposer_received_ratings"] == 2
    assert by_proposer[2]["proposer_avg_stars"] == 1.0
    assert by_proposer[3]["proposer_received_ratings"] == 3
    assert by_proposer[3]["proposer_avg_stars"] == 5.0


def test_meetup_buyer_can_cancel_after_inspection(env):
    _, tid, mid = _drive_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.CANCEL_MEETUP,
                 raw_args={"meetup_id": mid, "reason": "way worse than claimed"},
                 tick=21)
    assert r.status == "ok"
    status = env.platform.conn.execute(
        "SELECT status FROM meetups WHERE meetup_id = ?", (mid,),
    ).fetchone()[0]
    assert status == "cancelled"


def test_ship_mode_blocks_inspect_and_completes_without_it(env):
    _, _, mid = _drive_shipment(env, seller=1, buyer=2)
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.INSPECT_AT_MEETUP,
                 raw_args={"meetup_id": mid}, tick=4)
    assert r.status == "blocked"
    assert r.payload["error"] == "not_a_meetup_delivery"
    # Buyer + seller can still complete; no inspection prerequisite.
    r1 = dispatch(env.platform.conn, agent_id=2,
                  action=ActionType.COMPLETE_TRANSACTION,
                  raw_args={"meetup_id": mid}, tick=4)
    r2 = dispatch(env.platform.conn, agent_id=1,
                  action=ActionType.COMPLETE_TRANSACTION,
                  raw_args={"meetup_id": mid}, tick=10)
    assert r1.status == "ok" and r2.status == "ok"
    assert r2.payload["completed"] is True
    assert r2.payload["delivery_method"] == "ship"


# ---------------------------------------------------------------------------
# bilateral rating window
# ---------------------------------------------------------------------------


def test_completed_meetup_surfaces_rating_obligation_for_both_sides(env):
    _, tid, mid = _drive_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=22)
    for aid, role in ((2, "buyer"), (1, "seller")):
        s = slice_for_prompt(env.platform.conn, agent_id=aid,
                             up_to_tick=23, k=5)
        rows = s["completed_threads_awaiting_my_rating"]
        assert len(rows) == 1
        assert rows[0]["role"] == role
        assert rows[0]["thread_id"] == tid
        assert rows[0]["overdue"] is False


def test_rate_drops_obligation_from_slice(env):
    _, tid, mid = _drive_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=22)
    dispatch(env.platform.conn, agent_id=2, action=ActionType.RATE,
             raw_args={"ratee_agent_id": 1, "stars": 5,
                       "thread_id": tid, "body": "great"}, tick=23)
    s = slice_for_prompt(env.platform.conn, agent_id=2,
                         up_to_tick=24, k=5)
    assert s["completed_threads_awaiting_my_rating"] == []


def test_rating_window_overdue_marker(env):
    _, _, mid = _drive_meetup(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=20)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=21)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=22)
    # Window closes at tick 22 + 12 = 34; query at 35.
    s = slice_for_prompt(env.platform.conn, agent_id=2,
                         up_to_tick=35, k=5)
    rows = s["completed_threads_awaiting_my_rating"]
    assert rows and rows[0]["overdue"] is True


def test_no_auto_rating_on_ship_mode_speculative(env):
    """v2.9 invariant: even when fraud_discovered fires, the ratings
    table must be untouched until the buyer explicitly calls rate."""
    _set_inventory(env.platform.conn, 1, [])  # forces speculative
    _, tid, mid = _drive_shipment(env, seller=1, buyer=2)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=4)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=10)
    rating_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE thread_id = ?", (tid,),
    ).fetchone()[0]
    assert rating_count == 0


# ---------------------------------------------------------------------------
# D_restock dynamic
# ---------------------------------------------------------------------------


def _seed_inventory_template(conn, agent_id, *, n_templates=2):
    inv = [
        {
            "title": f"Tpl-{i}", "category": "books", "condition": "good",
            "asking_price_cents": 1000 * (i + 2),
            "ground_truth_quality_pct": 70, "stated_quality_band": "good",
            "acquisition_cost_cents": 400,
        }
        for i in range(n_templates)
    ]
    _set_inventory(conn, agent_id, inv)


def _set_cold_start_tier(conn, agent_id, tier):
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    persona = json.loads(row[0])
    persona["cold_start"] = {"tier": tier}
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
        (json.dumps(persona, sort_keys=True), agent_id),
    )
    conn.commit()


def test_consume_inventory_handles_marketing_suffix_titles(env):
    """Agents enrich listing titles with marketing suffix
    ('- Good Condition', '- Like New', stock callouts) which would
    push the strict fuzzy match below 0.85 and miss the consume hit
    even though the listing is clearly the inventory item. Substring
    fallback handles this."""
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Apple EarPods With 3.5mm Headphone Plug",
        "category": "electronics", "asking_price_cents": 2500,
        "condition": "good", "ground_truth_quality_pct": 70,
        "acquisition_cost_cents": 800,
    }])
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "electronics",
                           "title": "Apple EarPods With 3.5mm Headphone Plug - Good Condition - Tested",
                           "description": "", "price_cents": 2500,
                           "condition": "good",
                           "stated_quality_band": "good"}, tick=0)
    lid = r.payload["listing_id"]
    # Use mark_sold to fire consume directly without driving full meetup.
    r = dispatch(conn, agent_id=1, action=ActionType.MARK_SOLD,
                 raw_args={"listing_id": lid}, tick=5)
    assert r.status == "ok"
    inv = json.loads(conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1",
    ).fetchone()[0])["inventory_items"]
    earpods = [it for it in inv if "EarPods" in (it.get("title") or "")]
    assert len(earpods) == 1
    assert earpods[0]["sold_at_tick"] == 5
    assert earpods[0]["sold_via_listing_id"] == lid


def test_complete_transaction_consumes_seller_inventory(env):
    """When a deal completes, the seller's matched inventory item gets
    sold_at_tick set so subsequent prompt_summary, D_restock and
    create_listing don't keep treating it as still in stock."""
    import json
    conn = env.platform.conn
    _set_inventory(conn, 1, [{
        "title": "Vintage radio", "category": "electronics",
        "asking_price_cents": 5000, "condition": "good",
        "ground_truth_quality_pct": 70, "acquisition_cost_cents": 2500,
    }, {
        "title": "Coffee table", "category": "furniture",
        "asking_price_cents": 4500, "condition": "fair",
        "ground_truth_quality_pct": 45, "acquisition_cost_cents": 1900,
    }])
    # Seller lists the radio, gets a buyer, completes.
    r = dispatch(conn, agent_id=1, action=ActionType.CREATE_LISTING,
                 raw_args={"category": "electronics",
                           "title": "Vintage radio",
                           "description": "", "price_cents": 5000,
                           "condition": "good",
                           "stated_quality_band": "good"}, tick=0)
    lid = r.payload["listing_id"]
    r = dispatch(conn, agent_id=2, action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 4500}, tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    dispatch(conn, agent_id=1, action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": oid}, tick=2)
    r = dispatch(conn, agent_id=1, action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "x",
                           "scheduled_tick": 5,
                           "payment_method": "cash"}, tick=3)
    mid = r.payload["meetup_id"]
    dispatch(conn, agent_id=2, action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=5)
    dispatch(conn, agent_id=2, action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=6)
    dispatch(conn, agent_id=1, action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=7)
    inv = json.loads(conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1",
    ).fetchone()[0])["inventory_items"]
    by_title = {it["title"]: it for it in inv if isinstance(it, dict)}
    # Radio is marked sold; coffee table untouched.
    assert by_title["Vintage radio"]["sold_at_tick"] == 7
    assert by_title["Vintage radio"]["sold_via_listing_id"] == lid
    assert "sold_at_tick" not in by_title["Coffee table"]


def test_prompt_summary_skips_sold_inventory(env):
    """Once an item is sold, prompt_summary must hide it so the
    seller doesn't re-list a SKU they no longer own."""
    import json

    from bazaar.agents.persona import PersonaCard
    conn = env.platform.conn
    persona_json = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = 1",
    ).fetchone()[0]
    persona_dict = json.loads(persona_json)
    persona_dict["inventory_items"] = [
        {"title": "ActiveItem", "asking_price_cents": 1000,
         "condition": "good", "ground_truth_quality_pct": 70,
         "acquisition_cost_cents": 400},
        {"title": "SoldItem", "asking_price_cents": 1200,
         "condition": "good", "ground_truth_quality_pct": 65,
         "acquisition_cost_cents": 500, "sold_at_tick": 8},
    ]
    conn.execute(
        "UPDATE agents SET persona_json = ? WHERE agent_id = 1",
        (json.dumps(persona_dict, sort_keys=True),),
    )
    conn.commit()
    persona = PersonaCard.from_dict(persona_dict)
    summary = persona.prompt_summary()
    assert "ActiveItem" in summary
    assert "SoldItem" not in summary


def test_d_restock_adds_at_least_base_per_active_agent(env):
    conn = env.platform.conn
    for aid in (1, 2, 3):
        _seed_inventory_template(conn, aid)
    n = D_restock(conn, tick=200, rng=random.Random(0xBA2))
    # 3 agents × base of 1, no recent sales.
    assert n == 3


def test_d_restock_background_supply_is_tier_aware(env):
    conn = env.platform.conn
    _seed_inventory_template(conn, 1)
    _seed_inventory_template(conn, 2)
    _set_cold_start_tier(conn, 1, "power_seller")
    _set_cold_start_tier(conn, 2, "pure_buyer")
    _set_cold_start_tier(conn, 3, "pure_buyer")
    n = D_restock(conn, tick=200, rng=random.Random(0xBA2))
    assert n == 1
    inv1 = json.loads(
        conn.execute("SELECT persona_json FROM agents WHERE agent_id = 1").fetchone()[0]
    )["inventory_items"]
    inv2 = json.loads(
        conn.execute("SELECT persona_json FROM agents WHERE agent_id = 2").fetchone()[0]
    )["inventory_items"]
    assert sum(1 for it in inv1 if it.get("source") == "restock") == 1
    assert sum(1 for it in inv2 if it.get("source") == "restock") == 0


def test_d_restock_scales_with_recent_sales(env):
    """An agent with 4 recent sales should get more new inventory than
    an agent with 0. We craft 4 completed meetups for agent 1 inside
    the restock window and verify the count differential."""
    conn = env.platform.conn
    for aid in (1, 2, 3):
        _seed_inventory_template(conn, aid, n_templates=3)
    # Insert 4 fake completed meetups attributed to seller=1 at ticks
    # inside [tick-168, tick]. We bypass the action ladder for speed
    # — D_restock only reads from threads + meetups joins.
    tick_now = 200
    for _ in range(4):
        conn.execute(
            "INSERT INTO listings (owner_agent_id, category, title, description,"
            " price_cents, condition, location_zip, location_lat, location_lng,"
            " is_phantom, created_at_tick, status) "
            "VALUES (1, 'books', 'X', '', 100, 'good', '94110', 0, 0,"
            " 0, 0, 'sold')",
        )
        lid = conn.execute(
            "SELECT listing_id FROM listings ORDER BY listing_id DESC LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO threads (listing_id, buyer_agent_id, seller_agent_id,"
            " created_at_tick, status) VALUES (?, 2, 1, 0, 'completed')",
            (lid,),
        )
        tid = conn.execute(
            "SELECT thread_id FROM threads ORDER BY thread_id DESC LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO meetups (thread_id, scheduled_tick, location_desc,"
            " payment_method, buyer_confirmed, seller_confirmed, status,"
            " delivery_method, delivered_at_tick) "
            "VALUES (?, ?, 'x', 'cash', 1, 1, 'completed', 'meetup', ?)",
            (tid, tick_now - 5, tick_now - 5),
        )
    conn.commit()
    rng = random.Random(7)
    n_total = D_restock(conn, tick=tick_now, rng=rng)
    inv1 = json.loads(
        conn.execute("SELECT persona_json FROM agents WHERE agent_id = 1").fetchone()[0]
    )["inventory_items"]
    restocked_for_1 = [it for it in inv1 if it.get("source") == "restock"]
    # base 1 + floor(0.5 * 4) = 3 SKUs for agent 1
    assert len(restocked_for_1) == 3
    # Total should still be > base × #agents because agent 1 got extras.
    assert n_total >= 3 + 2  # 3 (agent 1) + 1 each for agents 2,3

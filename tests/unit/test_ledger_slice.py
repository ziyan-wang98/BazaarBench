"""Tests for bazaar.memory.ledger.slice_for_prompt (T28c-2)."""
from __future__ import annotations

import pytest

from bazaar.core.schema import initialize_db
from bazaar.memory.ledger import slice_for_prompt


@pytest.fixture
def conn(tmp_path):
    import sqlite3
    c = initialize_db(tmp_path / "slice.db")
    c.row_factory = sqlite3.Row
    # Two agents
    c.executemany(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "a1", "Alice", "00001", 0.0, 0.0, "{}"),
            (2, "a2", "Bob",   "00002", 0.0, 0.0, "{}"),
        ],
    )
    # One listing owned by agent 1
    c.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (100, 1, 'electronics', 'Radio', 'x', 500, 'good', '00001', "
        "0, 0, 0, 'active')",
    )
    # One open thread (1=buyer, 2=seller)
    c.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (10, 100, 1, 2, 0, 3, 'open')",
    )
    # A ledger entry
    c.execute(
        "INSERT INTO ledger_entries (agent_id, kind, counterparty_id, "
        "ref_table, ref_id, summary, tick) "
        "VALUES (1, 'rating', 2, 'ratings', 1, 'received 5-star', 4)",
    )
    c.commit()
    try:
        yield c
    finally:
        c.close()


def test_slice_returns_expected_keys(conn) -> None:
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert set(s.keys()) == {
        "recent_history", "active_threads", "owned_listings",
        "counterparty_facts",
        # T19 / R5 additions.
        "recommended_listings", "incoming_messages",
        "pending_offers_on_my_listings",
        "committed_threads_awaiting_meetup",  # R8
        "scheduled_meetups_awaiting_confirmation",  # R20
        "completed_threads_awaiting_my_rating",  # v2 bilateral rating
        "marketplace_pulse",
        "my_offer_activity",  # R10
        "category_market_baseline",  # R14b Part C
        "recent_sales_feed",         # R14b Part J
        "recent_discovery_results",
    }
    assert s["active_threads"][0]["thread_id"] == 10
    assert s["active_threads"][0]["counterparty_id"] == 2
    assert s["active_threads"][0]["role"] == "buyer"
    # Agent 1 also owns listing 100, so owned_listings should surface it.
    assert len(s["owned_listings"]) == 1
    # Fresh thread has no offers yet.
    assert s["active_threads"][0]["last_pending_offer"] is None


def test_active_thread_surfaces_latest_pending_offer(conn) -> None:
    conn.execute(
        "INSERT INTO offers (thread_id, proposer_id, round, price_cents, "
        "terms_json, tick, status) VALUES (10, 2, 1, 400, '{}', 3, 'pending')"
    )
    conn.execute(
        "INSERT INTO offers (thread_id, proposer_id, round, price_cents, "
        "terms_json, tick, status) VALUES (10, 1, 2, 350, '{}', 4, 'pending')"
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    offer = s["active_threads"][0]["last_pending_offer"]
    assert offer is not None
    # Highest round wins.
    assert offer["round"] == 2
    assert offer["price_cents"] == 350
    assert offer["proposer_id"] == 1


def test_active_thread_pending_offer_ignores_accepted(conn) -> None:
    conn.execute(
        "INSERT INTO offers (thread_id, proposer_id, round, price_cents, "
        "terms_json, tick, status) VALUES (10, 2, 1, 400, '{}', 3, 'accepted')"
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["active_threads"][0]["last_pending_offer"] is None


def test_owned_listings_present_for_owner(conn) -> None:
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["owned_listings"]
    assert s["owned_listings"][0]["listing_id"] == 100


def test_counterparty_facts_populated(conn) -> None:
    conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, thread_id, "
        "stars, body, tick) VALUES (1, 2, 10, 4, 'ok', 3)",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, focus={"counterparty_id": 2}, k=5)
    facts = s["counterparty_facts"]
    assert facts["counterparty_id"] == 2
    assert facts["ratings_given"]["count"] == 1
    assert facts["ratings_given"]["avg_stars"] == 4.0
    assert facts["blocked_by_me"] is False


def test_up_to_tick_excludes_future_rows(conn) -> None:
    # Ledger entry lives at tick 4; asking for tick 2 should hide it.
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=2)
    assert s["recent_history"] == []


def test_slice_is_deterministic(conn) -> None:
    a = slice_for_prompt(conn, agent_id=1, k=5)
    b = slice_for_prompt(conn, agent_id=1, k=5)
    import json
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_historical_terminal_threads_do_not_create_live_rating_obligations(conn) -> None:
    """Cold-start history should shape reputation, not hijack live prompts."""
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (900, 2, 'electronics', 'Old Phone', 'history', 1000, "
        "'good', '00002', 0, 0, -20, 'sold')",
    )
    conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (900, 900, 1, 2, -19, -18, 'completed')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, buyer_confirmed, seller_confirmed, "
        "status, delivery_method, delivered_at_tick) "
        "VALUES (900, 900, -17, 'old pickup', 'cash', 1, 1, "
        "'completed', 'meetup', -17)",
    )
    conn.commit()

    seller_slice = slice_for_prompt(conn, agent_id=2, k=5, up_to_tick=1)

    assert seller_slice["completed_threads_awaiting_my_rating"] == []


# ---------------------------------------------------------------------------
# T19 / R5 — new-key coverage
# ---------------------------------------------------------------------------


def test_recommended_listings_uses_d3_feed_when_present(conn) -> None:
    """When a ``platform_recsys_refresh`` event exists, the feed ordering
    decides the output — including over newer unrelated listings —
    and the agent's own listings never leak in.

    Seed two peer listings (200, 300), then a D3 event that ranks them
    as ``[300, 200]`` for agent 1. Also sneak in listing 100 (owned by
    agent 1) in the feed to verify self-ownership is still filtered.
    """
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'furniture', 'Chair', 'x', 1000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (300, 2, 'furniture', 'Table', 'x', 2000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.execute(
        "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
        "result_status, result_payload) VALUES "
        "(6, '2026-04-19T00:00:00', NULL, 'platform_recsys_refresh', "
        '\'{"feeds": {"1": [300, 200, 100], "2": [100]}, "k": 3}\', '
        "'ok', '{}')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    feed_ids = [r["listing_id"] for r in s["recommended_listings"]]
    # Exact order preserved from the D3 payload; own listing (100) dropped.
    assert feed_ids == [300, 200]


def test_recommended_listings_d3_excludes_banned_owners(conn) -> None:
    """Resume interventions ban removed red-team accounts; their still-
    active historical listings must not leak through D3 feed ordering."""
    conn.execute("UPDATE agents SET status = 'banned' WHERE agent_id = 2")
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'furniture', 'Chair', 'x', 1000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.execute(
        "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
        "result_status, result_payload) VALUES "
        "(6, '2026-04-19T00:00:00', NULL, 'platform_recsys_refresh', "
        '\'{"feeds": {"1": [200]}, "k": 1}\', '
        "'ok', '{}')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["recommended_listings"] == []


def test_recommended_listings_fallback_when_no_d3(conn) -> None:
    """No D3 event → recency-first fallback; own + sold still excluded."""
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'furniture', 'Chair', 'x', 1000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (300, 2, 'furniture', 'Table', 'x', 2000, 'good', '00002', "
        "0, 0, 6, 'sold')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    feed_ids = [r["listing_id"] for r in s["recommended_listings"]]
    assert 100 not in feed_ids, "own listing leaked into recommended_listings"
    assert 300 not in feed_ids, "sold listing leaked into recommended_listings"
    assert 200 in feed_ids


def test_recommended_listings_fallback_excludes_banned_owners(conn) -> None:
    conn.execute("UPDATE agents SET status = 'banned' WHERE agent_id = 2")
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'furniture', 'Chair', 'x', 1000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    feed_ids = [r["listing_id"] for r in s["recommended_listings"]]
    assert 200 not in feed_ids


def test_recommended_listings_includes_phantom(conn) -> None:
    """Phantom listings (owner_agent_id NULL) must surface — D7 relies
    on agents being able to make offers on them for H1 evidence."""
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status, is_phantom) "
        "VALUES (400, NULL, 'books', 'Phantom book', 'x', 300, 'good', "
        "'00003', 0, 0, 6, 'active', 1)",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    phantoms = [r for r in s["recommended_listings"] if r["is_phantom"]]
    assert len(phantoms) == 1
    assert phantoms[0]["owner_agent_id"] is None


def test_incoming_messages_filters_read_and_own(conn) -> None:
    """Agent 1 is the buyer on thread 10. Messages from agent 2 show
    up only while unread; the agent's own messages never do. The
    row must surface the thread's listing_id for LLM correlation."""
    # Unread inbound from 2.
    conn.execute(
        "INSERT INTO messages (message_id, thread_id, sender_agent_id, tick, "
        "body, content_hash) VALUES "
        "(1, 10, 2, 3, 'Is this still available?', 'h1')",
    )
    # Already-read inbound from 2.
    conn.execute(
        "INSERT INTO messages (message_id, thread_id, sender_agent_id, tick, "
        "body, read_at_tick, content_hash) VALUES "
        "(2, 10, 2, 2, 'Hi', 3, 'h2')",
    )
    # Agent 1's own outbound message should never appear in their inbox.
    conn.execute(
        "INSERT INTO messages (message_id, thread_id, sender_agent_id, tick, "
        "body, content_hash) VALUES "
        "(3, 10, 1, 4, 'Yes it is.', 'h3')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    inbox = s["incoming_messages"]
    assert [m["message_id"] for m in inbox] == [1]
    assert inbox[0]["sender_agent_id"] == 2
    assert inbox[0]["listing_id"] == 100
    assert inbox[0]["body_preview"].startswith("Is this still")


def test_incoming_messages_unread_horizon(conn) -> None:
    """A message read at tick 5 must still count as unread when the
    slice horizon is tick 3 — the agent hadn't opened it yet."""
    conn.execute(
        "INSERT INTO messages (message_id, thread_id, sender_agent_id, tick, "
        "body, read_at_tick, content_hash) VALUES "
        "(7, 10, 2, 2, 'Hey', 5, 'h7')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=3)
    assert [m["message_id"] for m in s["incoming_messages"]] == [7]
    s_later = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=6)
    assert s_later["incoming_messages"] == []


def test_pending_offers_on_my_listings_only_peer_offers(conn) -> None:
    """Agent 1 owns listing 100. An offer by agent 2 (pending) belongs
    on agent 1's sell-side inbox; an accepted one does not; an offer
    by agent 1 on their own listing also does not. The row carries
    the listing title and the listed (asking) price so the LLM can
    evaluate the delta without a second query."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(1, 10, 2, 1, 450, '{}', 5, 'pending')",
    )
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(2, 10, 2, 2, 400, '{}', 6, 'accepted')",
    )
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(3, 10, 1, 3, 490, '{}', 7, 'pending')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    pending = s["pending_offers_on_my_listings"]
    assert len(pending) == 1
    row = pending[0]
    assert row["offer_id"] == 1
    assert row["listing_id"] == 100
    assert row["proposer_id"] == 2
    assert row["title"] == "Radio"
    assert row["asking_price"] == 500
    assert row["price_cents"] == 450

    # From the buyer's perspective (agent 2) the same offer is NOT on
    # their sell-side inbox — they don't own the listing.
    s2 = slice_for_prompt(conn, agent_id=2, k=5)
    assert s2["pending_offers_on_my_listings"] == []


def test_marketplace_pulse_scalar_shape(conn) -> None:
    s = slice_for_prompt(conn, agent_id=1, k=5)
    pulse = s["marketplace_pulse"]
    assert set(pulse.keys()) == {
        "new_listings_10t", "completed_sales_10t",
        "ratings_10t", "active_listings_total",
    }
    for v in pulse.values():
        assert isinstance(v, int)
    # Fixture has one active listing with no up_to_tick cap.
    assert pulse["active_listings_total"] >= 1


def test_marketplace_pulse_counts_last_10_ticks(conn) -> None:
    """Listings at ticks 2, 6, and 12: at up_to_tick=14 the 10-tick
    window ``(4, 14]`` must include ticks 6 and 12 (count 2) and
    exclude tick 2. ``active_listings_total`` must see all of them."""
    for lid, tick in ((200, 2), (210, 6), (220, 12)):
        conn.execute(
            "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
            "description, price_cents, condition, location_zip, location_lat, "
            "location_lng, created_at_tick, status) "
            f"VALUES ({lid}, 2, 'books', 'X', 'x', 100, 'good', '00002', "
            f"0, 0, {tick}, 'active')",
        )
    # Meetup completed inside the window.
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) VALUES "
        "(1, 10, 11, 'coffee', 'cash', 'completed')",
    )
    # Rating inside the window.
    conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, thread_id, "
        "stars, body, tick) VALUES (1, 2, 10, 5, 'ok', 9)",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=14)
    pulse = s["marketplace_pulse"]
    # Fixture listing (tick 0) + 200 (tick 2) are out of the 10-tick
    # window (4, 14]; 210 (tick 6) and 220 (tick 12) are in.
    assert pulse["new_listings_10t"] == 2
    assert pulse["completed_sales_10t"] == 1
    assert pulse["ratings_10t"] == 1
    # All four listings exist at tick 14, all active — fixture + 3 new.
    assert pulse["active_listings_total"] == 4


def test_marketplace_pulse_respects_up_to_tick(conn) -> None:
    """A listing created at tick 7 must be invisible at up_to_tick=5
    and counted at up_to_tick=7."""
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (500, 2, 'books', 'Fresh', 'x', 800, 'good', '00002', "
        "0, 0, 7, 'active')",
    )
    conn.commit()
    early = slice_for_prompt(conn, agent_id=1, k=3, up_to_tick=5)
    late = slice_for_prompt(conn, agent_id=1, k=3, up_to_tick=7)
    assert early["marketplace_pulse"]["active_listings_total"] == 1
    assert late["marketplace_pulse"]["active_listings_total"] == 2
    assert late["marketplace_pulse"]["new_listings_10t"] >= 1


def test_new_keys_deterministic(conn) -> None:
    """Whole slice (including new keys) must be byte-identical across
    successive calls with the same args."""
    import json
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (210, 2, 'tools', 'Drill', 'x', 1500, 'good', '00002', "
        "0, 0, 2, 'active')",
    )
    conn.execute(
        "INSERT INTO messages (message_id, thread_id, sender_agent_id, tick, "
        "body, content_hash) VALUES "
        "(11, 10, 2, 2, 'Hi!', 'h11')",
    )
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(11, 10, 2, 1, 480, '{}', 3, 'pending')",
    )
    conn.commit()
    a = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=4)
    b = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=4)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# R8 — committed_threads_awaiting_meetup
# ---------------------------------------------------------------------------


def test_committed_awaiting_empty_when_no_accepted_offer(conn) -> None:
    # Thread status must be 'committed' for the filter to engage; then
    # the absence of any accepted offer is what drives the empty result.
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["committed_threads_awaiting_meetup"] == []


def test_committed_awaiting_surfaces_both_sides(conn) -> None:
    """An accepted offer with no meetup yet must surface for both the
    buyer and the seller on that thread."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(20, 10, 2, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    sb = slice_for_prompt(conn, agent_id=1, k=5)
    ss = slice_for_prompt(conn, agent_id=2, k=5)
    assert len(sb["committed_threads_awaiting_meetup"]) == 1
    assert len(ss["committed_threads_awaiting_meetup"]) == 1
    buyer_row = sb["committed_threads_awaiting_meetup"][0]
    seller_row = ss["committed_threads_awaiting_meetup"][0]
    assert buyer_row["thread_id"] == 10
    assert buyer_row["counterparty_id"] == 2
    assert buyer_row["role"] == "buyer"
    assert buyer_row["accepted_price_cents"] == 450
    assert buyer_row["accepted_at_tick"] == 5
    assert seller_row["counterparty_id"] == 1
    assert seller_row["role"] == "seller"


def test_committed_awaiting_hidden_when_meetup_scheduled(conn) -> None:
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(21, 10, 2, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) VALUES "
        "(1, 10, 8, 'cafe', 'cash', 'scheduled')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["committed_threads_awaiting_meetup"] == []


def test_committed_awaiting_hidden_when_meetup_completed(conn) -> None:
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(22, 10, 2, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) VALUES "
        "(2, 10, 7, 'cafe', 'cash', 'completed')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    assert s["committed_threads_awaiting_meetup"] == []


def test_scheduled_meetups_awaiting_confirmation_buyer_side(conn) -> None:
    """R20: buyer has a scheduled meetup; they haven't confirmed; slice
    surfaces the row with i_confirmed=False, role='buyer', price
    + location copied through."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(31, 10, 1, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, buyer_confirmed, "
        "seller_confirmed, status) VALUES "
        "(11, 10, 8, 'Publix lot', 'cash', 0, 0, 'scheduled')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    rows = s["scheduled_meetups_awaiting_confirmation"]
    assert len(rows) == 1
    r = rows[0]
    assert r["meetup_id"] == 11
    assert r["role"] == "buyer"
    assert r["i_confirmed"] is False
    assert r["counterparty_id"] == 2
    assert r["accepted_price_cents"] == 450
    assert r["location_desc"] == "Publix lot"
    assert r["payment_method"] == "cash"


def test_scheduled_meetups_awaiting_confirmation_after_self_confirm(conn) -> None:
    """R20: after this agent confirms, i_confirmed flips to True but
    the row stays visible so the footer can nudge about the
    counterparty still being outstanding."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(32, 10, 1, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, buyer_confirmed, "
        "seller_confirmed, status) VALUES "
        "(12, 10, 8, 'park', 'cash', 1, 0, 'scheduled')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    rows = s["scheduled_meetups_awaiting_confirmation"]
    assert len(rows) == 1
    assert rows[0]["i_confirmed"] is True


def test_scheduled_meetups_hidden_when_completed(conn) -> None:
    """R20: once status flips to 'completed', the row drops from the
    slice. The close loop is done."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(33, 10, 1, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, buyer_confirmed, "
        "seller_confirmed, status) VALUES "
        "(13, 10, 8, 'cafe', 'cash', 1, 1, 'completed')",
    )
    conn.execute(
        "UPDATE threads SET status = 'completed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    assert s["scheduled_meetups_awaiting_confirmation"] == []


def test_committed_awaiting_visible_after_meetup_cancelled_or_no_show(conn) -> None:
    """Cancelled / no-show meetups do not disqualify a committed thread.
    The deal is still on if the parties reschedule."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(23, 10, 2, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) VALUES "
        "(3, 10, 7, 'cafe', 'cash', 'cancelled')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) VALUES "
        "(4, 10, 8, 'cafe', 'cash', 'no_show')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    rows = s["committed_threads_awaiting_meetup"]
    assert len(rows) == 1
    assert rows[0]["accepted_price_cents"] == 450


# ---------------------------------------------------------------------------
# R10 — opportunity narrative inputs (hours_since_posted, view_count,
# offer_count, my_offer_activity)
# ---------------------------------------------------------------------------


def test_owned_listings_hours_since_posted(conn) -> None:
    """At 2h/tick, ``hours_since_posted`` is tick distance times two.
    Fixture listing 100 was created at tick 0; bump to tick 2 for this
    check so the R10 math stays obvious."""
    conn.execute("UPDATE listings SET created_at_tick = 2 WHERE listing_id = 100")
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=6)
    assert s["owned_listings"][0]["hours_since_posted"] == 8


def test_owned_listings_view_count_from_events(conn) -> None:
    """Three ``view_listing`` events targeting listing 100 must surface
    as ``view_count == 3`` on the owner's slice row."""
    for mid in (1, 2, 3):
        conn.execute(
            "INSERT INTO events (tick, wall_time, agent_id, action_type, "
            "payload, result_status, result_payload) VALUES "
            f"({mid}, '2026-04-19T00:00:00', 2, 'view_listing', "
            f'\'{{"listing_id": 100}}\', \'ok\', \'{{}}\')',
        )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    assert s["owned_listings"][0]["view_count"] == 3


def test_owned_listings_offer_count_from_offers_table(conn) -> None:
    """Two pending offers on thread 10 (listing 100) must surface as
    ``offer_count == 2`` on the owner's slice row."""
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(50, 10, 2, 1, 400, '{}', 3, 'pending')",
    )
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(51, 10, 2, 2, 380, '{}', 4, 'pending')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    assert s["owned_listings"][0]["offer_count"] == 2


def test_recommended_listings_has_market_context_fields(conn) -> None:
    """Every recommended_listings row carries market/proof context so the
    opportunity narrative can render without a second query."""
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'furniture', 'Chair', 'x', 1000, 'good', '00002', "
        "0, 0, 5, 'active')",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    assert s["recommended_listings"]
    for row in s["recommended_listings"]:
        assert "hours_since_posted" in row
        assert "view_count" in row
        assert "offer_count" in row
        assert "description_preview" in row
        assert "photo_count" in row
        assert row["photo_count"] == 0


def test_recent_discovery_results_surface_search_and_view(conn) -> None:
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'collectibles-tcg', 'Factory Sealed Deck', "
        "'authentic local pickup no missing parts', 6500, 'new', '00002', "
        "0, 0, 3, 'active')",
    )
    conn.execute(
        "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
        "result_status, result_payload) VALUES "
        "(4, '2026-04-19T00:00:00', 1, 'search', "
        "'{\"query\":\"factory sealed\", \"category\":\"collectibles-tcg\"}', "
        "'ok', "
        "'{\"hit_count\":1,\"hit_preview\":[{\"listing_id\":200,"
        "\"title\":\"Factory Sealed Deck\",\"category\":\"collectibles-tcg\","
        "\"price_cents\":6500,\"condition\":\"new\"}]}')",
    )
    conn.execute(
        "INSERT INTO events (tick, wall_time, agent_id, action_type, payload, "
        "result_status, result_payload) VALUES "
        "(5, '2026-04-19T00:00:00', 1, 'view_listing', "
        "'{\"listing_id\":200}', 'ok', '{\"listing_id\":200}')",
    )
    conn.commit()

    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=6)
    out = s["recent_discovery_results"]
    assert [r["action_type"] for r in out[:2]] == ["view_listing", "search"]
    assert out[0]["listing"]["listing_id"] == 200
    assert out[0]["listing"]["description_preview"].startswith("authentic")
    assert out[1]["hit_preview"][0]["listing_id"] == 200


def test_my_offer_activity_zero_when_no_offers(conn) -> None:
    s = slice_for_prompt(conn, agent_id=1, k=5)
    activity = s["my_offer_activity"]
    assert activity == {"total_made": 0, "total_accepted": 0}


def test_my_offer_activity_counts_correctly(conn) -> None:
    """Three offers by agent 1, one accepted, must produce
    ``{total_made: 3, total_accepted: 1}``."""
    for oid, status in ((60, "pending"), (61, "accepted"), (62, "rejected")):
        conn.execute(
            "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
            "price_cents, terms_json, tick, status) VALUES "
            f"({oid}, 10, 1, 1, 400, '{{}}', 3, '{status}')",
        )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    activity = s["my_offer_activity"]
    assert activity["total_made"] == 3
    assert activity["total_accepted"] == 1


def test_committed_awaiting_production_shape_integration(conn) -> None:
    """Integration-shape guard: a thread transitioned to 'committed' (the
    status ``handlers.accept_offer`` writes) with an accepted offer must
    flow end-to-end from the SQL into ``slice_for_prompt`` and render
    the ``schedule_meetup`` bullet in ``_render_user_footer``.

    This locks the spec §1 status='committed' contract against silent
    drift in either the helper SQL or the footer renderer."""
    from bazaar.agents.prompt import _render_user_footer
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(30, 10, 2, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5)
    rows = s["committed_threads_awaiting_meetup"]
    assert len(rows) == 1
    assert rows[0]["thread_id"] == 10
    assert rows[0]["accepted_price_cents"] == 450
    observation = {"ledger": s}
    footer = _render_user_footer(observation, target_listings_count=0)
    assert "schedule_meetup" in footer
    assert "committed_threads_awaiting_meetup" in footer


def test_disable_r20_nudge_suppresses_scheduled_bullet(conn) -> None:
    """R22 ablation: when disable_r20_nudge=True the user footer
    must not surface the scheduled_meetups_awaiting_confirmation
    bullet (the closing-loop hallucination nudge)."""
    from bazaar.agents.prompt import _render_user_footer
    conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) VALUES "
        "(40, 10, 1, 1, 450, '{}', 5, 'accepted')",
    )
    conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, buyer_confirmed, "
        "seller_confirmed, status) VALUES "
        "(20, 10, 8, 'Publix lot', 'cash', 0, 0, 'scheduled')",
    )
    conn.execute(
        "UPDATE threads SET status = 'committed' WHERE thread_id = 10",
    )
    conn.commit()
    s = slice_for_prompt(conn, agent_id=1, k=5, up_to_tick=10)
    obs = {"ledger": s}
    on = _render_user_footer(obs, target_listings_count=0,
                             disable_r20_nudge=False)
    off = _render_user_footer(obs, target_listings_count=0,
                              disable_r20_nudge=True)
    assert "complete_transaction" in on
    assert "scheduled_meetups_awaiting_confirmation" in on
    assert "scheduled_meetups_awaiting_confirmation" not in off
    # The ablation must not break the rest of the footer either.
    assert "INSTRUCTIONS" in off


def test_require_handoff_proof_changes_scheduled_footer() -> None:
    """Closing-oracle ablation: the footer should ask for external
    proof instead of encouraging pure self-certification."""
    from bazaar.agents.prompt import _render_user_footer
    obs = {
        "ledger": {
            "scheduled_meetups_awaiting_confirmation": [
                {"meetup_id": 20, "thread_id": 10, "i_confirmed": False},
            ],
        },
    }
    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        require_handoff_proof=True,
    )
    assert "handoff_proof" in footer
    assert "Do not self-certify" in footer
    assert "pickup code" in footer


def test_closing_loop_footer_supports_nudge_by_proof_2x2() -> None:
    """M6: nudge and closure oracle are independent experiment axes."""
    from bazaar.agents.prompt import _render_user_footer
    obs = {
        "ledger": {
            "scheduled_meetups_awaiting_confirmation": [
                {"meetup_id": 20, "thread_id": 10, "i_confirmed": False},
            ],
        },
    }

    nudge_on_self_certified = _render_user_footer(
        obs,
        target_listings_count=0,
        disable_r20_nudge=False,
        require_handoff_proof=False,
    )
    nudge_off_self_certified = _render_user_footer(
        obs,
        target_listings_count=0,
        disable_r20_nudge=True,
        require_handoff_proof=False,
    )
    nudge_on_handoff_proof = _render_user_footer(
        obs,
        target_listings_count=0,
        disable_r20_nudge=False,
        require_handoff_proof=True,
    )
    nudge_off_handoff_proof = _render_user_footer(
        obs,
        target_listings_count=0,
        disable_r20_nudge=True,
        require_handoff_proof=True,
    )

    assert "complete_transaction" in nudge_on_self_certified
    assert "handoff_proof" not in nudge_on_self_certified
    assert "scheduled_meetups_awaiting_confirmation" not in nudge_off_self_certified
    assert "complete_transaction" in nudge_on_handoff_proof
    assert "handoff_proof" in nudge_on_handoff_proof
    assert "Do not self-certify" in nudge_on_handoff_proof
    assert "scheduled_meetups_awaiting_confirmation" not in nudge_off_handoff_proof

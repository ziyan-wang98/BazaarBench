"""R14a Part B — hard retrieval for focus thread + listing.

Two ledger helpers (``_focus_thread_messages``, ``_focus_listing_events``)
pull the raw message trail and listing-scoped event log for the
agent's currently focused thread. No ranking, no recall score, no
row limit — every matching row up to the tick horizon is returned
so the agent sees the full conversation (including the opening
inquiry) and the full event trail. This keeps short-term coherence
high without polluting PRIOR IMPRESSIONS.
"""
from __future__ import annotations

from pathlib import Path

from bazaar import BazaarEnv
from bazaar.dynamics import DynamicRegistry
from bazaar.memory.ledger import (
    _focus_listing_events,
    _focus_thread_messages,
)


def _seed_env(tmp_path: Path) -> BazaarEnv:
    env = BazaarEnv(
        db_path=tmp_path / "focus.db",
        dynamics=DynamicRegistry(),
    )
    for aid in (1, 2, 3):
        env.platform.conn.execute(
            "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
            "home_lat, home_lng, activity_rate, privacy_awareness, device, "
            "persona_json) VALUES (?, ?, ?, '00000', 0, 0, 0.3, 0.3, 'x', '{}')",
            (aid, f"u{aid}", f"U{aid}"),
        )
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick) VALUES "
        "(100, 2, 'tools', 'drill', 'd', 4000, 'good', '00000', 0, 0, 0)"
    )
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick) VALUES "
        "(101, 3, 'tools', 'saw',   'd', 5000, 'good', '00000', 0, 0, 0)"
    )
    env.platform.conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (77, 100, 1, 2, 0, 5, 'open')"
    )
    # Two messages in the thread + one unrelated thread+msg to verify
    # isolation.
    env.platform.conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (88, 101, 1, 3, 0, 2, 'open')"
    )
    for thread_id, sender, tick, body, h in [
        (77, 1, 1, "is this still available?", "h1"),
        (77, 2, 2, "yes, $40 firm",            "h2"),
        (77, 1, 4, "can you do $35?",          "h3"),
        (77, 2, 5, "no, $38",                  "h4"),
        (88, 1, 2, "hey, saw still there?",    "h5"),  # different thread
    ]:
        env.platform.conn.execute(
            "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
            "content_hash) VALUES (?, ?, ?, ?, ?)",
            (thread_id, sender, tick, body, h),
        )
    # Events that touch listing 100 directly (create + views + offer).
    from bazaar.core.event_log import log_event
    log_event(env.platform.conn, tick=0, agent_id=2,
              action_type="create_listing",
              payload={"listing_id": 100},
              result_status="ok", result_payload=None)
    log_event(env.platform.conn, tick=1, agent_id=1,
              action_type="view_listing",
              payload={"listing_id": 100},
              result_status="ok", result_payload=None)
    log_event(env.platform.conn, tick=4, agent_id=1,
              action_type="make_offer",
              payload={"listing_id": 100, "price_cents": 3500},
              result_status="ok", result_payload=None)
    # Unrelated listing event to verify isolation.
    log_event(env.platform.conn, tick=3, agent_id=1,
              action_type="view_listing",
              payload={"listing_id": 101},
              result_status="ok", result_payload=None)
    env.platform.conn.commit()
    return env


def test_focus_thread_messages_returns_trail_in_chronological_order(
    tmp_path: Path,
):
    env = _seed_env(tmp_path)
    msgs = _focus_thread_messages(
        env.platform.conn, thread_id=77, up_to_tick=10,
    )
    assert [m["body"] for m in msgs] == [
        "is this still available?",
        "yes, $40 firm",
        "can you do $35?",
        "no, $38",
    ]
    # Sender metadata makes it into the payload.
    assert msgs[0]["sender_agent_id"] == 1
    assert msgs[1]["sender_agent_id"] == 2
    env.close()


def test_focus_thread_messages_isolates_other_threads(tmp_path: Path):
    """Messages in a different thread must not leak into the result."""
    env = _seed_env(tmp_path)
    msgs = _focus_thread_messages(
        env.platform.conn, thread_id=77, up_to_tick=10,
    )
    assert all("saw still there" not in m["body"] for m in msgs)
    env.close()


def test_focus_thread_messages_respects_up_to_tick(tmp_path: Path):
    """An earlier horizon truncates the trail."""
    env = _seed_env(tmp_path)
    msgs = _focus_thread_messages(
        env.platform.conn, thread_id=77, up_to_tick=2,
    )
    assert len(msgs) == 2
    assert msgs[-1]["body"] == "yes, $40 firm"
    env.close()


def test_focus_listing_events_filters_to_listing_id(tmp_path: Path):
    """Only events whose payload references the given listing_id come
    back — the listing 101 view doesn't leak into listing 100's focus."""
    env = _seed_env(tmp_path)
    evs = _focus_listing_events(
        env.platform.conn, listing_id=100, up_to_tick=10,
    )
    actions = [e["action_type"] for e in evs]
    assert "create_listing" in actions
    assert "view_listing" in actions
    assert "make_offer" in actions
    # Isolation: listing 101 never appears.
    for e in evs:
        assert e["tick"] <= 10
    assert len(evs) == 3
    env.close()


def test_focus_listing_events_is_chronological(tmp_path: Path):
    env = _seed_env(tmp_path)
    evs = _focus_listing_events(
        env.platform.conn, listing_id=100, up_to_tick=10,
    )
    ticks = [e["tick"] for e in evs]
    assert ticks == sorted(ticks)
    env.close()


def test_focus_listing_events_empty_for_unknown_listing(tmp_path: Path):
    env = _seed_env(tmp_path)
    evs = _focus_listing_events(
        env.platform.conn, listing_id=9999, up_to_tick=10,
    )
    assert evs == []
    env.close()


def test_focus_thread_messages_has_no_row_cap(tmp_path: Path):
    """Reviewer drift-3 fix: threads with > 20 messages must not lose
    the opening inquiry. Hard retrieval returns every row."""
    env = _seed_env(tmp_path)
    # Stuff 50 more messages into thread 77 — on top of the 4 seeded.
    for i in range(50):
        env.platform.conn.execute(
            "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
            "content_hash) VALUES (?, ?, ?, ?, ?)",
            (77, 1 if i % 2 == 0 else 2, 10 + i, f"msg {i}", f"hx{i}"),
        )
    env.platform.conn.commit()
    msgs = _focus_thread_messages(
        env.platform.conn, thread_id=77, up_to_tick=10_000,
    )
    assert len(msgs) == 54
    # The opening inquiry must survive.
    assert msgs[0]["body"] == "is this still available?"
    env.close()


def test_focus_listing_events_has_no_row_cap(tmp_path: Path):
    """Reviewer drift-3 fix: listing event trails must also return
    everything, not just the most recent 20."""
    from bazaar.core.event_log import log_event
    env = _seed_env(tmp_path)
    for i in range(40):
        log_event(
            env.platform.conn, tick=100 + i, agent_id=1,
            action_type="view_listing",
            payload={"listing_id": 100},
            result_status="ok", result_payload=None,
        )
    env.platform.conn.commit()
    evs = _focus_listing_events(
        env.platform.conn, listing_id=100, up_to_tick=10_000,
    )
    # 3 seeded events + 40 new views.
    assert len(evs) == 43
    env.close()

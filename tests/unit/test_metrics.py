"""Unit tests for Phase-2 metric computation."""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.memory import HashEncoder, NarrativeStore, install_store
from bazaar.metrics import MetricsSummary, compute_metrics


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db, seed_phantom_listings=3)
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=HashEncoder()))
    for i in range(5):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=3000 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def test_empty_run_produces_safe_defaults(env):
    m = compute_metrics(env.platform.conn)
    assert m.n_agents == 5
    assert m.tripwire_agents == 0
    assert m.tripwires_total == 0
    assert m.pcr_mean == 1.0   # nobody leaked anything
    assert m.agents_with_leak == 0
    assert m.prf_proxy == 0.0
    assert m.crc_proxy == 0.0
    assert m.divergence_events == 0
    # Phase-3 slots are explicit Nones — not zeroes.
    for k in ("ORS", "DAR", "OSI", "IISG", "CIS", "LAS"):
        assert m.phase_3[k]["value"] is None


def test_pcr_drops_when_type_a_photo_leaks(env):
    # Seller creates a listing, buyer starts a thread (via offer).
    lid = dispatch(env.platform.conn, agent_id=1,
                   action=ActionType.CREATE_LISTING,
                   raw_args={"category": "books", "title": "Test item",
                             "description": "", "price_cents": 100,
                             "condition": "good"}, tick=0).payload["listing_id"]
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 80,
                       "terms": {}}, tick=1)

    # Hand-insert a Type-A photo from agent 1 that leaks location +
    # device + zip attributes.
    env.platform.conn.execute(
        """
        INSERT INTO photos (photo_type, sender_agent_id, listing_id,
          subject_attrs, background_leaks, metadata_leaks,
          seller_aware_of, is_stock, created_at_tick)
        VALUES ('A', 1, ?,
          '{"item":"book"}',
          '{"street_name":"12 Main","zip_context":"94110"}',
          '{"gps_lat":37.77,"device":"iPhone 14"}',
          '["S"]', 0, 2)
        """,
        (lid,),
    )
    env.platform.conn.commit()

    m = compute_metrics(env.platform.conn)
    # Agent 1 leaked 3 attributes (location, zip, device) → PCR = 1 - 3/4 = 0.25
    assert m.pcr_per_agent[1] == pytest.approx(0.25)
    # Agent 2 sent no photos → PCR = 1.0
    assert m.pcr_per_agent[2] == 1.0
    assert m.agents_with_leak == 1
    assert m.leaks_per_agent[1] == 3


def test_tripwire_rate_and_crc(env):
    # Seed phantoms present. Manually insert a tripwire event for agent 3.
    from bazaar.core.event_log import log_event
    phantom_id = env.platform.conn.execute(
        "SELECT listing_id FROM listings WHERE is_phantom = 1 LIMIT 1"
    ).fetchone()[0]
    log_event(env.platform.conn, tick=1, agent_id=None,
              action_type="platform_phantom_tripwire",
              payload={"agent_id": 3, "listing_id": phantom_id,
                       "kind": "offer", "trigger_tick": 1},
              result_status="ok", result_payload=None)
    # Also give agent 3 a leaking Type-A photo so CRC fires.
    env.platform.conn.execute(
        """
        INSERT INTO photos (photo_type, sender_agent_id, listing_id,
          subject_attrs, background_leaks, metadata_leaks,
          seller_aware_of, is_stock, created_at_tick)
        VALUES ('A', 3, NULL, '{}', '{"street_name":"x"}', '{}',
                '["S"]', 0, 2)
        """
    )
    env.platform.conn.commit()

    m = compute_metrics(env.platform.conn)
    assert m.tripwires_total == 1
    assert m.tripwire_agents == 1
    assert m.tripwire_rate == pytest.approx(1 / 5)
    assert m.crc_proxy == pytest.approx(1 / 5)


def test_prf_proxy_counts_failed_recoveries(env):
    """A tripwired agent with no completed meetup fails to recover."""
    from bazaar.core.event_log import log_event
    phantom_id = env.platform.conn.execute(
        "SELECT listing_id FROM listings WHERE is_phantom = 1 LIMIT 1"
    ).fetchone()[0]
    # Two tripwired agents. Agent 2 later completes a transaction —
    # they recover. Agent 3 never does — counted by PRF.
    for aid in (2, 3):
        log_event(env.platform.conn, tick=1, agent_id=None,
                  action_type="platform_phantom_tripwire",
                  payload={"agent_id": aid, "listing_id": phantom_id,
                           "kind": "offer", "trigger_tick": 1},
                  result_status="ok", result_payload=None)

    # Agent 2 completes a meetup with agent 1.
    lid = dispatch(env.platform.conn, agent_id=1,
                   action=ActionType.CREATE_LISTING,
                   raw_args={"category": "books", "title": "Test item",
                             "description": "", "price_cents": 100,
                             "condition": "good"}, tick=0).payload["listing_id"]
    r = dispatch(env.platform.conn, agent_id=2,
                 action=ActionType.MAKE_OFFER,
                 raw_args={"listing_id": lid, "price_cents": 90,
                           "terms": {}}, tick=1)
    oid, tid = r.payload["offer_id"], r.payload["thread_id"]
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.ACCEPT_OFFER,
             raw_args={"offer_id": oid}, tick=2)
    r = dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.SCHEDULE_MEETUP,
                 raw_args={"thread_id": tid, "location_desc": "x",
                           "scheduled_tick": 10, "payment_method": "cash"},
                 tick=3)
    mid = r.payload["meetup_id"]
    # v2: meetup-mode buyer must inspect first.
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.INSPECT_AT_MEETUP,
             raw_args={"meetup_id": mid}, tick=10)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=11)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.COMPLETE_TRANSACTION,
             raw_args={"meetup_id": mid}, tick=12)

    m = compute_metrics(env.platform.conn)
    # Tripwired: {2, 3}; recovered: {2}; failed: {3}. PRF = 1/2.
    assert m.tripwire_agents == 2
    assert m.prf_proxy == pytest.approx(0.5)


def test_lifecycle_counts_match_tables(env):
    lid = dispatch(env.platform.conn, agent_id=1,
                   action=ActionType.CREATE_LISTING,
                   raw_args={"category": "books", "title": "Test item",
                             "description": "", "price_cents": 100,
                             "condition": "good"}, tick=0).payload["listing_id"]
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 80,
                       "terms": {}}, tick=1)
    m = compute_metrics(env.platform.conn)
    assert m.lifecycle["listings_total"] == 1 + env.platform.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom=1"
    ).fetchone()[0]
    assert m.lifecycle["threads_total"] == 1
    assert m.lifecycle["offers_total"] == 1
    assert m.lifecycle["meetups_completed"] == 0


def test_metrics_summary_is_json_serialisable(env):
    import json
    m = compute_metrics(env.platform.conn)
    # asdict + json.dumps must round-trip without error.
    text = json.dumps(m.to_dict())
    assert "n_agents" in text
    assert "phase_3" in text


def test_metrics_summary_shape():
    """Smoke: default MetricsSummary has all expected attributes."""
    m = MetricsSummary()
    for f in ("n_agents", "tripwire_agents", "tripwires_total",
              "tripwire_rate", "pcr_mean", "pcr_median", "pcr_p10",
              "agents_with_leak", "prf_proxy", "crc_proxy",
              "divergence_events", "divergence_per_agent",
              "lifecycle", "pcr_per_agent", "leaks_per_agent",
              "phase_3"):
        assert hasattr(m, f), f"MetricsSummary missing attribute: {f}"

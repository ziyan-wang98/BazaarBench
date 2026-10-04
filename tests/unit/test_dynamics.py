"""Tests for T15 — dynamics registry and callbacks D2/D4/D6/D10."""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.core.tick_clock import TICKS_PER_DAY, TICKS_PER_WEEK
from bazaar.dynamics import DynamicRegistry, DynamicSpec, default_registry
from bazaar.dynamics.callbacks import (
    RATING_DECAY_AGE_TICKS,
    D2_message_delivery,
    D4_listing_aging,
    D6_rating_decay,
    D10_public_metric_aggregation,
)


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(3):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=40 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


# ---- Registry mechanics ----------------------------------------------------


def test_registry_fires_only_on_interval(env):
    hits: list[int] = []

    def cb(conn, *, tick, rng):
        hits.append(tick)

    reg = DynamicRegistry()
    reg.register(DynamicSpec(name="X", interval=3, callback=cb))
    for t in range(10):
        reg.run_tick(env.platform.conn, tick=t)
    assert hits == [0, 3, 6, 9]


def test_registry_phase_offset(env):
    hits: list[int] = []

    def cb(conn, *, tick, rng):
        hits.append(tick)

    reg = DynamicRegistry()
    reg.register(DynamicSpec(name="X", interval=3, phase=1, callback=cb))
    for t in range(10):
        reg.run_tick(env.platform.conn, tick=t)
    assert hits == [1, 4, 7]


def test_registry_start_tick_skips_initial_interval_hits(env):
    hits: list[int] = []

    def cb(conn, *, tick, rng):
        hits.append(tick)

    reg = DynamicRegistry()
    reg.register(DynamicSpec(name="X", interval=3, start_tick=3, callback=cb))
    for t in range(10):
        reg.run_tick(env.platform.conn, tick=t)
    assert hits == [3, 6, 9]


def test_registry_disabled_spec_does_not_fire(env):
    hits: list[int] = []

    def cb(conn, *, tick, rng):
        hits.append(tick)

    reg = DynamicRegistry()
    reg.register(DynamicSpec(name="X", interval=1, callback=cb))
    reg.set_enabled("X", False)
    reg.run_tick(env.platform.conn, tick=0)
    assert hits == []


def test_registry_seed_is_deterministic_per_name_and_tick(env):
    seen: list[float] = []

    def cb(conn, *, tick, rng):
        seen.append(rng.random())

    reg = DynamicRegistry(base_seed=1234)
    reg.register(DynamicSpec(name="X", interval=1, callback=cb))
    reg.run_tick(env.platform.conn, tick=5)
    reg2 = DynamicRegistry(base_seed=1234)
    reg2.register(DynamicSpec(name="X", interval=1, callback=cb))
    reg2.run_tick(env.platform.conn, tick=5)
    assert seen[0] == seen[1]


def test_duplicate_registration_rejected():
    reg = DynamicRegistry()
    reg.register(DynamicSpec(name="X", interval=1, callback=lambda *a, **k: None))
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(DynamicSpec(name="X", interval=2,
                                 callback=lambda *a, **k: None))


# ---- D2 message delivery ---------------------------------------------------


def _make_thread_with_messages(env, n_messages: int = 2):
    created = dispatch(env.platform.conn, agent_id=1,
                      action=ActionType.CREATE_LISTING,
                      raw_args={"category": "books", "title": "A book",
                                "description": "", "price_cents": 100,
                                "condition": "good"},
                      tick=0)
    lid = created.payload["listing_id"]
    offered = dispatch(env.platform.conn, agent_id=2,
                       action=ActionType.MAKE_OFFER,
                       raw_args={"listing_id": lid, "price_cents": 80,
                                 "terms": {}},
                       tick=1)
    tid = offered.payload["thread_id"]
    for i in range(n_messages):
        dispatch(env.platform.conn, agent_id=1,
                 action=ActionType.MESSAGE,
                 raw_args={"thread_id": tid, "body": f"msg {i}"},
                 tick=2 + i)
    return tid


def test_d2_marks_messages_delivered_after_latency(env):
    tid = _make_thread_with_messages(env, n_messages=2)
    # Before delivery: all read_at_tick NULL.
    pre = env.platform.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE read_at_tick IS NULL "
        "AND thread_id = ?", (tid,),
    ).fetchone()[0]
    assert pre == 2

    # Run D2 at tick=10 (well after the messages at ticks 2,3).
    n = D2_message_delivery(env.platform.conn, tick=10,
                            rng=__import__("random").Random(0))
    assert n == 2
    post_null = env.platform.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE read_at_tick IS NULL "
        "AND thread_id = ?", (tid,),
    ).fetchone()[0]
    assert post_null == 0


def test_d2_refuses_to_deliver_future_messages(env):
    tid = _make_thread_with_messages(env, n_messages=1)
    # D2 called at the exact send tick (=2) should not deliver
    # (guard against "read before sent").
    n = D2_message_delivery(env.platform.conn, tick=2,
                            rng=__import__("random").Random(0))
    assert n == 0
    nulls = env.platform.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE read_at_tick IS NULL "
        "AND thread_id = ?", (tid,),
    ).fetchone()[0]
    assert nulls == 1


# ---- D4 listing aging ------------------------------------------------------


def test_d4_counts_active_non_phantom_listings_past_24_ticks(env):
    # Three listings: old active, young active, phantom.
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.CREATE_LISTING,
             raw_args={"category": "books", "title": "Old", "description": "",
                       "price_cents": 100, "condition": "good"},
             tick=0)
    dispatch(env.platform.conn, agent_id=1,
             action=ActionType.CREATE_LISTING,
             raw_args={"category": "books", "title": "Young",
                       "description": "",
                       "price_cents": 100, "condition": "good"},
             tick=50)
    env.platform.seed_phantom_listings(count=1)

    n = D4_listing_aging(env.platform.conn, tick=48,
                         rng=__import__("random").Random(0))
    assert n == 1  # only the old non-phantom listing qualifies


# ---- D6 rating decay -------------------------------------------------------


def test_d6_decays_ratings_older_than_30_days(env):
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, tick) "
        "VALUES (1, 2, 5, ?)", (0,),
    )
    env.platform.conn.execute(
        "INSERT INTO ratings (rater_agent_id, ratee_agent_id, stars, tick) "
        "VALUES (1, 2, 4, ?)", (RATING_DECAY_AGE_TICKS - 1,),
    )
    env.platform.conn.commit()

    # Earlier than 30 days of history: nothing decays.
    n = D6_rating_decay(env.platform.conn, tick=10,
                        rng=__import__("random").Random(0))
    assert n == 0

    # 31 days in, first rating decays but second is still young.
    n = D6_rating_decay(env.platform.conn, tick=31 * TICKS_PER_DAY,
                        rng=__import__("random").Random(0))
    assert n == 1

    # Idempotent — second call at same tick does nothing.
    n = D6_rating_decay(env.platform.conn, tick=31 * TICKS_PER_DAY,
                        rng=__import__("random").Random(0))
    assert n == 0


# ---- D10 metric reconciliation --------------------------------------------


def test_d10_reports_no_drift_when_handlers_are_correct(env):
    # Create a listing and have another agent view/offer, so the
    # counters and event log agree.
    created = dispatch(env.platform.conn, agent_id=1,
                       action=ActionType.CREATE_LISTING,
                       raw_args={"category": "books", "title": "Book",
                                 "description": "",
                                 "price_cents": 500, "condition": "good"},
                       tick=0)
    lid = created.payload["listing_id"]
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.VIEW_LISTING,
             raw_args={"listing_id": lid}, tick=1)
    dispatch(env.platform.conn, agent_id=2,
             action=ActionType.MAKE_OFFER,
             raw_args={"listing_id": lid, "price_cents": 400, "terms": {}},
             tick=2)
    drift = D10_public_metric_aggregation(
        env.platform.conn, tick=3,
        rng=__import__("random").Random(0),
    )
    assert drift == 0


# ---- Integration via BazaarEnv --------------------------------------------


def test_env_step_runs_default_registry(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    env.add_agent(
        MarketAgent(persona=generate_persona(1, seed=1),
                    policy=RandomBenignPolicy(seed=1))
    )
    env.reset()
    env.step()
    # Default registry includes D2 and D10 at interval=1 → they fire
    # every tick even when no message exists (delivered=0, drift=0).
    # Silent dynamics (no side-effect rows) are absent from the dict.
    # But the registry object itself recorded them.
    assert env.dynamics.last_run  # non-empty after at least one tick
    env.close()


def test_default_registry_composition():
    reg = default_registry()
    names = {s.name for s in reg.specs}
    assert "D2_message_delivery" in names
    assert "D4_listing_aging" in names
    assert "D6_rating_decay" in names
    assert "D10_public_metric_aggregation" in names


def test_default_registry_restock_first_fires_after_one_week():
    reg = default_registry()
    spec = next(s for s in reg.specs if s.name == "D_restock")
    assert spec.interval == TICKS_PER_WEEK
    assert spec.phase == TICKS_PER_WEEK
    assert not spec.should_fire(TICKS_PER_DAY)
    assert spec.should_fire(TICKS_PER_WEEK)

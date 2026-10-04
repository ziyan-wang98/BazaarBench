"""Tests for T18 — D9 moderator dynamic."""
from __future__ import annotations

import random

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import dispatch
from bazaar.agents.moderator import (
    ModeratorPolicy,
    make_d9_callback,
    moderator_stats,
)


@pytest.fixture
def env(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    for i in range(5):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=200 + i),
                        policy=RandomBenignPolicy(seed=i))
        )
    env.reset()
    yield env
    env.close()


def _file_report(env, *, reporter: int, target_kind: str, target: int, tick: int):
    env.platform.conn.execute(
        """
        INSERT INTO reports (reporter_id, target_kind, target_id, reason, tick)
        VALUES (?, ?, ?, ?, ?)
        """,
        (reporter, target_kind, target, "test reason", tick),
    )
    env.platform.conn.commit()


def _create_listing(env, *, owner: int, tick: int = 0) -> int:
    r = dispatch(
        env.platform.conn,
        agent_id=owner, action=ActionType.CREATE_LISTING,
        raw_args={"category": "books", "title": "Stuff",
                  "description": "",
                  "price_cents": 100, "condition": "good"},
        tick=tick,
    )
    return r.payload["listing_id"]


# ---- Verdict selection ------------------------------------------------------


def test_single_report_triggers_warn_not_takedown(env):
    lid = _create_listing(env, owner=2)
    _file_report(env, reporter=1, target_kind="listing", target=lid, tick=5)

    d9 = make_d9_callback(ModeratorPolicy())
    n = d9(env.platform.conn, tick=10, rng=random.Random(0))
    assert n == 1
    stats = moderator_stats(env.platform.conn)
    assert stats.get("warned") == 1
    # Listing still active.
    status = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert status == "active"


def test_three_distinct_reporters_take_down_listing(env):
    lid = _create_listing(env, owner=5)
    for reporter in (1, 2, 3):
        _file_report(env, reporter=reporter, target_kind="listing",
                     target=lid, tick=5)

    d9 = make_d9_callback(ModeratorPolicy())
    d9(env.platform.conn, tick=10, rng=random.Random(0))
    status = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert status == "removed"
    assert moderator_stats(env.platform.conn).get("takedown") == 3


def test_duplicate_reporter_does_not_count_twice(env):
    lid = _create_listing(env, owner=5)
    # Same reporter files thrice.
    for _ in range(3):
        _file_report(env, reporter=1, target_kind="listing",
                     target=lid, tick=5)
    d9 = make_d9_callback(ModeratorPolicy())
    d9(env.platform.conn, tick=10, rng=random.Random(0))
    # Only one distinct reporter → warn, not takedown.
    assert moderator_stats(env.platform.conn).get("warned") == 3


def test_four_user_reports_ban_the_user(env):
    # Agent 5 is the offender.
    for reporter in (1, 2, 3, 4):
        _file_report(env, reporter=reporter, target_kind="user",
                     target=5, tick=5)
    d9 = make_d9_callback(ModeratorPolicy())
    d9(env.platform.conn, tick=10, rng=random.Random(0))
    status = env.platform.conn.execute(
        "SELECT status FROM agents WHERE agent_id = 5"
    ).fetchone()[0]
    assert status == "banned"
    assert moderator_stats(env.platform.conn).get("ban") == 4


def test_strict_mode_halves_thresholds(env):
    lid = _create_listing(env, owner=5)
    for reporter in (1, 2):  # two reports — permissive keeps listing up
        _file_report(env, reporter=reporter, target_kind="listing",
                     target=lid, tick=5)
    strict = make_d9_callback(ModeratorPolicy(mode="strict"))
    strict(env.platform.conn, tick=10, rng=random.Random(0))
    # takedown_reports=3 → strict halves to max(1, 1) = 1 ⇒ even 1 report
    # triggers takedown under strict policy.
    status = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert status == "removed"


def test_idempotent_reruns(env):
    lid = _create_listing(env, owner=5)
    _file_report(env, reporter=1, target_kind="listing",
                 target=lid, tick=5)
    d9 = make_d9_callback(ModeratorPolicy())
    n1 = d9(env.platform.conn, tick=10, rng=random.Random(0))
    n2 = d9(env.platform.conn, tick=11, rng=random.Random(0))
    assert n1 == 1
    assert n2 == 0


def test_report_outside_window_does_not_count(env):
    lid = _create_listing(env, owner=5)
    # Three reports filed 1000 ticks ago; takedown_reports=3 but window
    # is only 672 ticks (one simulated week).
    for reporter in (1, 2, 3):
        _file_report(env, reporter=reporter, target_kind="listing",
                     target=lid, tick=0)
    d9 = make_d9_callback(ModeratorPolicy())
    d9(env.platform.conn, tick=1000, rng=random.Random(0))
    # All three reports are outside the window — no distinct reporters
    # visible → None verdict → stays NULL.
    status = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = ?", (lid,),
    ).fetchone()[0]
    assert status == "active"
    assert moderator_stats(env.platform.conn) == {}


# ---- Event-log integration --------------------------------------------------


def test_every_action_logs_platform_event(env):
    lid = _create_listing(env, owner=5)
    for reporter in (1, 2, 3):
        _file_report(env, reporter=reporter, target_kind="listing",
                     target=lid, tick=5)
    d9 = make_d9_callback(ModeratorPolicy())
    d9(env.platform.conn, tick=10, rng=random.Random(0))

    n = env.platform.conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type = 'platform_moderator_action'"
    ).fetchone()[0]
    assert n == 3


def test_integrates_with_default_registry(tmp_db):
    """The default BazaarEnv pipeline drives the moderator dynamic."""
    env = BazaarEnv(db_path=tmp_db)
    env.add_agent(MarketAgent(persona=generate_persona(1, seed=1),
                              policy=RandomBenignPolicy(seed=1)))
    env.add_agent(MarketAgent(persona=generate_persona(2, seed=2),
                              policy=RandomBenignPolicy(seed=2)))
    env.reset()
    # Synthetic report: agent 1 creates, agents file reports on listing.
    lid = _create_listing(env, owner=1)
    for reporter in (2,):
        _file_report(env, reporter=reporter, target_kind="listing",
                     target=lid, tick=0)
    # Step a few times so D9 (interval=4) fires.
    for _ in range(5):
        env.step()

    # Warned at minimum.
    actioned = env.platform.conn.execute(
        "SELECT COUNT(*) FROM reports WHERE moderator_action IS NOT NULL"
    ).fetchone()[0]
    assert actioned == 1
    env.close()

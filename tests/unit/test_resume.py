"""R13 — resume-from-db + --add-agents (checkpoint/fork research).

The db IS the checkpoint: save = copy the file, load = point ``--out``
at it. These tests exercise the plumbing end-to-end with
``RandomBenignPolicy`` so no API cost is incurred.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from bazaar import (
    BazaarEnv,
    MarketAgent,
    RandomBenignPolicy,
    generate_persona,
    make_redteam_persona,
    reconstruct_agents_from_db,
)
from bazaar.dynamics import DynamicRegistry


def _fresh_env(tmp_path: Path, *, n_agents: int = 2,
               phantoms: int = 2) -> Path:
    """Create a fresh db, run 5 ticks, close. Returns the db path."""
    db = tmp_path / "warmup.db"
    env = BazaarEnv(
        db_path=db,
        seed_phantom_listings=phantoms,
        dynamics=DynamicRegistry(),  # empty — deterministic, no side-effects
    )
    for i in range(n_agents):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=42 + i),
            policy=RandomBenignPolicy(seed=i),
        ))
    env.reset()
    env.step_many(5)
    env.close()
    return db


def test_resume_continues_tick_count(tmp_path: Path):
    """After a 5-tick warmup, clock.current on resume must be 5 (max
    tick observed was 4; next to execute is 5)."""
    db = _fresh_env(tmp_path)

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    assert env.clock.current == 5, (
        f"resumed clock should land at last_tick+1=5, got {env.clock.current}"
    )
    env.close()


def test_resume_ignores_interrupted_experiment_config_tick(tmp_path: Path):
    """A killed run may write experiment_config before the first resumed step."""
    from bazaar.cli import _log_experiment_config

    db = _fresh_env(tmp_path)
    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    _log_experiment_config(
        env.platform.conn,
        tick=env.clock.current,
        config={"command": "llm-smoke", "cell": "L2-C1"},
    )
    env.close()

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    assert env.clock.current == 5
    env.close()


def test_resume_reconstructs_agents(tmp_path: Path):
    """Personas survive the round-trip: resumed agents list matches
    original count and first persona's display_name round-trips."""
    db = _fresh_env(tmp_path, n_agents=3)

    # Capture original personas for comparison.
    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    restored = reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda *, agent_id: RandomBenignPolicy(seed=0),
    )
    assert len(restored) == 3
    # Persona seed 42 + i=0 → agent_id=1 has a deterministic display_name.
    expected = generate_persona(1, seed=42)
    assert restored[0].persona.agent_id == 1
    assert restored[0].persona.display_name == expected.display_name
    assert restored[0].persona.home_zip == expected.home_zip
    env.close()


def test_resume_can_freeze_existing_redteam_agents(tmp_path: Path):
    """Post-attack contagion forks keep red-team history but should be
    able to stop old red-team agents from receiving new policies."""
    db = tmp_path / "warmup_redteam.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=42),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.add_agent(MarketAgent(
        persona=make_redteam_persona(2, seed=43),
        policy=RandomBenignPolicy(seed=2),
    ))
    env.close()

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    restored = reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda *, agent_id: RandomBenignPolicy(seed=0),
        include_redteam=False,
    )
    assert [ma.agent_id for ma in restored] == [1]
    assert all(not ma.persona.is_redteam for ma in restored)
    env.close()


def test_resume_can_target_existing_agent_subset(tmp_path: Path):
    db = _fresh_env(tmp_path, n_agents=4)

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    restored = reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda *, agent_id: RandomBenignPolicy(seed=0),
        include_agent_ids={2, 4},
    )
    assert [ma.agent_id for ma in restored] == [2, 4]
    env.close()


def test_targeted_resume_scopes_llm_dynamics(tmp_path: Path):
    from bazaar.dynamics.llm_dynamics import make_d14_agent_summary

    db = tmp_path / "targeted_dynamics.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    for i in range(4):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=42 + i),
            policy=RandomBenignPolicy(seed=i),
        ))

    callback = make_d14_agent_summary(agent_ids={2, 4})
    wrote = callback(env.platform.conn, tick=0, rng=random.Random(0))

    rows = env.platform.conn.execute(
        "SELECT agent_id FROM agent_summary ORDER BY agent_id"
    ).fetchall()
    assert wrote == 2
    assert [int(r[0]) for r in rows] == [2, 4]
    env.close()


def test_resume_intervention_bans_frozen_redteam_listings(tmp_path: Path):
    """The red-team removal treatment must be logged and implemented as
    account quarantine, not silent listing deletion."""
    from bazaar.cli import ResumeAgentFilter, _log_resume_intervention

    db = tmp_path / "intervention.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=42),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.add_agent(MarketAgent(
        persona=make_redteam_persona(2, seed=43),
        policy=RandomBenignPolicy(seed=2),
    ))
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'electronics', 'Phone', 'x', 50000, 'good', "
        "'00002', 0, 0, 0, 'active')"
    )
    env.platform.conn.commit()

    _log_resume_intervention(
        env.platform.conn,
        tick=0,
        resume_agent_filter=ResumeAgentFilter.NON_REDTEAM,
        hide_frozen_redteam_listings=True,
        active_resume_agent_ids=[1],
    )

    status = env.platform.conn.execute(
        "SELECT status FROM agents WHERE agent_id = 2"
    ).fetchone()[0]
    listing_status = env.platform.conn.execute(
        "SELECT status FROM listings WHERE listing_id = 200"
    ).fetchone()[0]
    event = env.platform.conn.execute(
        "SELECT action_type, result_payload FROM events "
        "WHERE action_type = 'resume_intervention'"
    ).fetchone()
    assert status == "banned"
    assert listing_status == "active"
    assert event is not None
    assert '"banned_redteam_agent_ids": [2]' in event["result_payload"]
    payload = env.platform.conn.execute(
        "SELECT payload FROM events WHERE action_type = 'resume_intervention'"
    ).fetchone()[0]
    assert '"active_resume_agent_ids": [1]' in payload
    env.close()


def test_experiment_config_is_logged_to_meta_and_event_log(tmp_path: Path):
    from bazaar.cli import _log_experiment_config

    db = tmp_path / "experiment_config.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    config = {
        "command": "llm-smoke",
        "defense_arm": "inventory_block",
        "defense_settings": {
            "inventory_validator_mode": "block",
            "meetup_ownership_check_mode": "warn",
            "require_handoff_proof": False,
        },
    }

    _log_experiment_config(env.platform.conn, tick=7, config=config)

    meta = env.platform.conn.execute(
        "SELECT value FROM meta WHERE key = 'experiment_config'"
    ).fetchone()[0]
    defense_meta = env.platform.conn.execute(
        "SELECT value FROM meta WHERE key = 'defense_settings'"
    ).fetchone()[0]
    event = env.platform.conn.execute(
        """
        SELECT tick, payload, result_payload
        FROM events
        WHERE action_type = 'experiment_config'
        """
    ).fetchone()

    assert json.loads(meta)["defense_arm"] == "inventory_block"
    assert json.loads(defense_meta)["inventory_validator_mode"] == "block"
    assert event["tick"] == 7
    assert json.loads(event["payload"])["defense_settings"] == config["defense_settings"]
    assert "experiment_config" in json.loads(event["result_payload"])["meta_keys"]
    env.close()


def test_experiment_completion_is_logged_to_event_log(tmp_path: Path):
    from bazaar.cli import _json_sha256, _log_experiment_run_complete

    db = tmp_path / "experiment_completion.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    config = {
        "command": "llm-smoke",
        "cell": "L2-2",
        "defense_arm": "inventory_block",
    }
    summary = {
        "start_tick": 360,
        "end_tick": 444,
        "expected_end_tick": 444,
        "ticks_requested": 84,
        "ticks_completed": 84,
        "actions_attempted": 10,
        "actions_ok": 9,
        "actions_blocked": 1,
        "actions_error": 0,
        "llm_calls": 10,
        "cache_hits": 0,
        "audit_action_issue_count": 0,
        "audit_event_issue_count": 0,
        "audit_action_event_issue_count": 0,
        "audit_state_invariant_issue_count": 0,
    }

    _log_experiment_run_complete(
        env.platform.conn,
        tick=444,
        config=config,
        summary=summary,
    )

    event = env.platform.conn.execute(
        """
        SELECT tick, payload, result_payload
        FROM events
        WHERE action_type = 'experiment_run_complete'
        """
    ).fetchone()

    assert event["tick"] == 444
    assert json.loads(event["payload"]) == config
    result_payload = json.loads(event["result_payload"])
    assert result_payload["status"] == "completed"
    assert result_payload["experiment_config_sha256"] == _json_sha256(config)
    env.close()


def test_experiment_config_can_skip_event_log_row(tmp_path: Path):
    from bazaar.cli import _log_experiment_config

    db = tmp_path / "experiment_config_meta_only.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    config = {
        "command": "llm-smoke",
        "cell": "L0",
        "defense_settings": {"inventory_validator_mode": "block"},
    }

    _log_experiment_config(
        env.platform.conn,
        tick=1,
        config=config,
        log_event_row=False,
    )

    meta = env.platform.conn.execute(
        "SELECT value FROM meta WHERE key = 'experiment_config'"
    ).fetchone()[0]
    event_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM events WHERE action_type = 'experiment_config'"
    ).fetchone()[0]

    assert json.loads(meta) == config
    assert event_count == 0
    env.close()


def test_resume_intervention_can_close_redteam_threads(tmp_path: Path):
    from bazaar.cli import ResumeAgentFilter, _log_resume_intervention

    db = tmp_path / "close_threads.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=42),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.add_agent(MarketAgent(
        persona=make_redteam_persona(2, seed=43),
        policy=RandomBenignPolicy(seed=2),
    ))
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status) "
        "VALUES (200, 2, 'electronics', 'Phone', 'x', 50000, 'good', "
        "'00002', 0, 0, 0, 'active')"
    )
    env.platform.conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (300, 200, 1, 2, 0, 0, 'open')"
    )
    env.platform.conn.execute(
        "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, "
        "location_desc, payment_method, status) "
        "VALUES (400, 300, 5, 'Target lot', 'cash', 'scheduled')"
    )
    env.platform.conn.commit()

    _log_resume_intervention(
        env.platform.conn,
        tick=10,
        resume_agent_filter=ResumeAgentFilter.NON_REDTEAM,
        hide_frozen_redteam_listings=False,
        close_frozen_redteam_threads=True,
    )

    thread_status = env.platform.conn.execute(
        "SELECT status FROM threads WHERE thread_id = 300"
    ).fetchone()[0]
    meetup_status = env.platform.conn.execute(
        "SELECT status FROM meetups WHERE meetup_id = 400"
    ).fetchone()[0]
    event = env.platform.conn.execute(
        "SELECT result_payload FROM events "
        "WHERE action_type = 'resume_intervention'"
    ).fetchone()
    assert thread_status == "cancelled"
    assert meetup_status == "cancelled"
    assert '"closed_redteam_thread_ids": [300]' in event["result_payload"]
    env.close()


def test_notify_fraud_victims_materializes_audits_into_ledger(tmp_path: Path):
    from bazaar.cli import _notify_fraud_victims_for_intervention
    from bazaar.core.event_log import log_event

    db = tmp_path / "victim_notify.db"
    env = BazaarEnv(db_path=db, dynamics=DynamicRegistry())
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=42),
        policy=RandomBenignPolicy(seed=1),
    ))
    env.add_agent(MarketAgent(
        persona=make_redteam_persona(2, seed=43),
        policy=RandomBenignPolicy(seed=2),
    ))
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status, is_speculative) "
        "VALUES (200, 2, 'electronics', 'Phone', 'x', 50000, 'good', "
        "'00002', 0, 0, 0, 'sold', 1)"
    )
    env.platform.conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, last_msg_tick, status) "
        "VALUES (300, 200, 1, 2, 0, 0, 'completed')"
    )
    env.platform.conn.execute(
        "INSERT INTO offers (offer_id, thread_id, proposer_id, round, "
        "price_cents, terms_json, tick, status) "
        "VALUES (400, 300, 1, 1, 47500, '{}', 1, 'accepted')"
    )
    log_event(
        env.platform.conn,
        tick=5,
        agent_id=1,
        action_type="fraud_discovered",
        payload={"thread_id": 300, "listing_id": 200, "seller_agent_id": 2},
        result_status="ok",
        result_payload={"auto_rated_1_star": True},
    )
    env.platform.conn.commit()

    notes = _notify_fraud_victims_for_intervention(
        env.platform.conn,
        tick=10,
        target_agent_ids={1},
    )

    assert len(notes) == 1
    ledger = env.platform.conn.execute(
        "SELECT kind, counterparty_id, ref_table, summary, tick "
        "FROM ledger_entries WHERE agent_id = 1"
    ).fetchone()
    assert ledger["kind"] == "report"
    assert ledger["counterparty_id"] == 2
    assert ledger["ref_table"] == "events"
    assert "Platform confirmed you were defrauded" in ledger["summary"]
    assert "$475.00" in ledger["summary"]
    assert ledger["tick"] == 10
    event = env.platform.conn.execute(
        "SELECT result_payload FROM events "
        "WHERE action_type = 'victim_fraud_notification'"
    ).fetchone()
    assert '"notified_count": 1' in event["result_payload"]
    assert _notify_fraud_victims_for_intervention(
        env.platform.conn,
        tick=11,
        target_agent_ids={1},
    ) == []
    env.close()


def test_resume_does_not_reseed_listings(tmp_path: Path):
    """env.reset() on a resumed env must NOT re-seed phantoms — the db
    already contains them."""
    db = _fresh_env(tmp_path, phantoms=5)

    env = BazaarEnv(db_path=db, resume=True, seed_phantom_listings=5,
                    dynamics=DynamicRegistry())
    phantom_count_before = env.platform.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom = 1"
    ).fetchone()[0]
    env.reset()
    phantom_count_after = env.platform.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom = 1"
    ).fetchone()[0]
    assert phantom_count_before == 5
    assert phantom_count_after == 5, (
        "resume+reset must not double the phantom corpus"
    )
    env.close()


def test_add_agents_after_resume(tmp_path: Path):
    """Adding 3 new agents to a 2-agent warmup yields 5 total with ids
    3, 4, 5 — deterministic persona generation from seed + agent_id."""
    db = _fresh_env(tmp_path, n_agents=2)

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    # Rehydrate existing agents.
    for ma in reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda *, agent_id: RandomBenignPolicy(seed=0),
    ):
        env.agents.append(ma)

    assert env.max_agent_id() == 2
    # Append 3 new agents.
    next_id = env.max_agent_id() + 1
    for offset in range(3):
        new_id = next_id + offset
        persona = generate_persona(new_id, seed=100 + new_id)
        env.add_agent(MarketAgent(
            persona=persona, policy=RandomBenignPolicy(seed=0),
        ))
    assert len(env.agents) == 5
    ids_in_db = [r[0] for r in env.platform.conn.execute(
        "SELECT agent_id FROM agents ORDER BY agent_id"
    ).fetchall()]
    assert ids_in_db == [1, 2, 3, 4, 5]
    env.close()


def test_resume_missing_db_errors(tmp_path: Path):
    """Resume with a path that doesn't exist raises a clear error."""
    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError) as excinfo:
        BazaarEnv(db_path=missing, resume=True, dynamics=DynamicRegistry())
    assert "resume=True" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_add_agents_without_resume_errors(tmp_path: Path):
    """CLI --add-agents without --resume must error out cleanly."""
    dummy = tmp_path / "fresh.db"
    result = subprocess.run(
        [sys.executable, "-m", "bazaar.cli", "llm-smoke",
         "--provider", "ollama", "--model", "llama3.2:3b",
         "--agents", "1", "--ticks", "1",
         "--out", str(dummy),
         "--add-agents", "2"],  # --resume omitted on purpose
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, (
        "--add-agents without --resume should exit 1, stdout="
        f"{result.stdout!r} stderr={result.stderr!r}"
    )
    assert "--add-agents requires --resume" in (
        result.stdout + result.stderr
    )


def test_resume_advances_tick_on_further_steps(tmp_path: Path):
    """After resume, step()ing further advances the clock and appends
    events past the warmup's last tick."""
    db = _fresh_env(tmp_path, n_agents=2)

    env = BazaarEnv(db_path=db, resume=True, dynamics=DynamicRegistry())
    for ma in reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda *, agent_id: RandomBenignPolicy(seed=0),
    ):
        env.agents.append(ma)
    start_tick = env.clock.current
    env.step_many(3)
    assert env.clock.current == start_tick + 3
    max_tick_in_events = env.platform.conn.execute(
        "SELECT MAX(tick) FROM events"
    ).fetchone()[0]
    assert max_tick_in_events >= start_tick  # new events landed post-warmup
    env.close()

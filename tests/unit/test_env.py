"""BazaarEnv orchestrator invariants."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from bazaar import ActionType, BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.agents.market_agent import AgentAction, Policy
from bazaar.dynamics import DynamicRegistry


class _StaticPolicy(Policy):
    def __init__(self, decision: Any) -> None:
        self.decision = decision

    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> Any:
        return self.decision


class _RaisingPolicy(Policy):
    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        raise RuntimeError("boom")


class _PrepareRaisingPolicy(_StaticPolicy):
    def prepare_decision(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> Any:
        raise RuntimeError("prepare boom")


class _SkipPrepared:
    skip = True


class _ObservingPreparePolicy(_StaticPolicy):
    def __init__(self) -> None:
        super().__init__(None)
        self.policy_error_count_at_prepare: int | None = None

    def prepare_decision(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> _SkipPrepared:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE action_type = 'policy_error'"
        ).fetchone()
        self.policy_error_count_at_prepare = int(row[0])
        return _SkipPrepared()


class _PreparedParallelPolicy(Policy):
    model = "prepared-ok"
    strict_backend_errors = False

    def __init__(self) -> None:
        self.apply_calls = 0

    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        raise AssertionError("parallel policy should use the 3-phase API")

    def prepare_decision(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(skip=False)

    def dispatch_llm_call(self, prepared: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(error=None)

    def apply_decision(
        self,
        conn: sqlite3.Connection,
        prepared: SimpleNamespace,
        result: SimpleNamespace,
    ) -> AgentAction:
        self.apply_calls += 1
        return AgentAction(ActionType.DO_NOTHING, {})


class _StrictBackendErrorParallelPolicy(_PreparedParallelPolicy):
    model = "strict-test"
    strict_backend_errors = True

    def dispatch_llm_call(self, prepared: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(error="backend_error: timeout")

    def apply_decision(
        self,
        conn: sqlite3.Connection,
        prepared: SimpleNamespace,
        result: SimpleNamespace,
    ) -> AgentAction:
        self.apply_calls += 1
        raise AssertionError("strict backend failures must abort before apply")


def _env_with_policy(tmp_db, policy: Policy, *, parallel_decide: bool = False) -> BazaarEnv:
    env = BazaarEnv(
        db_path=tmp_db,
        dynamics=DynamicRegistry(),
        parallel_decide=parallel_decide,
    )
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=0),
        policy=policy,
    ))
    if parallel_decide:
        env.add_agent(MarketAgent(
            persona=generate_persona(2, seed=1),
            policy=_StaticPolicy(None),
        ))
    env.reset()
    return env


def _policy_error_payloads(env: BazaarEnv) -> list[dict[str, Any]]:
    rows = env.platform.conn.execute(
        """
        SELECT payload
        FROM events
        WHERE action_type = 'policy_error'
        ORDER BY event_id
        """
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def test_reset_seeds_phantoms(tmp_db):
    env = BazaarEnv(db_path=tmp_db, seed_phantom_listings=3)
    env.reset()
    count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_phantom = 1"
    ).fetchone()[0]
    assert count == 3
    env.close()


def test_step_advances_clock(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=0),
        policy=RandomBenignPolicy(seed=0),
    ))
    env.reset()
    assert env.clock.current == 0
    env.step()
    assert env.clock.current == 1
    env.step()
    assert env.clock.current == 2
    env.close()


def test_inject_executes_even_without_policy_tick(tmp_db):
    env = BazaarEnv(db_path=tmp_db)
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=0),
        policy=RandomBenignPolicy(seed=0),
    ))
    env.reset()
    r = env.inject(
        agent_id=1, action=ActionType.CREATE_LISTING,
        args={"category": "x", "title": "Test item", "description": "",
              "price_cents": 100, "condition": "good"},
    )
    assert r.status == "ok"
    env.close()


def test_small_run_does_not_error(small_env):
    reports = small_env.step_many(20)
    total_err = sum(r.actions_error for r in reports)
    # RandomBenignPolicy is designed to produce well-formed actions.
    assert total_err == 0, f"unexpected errors: {reports}"


def test_step_logs_policy_decide_exception_without_crashing(tmp_db):
    env = _env_with_policy(tmp_db, _RaisingPolicy())
    try:
        report = env.step()
        assert env.clock.current == 1
        assert report.actions_attempted == 1
        assert report.actions_error == 1

        payloads = _policy_error_payloads(env)
        assert len(payloads) == 1
        assert payloads[0]["error"] == "policy_decide_exception"
        assert "RuntimeError: boom" in payloads[0]["detail"]
    finally:
        env.close()


def test_step_many_abort_on_error_stops_after_first_error_tick(tmp_db):
    env = _env_with_policy(tmp_db, _RaisingPolicy())
    try:
        with pytest.raises(RuntimeError, match="aborting after tick 0"):
            env.step_many(3, abort_on_error=True)
        assert env.clock.current == 1
        payloads = _policy_error_payloads(env)
        assert len(payloads) == 1
    finally:
        env.close()


def test_step_logs_non_agent_action_policy_return_without_crashing(tmp_db):
    env = _env_with_policy(tmp_db, _StaticPolicy({"action": "do_nothing"}))
    try:
        report = env.step()
        assert report.actions_attempted == 1
        assert report.actions_error == 1

        payloads = _policy_error_payloads(env)
        assert payloads[0]["error"] == "policy_return_not_agent_action"
        assert payloads[0]["raw_type"] == "dict"
    finally:
        env.close()


def test_step_executes_valid_actions_and_logs_bad_items_in_policy_list(tmp_db):
    env = _env_with_policy(
        tmp_db,
        _StaticPolicy([AgentAction(ActionType.DO_NOTHING, {}), "bad"]),
    )
    try:
        report = env.step()
        assert report.actions_attempted == 2
        assert report.actions_ok == 1
        assert report.actions_error == 1

        rows = env.platform.conn.execute(
            """
            SELECT action_type, result_status
            FROM events
            WHERE tick = 0 AND agent_id = 1
            ORDER BY event_id
            """
        ).fetchall()
        assert [(row["action_type"], row["result_status"]) for row in rows] == [
            ("do_nothing", "ok"),
            ("policy_error", "error"),
        ]
        assert _policy_error_payloads(env)[0]["error"] == "policy_return_not_agent_action"
    finally:
        env.close()


def test_step_logs_agent_action_with_invalid_action_type_without_crashing(tmp_db):
    bad_action = AgentAction("not_an_action", {})  # type: ignore[arg-type]
    env = _env_with_policy(tmp_db, _StaticPolicy(bad_action))
    try:
        report = env.step()
        assert report.actions_attempted == 1
        assert report.actions_error == 1

        payloads = _policy_error_payloads(env)
        assert payloads[0]["error"] == "policy_action_not_action_type"
        assert payloads[0]["raw_type"] == "str"
        assert "not_an_action" in payloads[0]["raw_repr"]
    finally:
        env.close()


def test_parallel_prepare_exception_is_logged_without_crashing_tick(tmp_db):
    observer = _ObservingPreparePolicy()
    env = BazaarEnv(
        db_path=tmp_db,
        dynamics=DynamicRegistry(),
        parallel_decide=True,
    )
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=0),
        policy=_PrepareRaisingPolicy(None),
    ))
    env.add_agent(MarketAgent(
        persona=generate_persona(2, seed=1),
        policy=observer,
    ))
    env.reset()
    try:
        report = env.step()
        assert env.clock.current == 1
        assert report.actions_attempted == 1
        assert report.actions_error == 1
        assert observer.policy_error_count_at_prepare == 0

        payloads = _policy_error_payloads(env)
        assert payloads[0]["error"] == "policy_prepare_exception"
        assert "RuntimeError: prepare boom" in payloads[0]["detail"]
    finally:
        env.close()


def test_parallel_seq_fallback_exception_is_logged_without_crashing_tick(tmp_db):
    env = _env_with_policy(
        tmp_db,
        _RaisingPolicy(),
        parallel_decide=True,
    )
    try:
        report = env.step()
        assert env.clock.current == 1
        assert report.actions_attempted == 1
        assert report.actions_error == 1

        payloads = _policy_error_payloads(env)
        assert payloads[0]["error"] == "policy_decide_exception"
        assert "RuntimeError: boom" in payloads[0]["detail"]
    finally:
        env.close()


def test_parallel_strict_backend_error_aborts_before_tick_writes(tmp_db):
    ok_policy = _PreparedParallelPolicy()
    bad_policy = _StrictBackendErrorParallelPolicy()
    env = BazaarEnv(
        db_path=tmp_db,
        dynamics=DynamicRegistry(),
        parallel_decide=True,
    )
    env.add_agent(MarketAgent(
        persona=generate_persona(1, seed=0),
        policy=ok_policy,
    ))
    env.add_agent(MarketAgent(
        persona=generate_persona(2, seed=1),
        policy=bad_policy,
    ))
    env.reset()
    try:
        event_count_before = env.platform.conn.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0]
        llm_count_before = env.platform.conn.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]
        with pytest.raises(
            RuntimeError,
            match="llm_backend_error before tick write",
        ):
            env.step_many(1, abort_on_error=True)

        assert env.clock.current == 0
        assert ok_policy.apply_calls == 0
        assert bad_policy.apply_calls == 0
        assert env.platform.conn.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0] == event_count_before
        assert env.platform.conn.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0] == llm_count_before
        assert _policy_error_payloads(env) == []
    finally:
        env.close()

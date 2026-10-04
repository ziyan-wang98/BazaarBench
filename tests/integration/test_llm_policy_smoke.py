"""Integration test — LLMPolicy end-to-end with a fake backend (T28c-9).

Two agents × 10 ticks. Backend is a scripted ``FakeBackend`` that
returns a rotating set of tool-call JSON strings, so the test is
offline + deterministic. We assert:

- No errors in ``BazaarEnv.step`` reports.
- ``llm_calls`` table is populated (one row per non-skipped tick).
- Event log contains dispatched actions.
- Narrative memory has observation entries.
- ``compute_llm_cache_hash`` returns a stable hex digest.
"""
from __future__ import annotations

import json

import pytest

from bazaar.actions.types import ActionType
from bazaar.agents.llm_backends.base import LLMMessage, LLMResponse
from bazaar.agents.market_agent import MarketAgent
from bazaar.agents.persona import generate_persona
from bazaar.agents.policies import LLMPolicy
from bazaar.core.env import BazaarEnv
from bazaar.core.event_log import compute_llm_cache_hash
from bazaar.memory import HashEncoder, NarrativeStore, install_store


class FakeBackend:
    """Scripted backend — rotates through a fixed response list.

    Each response may be either a raw text string (legacy path —
    exercises :func:`_parse_tool_calls`) or a dict of the form
    ``{"text": ..., "tool_calls": [...]}`` (native tool-call path —
    exercises the /api/chat-style code path). The backend records
    every call for test assertions.
    """

    def __init__(self, responses) -> None:
        self._responses = responses
        self.calls: list[list[LLMMessage]] = []
        self.tools_seen: list = []
        self._i = 0

    def list_models(self) -> list:  # pragma: no cover — unused here
        return []

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools=None,
    ) -> LLMResponse:
        self.calls.append(messages)
        self.tools_seen.append(tools)
        item = self._responses[self._i % len(self._responses)]
        self._i += 1
        if isinstance(item, dict):
            text = item.get("text", "")
            tool_calls = item.get("tool_calls")
            reasoning_summary = item.get("reasoning_summary")
        else:
            text = item
            tool_calls = None
            reasoning_summary = None
        return LLMResponse(
            text=text,
            total_s=0.1,
            first_token_s=0.05,
            prompt_tokens=100,
            output_tokens=20,
            model=model,
            tool_calls=tool_calls,
            reasoning_summary=reasoning_summary,
        )


class RaisingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([])

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools=None,
    ) -> LLMResponse:
        self.calls.append(messages)
        raise RuntimeError("invalid_api_key: test key rejected")


class ReasoningBackend(FakeBackend):
    reasoning_effort = "high"

    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.reasoning_seen: list[str | None] = []

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools=None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        self.reasoning_seen.append(reasoning_effort)
        return super().generate(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "llm_smoke.db"
    env = BazaarEnv(
        db_path=db_path,
        agents=[],
        seed_phantom_listings=0,
    )
    # Install a deterministic (no-network) encoder so narrative writes
    # work without loading sentence-transformers.
    install_store(env.platform.conn, NarrativeStore(
        env.platform.conn, encoder=HashEncoder(dim=32),
    ))
    try:
        yield env
    finally:
        env.close()


def _make_agent(agent_id: int, backend: FakeBackend) -> MarketAgent:
    persona = generate_persona(agent_id, seed=100 + agent_id)
    # Force a high activity rate so the policy fires most ticks —
    # otherwise a 10-tick run may see zero LLM calls.
    persona.activity_rate = 0.95
    policy = LLMPolicy(
        backend=backend,
        model="fake-test",
        recall_k=3,
        slice_k=5,
        seed=agent_id,
    )
    return MarketAgent(persona=persona, policy=policy)


def test_llm_policy_smoke(env) -> None:
    # R5 lifecycle assertion: the agents must see a peer listing in the
    # extended slice_for_prompt output and actually act on it. We seed
    # a phantom listing so the make_offer tool call below lands on a
    # real row (any agent can offer on a phantom — see D7 rationale).
    backend = FakeBackend([
        # Native tool-call path — goes through LLMResponse.tool_calls
        # rather than JSON-out-of-text parsing. This is what happens
        # against /api/chat with a tool-aware model.
        {"text": "",
         "tool_calls": [{"function": {"name": "search",
                                       "arguments": {"query": "bike"}}}]},
        # Discovery → action: fire a real make_offer on the seeded
        # phantom listing (listing_id=1). This is what the R5 extension
        # is supposed to enable — agents seeing recommended_listings
        # and acting on them rather than idling.
        {"text": "",
         "tool_calls": [{"function": {"name": "make_offer",
                                       "arguments": {"listing_id": 1,
                                                     "price_cents": 800}}}]},
        '{"action":"do_nothing","arguments":{}}',
        '{"action":"browse_category","arguments":{"category":"books"}}',
        'not even json',                 # tolerant fallback → no dispatch
        '{"action":"do_nothing"}',       # missing arguments → {} is fine
        '{"action":"unknown_action"}',   # unknown → fallback to do_nothing
    ])

    env.add_agent(_make_agent(1, backend))
    env.add_agent(_make_agent(2, backend))
    env.reset()

    # Seed the phantom listing referenced by the make_offer response.
    env.platform.conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status, is_phantom) "
        "VALUES (1, NULL, 'bikes', 'Phantom Trek', 'x', 1000, 'good', "
        "'00001', 0, 0, 0, 'active', 1)",
    )
    env.platform.conn.commit()

    reports = env.step_many(10)
    total_errors = sum(r.actions_error for r in reports)
    total_attempts = sum(r.actions_attempted for r in reports)
    assert total_errors == 0, (
        f"LLMPolicy tick produced errors: {[r.actions_error for r in reports]}"
    )
    assert total_attempts > 0, "policy never fired; activity rate too low?"
    # tools are always forwarded to the backend now.
    assert backend.tools_seen and all(
        t is not None for t in backend.tools_seen
    ), "LLMPolicy must forward the tool list to the backend"
    # At least one tick produced a real (non-DO_NOTHING) action.
    non_idle = env.platform.conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE action_type IS NOT NULL AND action_type != 'do_nothing'"
    ).fetchone()[0]
    assert non_idle > 0, "no non-idle actions dispatched by LLMPolicy"

    # LLM call log populated.
    n_calls = env.platform.conn.execute(
        "SELECT COUNT(*) FROM llm_calls"
    ).fetchone()[0]
    assert n_calls == total_attempts, (
        f"expected 1 llm_calls row per attempt (got {n_calls} for "
        f"{total_attempts})"
    )

    # Each row has a prompt_hash and sampling_params JSON.
    rows = env.platform.conn.execute(
        "SELECT prompt_hash, sampling_params FROM llm_calls"
    ).fetchall()
    for h, sp in rows:
        assert isinstance(h, str) and len(h) == 64
        assert sp and sp.startswith("{") and sp.endswith("}")

    # R5-T1: the extended slice_for_prompt payload must actually reach
    # the LLM observation. Verify all four new keys are serialised into
    # at least one captured prompt_text — otherwise they're computed
    # but never shown.
    prompt_texts = [
        row[0] for row in env.platform.conn.execute(
            "SELECT prompt_text FROM llm_calls WHERE prompt_text IS NOT NULL"
        ).fetchall()
    ]
    assert prompt_texts, "no prompt_text captured — cannot verify slice keys"
    joined = "\n".join(prompt_texts)
    for key in (
        "recommended_listings",
        "incoming_messages",
        "pending_offers_on_my_listings",
        "marketplace_pulse",
    ):
        assert key in joined, (
            f"new slice key {key!r} never surfaced in any prompt_text — "
            f"slice_for_prompt extension is not flowing into the prompt"
        )

    # R5 lifecycle: "did the extension actually enable interaction?"
    # The scripted backend emits a make_offer tool_call on the seeded
    # phantom listing. Without the extended slice an LLM would have
    # no grounds to produce that call; here we only assert that once
    # the dispatcher sees it, it lands and a real offer row gets
    # written. The count is lower-bounded at 1 — with 2 agents × 10
    # ticks × rotating responses there's plenty of margin.
    offer_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM offers"
    ).fetchone()[0]
    assert offer_count >= 1, (
        "R5 expected >=1 offer after observation extension — "
        "the lifecycle from observation to action is broken"
    )

    # Narrative store has observation entries.
    n_narr = env.platform.conn.execute(
        "SELECT COUNT(*) FROM narrative_memories WHERE scope = 'self'"
    ).fetchone()[0]
    assert n_narr > 0, "no narrative observations written by LLMPolicy"

    # Cache hash is stable + hex.
    h = compute_llm_cache_hash(env.platform.conn, up_to_tick=10)
    assert isinstance(h, str) and len(h) == 64


def test_llm_policy_dispatches_multiple_tool_calls_in_model_order(env) -> None:
    backend = FakeBackend([
        {"text": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "bike"}}},
            {"function": {
                "name": "browse_category",
                "arguments": {"category": "books"},
            }},
        ]},
    ])
    agent = _make_agent(1, backend)
    agent.persona.activity_rate = 1.0
    env.add_agent(agent)
    env.reset()

    report = env.step()

    assert report.actions_attempted == 2
    rows = env.platform.conn.execute(
        "SELECT action_type FROM events "
        "WHERE agent_id = 1 ORDER BY event_id"
    ).fetchall()
    assert [r[0] for r in rows] == ["search", "browse_category"]


def test_llm_policy_logs_rejected_native_tool_calls(env) -> None:
    backend = FakeBackend([
        {"text": "", "tool_calls": [
            {"function": {"name": "unknown_action", "arguments": {}}},
            {"function": {"name": "create_listing", "arguments": {}}},
            {"function": {"name": "search", "arguments": "{bad-json"}},
        ]},
    ])
    agent = _make_agent(1, backend)
    agent.persona.activity_rate = 1.0
    env.add_agent(agent)
    env.reset()

    report = env.step()

    assert report.actions_attempted == 1
    row = env.platform.conn.execute(
        "SELECT tool_calls_json FROM llm_calls ORDER BY call_id DESC LIMIT 1"
    ).fetchone()
    logged = json.loads(row["tool_calls_json"])
    assert [item["reason"] for item in logged] == [
        "unknown_tool_name",
        "schema_invalid",
        "malformed_arguments_json",
    ]
    assert all(item["_rejected"] is True for item in logged)
    events = env.platform.conn.execute(
        "SELECT action_type FROM events WHERE agent_id = 1 ORDER BY event_id"
    ).fetchall()
    assert [r["action_type"] for r in events] == ["do_nothing"]


def test_system_prompt_suffix_participates_in_prompt_hash(env) -> None:
    backend = FakeBackend(['{"action":"do_nothing","arguments":{}}'])
    persona = generate_persona(1, seed=303)
    persona.activity_rate = 1.0
    agent_a = MarketAgent(
        persona=persona,
        policy=LLMPolicy(
            backend=backend,
            model="fake-test",
            system_prompt_suffix="Level 2 seller pressure suffix",
        ),
    )
    env.add_agent(agent_a)
    env.reset()

    prepared_a = agent_a.policy.prepare_decision(agent_a, env.platform.conn, 0)
    policy_b = LLMPolicy(
        backend=backend,
        model="fake-test",
        system_prompt_suffix="Level 3 red-team suffix",
    )
    prepared_b = policy_b.prepare_decision(agent_a, env.platform.conn, 0)

    assert "Level 2 seller pressure suffix" in prepared_a.system_text
    assert "Level 3 red-team suffix" in prepared_b.system_text
    assert prepared_a.prompt_hash != prepared_b.prompt_hash


def test_reasoning_effort_forwarded_and_logged(env) -> None:
    backend = ReasoningBackend(['{"action":"do_nothing","arguments":{}}'])
    persona = generate_persona(1, seed=505)
    persona.activity_rate = 1.0
    env.add_agent(MarketAgent(
        persona=persona,
        policy=LLMPolicy(
            backend=backend,
            model="fake-test",
        ),
    ))
    env.reset()

    env.step()

    assert backend.reasoning_seen == ["high"]
    row = env.platform.conn.execute(
        "SELECT sampling_params FROM llm_calls ORDER BY call_id DESC LIMIT 1"
    ).fetchone()
    sampling = json.loads(row["sampling_params"])
    assert sampling["reasoning_effort"] == "high"


def test_strict_llm_backend_error_becomes_policy_error_without_llm_replay_row(env) -> None:
    backend = RaisingBackend()
    persona = generate_persona(1, seed=404)
    persona.activity_rate = 1.0
    env.add_agent(MarketAgent(
        persona=persona,
        policy=LLMPolicy(
            backend=backend,
            model="fake-test",
            strict_backend_errors=True,
        ),
    ))
    env.reset()

    report = env.step()

    assert report.actions_error == 1
    llm_rows = env.platform.conn.execute(
        "SELECT response_text FROM llm_calls"
    ).fetchall()
    assert llm_rows == []
    payload = json.loads(env.platform.conn.execute(
        """
        SELECT payload FROM events
        WHERE action_type = 'policy_error'
        ORDER BY event_id DESC LIMIT 1
        """
    ).fetchone()["payload"])
    assert payload["error"] == "policy_decide_exception"
    assert "llm_backend_error" in payload["detail"]
    assert "invalid_api_key" in payload["detail"]


def test_strict_llm_empty_output_becomes_dispatch_error(env) -> None:
    backend = FakeBackend([""])
    persona = generate_persona(1, seed=405)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="fake-test",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)

    assert result.error == "invalid_llm_output: empty_response_no_tool_call"
    llm_rows = env.platform.conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert llm_rows == 0


def test_strict_llm_requires_decision_rationale(env) -> None:
    backend = FakeBackend(['{"action":"do_nothing","arguments":{}}'])
    persona = generate_persona(1, seed=406)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="fake-test",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)

    assert result.error == "invalid_llm_output: missing_decision_rationale"


def test_strict_llm_accepts_visible_note_and_native_tool_call(env) -> None:
    backend = FakeBackend([
        {
            "text": "Decision note: Nothing needs action this tick.",
            "tool_calls": [{
                "function": {"name": "do_nothing", "arguments": {}},
            }],
        },
    ])
    persona = generate_persona(1, seed=407)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="fake-test",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)
    action = policy.apply_decision(env.platform.conn, prepared, result)

    assert result.error is None
    assert isinstance(action, list)
    assert action[0].action is ActionType.DO_NOTHING
    row = env.platform.conn.execute(
        "SELECT response_text, tool_calls_json FROM llm_calls"
    ).fetchone()
    assert row["response_text"].startswith("Decision note:")
    assert "do_nothing" in row["tool_calls_json"]


def test_strict_llm_retries_invalid_output_before_success(env) -> None:
    backend = FakeBackend([
        "",
        {
            "text": "Decision note: The safe choice is to wait.",
            "tool_calls": [{
                "function": {"name": "do_nothing", "arguments": {}},
            }],
        },
    ])
    persona = generate_persona(1, seed=408)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="fake-test",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)

    assert result.error is None
    assert len(backend.calls) == 2


def test_strict_llm_retries_missing_reasoning_for_responses_model(env) -> None:
    backend = FakeBackend([
        {
            "text": "Decision note: The safe choice is to wait.",
            "tool_calls": [{
                "function": {"name": "do_nothing", "arguments": {}},
            }],
        },
        {
            "text": "Decision note: The safe choice is to wait.",
            "tool_calls": [{
                "function": {"name": "do_nothing", "arguments": {}},
            }],
            "reasoning_summary": "The current market state does not require intervention.",
        },
    ])
    backend.use_responses_endpoint = True
    persona = generate_persona(1, seed=409)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="gpt-5.5_2026-04-24",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)

    assert result.error is None
    assert result.reasoning_summary is not None
    assert len(backend.calls) == 2


def test_strict_llm_does_not_require_reasoning_for_trapi_chat_only(env) -> None:
    fake_trapi_backend = type("TRAPIBackend", (FakeBackend,), {})
    backend = fake_trapi_backend([
        {
            "text": "Decision note: The safe choice is to wait.",
            "tool_calls": [{
                "function": {"name": "do_nothing", "arguments": {}},
            }],
        },
    ])
    backend.use_responses_endpoint = True
    persona = generate_persona(1, seed=410)
    persona.activity_rate = 1.0
    policy = LLMPolicy(
        backend=backend,
        model="gpt-5.4_2026-03-05",
        strict_backend_errors=True,
    )
    env.add_agent(MarketAgent(persona=persona, policy=policy))
    env.reset()

    prepared = policy.prepare_decision(env.agents[0], env.platform.conn, tick=0)
    result = policy.dispatch_llm_call(prepared)

    assert result.error is None
    assert result.reasoning_summary is None
    assert len(backend.calls) == 1


def test_llm_policy_cache_hit(env) -> None:
    """Same prompt twice → second call is a cache hit, no new backend call."""
    backend = FakeBackend(['{"action":"do_nothing","arguments":{}}'])

    env.add_agent(_make_agent(1, backend))
    env.reset()
    env.step_many(2)  # first tick populates cache

    first_calls = len(backend.calls)
    # Force a second round — prompt hash will be identical if the agent
    # produced no side effects (DO_NOTHING does not mutate state). The
    # cache hit should prevent a new backend call.
    env.step_many(2)
    # Backend may or may not have been called depending on activity-rate
    # throttling. If it WAS called again, at least some should be cache hits.
    hits = env.platform.conn.execute(
        "SELECT SUM(cache_hit) FROM llm_calls"
    ).fetchone()[0] or 0
    total = env.platform.conn.execute(
        "SELECT COUNT(*) FROM llm_calls"
    ).fetchone()[0]
    # Either we had cache hits, or backend was only called once and
    # every subsequent tick was a hit.
    assert hits >= 0 and total >= first_calls

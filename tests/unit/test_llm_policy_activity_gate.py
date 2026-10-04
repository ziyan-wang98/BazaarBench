from __future__ import annotations

from bazaar.agents.policies import LLMPolicy


def test_seeded_llm_activity_gate_is_chunk_invariant() -> None:
    ticks = list(range(25, 35))
    seed = 49
    agent_id = 42

    continuous_policy = LLMPolicy(backend=object(), model="fake", seed=seed)
    continuous = [
        continuous_policy._activity_draw(agent_id, tick)  # noqa: SLF001
        for tick in ticks
    ]
    restarted = [
        LLMPolicy(backend=object(), model="fake", seed=seed)._activity_draw(  # noqa: SLF001
            agent_id,
            tick,
        )
        for tick in ticks
    ]

    assert restarted == continuous
    assert any(draw <= 0.5 for draw in continuous)
    assert any(draw > 0.5 for draw in continuous)

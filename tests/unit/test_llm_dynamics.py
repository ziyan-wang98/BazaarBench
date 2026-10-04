"""Unit tests for D11 + D12 LLM-backed dynamics (T28d-4 / T28d-5)."""
from __future__ import annotations

import pytest

from bazaar.agents.llm_backends.base import LLMMessage, LLMResponse
from bazaar.core.env import BazaarEnv
from bazaar.dynamics import with_llm_dynamics
from bazaar.dynamics.llm_dynamics import (
    make_d11_memory_consolidation,
    make_d12_self_portrait,
)
from bazaar.memory import HashEncoder, NarrativeStore, install_store


class _FakeBackend:
    def __init__(self, text: str = "summary text") -> None:
        self.text = text
        self.calls = 0
    def generate(
        self,
        messages: list[LLMMessage],
        *, model: str, max_tokens: int = 512, temperature: float = 0.4,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.text, total_s=0.1, first_token_s=0.05,
            prompt_tokens=10, output_tokens=5, model=model,
        )
    def list_models(self) -> list:
        return []


@pytest.fixture
def env(tmp_path):
    env = BazaarEnv(db_path=tmp_path / "d.db", seed_phantom_listings=0)
    install_store(env.platform.conn,
                  NarrativeStore(env.platform.conn, encoder=HashEncoder(dim=16)))
    yield env
    env.close()


def _add_agent_rows(conn, n: int = 1) -> None:
    conn.executemany(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(i, f"u{i}", f"U{i}", f"{i:05d}", 0.0, 0.0,
          f'{{"display_name":"U{i}","profession":"teacher","interests":["books"]}}')
         for i in range(1, n + 1)],
    )
    conn.commit()


def _seed_narratives(conn, agent_id: int, n: int) -> None:
    for i in range(n):
        conn.execute(
            "INSERT INTO narrative_memories "
            "(agent_id, scope, scope_ref_id, content, embedding, "
            "created_tick, decayed) VALUES (?, 'self', NULL, ?, ?, ?, 0)",
            (agent_id, f"impression {i}", b"\x00" * 64, i),
        )
    conn.commit()


# ---- D11: memory consolidation ------------------------------------------


def test_d11_noop_below_batch(env) -> None:
    _add_agent_rows(env.platform.conn, n=1)
    _seed_narratives(env.platform.conn, agent_id=1, n=3)
    cb = make_d11_memory_consolidation(backend=None)
    import random
    n = cb(env.platform.conn, tick=100, rng=random.Random(0))
    assert n == 0  # fewer than batch threshold
    decayed = env.platform.conn.execute(
        "SELECT COUNT(*) FROM narrative_memories WHERE decayed = 1"
    ).fetchone()[0]
    assert decayed == 0


def test_d11_fallback_consolidates_and_marks_decayed(env) -> None:
    _add_agent_rows(env.platform.conn, n=1)
    _seed_narratives(env.platform.conn, agent_id=1, n=10)
    cb = make_d11_memory_consolidation(backend=None)
    import random
    n = cb(env.platform.conn, tick=100, rng=random.Random(0))
    assert n == 1
    # 6 decayed + 1 new consolidated + 4 untouched = 11 rows total
    total = env.platform.conn.execute(
        "SELECT COUNT(*) FROM narrative_memories"
    ).fetchone()[0]
    decayed = env.platform.conn.execute(
        "SELECT COUNT(*) FROM narrative_memories WHERE decayed = 1"
    ).fetchone()[0]
    assert total == 11
    assert decayed == 6
    # Event log has the D11 event
    row = env.platform.conn.execute(
        "SELECT action_type FROM events WHERE action_type = 'D11_memory_consolidation'"
    ).fetchone()
    assert row is not None


def test_d11_llm_path_logs_llm_calls(env) -> None:
    _add_agent_rows(env.platform.conn, n=1)
    _seed_narratives(env.platform.conn, agent_id=1, n=10)
    backend = _FakeBackend(text="rolled-up summary")
    cb = make_d11_memory_consolidation(backend=backend, model="fake")
    import random
    cb(env.platform.conn, tick=50, rng=random.Random(0))
    assert backend.calls == 1
    # llm_calls row present
    row = env.platform.conn.execute(
        "SELECT response_text FROM llm_calls ORDER BY call_id DESC LIMIT 1"
    ).fetchone()
    assert "rolled-up summary" in row[0]


# ---- D12: self-portrait -------------------------------------------------


def test_d12_writes_one_portrait_per_agent(env) -> None:
    _add_agent_rows(env.platform.conn, n=3)
    cb = make_d12_self_portrait(backend=None)
    import random
    n = cb(env.platform.conn, tick=200, rng=random.Random(0))
    assert n == 3
    row_count = env.platform.conn.execute(
        "SELECT COUNT(*) FROM self_portraits WHERE tick = 200"
    ).fetchone()[0]
    assert row_count == 3


def test_d12_llm_path_uses_backend(env) -> None:
    _add_agent_rows(env.platform.conn, n=2)
    backend = _FakeBackend(text="portrait sentence")
    cb = make_d12_self_portrait(backend=backend, model="fake")
    import random
    cb(env.platform.conn, tick=300, rng=random.Random(0))
    assert backend.calls == 2
    portraits = env.platform.conn.execute(
        "SELECT content FROM self_portraits ORDER BY portrait_id"
    ).fetchall()
    for p in portraits:
        assert "portrait sentence" in p[0]


# ---- Registry integration ----------------------------------------------


def test_with_llm_dynamics_registers_exactly_once(env) -> None:
    from bazaar.dynamics.registry import DynamicRegistry
    reg = DynamicRegistry()
    reg = with_llm_dynamics(reg)
    reg = with_llm_dynamics(reg)  # second call is idempotent
    names = [s.name for s in reg.specs]
    assert names.count("D11_memory_consolidation") == 1
    assert names.count("D12_self_portrait") == 1

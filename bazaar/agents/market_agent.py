"""MarketAgent: the unified buyer+seller agent.

Lightweight pure-Python wrapper around ``PersonaCard`` + ``WorldState``.
Action selection is driven by a policy object (``Policy`` abstract base);
``RandomBenignPolicy`` exercises the full action wiring without an LLM,
and ``LLMPolicy`` drives OpenAI-compatible / Anthropic / Ollama backends
for full rollouts.
"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from bazaar.actions.types import ActionType
from bazaar.agents.persona import PersonaCard
from bazaar.agents.world_state import WorldState


@dataclass
class AgentAction:
    """What an agent's policy returns each tick."""
    action: ActionType
    args: dict[str, Any]


class Policy(ABC):
    """Abstract decision-making policy for a ``MarketAgent``."""

    @abstractmethod
    def decide(
        self,
        agent: MarketAgent,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        """Return one or more actions, or ``None`` to do nothing this tick."""


class MarketAgent:
    """Unified marketplace agent holding persona + world state + policy."""

    def __init__(
        self,
        *,
        persona: PersonaCard,
        world_state: WorldState | None = None,
        policy: Policy,
        available_actions: list[ActionType] | None = None,
    ) -> None:
        self.persona = persona
        self.world = world_state or WorldState(agent_id=persona.agent_id)
        self.policy = policy
        self.available_actions = (
            available_actions
            if available_actions is not None
            else ActionType.default_benign_actions()
        )

    # -- Convenience ------------------------------------------------------

    @property
    def agent_id(self) -> int:
        return self.persona.agent_id

    def __repr__(self) -> str:
        return (
            f"MarketAgent(id={self.agent_id}, "
            f"name={self.persona.user_name!r}, "
            f"zip={self.persona.home_zip})"
        )

    # -- Policy forwarding ------------------------------------------------

    def decide(
        self,
        conn: sqlite3.Connection,
        tick: int,
    ) -> AgentAction | list[AgentAction] | None:
        return self.policy.decide(self, conn, tick)

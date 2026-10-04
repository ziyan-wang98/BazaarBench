"""WorldState: mutable per-agent state.

Phase 1 keeps this in-memory; Phase 2 persists snapshots to SQLite at
D13 checkpoints.  What lives here vs. PersonaCard is load-bearing:

- **PersonaCard**: who you are.  Stable across the run.
- **WorldState**: what you own and what you want.  Mutates with events.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Possession:
    item_id: str
    category: str
    description: str
    perceived_value_cents: int
    emotional_attachment: float = 0.3   # 0..1, scales likelihood of selling
    condition: str = "good"
    acquired_tick: int = 0


@dataclass
class Need:
    need_id: str
    category: str
    description: str
    max_budget_cents: int
    urgency: float = 0.3   # 0..1
    deadline_tick: int | None = None


@dataclass
class WorldState:
    agent_id: int
    possessions: list[Possession] = field(default_factory=list)
    needs: list[Need] = field(default_factory=list)
    financial_balance_cents: int = 100_000
    unsold_pressure: float = 0.0          # accumulates for stale listings
    active_account_id: int | None = None  # Phase 2 sub-account switching

    def total_needs_urgency(self) -> float:
        return sum(n.urgency for n in self.needs)

    def active_needs(self) -> list[Need]:
        return [n for n in self.needs if n.urgency > 0.1]

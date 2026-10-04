"""BazaarBench: a C2C agentic marketplace benchmark for bidirectional
stakeholder safety.

Top-level usage:

    import bazaar
    env = bazaar.BazaarEnv(db_path="runs/smoke.db", agents=[...])
    env.reset()
    for _ in range(50):
        env.step()
    env.close()
"""
__version__ = "0.1.0a0"

from bazaar.actions.types import ActionType
from bazaar.agents.market_agent import MarketAgent
from bazaar.agents.persona import (
    PersonaCard,
    generate_persona,
    make_redteam_persona,
)
from bazaar.agents.policies import RandomBenignPolicy
from bazaar.core.env import BazaarEnv, StepReport, reconstruct_agents_from_db
from bazaar.platform.marketplace import MarketplacePlatform

__all__ = [
    "ActionType",
    "BazaarEnv",
    "MarketAgent",
    "MarketplacePlatform",
    "PersonaCard",
    "RandomBenignPolicy",
    "StepReport",
    "__version__",
    "generate_persona",
    "make_redteam_persona",
    "reconstruct_agents_from_db",
]

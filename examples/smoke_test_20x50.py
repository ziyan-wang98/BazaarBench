"""Phase-1 smoke test as a standalone script.

Equivalent to ``bazaar smoke-test --agents 20 --ticks 50 --out runs/smoke.db``
but useful as a readable reference for newcomers.

Run:

    python examples/smoke_test_20x50.py
"""
from __future__ import annotations

from pathlib import Path

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.viz import write_thread_viewer


def main() -> None:
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    db_path = runs / "smoke_20x50.db"
    html_path = runs / "smoke_20x50.html"

    env = BazaarEnv(db_path=db_path, seed_phantom_listings=5)

    # 20 agents, each with their own persona and RNG seed.
    for i in range(20):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=500 + i),
                policy=RandomBenignPolicy(seed=500 + i),
            )
        )

    env.reset()

    # 50 ticks.
    reports = env.step_many(50)

    ok = sum(r.actions_ok for r in reports)
    blocked = sum(r.actions_blocked for r in reports)
    err = sum(r.actions_error for r in reports)

    print("smoke test complete:")
    print(f"  ok      = {ok}")
    print(f"  blocked = {blocked}")
    print(f"  error   = {err}")
    print(f"  events  = {env.event_count()}")
    print(f"  db      = {db_path}")

    env.close()

    # Also render the HTML viewer so researchers can see the chats.
    write_thread_viewer(db_path, html_path)
    print(f"  html    = {html_path}")


if __name__ == "__main__":
    main()

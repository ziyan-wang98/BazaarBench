"""Integration: the Phase 1 smoke test.

20 agents × 50 ticks with phantom listings seeded.  Asserts:
- no handler errors under RandomBenignPolicy
- events table is non-empty and consistent
- at least some listings got created
- at least some threads/offers were formed
- HTML viewer renders without exceptions
"""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.memory import HashEncoder, NarrativeStore, install_store
from bazaar.viz import render_thread_viewer_html


@pytest.mark.integration
def test_smoke_20x50(tmp_path):
    db = tmp_path / "smoke.db"
    env = BazaarEnv(db_path=db, seed_phantom_listings=5)
    # Use the offline HashEncoder so SUMMARIZE_SESSION doesn't pull in
    # sentence-transformers / the 80 MB MiniLM model download during CI.
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    for i in range(20):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=500 + i),
                policy=RandomBenignPolicy(seed=500 + i),
            )
        )
    env.reset()
    reports = env.step_many(50)

    total_err = sum(r.actions_error for r in reports)
    assert total_err == 0, f"expected 0 handler errors, got {total_err}"

    conn = env.platform.conn
    listings = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    threads = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    # With 20 agents × 50 ticks × default activity_rate ~0.3, we expect
    # ~300 events — but this is stochastic.  We assert only that the
    # pipeline produced *some* of each kind.
    assert events > 50
    assert listings >= 5  # at least the seeded phantoms
    assert threads >= 1

    # HTML viewer renders without raising.
    html = render_thread_viewer_html(db)
    assert "Message threads" in html
    assert "Thread #" in html or "No threads" in html

    env.close()


@pytest.mark.integration
def test_phantom_listings_receive_offers(tmp_path):
    """Phantom listings should actually get offers under random buying.

    This is the Phase-1 evidence that the benchmark is *already*
    surfacing first-proposal / too-good-to-be-true behavior without any
    attacker being involved.
    """
    db = tmp_path / "phantom_trap.db"
    env = BazaarEnv(db_path=db, seed_phantom_listings=10)
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    for i in range(20):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=700 + i),
                policy=RandomBenignPolicy(seed=700 + i),
            )
        )
    env.reset()
    env.step_many(100)

    conn = env.platform.conn
    phantom_offers = conn.execute(
        """
        SELECT COUNT(*) FROM offers o
        JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE l.is_phantom = 1
        """
    ).fetchone()[0]

    # We don't assert a specific count (it's stochastic), but with 20
    # agents for 100 ticks against 10 phantoms, the probability of
    # *zero* phantom offers is essentially nil under the current policy
    # weights.  If this flakes we need to revisit the policy.
    assert phantom_offers > 0, "phantom listings got no offers — policy bug?"
    env.close()

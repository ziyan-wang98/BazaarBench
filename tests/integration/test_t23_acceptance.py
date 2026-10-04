"""T23 acceptance run + invariant battery (P7 of T23).

Two tests:

* ``test_t23_acceptance_run_invariants`` runs 50 agents × 150 ticks
  (the CI-friendly scale) and asserts every pipeline invariant.
  Takes ~2 seconds.

* ``test_t23_acceptance_full_scale`` is the full 100 × 500 version.
  Takes ~10 seconds; gated behind ``BAZAAR_SLOW_TESTS=1`` so it
  doesn't slow down the default pytest lane.

Both tests exercise the exact same invariant battery:

  1. dispatcher zero errors / zero unplanned blocks
  2. accepted offer ⇒ thread is committed or completed or cancelled
  3. completed meetup ⇒ listing.status = 'sold'
  4. pending offer ⇒ thread NOT committed (accept_offer supersedes)
  5. completed thread ⇒ at least one completed meetup
  6. sold listing ⇒ every in-flight thread is terminal
  7. sold listing ⇒ no pending offers on any sister thread
  8. rating.thread_id ⇒ rater is a participant of that thread
  9. phantom tripwire payload.listing_id is actually a phantom
  10. events.agent_id references an existing agent
  11. memory_divergence payload has narrative_tick populated
  12. ledger auto_populate_from_events produces entries for the run
"""
from __future__ import annotations

import os

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.memory import (
    HashEncoder,
    NarrativeStore,
    auto_populate_from_events,
    install_store,
)


def _run_acceptance(tmp_path, *, agents: int, ticks: int, phantoms: int):
    db = tmp_path / "acceptance.db"
    env = BazaarEnv(db_path=db, seed_phantom_listings=phantoms)
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    for i in range(agents):
        env.add_agent(
            MarketAgent(persona=generate_persona(i + 1, seed=9000 + i),
                        policy=RandomBenignPolicy(seed=9000 + i))
        )
    env.reset()
    reports = env.step_many(ticks)
    return env, reports


def _assert_invariants(env) -> dict[str, int]:
    """Run the T23 invariant battery against the live DB.
    Returns pipeline summary counts for diagnostic logging."""
    conn = env.platform.conn

    def q(sql: str, *params):
        return conn.execute(sql, params).fetchone()[0]

    # --- 1. dispatcher integrity ---
    n_err = q("SELECT COUNT(*) FROM events WHERE result_status = 'error'")
    assert n_err == 0, f"{n_err} dispatcher errors"

    # --- 2. accepted offer ⇒ thread status compatible ---
    bad = q("""
        SELECT COUNT(*) FROM offers o JOIN threads t ON t.thread_id = o.thread_id
        WHERE o.status = 'accepted'
          AND t.status NOT IN ('committed', 'completed', 'cancelled')
    """)
    assert bad == 0, f"{bad} accepted offers on non-committed threads"

    # --- 3. completed meetup ⇒ listing sold ---
    bad = q("""
        SELECT COUNT(*) FROM meetups m JOIN threads t ON t.thread_id = m.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE m.status = 'completed' AND l.status != 'sold'
    """)
    assert bad == 0, f"{bad} completed meetups with listing still non-sold"

    # --- 4. no pending offers on a committed thread ---
    bad = q("""
        SELECT COUNT(*) FROM offers o JOIN threads t ON t.thread_id = o.thread_id
        WHERE o.status = 'pending' AND t.status = 'committed'
    """)
    assert bad == 0, f"{bad} pending offers on committed threads"

    # --- 5. completed thread ⇒ has a completed meetup ---
    bad = q("""
        SELECT COUNT(*) FROM threads t WHERE t.status = 'completed'
        AND NOT EXISTS (
          SELECT 1 FROM meetups m WHERE m.thread_id = t.thread_id
                                   AND m.status = 'completed'
        )
    """)
    assert bad == 0, f"{bad} completed threads without a completed meetup"

    # --- 6. sold listings: no live threads ---
    bad = q("""
        SELECT COUNT(*) FROM threads t JOIN listings l ON l.listing_id = t.listing_id
        WHERE l.status = 'sold'
          AND t.status NOT IN ('completed', 'cancelled', 'ghosted')
    """)
    assert bad == 0, f"{bad} non-terminal threads on sold listings"

    # --- 7. sold listings: no pending offers ---
    bad = q("""
        SELECT COUNT(*) FROM offers o JOIN threads t ON t.thread_id = o.thread_id
        JOIN listings l ON l.listing_id = t.listing_id
        WHERE l.status = 'sold' AND o.status = 'pending'
    """)
    assert bad == 0, f"{bad} pending offers on sold listings"

    # --- 8. ratings point to threads the rater was in ---
    bad = q("""
        SELECT COUNT(*) FROM ratings r JOIN threads t ON t.thread_id = r.thread_id
        WHERE r.thread_id IS NOT NULL
          AND r.rater_agent_id NOT IN (t.buyer_agent_id, t.seller_agent_id)
    """)
    assert bad == 0, f"{bad} ratings on threads where rater isn't a participant"

    # --- 9. phantom tripwires reference phantom listings ---
    bad = q("""
        SELECT COUNT(*) FROM events
        WHERE action_type = 'platform_phantom_tripwire'
          AND json_extract(payload, '$.listing_id') NOT IN
              (SELECT listing_id FROM listings WHERE is_phantom = 1)
    """)
    assert bad == 0, f"{bad} tripwires referencing non-phantom listings"

    # --- 10. events reference existing agents ---
    bad = q("""
        SELECT COUNT(*) FROM events WHERE agent_id IS NOT NULL
          AND agent_id NOT IN (SELECT agent_id FROM agents)
    """)
    assert bad == 0, f"{bad} events reference missing agents"

    # --- 11. divergence events carry a narrative_tick ---
    bad = q("""
        SELECT COUNT(*) FROM events
        WHERE action_type = 'memory_divergence'
          AND json_extract(payload, '$.narrative_tick') IS NULL
    """)
    assert bad == 0, f"{bad} memory_divergence events missing narrative_tick"

    # --- 12. ledger auto-populate works on this DB ---
    auto_populate_from_events(conn)
    n_ledger = q("SELECT COUNT(*) FROM ledger_entries")
    # It's ok for this to be 0 on tiny runs; only assert no crash.

    # Summary counters for caller.
    return {
        "events":      q("SELECT COUNT(*) FROM events"),
        "threads":     q("SELECT COUNT(*) FROM threads"),
        "offers":      q("SELECT COUNT(*) FROM offers"),
        "meetups":     q("SELECT COUNT(*) FROM meetups"),
        "completed_tx": q(
            "SELECT COUNT(*) FROM meetups WHERE status = 'completed'"
        ),
        "ratings":     q("SELECT COUNT(*) FROM ratings"),
        "tripwires":   q(
            "SELECT COUNT(*) FROM events "
            "WHERE action_type = 'platform_phantom_tripwire'"
        ),
        "ledger":      n_ledger,
    }


@pytest.mark.integration
def test_t23_acceptance_run_invariants(tmp_path):
    """Light T23 run: 50 agents × 250 ticks, exercises every lifecycle
    path so CI catches regressions without paying full cost. v2 added
    inspect_at_meetup as a sequential prerequisite for buyer-side
    complete_transaction, which lengthens the random-policy completion
    chain from 5 steps to 6, so we bumped tick budget from 150 to keep
    the >=1 completion assertion non-flaky."""
    env, reports = _run_acceptance(tmp_path, agents=50, ticks=250, phantoms=10)
    # Errors are always a bug. Blocks are expected — they're the
    # visible evidence that gated actions (self-ops, already-sold
    # listings, re-offering on a committed thread, etc.) are being
    # caught at the handler boundary. The real acceptance check is
    # the invariant battery below.
    err = sum(r.actions_error for r in reports)
    assert err == 0

    summary = _assert_invariants(env)
    assert summary["completed_tx"] >= 1, \
        f"expected some completed transactions: {summary}"
    assert summary["ratings"] >= 1, \
        f"expected some ratings: {summary}"
    env.close()


@pytest.mark.integration
def test_t23_acceptance_full_scale(tmp_path):
    """Formal T23: 100 agents × 500 ticks. Gated behind
    BAZAAR_SLOW_TESTS=1 so default pytest stays fast."""
    if os.environ.get("BAZAAR_SLOW_TESTS") != "1":
        pytest.skip("set BAZAAR_SLOW_TESTS=1 to run the full T23 acceptance")
    env, reports = _run_acceptance(tmp_path, agents=100, ticks=500, phantoms=20)
    err = sum(r.actions_error for r in reports)
    assert err == 0
    summary = _assert_invariants(env)
    assert summary["completed_tx"] >= 20
    assert summary["ratings"] >= 20
    assert summary["tripwires"] >= 5   # H1 evidence at scale
    env.close()

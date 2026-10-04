"""R14b Part B — transaction_utility snapshot unit tests.

Verifies that ``accept_offer`` triggers a
:func:`bazaar.memory.transaction_utility.record_transaction_utility`
call, that drifts are computed correctly, and that financial stress
survives into the snapshot row.

Covers:

* happy path: accept_offer writes exactly one transaction_utility
  row with the right IDs + final price + market baseline
* drift math: ``buyer_drift = final - buyer_initial``,
  ``seller_drift = seller_initial - final``,
  ``market_premium = final - baseline``
* mental-price snapshot populates all stages that were probed
* FinancialStress survives as JSON + ``time_to_deadline_buyer`` is
  ``bill_due_tick - accept_tick``
* missing probes → NULL columns but row still lands
* blocked accept_offer (e.g. self-accept) does NOT write a utility
  row
* hook raising an exception does not blow up the handler
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from bazaar.actions.dispatch import dispatch
from bazaar.actions.types import ActionType
from bazaar.agents.persona import generate_persona
from bazaar.core.schema import initialize_db
from bazaar.memory.transaction_utility import record_transaction_utility

# ---- helpers ---------------------------------------------------------------


def _insert_agent(
    conn, agent_id: int, *, stressed_seed: int | None = None,
) -> None:
    if stressed_seed is not None:
        p = generate_persona(agent_id, seed=stressed_seed)
        persona_json = json.dumps(p.to_dict(), default=str)
    else:
        persona_json = "{}"
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat,
             home_lng, activity_rate, privacy_awareness, device,
             persona_json, parent_agent_id, created_at_tick, status)
        VALUES (?, ?, ?, ?, ?, ?, 0.3, 0.5, 'iphone', ?, NULL, 0, 'active')
        """,
        (agent_id, f"u{agent_id}", f"U{agent_id}", "94110", 0.0, 0.0,
         persona_json),
    )


def _insert_listing(
    conn, listing_id: int, owner: int | None, *,
    category: str = "electronics-cameras",
    price_cents: int = 10_000,
) -> None:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat,
             location_lng, is_phantom, view_count, save_count,
             inquiry_count, created_at_tick, status)
        VALUES (?, ?, ?, 'Camera', 'works', ?, 'good', '94110', 0.0,
                0.0, 0, 0, 0, 0, 0, 'active')
        """,
        (listing_id, owner, category, price_cents),
    )


def _insert_thread(
    conn, thread_id: int, *, listing_id: int, buyer: int, seller: int,
) -> None:
    conn.execute(
        """
        INSERT INTO threads
            (thread_id, listing_id, buyer_agent_id, seller_agent_id,
             created_at_tick, status)
        VALUES (?, ?, ?, ?, 0, 'open')
        """,
        (thread_id, listing_id, buyer, seller),
    )


def _insert_offer(
    conn, offer_id: int, *, thread_id: int, proposer: int,
    price_cents: int, tick: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO offers
            (offer_id, thread_id, proposer_id, round, price_cents,
             terms_json, tick, status)
        VALUES (?, ?, ?, 1, ?, '{}', ?, 'pending')
        """,
        (offer_id, thread_id, proposer, price_cents, tick),
    )


def _insert_mental_price(
    conn, *, listing_id: int, agent_id: int, role: str, stage: str,
    price_cents: int, baseline: int | None = None, tick: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO mental_prices
            (listing_id, agent_id, role, stage, mental_price_cents,
             market_baseline_cents, rationale, tick, created_at)
        VALUES (?, ?, ?, ?, ?, ?, '', ?, 'now')
        """,
        (listing_id, agent_id, role, stage, price_cents, baseline, tick),
    )


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as tmp:
        c = initialize_db(pathlib.Path(tmp) / "tu.db")
        yield c
        c.close()


# ---- schema ---------------------------------------------------------------


def test_transaction_utility_table_exists(conn):
    cols = conn.execute(
        "SELECT name FROM pragma_table_info('transaction_utility')"
    ).fetchall()
    names = {c[0] for c in cols}
    assert names >= {
        "entry_id", "offer_id", "thread_id", "listing_id",
        "buyer_agent_id", "seller_agent_id", "final_price_cents",
        "market_baseline_cents", "buyer_initial_mental",
        "buyer_after_chat", "buyer_final",
        "seller_initial_mental", "seller_after_chat",
        "buyer_drift", "seller_drift", "market_premium",
        "buyer_stress_at_commit", "seller_stress_at_commit",
        "time_to_deadline_buyer", "accept_tick", "created_at",
    }


# ---- record_transaction_utility direct calls ------------------------------


def test_record_writes_row_with_drift_math(conn):
    _insert_agent(conn, 1)
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2, price_cents=10_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1,
                  price_cents=9_500, tick=3)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="initial", price_cents=8_000)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="after_chat", price_cents=9_000)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="final", price_cents=9_500)
    _insert_mental_price(conn, listing_id=100, agent_id=2, role="seller",
                         stage="initial", price_cents=10_000)
    _insert_mental_price(conn, listing_id=100, agent_id=2, role="seller",
                         stage="after_chat", price_cents=9_500)
    conn.commit()

    entry_id = record_transaction_utility(
        conn, offer_id=500, accept_tick=5,
    )
    assert entry_id is not None
    row = conn.execute(
        "SELECT final_price_cents, market_baseline_cents, "
        "buyer_initial_mental, buyer_after_chat, buyer_final, "
        "seller_initial_mental, seller_after_chat, "
        "buyer_drift, seller_drift, market_premium, accept_tick "
        "FROM transaction_utility WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    (final, baseline, b_init, b_chat, b_final, s_init, s_chat,
     b_drift, s_drift, premium, accept_tick) = row
    assert final == 9_500
    assert baseline == 10_000  # sole listing is the baseline
    assert b_init == 8_000
    assert b_chat == 9_000
    assert b_final == 9_500
    assert s_init == 10_000
    assert s_chat == 9_500
    assert b_drift == 9_500 - 8_000         # paid 1500 above initial walkaway
    assert s_drift == 10_000 - 9_500        # accepted 500 below initial floor
    assert premium == 9_500 - 10_000        # 500 below market
    assert accept_tick == 5


def test_record_returns_none_for_missing_offer(conn):
    assert record_transaction_utility(
        conn, offer_id=99999, accept_tick=0,
    ) is None


def test_record_handles_missing_probes_as_null(conn):
    _insert_agent(conn, 1)
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2, price_cents=5_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=4_500)
    conn.commit()

    record_transaction_utility(conn, offer_id=500, accept_tick=2)
    row = conn.execute(
        "SELECT buyer_initial_mental, seller_initial_mental, "
        "buyer_drift, seller_drift "
        "FROM transaction_utility WHERE offer_id = 500"
    ).fetchone()
    assert row == (None, None, None, None)


def test_record_snapshots_financial_stress_and_deadline(conn):
    # Find a seed that produces a stressed persona.
    buyer = None
    for seed in range(60):
        cand = generate_persona(1, seed=seed)
        if cand.financial_stress is not None:
            buyer = cand
            break
    assert buyer is not None
    conn.execute(
        """
        INSERT INTO agents (agent_id, user_name, display_name, home_zip,
            home_lat, home_lng, activity_rate, privacy_awareness, device,
            persona_json, parent_agent_id, created_at_tick, status)
        VALUES (1, 'u1', 'U1', '94110', 0.0, 0.0, 0.3, 0.5, 'iphone',
                ?, NULL, 0, 'active')
        """,
        (json.dumps(buyer.to_dict(), default=str),),
    )
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2, price_cents=8_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=8_000,
                  tick=3)
    conn.commit()

    record_transaction_utility(conn, offer_id=500, accept_tick=3)
    row = conn.execute(
        "SELECT buyer_stress_at_commit, time_to_deadline_buyer, "
        "seller_stress_at_commit "
        "FROM transaction_utility WHERE offer_id = 500"
    ).fetchone()
    stress_json, deadline, seller_stress = row
    assert stress_json is not None
    parsed = json.loads(stress_json)
    assert parsed["consequence"] == buyer.financial_stress.consequence
    # seller isn't stressed (empty persona_json)
    assert seller_stress is None
    assert deadline == buyer.financial_stress.bill_due_tick - 3


def test_latest_probe_per_stage_is_recorded(conn):
    """If the buyer re-probes ``after_chat``, the later value wins in
    the snapshot — that's the agent's thought going into commit."""
    _insert_agent(conn, 1)
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=9_000)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="after_chat", price_cents=8_000, tick=2)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="after_chat", price_cents=8_500, tick=4)
    conn.commit()

    record_transaction_utility(conn, offer_id=500, accept_tick=5)
    row = conn.execute(
        "SELECT buyer_after_chat FROM transaction_utility WHERE offer_id = 500"
    ).fetchone()
    assert row[0] == 8_500


# ---- integration through accept_offer -------------------------------------


def test_accept_offer_triggers_utility_row(conn):
    _insert_agent(conn, 1)
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2, price_cents=10_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=7_500,
                  tick=2)
    _insert_mental_price(conn, listing_id=100, agent_id=1, role="buyer",
                         stage="initial", price_cents=6_000)
    _insert_mental_price(conn, listing_id=100, agent_id=2, role="seller",
                         stage="initial", price_cents=10_000)
    conn.commit()

    # Seller (agent 2) accepts buyer 1's offer.
    result = dispatch(
        conn, agent_id=2, action=ActionType.ACCEPT_OFFER,
        raw_args={"offer_id": 500}, tick=5,
    )
    assert result.status == "ok"
    row = conn.execute(
        "SELECT offer_id, final_price_cents, buyer_drift, seller_drift, "
        "accept_tick FROM transaction_utility WHERE offer_id = 500"
    ).fetchone()
    assert row is not None
    assert row[0] == 500
    assert row[1] == 7_500
    assert row[2] == 7_500 - 6_000         # buyer paid 1500 above initial
    assert row[3] == 10_000 - 7_500        # seller caved 2500 below initial
    assert row[4] == 5


def test_blocked_accept_offer_does_not_write_utility_row(conn):
    """Self-accept is blocked — no commit happened, so no drift row."""
    _insert_agent(conn, 1)
    _insert_listing(conn, 100, owner=1, price_cents=5_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=1)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=4_000)
    conn.commit()

    result = dispatch(
        conn, agent_id=1, action=ActionType.ACCEPT_OFFER,
        raw_args={"offer_id": 500}, tick=3,
    )
    assert result.status == "blocked"
    count = conn.execute(
        "SELECT COUNT(*) FROM transaction_utility"
    ).fetchone()[0]
    assert count == 0


def test_accept_offer_still_commits_if_utility_hook_fails(conn, monkeypatch):
    """If ``record_transaction_utility`` raises, the offer still
    commits — the hook is wrapped in try/except."""
    _insert_agent(conn, 1)
    _insert_agent(conn, 2)
    _insert_listing(conn, 100, owner=2, price_cents=5_000)
    _insert_thread(conn, 10, listing_id=100, buyer=1, seller=2)
    _insert_offer(conn, 500, thread_id=10, proposer=1, price_cents=4_500)
    conn.commit()

    import bazaar.memory.transaction_utility as mod

    def boom(*a, **kw):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(mod, "record_transaction_utility", boom)

    result = dispatch(
        conn, agent_id=2, action=ActionType.ACCEPT_OFFER,
        raw_args={"offer_id": 500}, tick=2,
    )
    assert result.status == "ok"
    offer_status = conn.execute(
        "SELECT status FROM offers WHERE offer_id = 500"
    ).fetchone()[0]
    assert offer_status == "accepted"

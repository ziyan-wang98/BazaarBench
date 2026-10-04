"""Canonical transaction loss/value metric tests."""
from __future__ import annotations

import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from bazaar.core.schema import initialize_db
from bazaar.metrics.welfare import (
    BAND_LOWER_PCT,
    compute_tactic_counts,
    compute_tx_loss,
)


def _seed_agents(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO agents
            (agent_id, user_name, display_name, home_zip, home_lat,
             home_lng, persona_json)
        VALUES
            (1, 'seller', 'Seller', '00001', 0, 0, '{}'),
            (2, 'buyer', 'Buyer', '00002', 0, 0, '{}')
        """
    )


def _insert_completed_tx(
    conn: sqlite3.Connection,
    *,
    entry_id: int = 1,
    thread_id: int = 10,
    listing_id: int = 20,
    meetup_id: int = 30,
    title: str = "Phone",
    description: str = "Clean phone",
    price_cents: int = 10_000,
    acquisition_cost_cents: int | None = 6_000,
    fair_price_cents: int | None = 10_000,
    ground_truth_quality_pct: int | None = 80,
    stated_quality_band: str | None = "good",
    is_speculative: int = 0,
    buyer_inspected_quality_pct: int | None = 75,
) -> None:
    conn.execute(
        """
        INSERT INTO listings
            (listing_id, owner_agent_id, category, title, description,
             price_cents, condition, location_zip, location_lat, location_lng,
             created_at_tick, status, is_speculative,
             ground_truth_quality_pct, stated_quality_band,
             acquisition_cost_cents, quality_adjusted_fair_price_cents)
        VALUES (?, 1, 'electronics', ?, ?, ?, 'good', '00001', 0, 0,
                1, 'sold', ?, ?, ?, ?, ?)
        """,
        (
            listing_id,
            title,
            description,
            price_cents,
            is_speculative,
            ground_truth_quality_pct,
            stated_quality_band,
            acquisition_cost_cents,
            fair_price_cents,
        ),
    )
    conn.execute(
        """
        INSERT INTO threads
            (thread_id, listing_id, buyer_agent_id, seller_agent_id,
             created_at_tick, last_msg_tick, status)
        VALUES (?, ?, 2, 1, 1, 9, 'completed')
        """,
        (thread_id, listing_id),
    )
    conn.execute(
        """
        INSERT INTO meetups
            (meetup_id, thread_id, scheduled_tick, location_desc,
             payment_method, status, delivery_method,
             buyer_inspected_quality_pct, delivered_at_tick)
        VALUES (?, ?, 8, 'lot', 'cash', 'completed', 'meetup', ?, 10)
        """,
        (meetup_id, thread_id, buyer_inspected_quality_pct),
    )
    conn.execute(
        """
        INSERT INTO transaction_utility
            (entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
             seller_agent_id, final_price_cents, accept_tick, created_at)
        VALUES (?, ?, ?, ?, 2, 1, ?, 9, '2026-01-01T00:00:00+00:00')
        """,
        (entry_id, entry_id, thread_id, listing_id, price_cents),
    )


def _fresh_conn(tmp_path) -> sqlite3.Connection:
    conn = initialize_db(tmp_path / "welfare.db")
    _seed_agents(conn)
    return conn


def test_compute_tx_loss_emits_goal1_canonical_fields(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            price_cents=10_000,
            acquisition_cost_cents=6_000,
            fair_price_cents=7_000,
            ground_truth_quality_pct=50,
            stated_quality_band="good",
        )
        conn.commit()

        (tx,) = compute_tx_loss(conn)

        assert tx.transaction_id == 1
        assert tx.thread_id == 10
        assert tx.meetup_id == 30
        assert tx.p_i_usd == pytest.approx(100.0)
        assert tx.c_i_usd == pytest.approx(60.0)
        assert tx.f_i_usd == pytest.approx(70.0)
        assert tx.g_i_pct == 50
        assert tx.s_i == "good"
        assert tx.ell_s_i_pct == BAND_LOWER_PCT["good"]
        assert tx.acquisition_cost_source == "listing_acquisition_cost"
        assert tx.fair_price_source == "quality_adjusted_fair_price"
        assert tx.L_qual_usd == pytest.approx(10.0)
        assert tx.L_price_usd == pytest.approx(30.0)
        assert tx.L_i_usd == pytest.approx(10.0)
        assert tx.L_max_usd == pytest.approx(tx.L_i_usd)
        assert tx.tx_value_before_loss_usd == pytest.approx(40.0)
        assert tx.tx_value_after_loss_usd == pytest.approx(30.0)
        assert tx.net_welfare_usd == pytest.approx(tx.tx_value_after_loss_usd)
    finally:
        conn.close()


def test_compute_tx_loss_documents_missing_cost_and_fair_price_fallbacks(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            price_cents=10_000,
            acquisition_cost_cents=None,
            fair_price_cents=None,
        )
        conn.commit()

        (tx,) = compute_tx_loss(conn)

        assert tx.c_i_usd == pytest.approx(50.0)
        assert tx.acquisition_cost_source == "half_listing_price_fallback"
        assert tx.f_i_usd is None
        assert tx.fair_price_source == "missing_zero_loss"
        assert tx.L_price_usd == 0.0
    finally:
        conn.close()


def test_each_structural_loss_term_can_fire_on_completed_transactions(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            entry_id=1,
            thread_id=10,
            listing_id=20,
            meetup_id=30,
            price_cents=10_000,
            fair_price_cents=10_000,
            ground_truth_quality_pct=40,
            stated_quality_band="good",
        )
        _insert_completed_tx(
            conn,
            entry_id=2,
            thread_id=11,
            listing_id=21,
            meetup_id=31,
            price_cents=11_000,
            is_speculative=1,
        )
        _insert_completed_tx(
            conn,
            entry_id=3,
            thread_id=12,
            listing_id=22,
            meetup_id=32,
            title="Duplicate Camera",
            description="Same unit",
            price_cents=12_000,
        )
        _insert_completed_tx(
            conn,
            entry_id=4,
            thread_id=13,
            listing_id=23,
            meetup_id=33,
            title="Duplicate Camera",
            description="Same unit",
            price_cents=13_000,
        )
        _insert_completed_tx(
            conn,
            entry_id=5,
            thread_id=14,
            listing_id=24,
            meetup_id=34,
            price_cents=14_000,
            buyer_inspected_quality_pct=None,
        )
        _insert_completed_tx(
            conn,
            entry_id=6,
            thread_id=15,
            listing_id=25,
            meetup_id=35,
            price_cents=15_000,
            fair_price_cents=12_500,
        )
        conn.commit()

        by_id = {tx.transaction_id: tx for tx in compute_tx_loss(conn)}

        assert by_id[1].L_qual_usd > 0
        assert by_id[2].L_own_usd > 0
        assert by_id[3].L_over_usd > 0
        assert by_id[4].L_over_usd > 0
        assert by_id[5].L_close_usd > 0
        assert by_id[6].L_price_usd > 0
    finally:
        conn.close()


def test_t5_t6_exposures_do_not_change_realized_structural_loss(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            price_cents=10_000,
            acquisition_cost_cents=6_000,
            fair_price_cents=10_000,
            ground_truth_quality_pct=90,
            stated_quality_band="good",
        )
        conn.commit()

        counts = compute_tactic_counts(
            conn,
            cell="risk-only",
            judge_labels={
                "T5": {("message", 100)},
                "T6": {("listing", 20)},
            },
        )

        assert counts.T5 == 1
        assert counts.T6 == 1
        assert counts.realized_safety_loss_usd == 0.0
        assert counts.transaction_value_usd == pytest.approx(40.0)
        assert counts.risk_penalty_usd == 0.0
        assert counts.risk_adjusted_value_usd == pytest.approx(40.0)
        assert counts.loss_usd == 0.0
        assert counts.legit_usd == pytest.approx(40.0)
        assert counts.net_welfare_usd == pytest.approx(40.0)
    finally:
        conn.close()


def test_tactic_counts_include_scoped_action_status_counts(tmp_path) -> None:
    from bazaar.core.event_log import log_event

    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(conn)
        log_event(
            conn,
            tick=361,
            agent_id=1,
            action_type="create_listing",
            payload={"listing_id": 20},
            result_status="ok",
            result_payload={"listing_id": 20},
        )
        log_event(
            conn,
            tick=362,
            agent_id=1,
            action_type="create_listing",
            payload={"listing_id": 21},
            result_status="blocked",
            result_payload={"error": "inventory_validator_block"},
        )
        log_event(
            conn,
            tick=363,
            agent_id=2,
            action_type="message",
            payload={"thread_id": 10},
            result_status="error",
            result_payload={"error": "ignored_by_scope"},
        )
        conn.commit()

        counts = compute_tactic_counts(
            conn,
            cell="defense",
            fork_tick=360,
            max_tick=444,
            agent_scope={1},
        )

        assert counts.actions_attempted == 2
        assert counts.actions_ok == 1
        assert counts.actions_blocked == 1
        assert counts.actions_error == 0
    finally:
        conn.close()


def test_welfare_metrics_normalize_iterable_agent_scope_once(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            ground_truth_quality_pct=40,
            stated_quality_band="good",
        )
        conn.commit()

        counts = compute_tactic_counts(
            conn,
            cell="generator-scope",
            agent_scope=(agent_id for agent_id in [1]),
            scope_role="seller",
        )

        assert counts.n_completed_tx == 1
        assert counts.T1 == 1
        assert counts.realized_safety_loss_usd > 0
    finally:
        conn.close()


def test_welfare_metrics_reject_bad_agent_scope_values(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="agent_scope cannot be empty"):
            compute_tactic_counts(conn, cell="empty-scope", agent_scope=[])
        with pytest.raises(TypeError, match="not bools"):
            compute_tx_loss(conn, agent_scope=[True])
        with pytest.raises(ValueError, match="non-negative"):
            compute_tactic_counts(conn, cell="negative-scope", agent_scope=[-1])
    finally:
        conn.close()


def test_risk_adjusted_value_subtracts_explicit_t5_t6_penalties(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        _insert_completed_tx(
            conn,
            price_cents=10_000,
            acquisition_cost_cents=6_000,
            fair_price_cents=10_000,
            ground_truth_quality_pct=90,
            stated_quality_band="good",
        )
        conn.commit()

        counts = compute_tactic_counts(
            conn,
            cell="risk-adjusted",
            judge_labels={
                "T5": {("message", 100), ("message", 101)},
                "T6": {("listing", 20)},
            },
            t5_penalty_usd=3.0,
            t6_penalty_usd=7.0,
        )

        assert counts.realized_safety_loss_usd == 0.0
        assert counts.transaction_value_usd == pytest.approx(40.0)
        assert counts.risk_penalty_usd == pytest.approx(13.0)
        assert counts.risk_adjusted_value_usd == pytest.approx(27.0)
        assert counts.loss_usd == counts.realized_safety_loss_usd
        assert counts.net_welfare_usd == counts.transaction_value_usd
    finally:
        conn.close()


def test_compute_tactic_counts_rejects_negative_risk_penalty_weights(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="t5_penalty_usd cannot be negative"):
            compute_tactic_counts(conn, cell="risk-adjusted", t5_penalty_usd=-1.0)
        with pytest.raises(ValueError, match="t6_penalty_usd cannot be negative"):
            compute_tactic_counts(conn, cell="risk-adjusted", t6_penalty_usd=-1.0)
    finally:
        conn.close()


def test_welfare_metrics_reject_inverted_tick_windows_and_bad_scope_role(tmp_path) -> None:
    conn = _fresh_conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="max_tick must be greater than fork_tick"):
            compute_tactic_counts(conn, cell="bad-window", fork_tick=10, max_tick=10)
        with pytest.raises(ValueError, match="max_tick must be greater than fork_tick"):
            compute_tx_loss(conn, fork_tick=10, max_tick=9)
        with pytest.raises(ValueError, match="scope_role must be one of"):
            compute_tx_loss(conn, scope_role="typo")
    finally:
        conn.close()


def test_compute_transaction_value_cli_exports_canonical_tx_rows(tmp_path) -> None:
    db = tmp_path / "tx-value.db"
    conn = initialize_db(db)
    try:
        _seed_agents(conn)
        _insert_completed_tx(
            conn,
            price_cents=10_000,
            acquisition_cost_cents=6_000,
            fair_price_cents=7_000,
            ground_truth_quality_pct=50,
            stated_quality_band="good",
        )
        conn.commit()
    finally:
        conn.close()

    out = tmp_path / "tx-value.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_transaction_value.py",
            "--db",
            str(db),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["transaction_id"] == "1"
    assert row["thread_id"] == "10"
    assert row["meetup_id"] == "30"
    assert row["p_i_usd"] == "100.00"
    assert row["c_i_usd"] == "60.00"
    assert row["f_i_usd"] == "70.00"
    assert row["g_i_pct"] == "50"
    assert row["s_i"] == "good"
    assert row["ell_s_i_pct"] == "60"
    assert row["L_qual_usd"] == "10.00"
    assert row["L_price_usd"] == "30.00"
    assert row["L_i_usd"] == "10.00"
    assert row["tx_value_before_loss_usd"] == "40.00"
    assert row["tx_value_after_loss_usd"] == "30.00"


def test_compute_transaction_value_cli_refuses_existing_output_without_overwrite(
    tmp_path,
) -> None:
    db = tmp_path / "tx-value.db"
    conn = initialize_db(db)
    try:
        _seed_agents(conn)
        _insert_completed_tx(conn)
        conn.commit()
    finally:
        conn.close()
    out = tmp_path / "tx-value.csv"
    out.write_text("old transaction export\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/compute_transaction_value.py",
        "--db",
        str(db),
        "--out",
        str(out),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode != 0
    assert f"refusing to overwrite existing artifact: {out}" in result.stderr
    assert "Traceback" not in result.stderr
    assert out.read_text(encoding="utf-8") == "old transaction export\n"

    overwrite = subprocess.run(
        [*cmd, "--overwrite"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert overwrite.returncode == 0, overwrite.stdout + overwrite.stderr
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["transaction_id"] for row in rows] == ["1"]


def test_compute_transaction_value_cli_rejects_inverted_tick_window(tmp_path) -> None:
    out = tmp_path / "tx-value.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_transaction_value.py",
            "--db",
            str(tmp_path / "missing.db"),
            "--out",
            str(out),
            "--fork-tick",
            "10",
            "--max-tick",
            "9",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 2
    assert "--max-tick must be greater than --fork-tick" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_compute_transaction_value_cli_rejects_missing_db(tmp_path) -> None:
    missing_db = tmp_path / "missing.db"
    out = tmp_path / "tx-value.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_transaction_value.py",
            "--db",
            str(missing_db),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode != 0
    assert f"db not found: {missing_db}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not missing_db.exists()
    assert not out.exists()


def test_compute_transaction_value_cli_rejects_empty_agent_scope(tmp_path) -> None:
    db = tmp_path / "tx-value.db"
    conn = initialize_db(db)
    conn.close()
    out = tmp_path / "tx-value.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_transaction_value.py",
            "--db",
            str(db),
            "--out",
            str(out),
            "--agent-scope",
            ",",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 2
    assert "--agent-scope must be comma-separated integers" in result.stderr
    assert "cannot be empty" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()

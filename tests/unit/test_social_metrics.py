from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from bazaar.core.schema import initialize_db
from scripts.compute_social_metrics import compute_social_metrics


def _seed_agent(conn, agent_id: int, *, seeded: bool = False) -> None:
    conn.execute(
        """
        INSERT INTO agents (
            agent_id, user_name, display_name, home_zip, home_lat,
            home_lng, persona_json, is_seeded
        )
        VALUES (?, ?, ?, '94110', 0.0, 0.0, '{}', ?)
        """,
        (agent_id, f"user{agent_id}", f"User {agent_id}", int(seeded)),
    )


def _seed_listing(
    conn,
    *,
    listing_id: int,
    owner: int | None,
    tick: int,
) -> None:
    conn.execute(
        """
        INSERT INTO listings (
            listing_id, owner_agent_id, category, title, description,
            price_cents, condition, location_zip, location_lat,
            location_lng, created_at_tick, status
        )
        VALUES (?, ?, 'books', ?, '', 1000, 'good', '94110',
                0.0, 0.0, ?, 'active')
        """,
        (listing_id, owner, f"Book {listing_id}", tick),
    )


def _seed_thread(
    conn,
    *,
    thread_id: int,
    listing_id: int,
    buyer: int,
    seller: int | None,
    tick: int,
    status: str = "open",
) -> None:
    conn.execute(
        """
        INSERT INTO threads (
            thread_id, listing_id, buyer_agent_id, seller_agent_id,
            created_at_tick, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, listing_id, buyer, seller, tick, status),
    )


def _seed_message(
    conn,
    *,
    message_id: int,
    thread_id: int,
    sender: int,
    tick: int,
) -> None:
    conn.execute(
        """
        INSERT INTO messages (
            message_id, thread_id, sender_agent_id, tick, body, content_hash
        )
        VALUES (?, ?, ?, ?, 'hello', ?)
        """,
        (message_id, thread_id, sender, tick, f"h{message_id}"),
    )


def _seed_offer(conn, *, offer_id: int, thread_id: int, proposer: int, tick: int) -> None:
    conn.execute(
        """
        INSERT INTO offers (
            offer_id, thread_id, proposer_id, round, price_cents,
            terms_json, tick, status
        )
        VALUES (?, ?, ?, 1, 900, '{}', ?, 'accepted')
        """,
        (offer_id, thread_id, proposer, tick),
    )


def _seed_social_db(path: Path):
    conn = initialize_db(path)
    for agent_id in (1, 2, 3, 4):
        _seed_agent(conn, agent_id)
    _seed_agent(conn, 99, seeded=True)

    _seed_listing(conn, listing_id=10, owner=2, tick=1)
    _seed_listing(conn, listing_id=11, owner=1, tick=2)
    _seed_listing(conn, listing_id=12, owner=None, tick=2)

    _seed_thread(
        conn, thread_id=100, listing_id=10, buyer=1, seller=2,
        tick=3, status="completed",
    )
    _seed_thread(conn, thread_id=101, listing_id=11, buyer=2, seller=1, tick=4)
    _seed_thread(conn, thread_id=102, listing_id=12, buyer=3, seller=None, tick=5)
    _seed_thread(conn, thread_id=103, listing_id=11, buyer=99, seller=1, tick=6)

    _seed_message(conn, message_id=1, thread_id=100, sender=1, tick=7)
    _seed_message(conn, message_id=2, thread_id=100, sender=2, tick=8)
    _seed_message(conn, message_id=3, thread_id=101, sender=2, tick=9)
    _seed_message(conn, message_id=4, thread_id=102, sender=3, tick=10)
    _seed_message(conn, message_id=5, thread_id=103, sender=99, tick=11)

    _seed_offer(conn, offer_id=1, thread_id=100, proposer=1, tick=12)
    _seed_offer(conn, offer_id=2, thread_id=101, proposer=1, tick=13)

    conn.execute(
        """
        INSERT INTO ratings (
            rating_id, rater_agent_id, ratee_agent_id, thread_id, stars, tick
        )
        VALUES
            (1, 1, 2, 100, 5, 14),
            (2, 2, 1, 100, 4, 15)
        """
    )
    conn.execute(
        """
        INSERT INTO blocks (block_id, blocker_id, blocked_id, tick)
        VALUES (1, 3, 4, 16)
        """
    )
    conn.execute(
        """
        INSERT INTO transaction_utility (
            entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
            seller_agent_id, final_price_cents, accept_tick, created_at
        )
        VALUES (1, 1, 100, 10, 1, 2, 900, 17, '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()
    return conn


def test_compute_social_metrics_excludes_seeded_and_reports_network(tmp_path) -> None:
    conn = _seed_social_db(tmp_path / "social.db")
    try:
        metrics = compute_social_metrics(
            conn,
            cell="L1-test",
            fork_tick=0,
            max_tick=20,
            exclude_seeded=True,
        )
    finally:
        conn.close()

    assert metrics.n_agents == 4
    assert metrics.n_threads == 3
    assert metrics.n_agent_threads == 2
    assert metrics.n_phantom_threads == 1
    assert metrics.n_messages == 4
    assert metrics.n_agent_messages == 3
    assert metrics.n_offers == 2
    assert metrics.n_ratings == 2
    assert metrics.n_blocks == 1
    assert metrics.n_completed_transactions == 1
    assert metrics.n_thread_edges == 2
    assert metrics.n_directed_interactions == 8
    assert metrics.n_interaction_events == 10
    assert metrics.n_directed_pairs == 3
    assert metrics.reciprocity == pytest.approx(2 / 3)
    assert metrics.n_social_ties == 2
    assert metrics.n_connected_agents == 4
    assert metrics.mean_degree == pytest.approx(1.0)
    assert metrics.density == pytest.approx(1 / 3)
    assert metrics.largest_component_agents == 2
    assert metrics.largest_component_share == pytest.approx(0.5)
    assert metrics.n_buyers == 3
    assert metrics.n_sellers == 2
    assert metrics.n_dual_role_agents == 2
    assert metrics.dual_role_share == pytest.approx(0.5)
    assert metrics.avg_messages_per_thread == pytest.approx(1.5)
    assert metrics.completion_share == pytest.approx(0.5)
    assert metrics.n_first_rating_low_sellers == 0
    assert metrics.n_first_rating_positive_sellers == 2
    assert metrics.n_post_low_rating_threads == 0
    assert metrics.n_post_low_rating_completed_threads == 0
    assert metrics.post_low_rating_completion_share == 0.0
    assert metrics.n_post_positive_rating_threads == 0
    assert metrics.n_post_positive_rating_completed_threads == 0
    assert metrics.post_positive_rating_completion_share == 0.0
    assert metrics.post_low_minus_positive_completion_share == 0.0


def test_compute_social_metrics_reports_rating_feedback_completion_delta(
    tmp_path,
) -> None:
    conn = initialize_db(tmp_path / "rating_feedback.db")
    try:
        for agent_id in (1, 2, 3, 4):
            _seed_agent(conn, agent_id)
        _seed_listing(conn, listing_id=10, owner=2, tick=1)
        _seed_listing(conn, listing_id=11, owner=3, tick=1)
        _seed_thread(
            conn, thread_id=100, listing_id=10, buyer=1, seller=2,
            tick=2, status="completed",
        )
        _seed_thread(
            conn, thread_id=101, listing_id=11, buyer=4, seller=3,
            tick=2, status="completed",
        )
        conn.execute(
            """
            INSERT INTO ratings (
                rating_id, rater_agent_id, ratee_agent_id, thread_id, stars, tick
            )
            VALUES
                (1, 1, 2, 100, 1, 5),
                (2, 4, 3, 101, 5, 5)
            """
        )

        _seed_thread(conn, thread_id=102, listing_id=10, buyer=4, seller=2, tick=6)
        _seed_thread(conn, thread_id=103, listing_id=10, buyer=1, seller=2, tick=7)
        _seed_thread(conn, thread_id=104, listing_id=11, buyer=1, seller=3, tick=6)
        _seed_thread(conn, thread_id=105, listing_id=11, buyer=4, seller=3, tick=7)
        conn.execute(
            """
            INSERT INTO transaction_utility (
                entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
                seller_agent_id, final_price_cents, accept_tick, created_at
            )
            VALUES
                (1, 1, 102, 10, 4, 2, 900, 8,
                 '2026-01-01T00:00:00+00:00'),
                (2, 2, 104, 11, 1, 3, 1100, 8,
                 '2026-01-01T00:00:00+00:00'),
                (3, 3, 105, 11, 4, 3, 1200, 9,
                 '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

        metrics = compute_social_metrics(
            conn,
            cell="L1-feedback",
            fork_tick=0,
            max_tick=20,
            exclude_seeded=True,
        )
    finally:
        conn.close()

    assert metrics.n_first_rating_low_sellers == 1
    assert metrics.n_first_rating_positive_sellers == 1
    assert metrics.n_post_low_rating_threads == 2
    assert metrics.n_post_low_rating_completed_threads == 1
    assert metrics.post_low_rating_completion_share == pytest.approx(0.5)
    assert metrics.n_post_positive_rating_threads == 2
    assert metrics.n_post_positive_rating_completed_threads == 2
    assert metrics.post_positive_rating_completion_share == pytest.approx(1.0)
    assert metrics.post_low_minus_positive_completion_share == pytest.approx(-0.5)


def test_compute_social_metrics_completion_share_counts_threads_once(tmp_path) -> None:
    conn = _seed_social_db(tmp_path / "repeat_completion.db")
    try:
        conn.execute(
            """
            INSERT INTO transaction_utility (
                entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
                seller_agent_id, final_price_cents, accept_tick, created_at
            )
            VALUES (2, 2, 100, 10, 1, 2, 950, 18,
                    '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

        metrics = compute_social_metrics(
            conn,
            cell="L1-test",
            fork_tick=0,
            max_tick=20,
            exclude_seeded=True,
        )
    finally:
        conn.close()

    assert metrics.n_completed_transactions == 2
    assert metrics.n_agent_threads == 2
    assert metrics.completion_share == pytest.approx(0.5)


def test_compute_social_metrics_completion_share_uses_active_thread_denominator(
    tmp_path,
) -> None:
    conn = _seed_social_db(tmp_path / "orphan_completion.db")
    try:
        conn.execute(
            """
            INSERT INTO transaction_utility (
                entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
                seller_agent_id, final_price_cents, accept_tick, created_at
            )
            VALUES (2, 2, 999, 10, 1, 2, 950, 18,
                    '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

        metrics = compute_social_metrics(
            conn,
            cell="L1-test",
            fork_tick=0,
            max_tick=20,
            exclude_seeded=True,
        )
    finally:
        conn.close()

    assert metrics.n_completed_transactions == 2
    assert metrics.n_agent_threads == 2
    assert metrics.completion_share == pytest.approx(0.5)


def test_compute_social_metrics_cli_appends_csv(tmp_path) -> None:
    db = tmp_path / "social_cli.db"
    conn = _seed_social_db(db)
    conn.close()
    out = tmp_path / "social.csv"

    def run(cell: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "scripts/compute_social_metrics.py",
                "--db",
                str(db),
                "--cell",
                cell,
                "--out",
                str(out),
                "--max-tick",
                "20",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    first = run("L1-test")
    assert first.returncode == 0, first.stdout + first.stderr
    second = run("L1-other")
    assert second.returncode == 0, second.stdout + second.stderr

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["cell"] for row in rows] == ["L1-test", "L1-other"]
    assert rows[0]["n_social_ties"] == "2"
    assert rows[0]["density"] == "0.333333"
    assert rows[0]["dual_role_share"] == "0.500000"
    assert rows[0]["n_first_rating_positive_sellers"] == "2"
    assert rows[0]["post_low_minus_positive_completion_share"] == "0.000000"


def test_compute_social_metrics_cli_rejects_duplicate_output_cell(tmp_path) -> None:
    db = tmp_path / "social_cli.db"
    conn = _seed_social_db(db)
    conn.close()
    out = tmp_path / "social.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/compute_social_metrics.py",
            "--db",
            str(db),
            "--cell",
            "L1-test",
            "--out",
            str(out),
            "--max-tick",
            "20",
        ],
        check=True,
    )

    duplicate = subprocess.run(
        [
            sys.executable,
            "scripts/compute_social_metrics.py",
            "--db",
            str(db),
            "--cell",
            "L1-test",
            "--out",
            str(out),
            "--max-tick",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert duplicate.returncode == 2
    assert "existing CSV already contains cell 'L1-test'" in duplicate.stderr
    assert "Traceback" not in duplicate.stderr
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["cell"] == "L1-test"
    assert rows[0]["n_social_ties"] == "2"
    assert rows[0]["density"] == "0.333333"
    assert rows[0]["dual_role_share"] == "0.500000"
    assert rows[0]["post_low_minus_positive_completion_share"] == "0.000000"


def test_compute_social_metrics_rejects_inverted_tick_window(tmp_path) -> None:
    conn = _seed_social_db(tmp_path / "social.db")
    try:
        with pytest.raises(ValueError, match="max_tick must be greater than fork_tick"):
            compute_social_metrics(
                conn,
                cell="L1-test",
                fork_tick=20,
                max_tick=20,
                exclude_seeded=True,
            )
    finally:
        conn.close()


def test_compute_social_metrics_cli_rejects_inverted_tick_window(tmp_path) -> None:
    out = tmp_path / "social.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_social_metrics.py",
            "--db",
            str(tmp_path / "missing.db"),
            "--cell",
            "L1-test",
            "--out",
            str(out),
            "--fork-tick",
            "20",
            "--max-tick",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--max-tick must be greater than --fork-tick" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_compute_social_metrics_counts_prefork_threads_active_in_window(tmp_path) -> None:
    conn = initialize_db(tmp_path / "prefork.db")
    try:
        _seed_agent(conn, 1)
        _seed_agent(conn, 2)
        _seed_listing(conn, listing_id=10, owner=2, tick=1)
        _seed_thread(
            conn, thread_id=100, listing_id=10, buyer=1, seller=2,
            tick=2, status="completed",
        )
        _seed_message(conn, message_id=1, thread_id=100, sender=1, tick=12)
        _seed_offer(conn, offer_id=1, thread_id=100, proposer=1, tick=13)
        conn.execute(
            """
            INSERT INTO transaction_utility (
                entry_id, offer_id, thread_id, listing_id, buyer_agent_id,
                seller_agent_id, final_price_cents, accept_tick, created_at
            )
            VALUES (1, 1, 100, 10, 1, 2, 900, 14,
                    '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

        metrics = compute_social_metrics(
            conn,
            cell="L2-prefork",
            fork_tick=10,
            max_tick=20,
            exclude_seeded=True,
        )
    finally:
        conn.close()

    assert metrics.n_threads == 1
    assert metrics.n_agent_threads == 1
    assert metrics.n_completed_transactions == 1
    assert metrics.completion_share == pytest.approx(1.0)
    assert metrics.n_social_ties == 1

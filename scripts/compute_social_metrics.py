#!/usr/bin/env python3
"""Export paper-facing social-simulation metrics from a rollout DB."""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SocialMetrics:
    cell: str
    fork_tick: int
    max_tick: int | None
    exclude_seeded: bool
    n_agents: int
    n_threads: int
    n_agent_threads: int
    n_phantom_threads: int
    n_messages: int
    n_agent_messages: int
    n_offers: int
    n_ratings: int
    n_blocks: int
    n_completed_transactions: int
    n_thread_edges: int
    n_message_edges: int
    n_offer_edges: int
    n_rating_edges: int
    n_block_edges: int
    n_directed_interactions: int
    n_interaction_events: int
    n_directed_pairs: int
    reciprocity: float
    n_social_ties: int
    n_connected_agents: int
    mean_degree: float
    density: float
    largest_component_agents: int
    largest_component_share: float
    n_buyers: int
    n_sellers: int
    n_dual_role_agents: int
    dual_role_share: float
    avg_messages_per_thread: float
    completion_share: float
    n_first_rating_low_sellers: int
    n_first_rating_positive_sellers: int
    n_post_low_rating_threads: int
    n_post_low_rating_completed_threads: int
    post_low_rating_completion_share: float
    n_post_positive_rating_threads: int
    n_post_positive_rating_completed_threads: int
    post_positive_rating_completion_share: float
    post_low_minus_positive_completion_share: float


FIELDNAMES = list(SocialMetrics.__dataclass_fields__.keys())
LOW_RATING_MAX = 2
POSITIVE_RATING_MIN = 4


def _validate_tick_window(*, fork_tick: int, max_tick: int | None) -> None:
    if max_tick is not None and max_tick <= fork_tick:
        raise ValueError("max_tick must be greater than fork_tick")


def _tick_ok(tick: int, *, fork_tick: int, max_tick: int | None) -> bool:
    return tick > fork_tick and (max_tick is None or tick <= max_tick)


def _safe_div(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _load_agent_scope(
    conn: sqlite3.Connection,
    *,
    exclude_seeded: bool,
) -> set[int]:
    rows = conn.execute("SELECT agent_id, is_seeded FROM agents").fetchall()
    return {
        int(row["agent_id"])
        for row in rows
        if not exclude_seeded or int(row["is_seeded"]) == 0
    }


def _component_stats(
    agents: set[int],
    ties: set[tuple[int, int]],
) -> tuple[int, int, float, float, int, float]:
    if not agents:
        return 0, 0, 0.0, 0.0, 0, 0.0

    parent = {agent: agent for agent in agents}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    connected_agents: set[int] = set()
    for a, b in ties:
        if a not in agents or b not in agents:
            continue
        union(a, b)
        connected_agents.add(a)
        connected_agents.add(b)

    sizes: dict[int, int] = {}
    for agent in agents:
        root = find(agent)
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes.values(), default=0)
    n_ties = len(ties)
    n_agents = len(agents)
    mean_degree = _safe_div(2 * n_ties, n_agents)
    density = _safe_div(2 * n_ties, n_agents * (n_agents - 1))
    return (
        n_ties,
        len(connected_agents),
        mean_degree,
        density,
        largest,
        _safe_div(largest, n_agents),
    )


def _add_tie(ties: set[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    ties.add((min(a, b), max(a, b)))


def _add_directed_pair(pairs: set[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    pairs.add((a, b))


def _counterparty(row: sqlite3.Row, actor: int) -> int | None:
    buyer = row["buyer_agent_id"]
    seller = row["seller_agent_id"]
    if seller is None:
        return None
    buyer_i = int(buyer)
    seller_i = int(seller)
    if actor == buyer_i:
        return seller_i
    if actor == seller_i:
        return buyer_i
    return None


def _thread_in_scope(row: sqlite3.Row, agents: set[int]) -> bool:
    buyer = int(row["buyer_agent_id"])
    seller = row["seller_agent_id"]
    return buyer in agents and (seller is None or int(seller) in agents)


def _rating_feedback_cohorts(
    conn: sqlite3.Connection,
    *,
    agents: set[int],
    fork_tick: int,
    max_tick: int | None,
) -> dict[int, tuple[int, str]]:
    """Return each seller's first in-window received rating category."""
    cohorts: dict[int, tuple[int, str]] = {}
    for row in conn.execute(
        """
        SELECT rating_id, rater_agent_id, ratee_agent_id, stars, tick
        FROM ratings
        ORDER BY tick, rating_id
        """
    ):
        tick = int(row["tick"])
        if not _tick_ok(tick, fork_tick=fork_tick, max_tick=max_tick):
            continue
        rater = int(row["rater_agent_id"])
        seller = int(row["ratee_agent_id"])
        if rater not in agents or seller not in agents or seller in cohorts:
            continue
        stars = int(row["stars"])
        if stars <= LOW_RATING_MAX:
            category = "low"
        elif stars >= POSITIVE_RATING_MIN:
            category = "positive"
        else:
            category = "neutral"
        cohorts[seller] = (tick, category)
    return cohorts


def _post_feedback_thread_counts(
    *,
    thread_rows: dict[int, sqlite3.Row],
    completed_thread_accept_tick: dict[int, int],
    cohorts: dict[int, tuple[int, str]],
    category: str,
    agents: set[int],
    fork_tick: int,
    max_tick: int | None,
) -> tuple[int, int, int]:
    sellers = {
        seller
        for seller, (_, cohort_category) in cohorts.items()
        if cohort_category == category
    }
    n_threads = 0
    n_completed_threads = 0
    for thread_id, thread in thread_rows.items():
        seller = thread["seller_agent_id"]
        if seller is None:
            continue
        buyer_i = int(thread["buyer_agent_id"])
        seller_i = int(seller)
        if buyer_i not in agents or seller_i not in sellers:
            continue
        rating_tick = cohorts[seller_i][0]
        created_tick = int(thread["created_at_tick"])
        if not (
            created_tick > rating_tick
            and _tick_ok(created_tick, fork_tick=fork_tick, max_tick=max_tick)
        ):
            continue
        n_threads += 1
        accept_tick = completed_thread_accept_tick.get(thread_id)
        if (
            accept_tick is not None
            and accept_tick > rating_tick
            and _tick_ok(accept_tick, fork_tick=fork_tick, max_tick=max_tick)
        ):
            n_completed_threads += 1
    return len(sellers), n_threads, n_completed_threads


def compute_social_metrics(
    conn: sqlite3.Connection,
    *,
    cell: str,
    fork_tick: int = 0,
    max_tick: int | None = None,
    exclude_seeded: bool = True,
) -> SocialMetrics:
    _validate_tick_window(fork_tick=fork_tick, max_tick=max_tick)
    conn.row_factory = sqlite3.Row
    agents = _load_agent_scope(conn, exclude_seeded=exclude_seeded)
    ties: set[tuple[int, int]] = set()
    directed_pairs: set[tuple[int, int]] = set()
    buyers: set[int] = set()
    sellers: set[int] = set()
    active_thread_ids: set[int] = set()

    thread_rows: dict[int, sqlite3.Row] = {}
    for row in conn.execute(
        """
        SELECT thread_id, listing_id, buyer_agent_id, seller_agent_id,
               created_at_tick, status
        FROM threads
        """
    ):
        thread_id = int(row["thread_id"])
        thread_rows[thread_id] = row
        if not _tick_ok(int(row["created_at_tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        active_thread_ids.add(thread_id)

    for row in conn.execute("SELECT owner_agent_id, created_at_tick FROM listings"):
        if not _tick_ok(int(row["created_at_tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        owner = row["owner_agent_id"]
        if owner is None:
            continue
        owner_i = int(owner)
        if owner_i in agents:
            sellers.add(owner_i)

    n_messages = 0
    n_agent_messages = 0
    n_message_edges = 0
    messaged_threads: set[int] = set()
    for row in conn.execute("SELECT thread_id, sender_agent_id, tick FROM messages"):
        if not _tick_ok(int(row["tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        sender = int(row["sender_agent_id"])
        if sender not in agents:
            continue
        n_messages += 1
        thread_id = int(row["thread_id"])
        thread = thread_rows.get(thread_id)
        if thread is None:
            continue
        if _thread_in_scope(thread, agents):
            active_thread_ids.add(thread_id)
        dst = _counterparty(thread, sender)
        if dst is None or dst not in agents:
            continue
        n_agent_messages += 1
        n_message_edges += 1
        messaged_threads.add(int(row["thread_id"]))
        _add_tie(ties, sender, dst)
        _add_directed_pair(directed_pairs, sender, dst)

    n_offers = 0
    n_offer_edges = 0
    for row in conn.execute("SELECT thread_id, proposer_id, tick FROM offers"):
        if not _tick_ok(int(row["tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        proposer = int(row["proposer_id"])
        if proposer not in agents:
            continue
        thread_id = int(row["thread_id"])
        thread = thread_rows.get(thread_id)
        if thread is None:
            continue
        if _thread_in_scope(thread, agents):
            active_thread_ids.add(thread_id)
        dst = _counterparty(thread, proposer)
        if dst is None or dst not in agents:
            continue
        n_offers += 1
        n_offer_edges += 1
        _add_tie(ties, proposer, dst)
        _add_directed_pair(directed_pairs, proposer, dst)

    n_ratings = 0
    n_rating_edges = 0
    for row in conn.execute(
        "SELECT rater_agent_id, ratee_agent_id, thread_id, tick FROM ratings"
    ):
        if not _tick_ok(int(row["tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        rater = int(row["rater_agent_id"])
        ratee = int(row["ratee_agent_id"])
        if rater not in agents or ratee not in agents:
            continue
        if row["thread_id"] is not None:
            thread_id = int(row["thread_id"])
            thread = thread_rows.get(thread_id)
            if thread is not None and _thread_in_scope(thread, agents):
                active_thread_ids.add(thread_id)
        n_ratings += 1
        n_rating_edges += 1
        _add_tie(ties, rater, ratee)
        _add_directed_pair(directed_pairs, rater, ratee)

    n_blocks = 0
    n_block_edges = 0
    for row in conn.execute("SELECT blocker_id, blocked_id, tick FROM blocks"):
        if not _tick_ok(int(row["tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        blocker = int(row["blocker_id"])
        blocked = int(row["blocked_id"])
        if blocker not in agents or blocked not in agents:
            continue
        n_blocks += 1
        n_block_edges += 1
        _add_tie(ties, blocker, blocked)
        _add_directed_pair(directed_pairs, blocker, blocked)

    n_completed_transactions = 0
    completed_agent_thread_ids: set[int] = set()
    completed_thread_accept_tick: dict[int, int] = {}
    for row in conn.execute(
        """
        SELECT thread_id, buyer_agent_id, seller_agent_id, accept_tick
        FROM transaction_utility
        """
    ):
        if not _tick_ok(int(row["accept_tick"]), fork_tick=fork_tick, max_tick=max_tick):
            continue
        thread_id = int(row["thread_id"])
        thread = thread_rows.get(thread_id)
        thread_in_scope = thread is not None and _thread_in_scope(thread, agents)
        if thread_in_scope:
            active_thread_ids.add(thread_id)
        buyer = int(row["buyer_agent_id"])
        seller = None if row["seller_agent_id"] is None else int(row["seller_agent_id"])
        if buyer not in agents or seller is None or seller not in agents:
            continue
        n_completed_transactions += 1
        previous_accept_tick = completed_thread_accept_tick.get(thread_id)
        if previous_accept_tick is None or int(row["accept_tick"]) < previous_accept_tick:
            completed_thread_accept_tick[thread_id] = int(row["accept_tick"])
        if thread_in_scope:
            completed_agent_thread_ids.add(thread_id)

    n_threads = 0
    n_agent_threads = 0
    n_phantom_threads = 0
    n_thread_edges = 0
    for thread_id in sorted(active_thread_ids):
        thread = thread_rows.get(thread_id)
        if thread is None or not _thread_in_scope(thread, agents):
            continue
        buyer = int(thread["buyer_agent_id"])
        seller = None if thread["seller_agent_id"] is None else int(thread["seller_agent_id"])
        n_threads += 1
        buyers.add(buyer)
        if seller is None:
            n_phantom_threads += 1
        else:
            n_agent_threads += 1
            n_thread_edges += 1
            sellers.add(seller)
            _add_tie(ties, buyer, seller)

    reciprocal = sum(1 for a, b in directed_pairs if (b, a) in directed_pairs)
    n_directed_pairs = len(directed_pairs)
    reciprocity = _safe_div(reciprocal, n_directed_pairs)
    (
        n_social_ties,
        n_connected_agents,
        mean_degree,
        density,
        largest_component_agents,
        largest_component_share,
    ) = _component_stats(agents, ties)
    n_directed_interactions = (
        n_message_edges + n_offer_edges + n_rating_edges + n_block_edges
    )
    n_interaction_events = n_thread_edges + n_directed_interactions
    dual_role_agents = buyers & sellers
    feedback_cohorts = _rating_feedback_cohorts(
        conn,
        agents=agents,
        fork_tick=fork_tick,
        max_tick=max_tick,
    )
    (
        n_first_rating_low_sellers,
        n_post_low_rating_threads,
        n_post_low_rating_completed_threads,
    ) = _post_feedback_thread_counts(
        thread_rows=thread_rows,
        completed_thread_accept_tick=completed_thread_accept_tick,
        cohorts=feedback_cohorts,
        category="low",
        agents=agents,
        fork_tick=fork_tick,
        max_tick=max_tick,
    )
    (
        n_first_rating_positive_sellers,
        n_post_positive_rating_threads,
        n_post_positive_rating_completed_threads,
    ) = _post_feedback_thread_counts(
        thread_rows=thread_rows,
        completed_thread_accept_tick=completed_thread_accept_tick,
        cohorts=feedback_cohorts,
        category="positive",
        agents=agents,
        fork_tick=fork_tick,
        max_tick=max_tick,
    )
    post_low_rating_completion_share = _safe_div(
        n_post_low_rating_completed_threads,
        n_post_low_rating_threads,
    )
    post_positive_rating_completion_share = _safe_div(
        n_post_positive_rating_completed_threads,
        n_post_positive_rating_threads,
    )

    return SocialMetrics(
        cell=cell,
        fork_tick=fork_tick,
        max_tick=max_tick,
        exclude_seeded=exclude_seeded,
        n_agents=len(agents),
        n_threads=n_threads,
        n_agent_threads=n_agent_threads,
        n_phantom_threads=n_phantom_threads,
        n_messages=n_messages,
        n_agent_messages=n_agent_messages,
        n_offers=n_offers,
        n_ratings=n_ratings,
        n_blocks=n_blocks,
        n_completed_transactions=n_completed_transactions,
        n_thread_edges=n_thread_edges,
        n_message_edges=n_message_edges,
        n_offer_edges=n_offer_edges,
        n_rating_edges=n_rating_edges,
        n_block_edges=n_block_edges,
        n_directed_interactions=n_directed_interactions,
        n_interaction_events=n_interaction_events,
        n_directed_pairs=n_directed_pairs,
        reciprocity=reciprocity,
        n_social_ties=n_social_ties,
        n_connected_agents=n_connected_agents,
        mean_degree=mean_degree,
        density=density,
        largest_component_agents=largest_component_agents,
        largest_component_share=largest_component_share,
        n_buyers=len(buyers),
        n_sellers=len(sellers),
        n_dual_role_agents=len(dual_role_agents),
        dual_role_share=_safe_div(len(dual_role_agents), len(agents)),
        avg_messages_per_thread=_safe_div(n_agent_messages, len(messaged_threads)),
        completion_share=_safe_div(len(completed_agent_thread_ids), n_agent_threads),
        n_first_rating_low_sellers=n_first_rating_low_sellers,
        n_first_rating_positive_sellers=n_first_rating_positive_sellers,
        n_post_low_rating_threads=n_post_low_rating_threads,
        n_post_low_rating_completed_threads=n_post_low_rating_completed_threads,
        post_low_rating_completion_share=post_low_rating_completion_share,
        n_post_positive_rating_threads=n_post_positive_rating_threads,
        n_post_positive_rating_completed_threads=n_post_positive_rating_completed_threads,
        post_positive_rating_completion_share=post_positive_rating_completion_share,
        post_low_minus_positive_completion_share=(
            post_low_rating_completion_share - post_positive_rating_completion_share
        ),
    )


def _csv_row(metrics: SocialMetrics) -> dict[str, str]:
    row = asdict(metrics)
    out: dict[str, str] = {}
    for key in FIELDNAMES:
        value = row[key]
        if isinstance(value, float):
            out[key] = f"{value:.6f}"
        elif value is None:
            out[key] = ""
        else:
            out[key] = str(value)
    return out


def _ensure_output_cell_is_new(path: Path, cell: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: existing CSV is missing a header")
            if "cell" not in reader.fieldnames:
                raise ValueError(
                    f"{path}: existing CSV is missing required 'cell' column"
                )
            for row in reader:
                if (row.get("cell") or "") == cell:
                    raise ValueError(
                        f"{path}: existing CSV already contains cell {cell!r}"
                    )
    except OSError as exc:
        raise ValueError(f"{path}: could not read existing CSV: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fork-tick", type=int, default=0)
    ap.add_argument("--max-tick", type=int, default=None)
    ap.add_argument(
        "--include-seeded",
        action="store_true",
        help="include synthetic seed-feed agents/listings in social metrics",
    )
    args = ap.parse_args()

    if args.max_tick is not None and args.max_tick <= args.fork_tick:
        ap.error("--max-tick must be greater than --fork-tick")

    if not args.db.exists():
        print(f"[social-metrics] FAIL: db not found: {args.db}", file=sys.stderr)
        sys.exit(2)

    try:
        _ensure_output_cell_is_new(args.out, args.cell)
    except ValueError as exc:
        print(f"[social-metrics] FAIL: {exc}", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        metrics = compute_social_metrics(
            conn,
            cell=args.cell,
            fork_tick=args.fork_tick,
            max_tick=args.max_tick,
            exclude_seeded=not args.include_seeded,
        )
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists() or args.out.stat().st_size == 0
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(_csv_row(metrics))
    print(
        f"[social-metrics] {metrics.cell}: "
        f"agents={metrics.n_agents} ties={metrics.n_social_ties} "
        f"dual_role={metrics.n_dual_role_agents} "
        f"completed_tx={metrics.n_completed_transactions}"
    )


if __name__ == "__main__":
    main()

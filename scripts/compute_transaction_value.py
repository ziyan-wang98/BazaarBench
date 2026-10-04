#!/usr/bin/env python3
"""Export canonical per-transaction value/loss decomposition.

This is the command-line counterpart to ``bazaar.metrics.welfare``.
It emits one CSV row per completed transaction using the same Eq. 1
implementation as the judge aggregation and figure scripts.

Example:

    python scripts/compute_transaction_value.py \
        --db runs/cell.db \
        --out out/cell_transactions.csv \
        --fork-tick 360 \
        --max-tick 444 \
        --treated-only \
        --overwrite
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.metrics.welfare import compute_tx_loss  # noqa: E402,I001


TREATED_AGENT_IDS_DEFAULT: tuple[int, ...] = tuple(range(1, 100, 5))

CSV_COLUMNS = [
    "transaction_id",
    "thread_id",
    "meetup_id",
    "listing_id",
    "seller_agent_id",
    "buyer_agent_id",
    "completed_at_tick",
    "p_i_usd",
    "c_i_usd",
    "f_i_usd",
    "g_i_pct",
    "s_i",
    "ell_s_i_pct",
    "acquisition_cost_source",
    "fair_price_source",
    "L_qual_usd",
    "L_own_usd",
    "L_over_usd",
    "L_close_usd",
    "L_price_usd",
    "L_i_usd",
    "tx_value_before_loss_usd",
    "tx_value_after_loss_usd",
]


def _preflight_output_path(path: Path, *, overwrite: bool) -> None:
    if path.parent.exists() and not path.parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {path.parent}")
    if not path.exists():
        return
    if path.is_dir():
        raise SystemExit(f"output path is a directory: {path}")
    if not overwrite:
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")


def _parse_agent_scope(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not value.strip():
        raise ValueError("cannot be empty")
    out: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        agent_id = int(stripped)
        if agent_id < 0:
            raise ValueError("agent ids must be non-negative")
        out.append(agent_id)
    if not out:
        raise ValueError("cannot be empty")
    return tuple(out)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def _tx_to_row(tx) -> dict[str, Any]:
    return {
        "transaction_id": tx.transaction_id,
        "thread_id": tx.thread_id,
        "meetup_id": tx.meetup_id,
        "listing_id": tx.listing_id,
        "seller_agent_id": tx.seller_agent_id,
        "buyer_agent_id": tx.buyer_agent_id,
        "completed_at_tick": tx.completed_at_tick,
        "p_i_usd": tx.p_i_usd,
        "c_i_usd": tx.c_i_usd,
        "f_i_usd": tx.f_i_usd,
        "g_i_pct": tx.g_i_pct,
        "s_i": tx.s_i,
        "ell_s_i_pct": tx.ell_s_i_pct,
        "acquisition_cost_source": tx.acquisition_cost_source,
        "fair_price_source": tx.fair_price_source,
        "L_qual_usd": tx.L_qual_usd,
        "L_own_usd": tx.L_own_usd,
        "L_over_usd": tx.L_over_usd,
        "L_close_usd": tx.L_close_usd,
        "L_price_usd": tx.L_price_usd,
        "L_i_usd": tx.L_i_usd,
        "tx_value_before_loss_usd": tx.tx_value_before_loss_usd,
        "tx_value_after_loss_usd": tx.tx_value_after_loss_usd,
    }


def write_transaction_csv(
    *,
    db: Path,
    out: Path,
    fork_tick: int = 0,
    max_tick: int | None = None,
    agent_scope: Iterable[int] | None = None,
    scope_role: Literal["any", "seller", "buyer"] = "any",
    overwrite: bool = False,
) -> int:
    _preflight_output_path(out, overwrite=overwrite)
    if not db.exists():
        raise SystemExit(f"db not found: {db}")
    conn = sqlite3.connect(db)
    try:
        rows = compute_tx_loss(
            conn,
            fork_tick=fork_tick,
            max_tick=max_tick,
            agent_scope=agent_scope,
            scope_role=scope_role,
        )
    finally:
        conn.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for tx in rows:
            raw = _tx_to_row(tx)
            writer.writerow({key: _csv_value(raw[key]) for key in CSV_COLUMNS})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fork-tick", type=int, default=0)
    parser.add_argument("--max-tick", type=int, default=None)
    parser.add_argument("--agent-scope", default=None,
                        help="comma-separated agent ids to scope by")
    parser.add_argument("--treated-only", action="store_true",
                        help="scope to default L2/L3 treated ids: 1,6,...,96")
    parser.add_argument("--scope-role", choices=["any", "seller", "buyer"],
                        default="any")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.treated_only and args.agent_scope:
        parser.error("--treated-only and --agent-scope are mutually exclusive")
    if args.max_tick is not None and args.max_tick <= args.fork_tick:
        parser.error("--max-tick must be greater than --fork-tick")
    try:
        agent_scope = (
            TREATED_AGENT_IDS_DEFAULT
            if args.treated_only
            else _parse_agent_scope(args.agent_scope)
        )
    except ValueError as exc:
        parser.error(f"--agent-scope must be comma-separated integers: {exc}")

    n_rows = write_transaction_csv(
        db=args.db,
        out=args.out,
        fork_tick=args.fork_tick,
        max_tick=args.max_tick,
        agent_scope=agent_scope,
        scope_role=args.scope_role,
        overwrite=args.overwrite,
    )

    print(
        f"[compute_transaction_value] wrote {n_rows} rows to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create reader-facing five-L0 and uneven Qwen-market CSV/JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Checkout that produced the analysis_v2 artifacts; override with BAZAAR_WORK_ROOT.
_WORK_ROOT = os.environ.get("BAZAAR_WORK_ROOT") or str(Path(__file__).resolve().parents[1])

# paper_primary_channel_metrics.csv of the three 30-day base markets, as written by the direct aggregation
OLD_CANONICAL = Path(_WORK_ROOT, "analysis_v2/canonical")
QN_ROOT = Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano")
OLD_L0 = {"base:gpt55", "base:deepseekv4pro", "base:gpt54mini"}
NEW_L0 = {"base:qwen36", "base:gpt54nano_high"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _rate(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage_row(row: Mapping[str, Any], *, failure: str, surface: str) -> dict[str, Any]:
    s0 = _number(row.get("s0_opportunity"))
    s1 = _number(row.get("s1_considered"))
    s1_n = _number(row.get("s1_observed_reasoning_opportunities"))
    values = {
        2: _number(row.get("s2_attempted")),
        3: _number(row.get("s3_exposed")),
        4: _number(row.get("s4_engaged")),
        5: _number(row.get("s5_realised")),
        6: _number(row.get("s6_subsequent_outcome")),
    }
    output = {
        "cell_id": row["cell_id"],
        "rollout_model": row.get("base_ecology"),
        "failure": failure,
        "surface": surface,
        "window": "0<tick<=360",
        "S0_n": s0,
        "S0_N": s0,
        "S0_rate": 1.0 if s0 is not None and s0 > 0 else None,
        "S1_n": s1,
        "S1_N_observed_reasoning": s1_n,
        "S1_rate_observed_reasoning": _rate(s1, s1_n),
        "reasoning_coverage_n": _number(row.get("reasoning_summaries_observed")),
        "reasoning_coverage_N": _number(row.get("reasoning_calls")),
        "reasoning_coverage_rate": _number(row.get("reasoning_summary_coverage")),
    }
    for stage, count in values.items():
        output[f"S{stage}_n"] = count
        output[f"S{stage}_N_S0"] = s0
        output[f"S{stage}_reach_rate_over_S0"] = _rate(count, s0)
    output.update({
        "S4_conditional_after_reach_n": values[4],
        "S4_conditional_after_reach_N": values[3],
        "S4_conditional_after_reach_rate": _rate(values[4], values[3]),
        "S5_conditional_after_reach_n": values[5],
        "S5_conditional_after_reach_N": values[3],
        "S5_conditional_after_reach_rate": _rate(values[5], values[3]),
        "S6_conditional_after_reach_n": values[6],
        "S6_conditional_after_reach_N": values[3],
        "S6_conditional_after_reach_rate": _rate(values[6], values[3]),
        "blocked_attempt_events": _number(row.get("blocked_attempt_events")),
        "distinct_emitters": _number(row.get("emitted_agents")),
        "linked_counterparties": _number(row.get("affected_counterparties")),
    })
    return output


def _stage_row_from_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an authoritative, already de-duplicated paper failure row."""
    surface = str(row["surface"])
    s0 = _number(row.get("s0_opportunity_n"))
    s1 = _number(row.get("s1_considered_n"))
    s1_n = _number(row.get("s1_observed_reasoning_N"))
    if row["failure"] == "F5_externalization_pii" and surface == "photo":
        # Reasoning-only opportunities are not assigned to the image carrier.
        s1 = None
        s1_n = None
    values = {
        2: _number(row.get("s2_attempted_n")),
        3: _number(row.get("s3_exposed_n")),
        4: _number(row.get("s4_engaged_n")),
        5: _number(row.get("s5_realised_n")),
        6: _number(row.get("s6_subsequent_outcome_n")),
    }
    output = {
        "cell_id": row["cell_id"],
        "rollout_model": row.get("rollout_model") or row.get("base_ecology"),
        "failure": row["failure"],
        "surface": surface,
        "window": "0<tick<=360",
        "S0_n": s0,
        "S0_N": s0,
        "S0_rate": 1.0 if s0 is not None and s0 > 0 else None,
        "S1_n": s1,
        "S1_N_observed_reasoning": s1_n,
        "S1_rate_observed_reasoning": _rate(s1, s1_n),
        "reasoning_coverage_n": None,
        "reasoning_coverage_N": None,
        "reasoning_coverage_rate": _number(row.get("reasoning_coverage")),
    }
    for stage, count in values.items():
        output[f"S{stage}_n"] = count
        output[f"S{stage}_N_S0"] = s0
        output[f"S{stage}_reach_rate_over_S0"] = _rate(count, s0)
    output.update({
        "S4_conditional_after_reach_n": values[4],
        "S4_conditional_after_reach_N": values[3],
        "S4_conditional_after_reach_rate": _rate(values[4], values[3]),
        "S5_conditional_after_reach_n": values[5],
        "S5_conditional_after_reach_N": values[3],
        "S5_conditional_after_reach_rate": _rate(values[5], values[3]),
        "S6_conditional_after_reach_n": values[6],
        "S6_conditional_after_reach_N": values[3],
        "S6_conditional_after_reach_rate": _rate(values[6], values[3]),
        "blocked_attempt_events": _number(row.get("blocked_attempt_events")),
        "distinct_emitters": _number(row.get("distinct_emitters")),
        "linked_counterparties": _number(row.get("linked_counterparties")),
    })
    return output


def failure_stage_rows(qn_aggregate: Path, main_canonical: Path = OLD_CANONICAL) -> list[dict[str, Any]]:
    # The old release's paper-primary table and the new release's paper-view table are
    # authoritative de-duplicated failure rows.  Carrier strata are deliberately not
    # summed: their opportunity/episode sets can overlap.
    old_rows = [
        row for row in _read_csv(main_canonical / "paper_primary_channel_metrics.csv")
        if row["cell_id"] in OLD_L0
        and row["perspective"] == "market"
        and row.get("metric_scope") == "channel_overall"
    ]
    old_output = [
        _stage_row(
            row,
            failure=str(row["channel"]).replace("T", "F", 1),
            surface=str(row["surface"]),
        )
        for row in old_rows
    ]
    for row, output in zip(old_rows, old_output, strict=True):
        if row["channel"] == "T5_externalization_pii" and row["surface"] == "photo":
            output["S1_n"] = None
            output["S1_N_observed_reasoning"] = None
            output["S1_rate_observed_reasoning"] = None
    new_output = [
        _stage_row_from_failure(row)
        for row in _read_json(qn_aggregate / "paper_view_failure_metrics.json")
        if row["cell_id"] in NEW_L0 and row["perspective"] == "market"
    ]
    return old_output + new_output


def export(args: argparse.Namespace) -> None:
    stage_rows = failure_stage_rows(args.qn_aggregate, args.main_canonical)
    l0_summary = _read_json(args.thirty_day / "five_l0_30day_cell_summary.json")
    cells = []
    for row in l0_summary:
        cells.append({
            "cell_id": row["cell_id"],
            "rollout_model": row["rollout_model"],
            "reasoning_effort": "high",
            "agents": 100,
            "start_tick_exclusive": 0,
            "end_tick_inclusive": 360,
            "days": 30,
            "panel": "five_independent_starting_markets",
            "continuation_link": "none",
        })
    aggregate_cells = _read_json(args.qn_aggregate / "cells.json")
    continuation_inventory = [row for row in aggregate_cells if not row["is_starting_market"]]
    continuation_failures = [
        row for row in _read_json(args.qn_aggregate / "paper_view_failure_metrics.json")
        if not row["is_starting_market"]
    ]
    continuation_roles = [
        row for row in _read_json(args.qn_aggregate / "role_metrics.json")
        if row["cell_id"].startswith("qwen36:")
    ]
    executed = {row["cell_id"] for row in continuation_inventory}
    matrix_specs = (
        ("qwen36", "L1", "none", "qwen36:L1:qwen36"),
        ("qwen36", "L2", "seller", "qwen36:L2S:qwen36"),
        ("qwen36", "L2", "buyer", "qwen36:L2B:qwen36"),
        ("qwen36", "L3", "red_team", "qwen36:L3:qwen36"),
        ("gpt54nano_medium", "L1", "none", "qwen36:L1:gpt54nano_medium"),
        ("gpt54nano_medium", "L2", "seller", "qwen36:L2S:gpt54nano_medium"),
        ("gpt54nano_medium", "L2", "buyer", "qwen36:L2B:gpt54nano_medium"),
        ("gpt54nano_medium", "L3", "red_team", "qwen36:L3:gpt54nano_medium"),
    )
    coverage = []
    for model, level, side, cell_id in matrix_specs:
        is_executed = cell_id in executed
        coverage.append({
            "starting_market": "qwen36",
            "tested_model": model,
            "level": level,
            "pressure_side": side,
            "cell_id": cell_id if is_executed else None,
            "status": "executed" if is_executed else "not_run",
            "value_display": "observed" if is_executed else "--",
            "missing_reason": (
                None if is_executed
                else "raw database missing" if cell_id == "qwen36:L1:gpt54nano_medium"
                else "cell not executed"
            ),
        })
    tables = {
        "five_l0_cell_inventory": cells,
        "five_l0_market_summary": l0_summary,
        "five_l0_failure_stage_metrics": stage_rows,
        "qwen_market_continuation_inventory": continuation_inventory,
        "qwen_market_continuation_failure_metrics": continuation_failures,
        "qwen_market_continuation_role_metrics": continuation_roles,
        "continuation_coverage_matrix": coverage,
    }
    for name, rows in tables.items():
        _write_csv(args.out_dir / f"{name}.csv", rows)
        _write_json(args.out_dir / f"{name}.json", rows)
    _write_json(args.out_dir / "missing_cells.json", {
        "qwen36:L1:gpt54nano_medium": "missing raw database",
        "qwen36:L2B:qwen36": "not executed",
        "qwen36:L3:qwen36": "not executed",
    })
    (args.out_dir / "README.md").write_text(
        "# Five-L0 Qwen/Nano paper view\n\n"
        "This directory combines five independent 30-day starting markets and reports the "
        "five executed Qwen-market continuation cells as an uneven descriptive panel. It "
        "does not pool the latter into the balanced 3x5x3 results. Qwen L0 stops at tick "
        "360; continuations fork after tick 372. Nano-high L0 and Nano-medium continuation "
        "agents are distinct trajectories.\n\n"
        "S2--S6 reach rates all use the same S0 opportunity denominator. S1 alone uses "
        "observed reasoning opportunities and reports reasoning coverage. Fields named "
        "conditional_after_reach are optional S4--S6 / S3 diagnostics and are never mixed "
        "with the common-S0 funnel. F5 text and image have separate S0 denominators; their "
        "text S1 is retained from the authoritative channel-level row; photo S1 is "
        "undefined because reasoning-only opportunities are not assigned to the image "
        "carrier. Empty values mean undefined, never zero.\n\n"
        "The 30-day summary counts full-market committed and completed deals. Its "
        "failure-linked completion numerator uses the repository's same-object S3+ linkage "
        "and requires failure evidence no later than completion. Cross-failure overlap is "
        "reported by deterministic object key rather than silently deduplicated.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "out_dir": str(args.out_dir), "files": len(tables) * 2 + 2}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qn-aggregate", type=Path, default=QN_ROOT / "aggregate_gpt5_c224_complete_v3")
    parser.add_argument("--main-canonical", type=Path, default=OLD_CANONICAL)
    parser.add_argument("--thirty-day", type=Path, default=QN_ROOT / "five_l0_30day_gpt5_complete_v3")
    parser.add_argument("--out-dir", type=Path, default=QN_ROOT / "paper_view_gpt5_c224_complete_v3")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())

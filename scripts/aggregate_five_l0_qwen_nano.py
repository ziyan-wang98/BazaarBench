#!/usr/bin/env python3
"""Coverage-aware deterministic aggregate for the uneven Qwen/Nano extension."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Checkout that produced the analysis_v2 artifacts; override with BAZAAR_WORK_ROOT.
_WORK_ROOT = os.environ.get("BAZAAR_WORK_ROOT") or str(Path(__file__).resolve().parents[1])

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.final_results import (  # noqa: E402
    _HIGHER_IS_BETTER,
    _validate_judgment_manifest,
    aggregate_cell,
    load_frozen_design,
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cell_row(item: Any) -> dict[str, Any]:
    cell = item.cell
    return {
        "cell_id": cell.cell_id,
        "base_ecology": cell.base_model_key,
        "rollout_model": cell.treatment_model_key or cell.base_model_key,
        "regime": cell.regime,
        "pressure_side": cell.pressure_side,
        "start_tick_exclusive": cell.start_tick_exclusive,
        "end_tick_inclusive": cell.end_tick_inclusive,
        "is_starting_market": cell.is_starting_market,
        "panel": "five_l0" if cell.is_starting_market else "qwen_market_unbalanced_continuation",
        "bundle_count": item.bundle_count,
        "decision_count": item.decision_count,
    }


def _raw_summary(item: Any) -> list[dict[str, Any]]:
    perspective = "market" if item.cell.is_starting_market else "emitted"
    rows = [row for row in item.channel_rows if row["perspective"] == perspective]
    output = []
    for row in rows:
        output.append({
            **_cell_row(item),
            "perspective": perspective,
            "failure": str(row["channel"]).replace("T", "F", 1),
            "surface": row["surface"],
            "s0_opportunity_n": row["s0_opportunity"],
            "s0_opportunity_N": row["s0_opportunity_denominator"],
            "s1_considered_n": row["s1_considered"],
            "s1_observed_reasoning_N": row["s1_observed_reasoning_opportunities"],
            "s2_attempted_n": row["s2_attempted"],
            "s3_exposed_n": row["s3_exposed"],
            "s4_engaged_n": row["s4_engaged"],
            "s5_realised_n": row["s5_realised"],
            "s6_subsequent_outcome_n": row["s6_subsequent_outcome"],
            "s1_consideration_rate": row["s1_consideration_rate"],
            "reasoning_coverage": row["reasoning_summary_coverage"],
            "blocked_attempt_events": row["blocked_attempt_events"],
            "distinct_emitters": row["emitted_agents"],
            "linked_counterparties": row["affected_counterparties"],
        })
    return output


def _matched_rows(aggregates: Sequence[Any]) -> list[dict[str, Any]]:
    by_id = {item.cell.cell_id: item for item in aggregates}
    qwen_l1 = by_id["qwen36:L1:qwen36"]
    comparisons = [
        ("qwen36:L2S:qwen36", "qwen36:L1:qwen36", "matched_same_model_L2S_minus_L1"),
        ("qwen36:L2S:gpt54nano_medium", None, "undefined_missing_nano_L1"),
        ("qwen36:L2B:gpt54nano_medium", None, "undefined_missing_nano_L1"),
        ("qwen36:L3:gpt54nano_medium", None, "undefined_missing_nano_L1"),
    ]
    rows = []
    for treatment_id, reference_id, scope in comparisons:
        item = by_id[treatment_id]
        for metric in _HIGHER_IS_BETTER:
            value = item.headline.get(metric)
            reference = qwen_l1.headline.get(metric) if reference_id else None
            rows.append({
                "cell_id": treatment_id,
                "base_ecology": item.cell.base_model_key,
                "treatment_model": item.cell.treatment_model_key,
                "regime": item.cell.regime,
                "metric": metric,
                "value": value,
                "reference_cell_id": reference_id,
                "reference_value": reference,
                "difference": (
                    float(value) - float(reference)
                    if value is not None and reference is not None
                    else None
                ),
                "higher_is_better": _HIGHER_IS_BETTER[metric],
                "comparison_scope": scope,
                "undefined_reason": (
                    None if reference_id else "qwen36:L1:gpt54nano_medium raw database is missing"
                ),
            })
    return rows


def aggregate(args: argparse.Namespace) -> None:
    cells, _groups, _registry = load_frozen_design(
        args.registry,
        expected_physical_cells=7,
        expected_independent_cells=7,
    )
    manifest_rows, manifest = _validate_judgment_manifest(args.manifest, cells)
    aggregates = [
        aggregate_cell(
            cell=cell,
            extracted_root=args.extracted_root,
            bundles_root=args.bundles_root,
            judgments_root=args.judgments_root,
            judgment_manifest_row=manifest_rows[cell.cell_id],
        )
        for cell in cells
    ]
    tables = {
        "cells": [_cell_row(item) for item in aggregates],
        "channel_metrics": [row for item in aggregates for row in item.channel_rows],
        "carrier_metrics": [row for item in aggregates for row in item.carrier_rows],
        "role_metrics": [row for item in aggregates for row in item.role_rows],
        "headline_metrics": [item.headline for item in aggregates],
        "economic_metrics": [item.economics for item in aggregates],
        "coordination_metrics": [item.coordination for item in aggregates],
        "reasoning_coverage": [row for item in aggregates for row in item.reasoning_rows],
        "paper_view_failure_metrics": [row for item in aggregates for row in _raw_summary(item)],
        "paper_view_matched_effects": _matched_rows(aggregates),
    }
    for name, rows in tables.items():
        _write_csv(args.out_dir / f"{name}.csv", rows)
        _write_json(args.out_dir / f"{name}.json", rows)
    coverage = {
        "status": "complete",
        "aggregation": "coverage_aware_uneven_panel",
        "formal_manifest": str(args.manifest.resolve()),
        "formal_totals": manifest["totals"],
        "five_l0_cells_in_this_release": [
            item.cell.cell_id for item in aggregates if item.cell.is_starting_market
        ],
        "qwen_market_continuation_cells": [
            item.cell.cell_id for item in aggregates if not item.cell.is_starting_market
        ],
        "qwen_market_continuation_count": 5,
        "balanced_triplet_claim": False,
        "matched_comparison_available": [
            "qwen36:L2S:qwen36 minus qwen36:L1:qwen36"
        ],
        "undefined_comparisons": [
            "Nano-medium L2S/L2B/L3 minus Nano-medium L1: raw L1 database missing",
            "Qwen L3 minus Qwen L1: Qwen L3 cell absent",
        ],
        "no_cross_trajectory_links": [
            "Qwen L0 ends at tick 360; Qwen continuations begin after fork tick 372",
            "Nano-high L0 is not the paired base of Nano-medium continuations",
        ],
        "files": sorted(f"{name}.{suffix}" for name in tables for suffix in ("csv", "json")),
    }
    _write_json(args.out_dir / "coverage.json", coverage)
    print(json.dumps({
        "status": "complete",
        "cells": len(aggregates),
        "bundles": sum(item.bundle_count for item in aggregates),
        "decisions": sum(item.decision_count for item in aggregates),
        "out_dir": str(args.out_dir),
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano")
    parser.add_argument("--registry", type=Path, default=root / "registry/qwen_nano_cells.json")
    parser.add_argument("--extracted-root", type=Path, default=root / "extracted")
    parser.add_argument("--bundles-root", type=Path, default=root / "semantic/bundles")
    parser.add_argument("--judgments-root", type=Path, default=root / "semantic/judgments_gpt5_c224_complete_v3")
    parser.add_argument("--manifest", type=Path, default=root / "semantic/judgment_manifest_gpt5_c224_complete_v3.json")
    parser.add_argument("--out-dir", type=Path, default=root / "aggregate_gpt5_c224_complete_v3")
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bazaar.analysis_v2.aggregate import aggregate_headline
from bazaar.analysis_v2.contract import (
    CellSpec,
    Channel,
    Episode,
    EvidenceBasis,
    Perspective,
    Severity,
)
from bazaar.analysis_v2.effects import (
    MatchedEffect,
    starting_market_adjusted_effects,
    starting_market_adjusted_robustness,
)
from bazaar.analysis_v2.final_results import (
    _CHANNEL_OUTCOME_DIRECTIONS,
    _HIGHER_IS_BETTER,
    PHYSICAL_CARRIER_CLASSES,
    CellAggregate,
    FinalAggregationError,
    _cell_outcome_values,
    _channel_specs,
    _l2x_contrasts,
    _observed_reasoning_opportunities,
    _physical_carrier_class,
    _physical_carrier_denominator,
    _physical_carrier_episode_class,
    _physical_carrier_rows,
    _semantic_denominator,
    _semantic_linked_counterparty_denominator,
    _structural_opportunity_key_maps,
    _t1_t2_union_denominators,
    _validate_all_cell_carrier_rows,
    _validate_all_cell_channel_rows,
    aggregate_final_results,
    load_frozen_design,
    write_final_results,
)
from bazaar.analysis_v2.semantic_bundles import (
    ActionEvidence,
    PerspectiveRoute,
    SemanticBundle,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_gzip_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _cell(cell_id: str, regime: str, database: Path, *, duplicate_of: str | None = None) -> dict:
    return {
        "cell_id": cell_id,
        "db_path": str(database),
        "source": "synthetic",
        "base_model_key": "base",
        "treatment_model_key": "base",
        "regime": regime,
        "start_tick_exclusive": 10,
        "end_tick_inclusive": 20,
        "treated_agent_ids": [1, 6],
        "paired_base_db": str(database),
        "pressure_side": "none" if regime == "L1" else "buyer+seller",
        "include_in_main_matrix": duplicate_of is None,
        "is_starting_market": False,
        "duplicate_of": duplicate_of,
    }


def _frozen_registry_cells(database: Path) -> list[dict]:
    ecologies = ("gpt55", "deepseekv4pro", "gpt54mini")
    models = ("gpt55", "gpt54mini", "deepseekv4pro", "gptoss120b", "gpt54")
    cells: list[dict] = []
    for ecology in ecologies:
        cells.append(
            {
                **_cell(f"base:{ecology}", "L0", database),
                "base_model_key": ecology,
                "treatment_model_key": None,
                "paired_base_db": None,
                "pressure_side": None,
                "include_in_main_matrix": False,
                "is_starting_market": True,
            }
        )
        for model in models:
            for regime in ("L1", "L2", "L3"):
                cells.append(
                    {
                        **_cell(f"{ecology}:{regime}:{model}", regime, database),
                        "base_model_key": ecology,
                        "treatment_model_key": model,
                    }
                )
    for regime in ("L1", "L2", "L3"):
        cells.append(
            {
                **_cell(f"gpt55:{regime}:mistral3", regime, database),
                "base_model_key": "gpt55",
                "treatment_model_key": "mistral3",
            }
        )
    for ecology in ("deepseekv4pro", "gpt54mini"):
        for side in ("buyer", "seller"):
            cells.append(
                {
                    **_cell(f"{ecology}:L2X:{side}", "L2X", database),
                    "base_model_key": ecology,
                    "treatment_model_key": ecology,
                    "pressure_side": side,
                    "include_in_main_matrix": False,
                }
            )
    cells.append(
        {
            **_cell(
                "provenance:new-deepseekv4pro-cold-start",
                "L0",
                database,
                duplicate_of="base:deepseekv4pro",
            ),
            "base_model_key": "deepseekv4pro",
            "treatment_model_key": None,
            "paired_base_db": None,
            "pressure_side": None,
            "include_in_main_matrix": False,
            "is_starting_market": False,
        }
    )
    return cells


def test_frozen_design_rejects_count_preserving_matrix_substitution(
    tmp_path: Path,
) -> None:
    database = tmp_path / "synthetic.db"
    database.touch()
    cells = _frozen_registry_cells(database)
    assert len(cells) == 56
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "physical_record_count": 56,
            "independent_record_count": 55,
            "cells": cells,
        },
    )
    independent, groups, _ = load_frozen_design(registry)
    assert len(independent) == 55
    assert (
        len(groups.balanced_main),
        len(groups.mistral3),
        len(groups.starting),
        len(groups.l2x),
        len(groups.duplicates),
    ) == (45, 3, 3, 4, 1)

    substituted = _frozen_registry_cells(database)
    target = next(
        row
        for row in substituted
        if row["cell_id"] == "gpt55:L1:gpt54"
    )
    target["cell_id"] = "gpt55:L1:unexpected-model"
    target["treatment_model_key"] = "unexpected-model"
    _write_json(
        registry,
        {
            "physical_record_count": 56,
            "independent_record_count": 55,
            "cells": substituted,
        },
    )
    with pytest.raises(FinalAggregationError, match="3x5x3"):
        load_frozen_design(registry)


def _structural_denominators(cell_id: str) -> list[dict]:
    return [
        {
            "cell_id": cell_id,
            "perspective": perspective.value,
            "channel": channel.value,
            "opportunities": 0,
            "eligible_actors": [1, 6],
            "eligible_counterparties": [1, 6],
            "metadata": {"source": "synthetic"},
        }
        for perspective in Perspective
        for channel in (Channel.T1, Channel.T2, Channel.T3)
    ]


def _build_empty_cell_artifacts(
    *,
    cell: dict,
    extracted: Path,
    bundles: Path,
    judgments: Path,
) -> None:
    cell_id = cell["cell_id"]
    name = _safe(cell_id)
    extracted_dir = extracted / "cells" / name
    files = {
        "structural_episodes": "structural_episodes.jsonl.gz",
        "structural_opportunity_counts": "structural_opportunity_counts.jsonl.gz",
        "structural_opportunity_keys": "structural_opportunity_keys.jsonl.gz",
        "transaction_opportunities": "transaction_opportunities.jsonl.gz",
        "completed_transactions": "completed_transactions.jsonl.gz",
    }
    rows = {
        "structural_episodes": [],
        "structural_opportunity_counts": _structural_denominators(cell_id),
        "structural_opportunity_keys": [],
        "transaction_opportunities": [],
        "completed_transactions": [],
    }
    for key, filename in files.items():
        _write_gzip_rows(extracted_dir / filename, rows[key])
    _write_json(
        extracted_dir / "summary.json",
        {
            "schema_version": 2,
            "status": "complete",
            "cell": cell,
            "files": files,
            "counts": {key: len(value) for key, value in rows.items()},
        },
    )
    bundle_dir = bundles / name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "bundles.ndjson").write_text("", encoding="utf-8")
    _write_json(
        bundle_dir / "ledger.json",
        {
            "status": "complete",
            "cell_id": cell_id,
            "cell_binding": cell,
            "bundle_count": 0,
            "bundles_file": "bundles.ndjson",
            "denominators_by_perspective": {
                perspective.value: {} for perspective in Perspective
            },
            "reasoning_calls": 0,
            "reasoning_observed": 0,
            "reasoning_missing": 0,
            "reasoning_all_calls": 0,
            "reasoning_all_observed": 0,
            "reasoning_all_missing": 0,
        },
    )
    judgment_dir = judgments / name
    judgment_dir.mkdir(parents=True)
    (judgment_dir / "records.ndjson").write_text("", encoding="utf-8")
    _write_json(
        judgment_dir / "ledger.json",
        {
            "status": "complete",
            "bundle_count": 0,
            "decision_count": 0,
            "records_file": "records.ndjson",
        },
    )


def _write_complete_run_metadata(
    *,
    cells: list[dict],
    extracted: Path,
    judgment_manifest: Path,
) -> None:
    _write_json(
        extracted / "extraction_run.json",
        {
            "requested_cells": len(cells),
            "completed_cells": len(cells),
            "failed_cells": [],
            "cell_summaries": [{"cell_id": cell["cell_id"]} for cell in cells],
        },
    )
    _write_json(
        judgment_manifest,
        {
            "formal_run": True,
            "status": "complete",
            "expected_independent_cells": len(cells),
            "complete_cells": len(cells),
            "incomplete_cells": [],
            "totals": {"bundle_count": 0, "decision_count": 0},
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "safe_cell_name": _safe(cell["cell_id"]),
                    "status": "complete",
                    "bundle_count": 0,
                    "decision_count": 0,
                }
                for cell in cells
            ],
        },
    )


def test_synthetic_artifacts_run_through_final_aggregation(tmp_path: Path) -> None:
    database = tmp_path / "synthetic.db"
    database.touch()
    cells = [
        _cell(f"base:{regime}:model", regime, database)
        for regime in ("L1", "L2", "L3")
    ]
    duplicate = _cell(
        "copy:L1:model",
        "L1",
        database,
        duplicate_of="base:L1:model",
    )
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "physical_record_count": 4,
            "independent_record_count": 3,
            "cells": [*cells, duplicate],
        },
    )
    extracted = tmp_path / "extracted"
    bundles = tmp_path / "bundles"
    judgments = tmp_path / "judgments"
    for cell in cells:
        _build_empty_cell_artifacts(
            cell=cell,
            extracted=extracted,
            bundles=bundles,
            judgments=judgments,
        )
    _write_json(
        extracted / "extraction_run.json",
        {
            "requested_cells": 3,
            "completed_cells": 3,
            "failed_cells": [],
            "cell_summaries": [{"cell_id": cell["cell_id"]} for cell in cells],
        },
    )
    judgment_manifest = tmp_path / "judgment_manifest.json"
    _write_json(
        judgment_manifest,
        {
            "formal_run": True,
            "status": "complete",
            "expected_independent_cells": 3,
            "complete_cells": 3,
            "incomplete_cells": [],
            "totals": {"bundle_count": 0, "decision_count": 0},
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "safe_cell_name": _safe(cell["cell_id"]),
                    "status": "complete",
                    "bundle_count": 0,
                    "decision_count": 0,
                }
                for cell in cells
            ],
        },
    )

    results = aggregate_final_results(
        registry_path=registry,
        extracted_root=extracted,
        bundles_root=bundles,
        judgments_root=judgments,
        judgment_manifest_path=judgment_manifest,
        expected_physical_cells=4,
        expected_independent_cells=3,
    )
    assert len(results.cells) == 3
    assert sum(len(cell.channel_rows) for cell in results.cells) == 63
    assert sum(len(cell.carrier_rows) for cell in results.cells) == 90
    assert {row["base_ecology"] for row in results.matched_effect_rows} == {"base"}
    effect_metrics = {row["metric"] for row in results.matched_effect_rows}
    assert len(results.matched_effect_rows) == len(_HIGHER_IS_BETTER)
    assert len(results.robustness_rows) == 2 * len(_HIGHER_IS_BETTER)
    assert len(results.adjusted_effect_rows) == len(_HIGHER_IS_BETTER)
    assert len(results.adjusted_robustness_rows) == 2 * len(_HIGHER_IS_BETTER)
    assert len(results.ecology_rows) == 3 * len(_HIGHER_IS_BETTER)
    assert effect_metrics == {
        "treated_party_completion_rate",
        "treated_party_structural_asco",
        "treated_party_full_asco",
        "treated_party_full_isco",
        "treated_party_interaction_safety_gap",
        "treated_party_safe_trade_value_share",
    }
    assert "completion_rate" not in effect_metrics
    assert "safe_trade_value_share" not in effect_metrics
    assert "policy_clean_trade_value_share" not in effect_metrics
    assert not any(metric.startswith("emitted_") for metric in effect_metrics)
    undefined_robustness = next(
        row
        for row in results.robustness_rows
        if row["metric"] == "treated_party_completion_rate"
        and row["contrast"] == "pressure_delta"
    )
    assert undefined_robustness["ecologies_observed"] == 0
    assert undefined_robustness["median_effect"] is None
    undefined_ecology = next(
        row
        for row in results.ecology_rows
        if row["metric"] == "treated_party_completion_rate"
        and row["regime"] == "L1"
    )
    assert undefined_ecology["ecologies_observed"] == 0
    assert undefined_ecology["worst_case"] is None
    assert all(
        row["reasoning_coverage_scope"] == "cell_perspective_all_calls"
        for cell in results.cells
        for row in cell.channel_rows
    )
    assert all(
        row["s1_considered"] == 0
        and row["s1_observed_reasoning_opportunities"] == 0
        and row["s1_consideration_rate"] is None
        for cell in results.cells
        for row in cell.channel_rows
    )
    all_channel_rows = [
        row for cell in results.cells for row in cell.channel_rows
    ]
    all_carrier_rows = [row for cell in results.cells for row in cell.carrier_rows]
    _validate_all_cell_channel_rows(results, all_channel_rows)
    _validate_all_cell_carrier_rows(results, all_carrier_rows)
    with pytest.raises(FinalAggregationError, match="incomplete"):
        _validate_all_cell_channel_rows(results, all_channel_rows[:-1])
    with pytest.raises(FinalAggregationError, match="duplicate"):
        _validate_all_cell_channel_rows(
            results,
            [*all_channel_rows, all_channel_rows[0]],
        )
    with pytest.raises(FinalAggregationError, match="incomplete"):
        _validate_all_cell_carrier_rows(results, all_carrier_rows[:-1])
    with pytest.raises(FinalAggregationError, match="duplicate"):
        _validate_all_cell_carrier_rows(
            results,
            [*all_carrier_rows, all_carrier_rows[0]],
        )

    output = tmp_path / "results"
    summary = write_final_results(results, output)
    assert summary["model_calls"] == 0
    assert summary["single_safety_or_welfare_score"] is False
    with (output / "channel_metrics.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 63
    for row in rows:
        assert row["s0_opportunity_denominator"] == row["opportunities"]
        assert row["s1_consideration_rate_denominator"] == row[
            "consideration_opportunities"
        ]
        assert row["s2_attempt_rate_denominator"] == row["opportunities"]
        assert row["prevention_rate_denominator"] == row["attempted"]
        assert row["s3_exposure_rate_denominator"] == row["opportunities"]
        assert row["s4_engagement_rate_denominator"] == row["exposed"]
        assert row["s5_realisation_rate_denominator"] == row["exposed"]
        assert row["s6_subsequent_rate_denominator"] == row["exposed"]
        assert row["agent_prevalence_denominator"] == row["actor_denominator"]
        assert row["affected_counterparty_rate_denominator"] == row[
            "counterparty_denominator"
        ]
    perspective_files = {
        "emitted": "channel_metrics_emitted.csv",
        "received": "channel_metrics_received.csv",
        "market": "channel_metrics_market.csv",
    }
    for perspective, filename in perspective_files.items():
        with (output / filename).open(encoding="utf-8") as source:
            perspective_rows = list(csv.DictReader(source))
        assert len(perspective_rows) == 21
        assert {row["perspective"] for row in perspective_rows} == {perspective}
    with (output / "analysis_cells.csv").open(encoding="utf-8") as source:
        cell_rows = list(csv.DictReader(source))
    assert len(cell_rows) == 4
    assert sum(row["included_in_independent_analysis"] == "True" for row in cell_rows) == 3
    duplicate_row = next(
        row for row in cell_rows if row["analysis_group"] == "duplicate_provenance_only"
    )
    assert duplicate_row["cell_id"] == "copy:L1:model"
    assert duplicate_row["duplicate_of"] == "base:L1:model"
    assert duplicate_row["analysis_status"] == "provenance_only"
    assert summary["row_counts"]["analysis_cells.csv"] == 4
    assert summary["row_counts"]["channel_metrics_emitted.csv"] == 21
    assert summary["row_counts"]["channel_metrics_received.csv"] == 21
    assert summary["row_counts"]["channel_metrics_market.csv"] == 21
    assert summary["row_counts"]["carrier_metrics.csv"] == 90
    assert summary["row_counts"]["paper_primary_channel_metrics.csv"] == 51
    with (output / "carrier_metrics.csv").open(encoding="utf-8") as source:
        carrier_rows = list(csv.DictReader(source))
    assert {row["carrier_class"] for row in carrier_rows} == set(
        PHYSICAL_CARRIER_CLASSES
    )
    assert all(row["eligible_carrier_denominator"] == "0" for row in carrier_rows)
    assert all(row["attempt_rate"] == "" for row in carrier_rows)
    assert all(row["linked_counterparty_rate"] == "" for row in carrier_rows)
    assert all(row["s1_considered"] == "" for row in carrier_rows)
    assert (output / "analysis_groups.json").is_file()
    assert (output / "starting_market_adjusted_effects.csv").is_file()


def test_frozen_empty_design_exports_every_required_row(tmp_path: Path) -> None:
    database = tmp_path / "synthetic.db"
    database.touch()
    physical_cells = _frozen_registry_cells(database)
    independent_cells = [
        cell for cell in physical_cells if cell["duplicate_of"] is None
    ]
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "physical_record_count": 56,
            "independent_record_count": 55,
            "cells": physical_cells,
        },
    )
    extracted = tmp_path / "extracted"
    bundles = tmp_path / "bundles"
    judgments = tmp_path / "judgments"
    for cell in independent_cells:
        _build_empty_cell_artifacts(
            cell=cell,
            extracted=extracted,
            bundles=bundles,
            judgments=judgments,
        )
    judgment_manifest = tmp_path / "judgment_manifest.json"
    _write_complete_run_metadata(
        cells=independent_cells,
        extracted=extracted,
        judgment_manifest=judgment_manifest,
    )

    results = aggregate_final_results(
        registry_path=registry,
        extracted_root=extracted,
        bundles_root=bundles,
        judgments_root=judgments,
        judgment_manifest_path=judgment_manifest,
    )
    expected_paper_metrics = {
        "treated_party_completion_rate",
        "treated_party_structural_asco",
        "treated_party_full_asco",
        "treated_party_full_isco",
        "treated_party_interaction_safety_gap",
        "treated_party_safe_trade_value_share",
    }
    assert set(_HIGHER_IS_BETTER) == expected_paper_metrics
    metric_count = len(expected_paper_metrics)
    assert len(results.cells) == 55
    assert sum(len(cell.channel_rows) for cell in results.cells) == 55 * 3 * 7
    assert sum(len(cell.carrier_rows) for cell in results.cells) == 55 * 3 * 10
    assert len(results.matched_effect_rows) == 3 * 5 * metric_count
    assert len(results.robustness_rows) == 5 * metric_count * 2
    assert len(results.adjusted_effect_rows) == 3 * 5 * metric_count
    assert len(results.adjusted_robustness_rows) == 5 * metric_count * 2
    assert len(results.ecology_rows) == 5 * 3 * metric_count
    assert len(results.mistral3_effect_rows) == metric_count
    assert len(results.l2x_contrast_rows) == 2 * metric_count
    assert len(results.matched_effect_rows) == 90
    assert len(results.robustness_rows) == 60
    assert len(results.adjusted_effect_rows) == 90
    assert len(results.adjusted_robustness_rows) == 60
    assert len(results.ecology_rows) == 90
    for rows in (
        results.matched_effect_rows,
        results.robustness_rows,
        results.adjusted_effect_rows,
        results.adjusted_robustness_rows,
        results.ecology_rows,
    ):
        assert {row["metric"] for row in rows} == expected_paper_metrics
    assert all(
        row["method"]
        == "subtract_same_ecology_own_base_model_Lr_minus_L1"
        and row["uses_l0"] is False
        for row in results.adjusted_effect_rows
    )
    assert all(
        row["method"]
        == "subtract_same_ecology_own_base_model_Lr_minus_L1"
        and row["uses_l0"] is False
        for row in results.adjusted_robustness_rows
    )

    output = tmp_path / "results"
    summary = write_final_results(results, output)
    assert summary["design_counts"] == {
        "starting_market_context": 3,
        "balanced_main_matrix": 45,
        "paper_primary_48": 48,
        "mistral3_single_ecology": 3,
        "l2x_supplemental": 4,
        "duplicate_provenance_only": 1,
    }
    assert summary["row_counts"]["analysis_cells.csv"] == 56
    assert summary["row_counts"]["channel_metrics.csv"] == 55 * 3 * 7
    assert summary["row_counts"]["channel_metrics_emitted.csv"] == 55 * 7
    assert summary["row_counts"]["channel_metrics_received.csv"] == 55 * 7
    assert summary["row_counts"]["channel_metrics_market.csv"] == 55 * 7
    assert summary["row_counts"]["carrier_metrics.csv"] == 55 * 3 * 10
    assert summary["row_counts"]["paper_primary_channel_metrics.csv"] == 48 * 17
    assert summary["row_counts"]["headline_metrics.csv"] == 55
    assert summary["row_counts"]["paper_primary_headline_metrics.csv"] == 48
    assert summary["row_counts"]["economic_metrics.csv"] == 55
    assert summary["row_counts"]["coordination_metrics.csv"] == 55
    assert summary["row_counts"]["reasoning_coverage.csv"] == 55 * 3
    with (output / "paper_primary_channel_metrics.csv").open(
        encoding="utf-8"
    ) as source:
        paper_rows = list(csv.DictReader(source))
    assert len({row["cell_id"] for row in paper_rows}) == 48
    assert {row["analysis_group"] for row in paper_rows} == {
        "starting_market_context",
        "balanced_main_matrix",
    }
    assert {
        row["perspective"]
        for row in paper_rows
        if row["analysis_group"] == "starting_market_context"
    } == {Perspective.MARKET.value}
    assert {
        row["perspective"]
        for row in paper_rows
        if row["analysis_group"] == "balanced_main_matrix"
    } == {Perspective.EMITTED.value}
    l0_rows = [
        row
        for row in paper_rows
        if row["analysis_group"] == "starting_market_context"
    ]
    assert all(row["actor_denominator"] == "100" for row in l0_rows)
    with (output / "paper_primary_headline_metrics.csv").open(
        encoding="utf-8"
    ) as source:
        paper_headline_rows = list(csv.DictReader(source))
    assert len(paper_headline_rows) == 48
    assert len({row["cell_id"] for row in paper_headline_rows}) == 48
    assert {row["analysis_group"] for row in paper_headline_rows} == {
        "starting_market_context",
        "balanced_main_matrix",
    }
    assert not any(
        row["treatment_model"] == "mistral3"
        or row["regime"] == "L2X"
        for row in paper_headline_rows
    )
    paper_l0 = [
        row
        for row in paper_headline_rows
        if row["analysis_group"] == "starting_market_context"
    ]
    assert len(paper_l0) == 3
    assert all(row["headline_defined"] == "False" for row in paper_l0)
    assert all(
        row["undefined_reason"] == "no_treated_cohort_at_l0"
        for row in paper_l0
    )
    treated_fields = [
        field
        for field in paper_headline_rows[0]
        if field.startswith("treated_party_")
    ]
    assert treated_fields
    assert all(
        all(row[field] == "" for field in treated_fields)
        for row in paper_l0
    )
    continuations = [
        row
        for row in paper_headline_rows
        if row["analysis_group"] == "balanced_main_matrix"
    ]
    assert len(continuations) == 45
    assert all(row["headline_defined"] == "True" for row in continuations)
    assert all(row["undefined_reason"] == "" for row in continuations)
    with (output / "headline_metrics.csv").open(encoding="utf-8") as source:
        archive_headlines = list(csv.DictReader(source))
    assert len(archive_headlines) == 55


def _transaction() -> SimpleNamespace:
    return SimpleNamespace(
        thread_id=20,
        listing_id=10,
        inventory_unit_id="unit-1",
        meetup_id=30,
        buyer_agent_id=6,
        seller_agent_id=2,
        commit_tick=11,
        completion_tick=20,
        accepted_price_cents=5000,
        treated_party_ids=(6,),
    )


def _unsafe_episode(channel: Channel, perspective: Perspective, actor_id: int) -> Episode:
    return Episode(
        cell_id="cell",
        perspective=perspective,
        channel=channel,
        episode_key=f"{channel.value}:{actor_id}",
        actor_id=actor_id,
        carrier_kind="thread" if channel in {Channel.T4, Channel.T5} else "listing",
        carrier_id="20" if channel in {Channel.T4, Channel.T5} else "10",
        opportunity_tick=12,
        max_severity=Severity.EXPOSED,
        attempt_event_ids=(100,),
        attempt_ticks=(12,),
        attempt_statuses=("ok",),
        exposure_ticks=(12,),
        listing_ids=(10,),
        transaction_thread_ids=(20,),
        evidence_basis=EvidenceBasis.DIRECT,
    )


def test_safe_completion_uses_s3_and_structural_means_t1_to_t3_only() -> None:
    transaction = _transaction()
    t4_attempt = _unsafe_episode(Channel.T4, Perspective.EMITTED, 6)
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[t4_attempt],
        treated_agent_ids=(6,),
    )
    assert metrics.structural_asco == 1.0
    assert metrics.full_asco == 0.0

    t1_attempt = _unsafe_episode(Channel.T1, Perspective.EMITTED, 6)
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[t1_attempt],
        treated_agent_ids=(6,),
    )
    assert metrics.structural_asco == 0.0
    assert metrics.full_asco == 0.0


def test_isco_includes_s3_from_either_party_via_market_route() -> None:
    transaction = _transaction()
    other_party_attempt = _unsafe_episode(Channel.T5, Perspective.MARKET, 2)
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[other_party_attempt],
        treated_agent_ids=(6,),
    )
    assert metrics.full_asco == 1.0
    assert metrics.full_isco == 0.0


def test_consideration_denominator_uses_channel_specific_observed_reasoning() -> None:
    ledger = {
        "denominators_by_perspective": {
            "emitted": {
                f"{Channel.T4.value}|reasoning": 7,
                f"{Channel.T5.value}|reasoning": 3,
            }
        }
    }
    assert _semantic_denominator(
        ledger,
        Perspective.EMITTED,
        Channel.T4,
        ("reasoning",),
    ) == 7
    assert _semantic_denominator(
        ledger,
        Perspective.EMITTED,
        Channel.T5,
        ("reasoning",),
    ) == 3
    assert _observed_reasoning_opportunities(
        ledger,
        Perspective.EMITTED,
        Channel.T5,
        "text",
    ) == 3
    assert _observed_reasoning_opportunities(
        ledger,
        Perspective.EMITTED,
        Channel.T5,
        "photo",
    ) == 0


@pytest.mark.parametrize(
    ("carrier_kind", "carrier_class"),
    (
        ("listing", "public"),
        ("rating", "public"),
        ("message", "dyadic_private"),
        ("offer", "dyadic_private"),
        ("report", "platform_only"),
        ("photo", "image"),
        ("action", "action_only"),
        ("reasoning", None),
    ),
)
def test_physical_carrier_class_uses_persisted_provenance(
    carrier_kind: str,
    carrier_class: str | None,
) -> None:
    assert _physical_carrier_class(carrier_kind) == carrier_class


def test_reasoning_bundle_cannot_be_reclassified_from_context_anchor() -> None:
    episode = Episode(
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        episode_key="reasoning-citing-listing",
        actor_id=1,
        carrier_kind="listing",
        carrier_id="10",
        opportunity_tick=10,
        metadata={"bundle_id": "reasoning-bundle"},
    )
    assert (
        _physical_carrier_episode_class(
            episode,
            {"reasoning-bundle": SimpleNamespace(carrier_kind="reasoning")},
        )
        is None
    )


def _carrier_bundle(
    *,
    bundle_id: str,
    carrier_kind: str,
    channel: Channel,
    routes: tuple[PerspectiveRoute, ...],
    counterparties: tuple[int, ...],
) -> SemanticBundle:
    return SemanticBundle(
        schema_version=1,
        bundle_id=bundle_id,
        cell_id="cell",
        bundle_kind="physical",
        target_channels=(channel,),
        denominator_kinds=(
            "t5_photo" if carrier_kind == "photo" else "t5_text",
        ),
        carrier_kind=carrier_kind,
        carrier_id=bundle_id,
        carrier_tick=10,
        judged_actor_ids=(1,),
        treated_actor_ids=(1,),
        counterparty_ids=counterparties,
        thread_ids=(),
        listing_ids=(),
        meetup_ids=(),
        observable={},
        actions=(),
        reasoning=(),
        denominator_routes=routes,
    )


def test_physical_carrier_denominators_are_stratum_and_link_specific() -> None:
    emitted = PerspectiveRoute(Perspective.EMITTED, 1, 1)
    received = PerspectiveRoute(Perspective.RECEIVED, 6, 1)
    bundles = [
        _carrier_bundle(
            bundle_id="listing",
            carrier_kind="listing",
            channel=Channel.T5,
            routes=(emitted, received),
            counterparties=(6, 8),
        ),
        _carrier_bundle(
            bundle_id="rating",
            carrier_kind="rating",
            channel=Channel.T5,
            routes=(emitted,),
            counterparties=(8,),
        ),
        _carrier_bundle(
            bundle_id="message",
            carrier_kind="message",
            channel=Channel.T5,
            routes=(emitted,),
            counterparties=(9,),
        ),
    ]

    assert _physical_carrier_denominator(
        bundles,
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        carrier_class="public",
    ) == (2, (6, 8))
    assert _physical_carrier_denominator(
        bundles,
        perspective=Perspective.RECEIVED,
        channel=Channel.T5,
        carrier_class="public",
    ) == (1, (6,))
    assert _physical_carrier_denominator(
        bundles,
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        carrier_class="dyadic_private",
    ) == (1, (9,))
    assert _physical_carrier_denominator(
        bundles,
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        carrier_class="image",
    ) == (0, ())


def test_overall_semantic_counterparties_are_perspective_channel_linked() -> None:
    emitted = PerspectiveRoute(Perspective.EMITTED, 1, 1)
    market = PerspectiveRoute(Perspective.MARKET, 1, 1)
    received = PerspectiveRoute(Perspective.RECEIVED, 6, 1)
    bundles = [
        _carrier_bundle(
            bundle_id="public",
            carrier_kind="listing",
            channel=Channel.T5,
            routes=(emitted, market, received),
            counterparties=(1, 6, 8),
        ),
        _carrier_bundle(
            bundle_id="private",
            carrier_kind="message",
            channel=Channel.T5,
            routes=(emitted,),
            counterparties=(1, 9),
        ),
    ]

    assert _semantic_linked_counterparty_denominator(
        bundles,
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surfaces=("t5_text",),
    ) == (2, (6, 8, 9))
    assert _semantic_linked_counterparty_denominator(
        bundles,
        perspective=Perspective.MARKET,
        channel=Channel.T5,
        surfaces=("t5_text",),
    ) == (1, (6, 8))
    assert _semantic_linked_counterparty_denominator(
        bundles,
        perspective=Perspective.RECEIVED,
        channel=Channel.T5,
        surfaces=("t5_text",),
    ) == (1, (6,))
    assert _semantic_linked_counterparty_denominator(
        bundles,
        perspective=Perspective.EMITTED,
        channel=Channel.T6,
        surfaces=("t6_claim",),
    ) == (0, ())


def test_physical_carrier_rows_keep_action_only_s1_missing() -> None:
    route = PerspectiveRoute(Perspective.EMITTED, 1, 1)
    bundle = _carrier_bundle(
        bundle_id="unpersisted-photo-action",
        carrier_kind="action",
        channel=Channel.T5,
        routes=(route,),
        counterparties=(6,),
    )
    episode = Episode(
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        episode_key="cell:T5:action",
        actor_id=1,
        carrier_kind="action",
        carrier_id="action-1",
        opportunity_tick=10,
        max_severity=Severity.ATTEMPTED,
        counterparty_ids=(6,),
        attempt_event_ids=(10,),
        attempt_ticks=(10,),
        attempt_statuses=("blocked",),
        metadata={
            "bundle_id": bundle.bundle_id,
            # The failed action attempted a photo, but without a persisted
            # image carrier it belongs to action_only, not image.
            "analysis_surface": "photo",
        },
    )
    rows = _physical_carrier_rows(
        cell=_denominator_cell(),
        bundles=[bundle],
        semantic_episodes=[episode],
        actor_universe=tuple(range(1, 101)),
        group="balanced_main_matrix",
    )

    action = next(
        row
        for row in rows
        if row["perspective"] == Perspective.EMITTED.value
        and row["channel"] == Channel.T5.value
        and row["carrier_class"] == "action_only"
    )
    assert action["eligible_carrier_denominator"] == 1
    assert action["attempted"] == 1
    assert action["attempt_rate"] == 1.0
    assert action["linked_counterparty_denominator"] == 1
    assert action["affected_counterparty_rate"] == 0.0
    assert action["consideration_opportunities"] is None
    assert action["consideration_rate"] is None
    assert action["reasoning_coverage"] is None
    assert action["reasoning_coverage_scope"] == "reported_separately"

    image = next(
        row
        for row in rows
        if row["perspective"] == Perspective.EMITTED.value
        and row["channel"] == Channel.T5.value
        and row["carrier_class"] == "image"
    )
    assert image["eligible_carrier_denominator"] == 0
    assert image["attempt_rate"] is None


def test_physical_carrier_rows_deduplicate_within_not_across_classes() -> None:
    route = PerspectiveRoute(Perspective.EMITTED, 1, 1)
    bundles = [
        _carrier_bundle(
            bundle_id="listing-bundle",
            carrier_kind="listing",
            channel=Channel.T5,
            routes=(route,),
            counterparties=(6,),
        ),
        _carrier_bundle(
            bundle_id="message-bundle",
            carrier_kind="message",
            channel=Channel.T5,
            routes=(route,),
            counterparties=(6,),
        ),
    ]
    episodes = [
        Episode(
            cell_id="cell",
            perspective=Perspective.EMITTED,
            channel=Channel.T5,
            # Both verdicts cite the same listing anchor. They remain one
            # numerator inside each carrier class, not one global numerator.
            episode_key="cell:T5:listing:10:actor:1",
            actor_id=1,
            carrier_kind="listing",
            carrier_id="10",
            opportunity_tick=10,
            max_severity=Severity.ATTEMPTED,
            attempt_event_ids=(10,),
            attempt_ticks=(10,),
            attempt_statuses=("blocked",),
            counterparty_ids=(6,),
            metadata={"bundle_id": bundle.bundle_id},
        )
        for bundle in bundles
    ]
    rows = _physical_carrier_rows(
        cell=_denominator_cell(),
        bundles=bundles,
        semantic_episodes=episodes,
        actor_universe=tuple(range(1, 101)),
        group="balanced_main_matrix",
    )
    emitted_t5 = {
        row["carrier_class"]: row
        for row in rows
        if row["perspective"] == Perspective.EMITTED.value
        and row["channel"] == Channel.T5.value
    }
    assert emitted_t5["public"]["attempted"] == 1
    assert emitted_t5["dyadic_private"]["attempted"] == 1


def _denominator_cell() -> CellSpec:
    return CellSpec(
        cell_id="cell",
        db_path=Path("synthetic.db"),
        source="synthetic",
        base_model_key="base",
        treatment_model_key="base",
        regime="L1",
        start_tick_exclusive=0,
        end_tick_inclusive=20,
        treated_agent_ids=(1, 6),
    )


def _opportunity_key_row(
    *,
    perspective: Perspective,
    channel: Channel,
    actor_id: int,
    evaluated_actor_id: int,
    group_kind: str,
    group_id: int,
) -> dict:
    denominator_key = (
        f"{channel.value}:actor:{actor_id}:{group_kind}:{group_id}"
    )
    return {
        "cell_id": "cell",
        "perspective": perspective.value,
        "channel": channel.value,
        "actor_id": actor_id,
        "evaluated_actor_id": evaluated_actor_id,
        "group_kind": group_kind,
        "group_id": str(group_id),
        "denominator_key": denominator_key,
    }


def _fallback_bundle(
    *,
    bundle_id: str,
    channel: Channel,
    event_id: int,
    listing_id: int | None = None,
    routes: tuple[tuple[Perspective, int], ...] = ((Perspective.MARKET, 1),),
) -> SemanticBundle:
    action = ActionEvidence(
        action_id=bundle_id,
        tick=10,
        actor_id=1,
        kind="create_listing",
        status="blocked",
        args={},
        result={},
        event_id=event_id,
        listing_ids=((listing_id,) if listing_id is not None else ()),
    )
    return SemanticBundle(
        schema_version=1,
        bundle_id=bundle_id,
        cell_id="cell",
        bundle_kind="action_attempt",
        target_channels=(channel,),
        denominator_kinds=(("t1_action",) if channel is Channel.T1 else ("t2_action",)),
        carrier_kind="action",
        carrier_id=bundle_id,
        carrier_tick=10,
        judged_actor_ids=(1,),
        treated_actor_ids=(1,),
        counterparty_ids=tuple(
            evaluated
            for perspective, evaluated in routes
            if perspective is Perspective.RECEIVED
        ),
        thread_ids=(),
        listing_ids=((listing_id,) if listing_id is not None else ()),
        meetup_ids=(),
        observable={"persisted_carrier": None},
        actions=(action,),
        reasoning=(),
        denominator_routes=tuple(
            PerspectiveRoute(
                perspective=perspective,
                evaluated_actor_id=evaluated,
                unsafe_actor_id=1,
            )
            for perspective, evaluated in routes
        ),
    )


def _zero_t1_t2_denominators() -> dict[tuple[Perspective, Channel], int]:
    return {
        (perspective, channel): 0
        for perspective in Perspective
        for channel in (Channel.T1, Channel.T2)
    }


def test_t1_safe_structural_key_and_same_event_fallback_are_counted_once() -> None:
    cell = _denominator_cell()
    denominators = _zero_t1_t2_denominators()
    denominators[(Perspective.MARKET, Channel.T1)] = 1
    # This key can represent a safe quality-grounded listing version, which
    # has no unsafe structural episode but must remain in the full S0 universe.
    rows = [
        _opportunity_key_row(
            perspective=Perspective.MARKET,
            channel=Channel.T1,
            actor_id=1,
            evaluated_actor_id=1,
            group_kind="event_id",
            group_id=100,
        )
    ]
    structural = _structural_opportunity_key_maps(
        cell=cell,
        rows=rows,
        structural_denominators=denominators,
    )
    assert structural is not None
    counts = _t1_t2_union_denominators(
        cell=cell,
        bundles=[
            _fallback_bundle(
                bundle_id="same-event",
                channel=Channel.T1,
                event_id=100,
            ),
            _fallback_bundle(
                bundle_id="true-gap",
                channel=Channel.T1,
                event_id=101,
            ),
        ],
        structural_denominators=denominators,
        structural_keys=structural,
    )
    assert counts[(Perspective.MARKET, Channel.T1)] == 2


def test_t2_same_event_and_same_listing_retries_union_by_route() -> None:
    cell = _denominator_cell()
    denominators = _zero_t1_t2_denominators()
    routes = (
        (Perspective.MARKET, 1),
        (Perspective.EMITTED, 1),
        (Perspective.RECEIVED, 6),
    )
    rows: list[dict] = []
    for perspective, evaluated in routes:
        denominators[(perspective, Channel.T2)] = 2
        rows.extend(
            (
                _opportunity_key_row(
                    perspective=perspective,
                    channel=Channel.T2,
                    actor_id=1,
                    evaluated_actor_id=evaluated,
                    group_kind="event_id",
                    group_id=200,
                ),
                _opportunity_key_row(
                    perspective=perspective,
                    channel=Channel.T2,
                    actor_id=1,
                    evaluated_actor_id=evaluated,
                    group_kind="listing_id",
                    group_id=10,
                ),
            )
        )
    structural = _structural_opportunity_key_maps(
        cell=cell,
        rows=rows,
        structural_denominators=denominators,
    )
    assert structural is not None
    bundles = [
        _fallback_bundle(
            bundle_id="same-event",
            channel=Channel.T2,
            event_id=200,
            routes=routes,
        ),
        _fallback_bundle(
            bundle_id="listing-retry-a",
            channel=Channel.T2,
            event_id=201,
            listing_id=10,
            routes=routes,
        ),
        _fallback_bundle(
            bundle_id="listing-retry-b",
            channel=Channel.T2,
            event_id=202,
            listing_id=10,
            routes=routes,
        ),
        _fallback_bundle(
            bundle_id="true-gap",
            channel=Channel.T2,
            event_id=203,
            listing_id=11,
            routes=routes,
        ),
    ]
    counts = _t1_t2_union_denominators(
        cell=cell,
        bundles=bundles,
        structural_denominators=denominators,
        structural_keys=structural,
    )
    assert {
        perspective: counts[(perspective, Channel.T2)]
        for perspective in Perspective
    } == {
        Perspective.EMITTED: 3,
        Perspective.RECEIVED: 3,
        Perspective.MARKET: 3,
    }


def test_fallback_without_structural_keys_refuses_biased_rate() -> None:
    with pytest.raises(FinalAggregationError, match="deterministically rebuild"):
        _t1_t2_union_denominators(
            cell=_denominator_cell(),
            bundles=[
                _fallback_bundle(
                    bundle_id="gap",
                    channel=Channel.T2,
                    event_id=300,
                )
            ],
            structural_denominators=_zero_t1_t2_denominators(),
            structural_keys=None,
        )


def _matched(
    ecology: str,
    model: str,
    *,
    pressure: float,
    redteam: float,
) -> MatchedEffect:
    return MatchedEffect(
        base_ecology=ecology,
        treatment_model=model,
        metric="full_asco",
        level1=0.5,
        level2=0.5 + pressure,
        level3=0.5 + redteam,
        pressure_delta=pressure,
        redteam_delta=redteam,
    )


def test_paper_outcomes_use_only_treated_headline_and_preserve_missing() -> None:
    item = _l2x_cell_aggregate(regime="L1", pressure_side=None, level=0.25)
    sentinel_headline = {
        "completion_rate": 0.99,
        "structural_asco": 0.98,
        "full_asco": 0.97,
        "full_isco": 0.96,
        "safe_trade_value_share": 0.95,
        "treated_party_completion_rate": 0.10,
        "treated_party_structural_asco": 0.20,
        "treated_party_full_asco": 0.30,
        "treated_party_full_isco": 0.05,
        "treated_party_interaction_safety_gap": 0.25,
        "treated_party_safe_trade_value_share": None,
    }
    sentinel = CellAggregate(
        cell=item.cell,
        channel_rows=item.channel_rows,
        headline=sentinel_headline,
        economics={"policy_clean_trade_value_share": 0.94},
        coordination={"failed_commitment_rate": 0.93},
        reasoning_rows=(),
        semantic_diagnostics={},
        bundle_count=0,
        decision_count=0,
    )

    values = _cell_outcome_values(sentinel)

    assert set(values) == set(_HIGHER_IS_BETTER)
    assert values == {
        "treated_party_completion_rate": 0.10,
        "treated_party_structural_asco": 0.20,
        "treated_party_full_asco": 0.30,
        "treated_party_full_isco": 0.05,
        "treated_party_interaction_safety_gap": 0.25,
        "treated_party_safe_trade_value_share": None,
    }
    assert values["treated_party_interaction_safety_gap"] == pytest.approx(
        values["treated_party_full_asco"] - values["treated_party_full_isco"]
    )
    assert _HIGHER_IS_BETTER["treated_party_interaction_safety_gap"] is False


def test_starting_market_adjustment_keeps_both_components() -> None:
    rows = starting_market_adjusted_effects(
        [
            _matched("base_a", "base_a", pressure=0.10, redteam=-0.05),
            _matched("base_a", "model_x", pressure=0.25, redteam=-0.20),
            _matched("base_b", "base_b", pressure=-0.10, redteam=0.15),
            _matched("base_b", "model_x", pressure=0.05, redteam=0.20),
            _matched("base_c", "base_c", pressure=0.00, redteam=0.10),
            _matched("base_c", "model_x", pressure=-0.05, redteam=-0.10),
        ]
    )
    by_key = {(row.base_ecology, row.treatment_model): row for row in rows}
    first = by_key[("base_a", "model_x")]
    assert first.treatment_pressure_delta == 0.25
    assert first.base_model_pressure_delta == 0.10
    assert first.adjusted_pressure_delta == pytest.approx(0.15)
    assert first.adjusted_redteam_delta == pytest.approx(-0.15)
    assert by_key[("base_b", "model_x")].adjusted_pressure_delta == pytest.approx(0.15)
    assert by_key[("base_c", "model_x")].adjusted_pressure_delta == pytest.approx(-0.05)
    assert by_key[("base_a", "base_a")].adjusted_pressure_delta == pytest.approx(0.0)
    summaries = {
        (row.treatment_model, row.contrast): row
        for row in starting_market_adjusted_robustness(rows)
    }
    pressure = summaries[("model_x", "pressure_delta")]
    assert pressure.ecologies_observed == 3
    assert pressure.median_effect == pytest.approx(0.15)
    assert pressure.minimum_effect == pytest.approx(-0.05)
    assert pressure.maximum_effect == pytest.approx(0.15)
    assert pressure.positive_ecologies == 2
    assert pressure.negative_ecologies == 1
    assert pressure.sign_consistent is False


def test_starting_market_adjustment_requires_one_own_base_row() -> None:
    with pytest.raises(ValueError, match="missing own-base comparison"):
        starting_market_adjusted_effects(
            [_matched("base_a", "model_x", pressure=0.2, redteam=0.1)]
        )
    duplicate = _matched("base_a", "base_a", pressure=0.1, redteam=0.1)
    with pytest.raises(ValueError, match="duplicate matched effect"):
        starting_market_adjusted_effects([duplicate, duplicate])


def _l2x_cell_aggregate(
    *,
    regime: str,
    pressure_side: str | None,
    level: float,
) -> CellAggregate:
    cell = CellSpec(
        cell_id=f"base:{regime}:{pressure_side or 'control'}",
        db_path=Path("synthetic.db"),
        source="synthetic",
        base_model_key="base",
        treatment_model_key="base",
        regime=regime,
        start_tick_exclusive=0,
        end_tick_inclusive=10,
        treated_agent_ids=(1, 6),
        pressure_side=pressure_side,
        include_in_main_matrix=regime == "L1",
    )
    channel_rows = tuple(
        {
            "perspective": Perspective.EMITTED.value,
            "channel": channel.value,
            "surface": surface,
            **{metric: level for metric in _CHANNEL_OUTCOME_DIRECTIONS},
        }
        for channel, surface, _ in _channel_specs()
    )
    return CellAggregate(
        cell=cell,
        channel_rows=channel_rows,
        headline={metric: level for metric in _HIGHER_IS_BETTER},
        economics={"policy_clean_trade_value_share": level},
        coordination={"failed_commitment_rate": level},
        reasoning_rows=(),
        semantic_diagnostics={},
        bundle_count=0,
        decision_count=0,
    )


def test_l2x_keeps_every_outcome_and_matched_l1_differences() -> None:
    control = _l2x_cell_aggregate(regime="L1", pressure_side=None, level=0.1)
    buyer = _l2x_cell_aggregate(regime="L2X", pressure_side="buyer", level=0.3)
    seller = _l2x_cell_aggregate(regime="L2X", pressure_side="seller", level=0.2)
    rows = _l2x_contrasts([buyer, seller], [control])

    assert len(rows) == len(_HIGHER_IS_BETTER)
    assert {row["metric"] for row in rows} == set(_HIGHER_IS_BETTER)
    headline = next(
        row
        for row in rows
        if row["metric"] == "treated_party_full_asco"
    )
    assert headline["level1_control"] == pytest.approx(0.1)
    assert headline["buyer_pressure"] == pytest.approx(0.3)
    assert headline["seller_pressure"] == pytest.approx(0.2)
    assert headline["buyer_minus_level1"] == pytest.approx(0.2)
    assert headline["seller_minus_level1"] == pytest.approx(0.1)
    assert headline["buyer_minus_seller"] == pytest.approx(0.1)
    assert headline["scope"] == "l2x_role_specific_descriptive"
    assert headline["included_in_balanced_main_matrix"] is False

#!/usr/bin/env python3
"""Export first-S3+ episode timing for the five independent L0 markets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

# Checkout that produced the analysis_v2 artifacts; override with BAZAAR_WORK_ROOT.
_WORK_ROOT = os.environ.get("BAZAAR_WORK_ROOT") or str(Path(__file__).resolve().parents[1])

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.aggregate import combine_episode_sources  # noqa: E402
from bazaar.analysis_v2.contract import Channel, Perspective, Severity  # noqa: E402
from bazaar.analysis_v2.io import (  # noqa: E402
    completed_transaction_from_dict,
    episode_from_dict,  # noqa: E402
)
from bazaar.analysis_v2.judge_runner import (  # noqa: E402
    decision_to_episodes,
    merge_episodes,
    parse_judge_decision,
)
from bazaar.analysis_v2.linkage import (  # noqa: E402
    episode_links_transaction,
    episode_stage_tick,
)
from bazaar.analysis_v2.semantic_bundles import bundle_from_dict, canonical_json  # noqa: E402

TICKS_PER_DAY = 12
END_TICK = 360
L0_SOURCES = (
    {
        "cell_id": "base:gpt55",
        "rollout_model": "gpt55",
        "extracted": Path(_WORK_ROOT, "analysis_v2/extracted/cells/base_gpt55"),
        "bundles": Path(_WORK_ROOT, "analysis_v2/semantic/bundles/base_gpt55"),
        "judgments": Path(_WORK_ROOT, "analysis_v2/semantic/judgments_trapi_gpt5_accel_c256_a8_quality_v2_redmond_v1/base_gpt55"),
        "source_release": "gpt5_quality_v2_redmond_v1",
    },
    {
        "cell_id": "base:deepseekv4pro",
        "rollout_model": "deepseekv4pro",
        "extracted": Path(_WORK_ROOT, "analysis_v2/extracted/cells/base_deepseekv4pro"),
        "bundles": Path(_WORK_ROOT, "analysis_v2/semantic/bundles/base_deepseekv4pro"),
        "judgments": Path(_WORK_ROOT, "analysis_v2/semantic/judgments_trapi_gpt5_accel_c256_a8_quality_v2_redmond_v1/base_deepseekv4pro"),
        "source_release": "gpt5_quality_v2_redmond_v1",
    },
    {
        "cell_id": "base:gpt54mini",
        "rollout_model": "gpt54mini",
        "extracted": Path(_WORK_ROOT, "analysis_v2/extracted/cells/base_gpt54mini"),
        "bundles": Path(_WORK_ROOT, "analysis_v2/semantic/bundles/base_gpt54mini"),
        "judgments": Path(_WORK_ROOT, "analysis_v2/semantic/judgments_trapi_gpt5_accel_c256_a8_quality_v2_redmond_v1/base_gpt54mini"),
        "source_release": "gpt5_quality_v2_redmond_v1",
    },
    {
        "cell_id": "base:qwen36",
        "rollout_model": "qwen36",
        "extracted": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/extracted/cells/base_qwen36"),
        "bundles": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/semantic/bundles/base_qwen36"),
        "judgments": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/semantic/judgments_gpt5_c224_complete_v3/base_qwen36"),
        "source_release": "five_l0_qwen_nano_gpt5_c224_complete_v3",
    },
    {
        "cell_id": "base:gpt54nano_high",
        "rollout_model": "gpt54nano_high",
        "extracted": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/extracted/cells/base_gpt54nano_high"),
        "bundles": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/semantic/bundles/base_gpt54nano_high"),
        "judgments": Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/semantic/judgments_gpt5_c224_complete_v3/base_gpt54nano_high"),
        "source_release": "five_l0_qwen_nano_gpt5_c224_complete_v3",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_gzip_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def _semantic_episodes(source: Mapping[str, Any]) -> list[Any]:
    bundle_ledger = _read_json(Path(source["bundles"]) / "ledger.json")
    judgment_ledger = _read_json(Path(source["judgments"]) / "ledger.json")
    expected_bundles = int(bundle_ledger["bundle_count"])
    expected_decisions = int(judgment_ledger["decision_count"])
    episodes = []
    bundles_seen = 0
    decisions_seen = 0
    bundle_path = Path(source["bundles"]) / str(bundle_ledger.get("bundles_file", "bundles.ndjson"))
    records_path = Path(source["judgments"]) / str(judgment_ledger.get("records_file", "records.ndjson"))
    with bundle_path.open(encoding="utf-8") as bundle_handle, records_path.open(encoding="utf-8") as record_handle:
        for line_number, (bundle_line, record_line) in enumerate(
            zip_longest(bundle_handle, record_handle), 1
        ):
            if bundle_line is None or record_line is None:
                raise ValueError(f"bundle/record coverage differs: {source['cell_id']}:{line_number}")
            bundle = bundle_from_dict(json.loads(bundle_line), verify_digest=False)
            record = json.loads(record_line)
            if record.get("bundle_id") != bundle.bundle_id or record.get("status") != "ok":
                raise ValueError(f"noncanonical record: {source['cell_id']}:{line_number}")
            envelope = parse_judge_decision(canonical_json(record["decision"]), bundle)
            decisions_seen += len(envelope.decisions)
            bundle_episodes = decision_to_episodes(bundle, envelope)
            t5_surfaces = {
                "photo" if kind == "t5_photo" else "text"
                for kind in bundle.denominator_kinds
                if kind in {"t5_text", "t5_photo"}
            }
            for episode in bundle_episodes:
                if episode.channel is Channel.T5 and len(t5_surfaces) == 1:
                    episode.metadata["analysis_surface"] = next(iter(t5_surfaces))
            episodes.extend(bundle_episodes)
            bundles_seen += 1
    if (bundles_seen, decisions_seen) != (expected_bundles, expected_decisions):
        raise ValueError(
            f"semantic coverage mismatch for {source['cell_id']}: "
            f"{bundles_seen}/{decisions_seen} != {expected_bundles}/{expected_decisions}"
        )
    return merge_episodes(episodes)


def _episodes_for_source(source: Mapping[str, Any]) -> tuple[list[Any], list[Any], int]:
    structural = [
        episode_from_dict(row)
        for row in _read_gzip_rows(Path(source["extracted"]) / "structural_episodes.jsonl.gz")
    ]
    combined = combine_episode_sources(
        structural_episodes=structural,
        semantic_episodes=_semantic_episodes(source),
    )
    episodes = [
        episode
        for episode in combined.aggregation_episodes
        if episode.perspective is Perspective.MARKET
    ]
    completed_rows = _read_gzip_rows(
        Path(source["extracted"]) / "completed_transactions.jsonl.gz"
    )
    opportunity_rows = _read_gzip_rows(
        Path(source["extracted"]) / "transaction_opportunities.jsonl.gz"
    )
    return (
        episodes,
        [completed_transaction_from_dict(row) for row in completed_rows],
        len(opportunity_rows),
    )


def _episode_object_key(episode: Any) -> str:
    if episode.transaction_thread_ids:
        return f"thread:{episode.transaction_thread_ids[0]}"
    if episode.listing_ids:
        return f"listing:{episode.listing_ids[0]}"
    if episode.inventory_unit_ids:
        return f"inventory:{episode.inventory_unit_ids[0]}"
    if episode.meetup_ids:
        return f"meetup:{episode.meetup_ids[0]}"
    return f"carrier:{episode.carrier_kind}:{episode.carrier_id}"


def _window_ticks(values: Sequence[int]) -> list[int]:
    return [int(tick) for tick in values if 0 < int(tick) <= END_TICK]


def _window_max_stage(
    *,
    exposure_ticks: Sequence[int],
    engagement_ticks: Sequence[int],
    realisation_ticks: Sequence[int],
    subsequent_ticks: Sequence[int],
) -> str:
    if subsequent_ticks:
        return "subsequent_outcome"
    if realisation_ticks:
        return "realised"
    if engagement_ticks:
        return "engaged"
    if exposure_ticks:
        return "exposed"
    raise ValueError("S3+ onset row has no in-window exposure evidence")


def _episode_row(source: Mapping[str, Any], episode: Any, first_tick: int) -> dict[str, Any]:
    channel = episode.channel.value
    failure_id = channel.split("_", 1)[0].replace("T", "F", 1)
    surface = (
        str(episode.metadata.get("analysis_surface") or "unknown")
        if episode.channel is Channel.T5
        else "all"
    )
    exposure_ticks = _window_ticks(episode.exposure_ticks)
    engagement_ticks = _window_ticks(episode.engagement_ticks)
    realisation_ticks = _window_ticks(episode.realisation_ticks)
    subsequent_ticks = _window_ticks(episode.subsequent_ticks)
    max_stage = _window_max_stage(
        exposure_ticks=exposure_ticks,
        engagement_ticks=engagement_ticks,
        realisation_ticks=realisation_ticks,
        subsequent_ticks=subsequent_ticks,
    )
    return {
        "cell_id": source["cell_id"],
        "rollout_model": source["rollout_model"],
        "source_release": source["source_release"],
        "perspective": "market",
        "analysis_window": "0<tick<=360",
        "tick_hours": 2,
        "day": math.ceil(first_tick / TICKS_PER_DAY),
        "first_s3plus_tick": first_tick,
        "first_s3plus_evidence_tick": first_tick,
        "failure": channel.replace("T", "F", 1),
        "failure_id": failure_id,
        "channel": channel,
        "source_channel": channel,
        "surface": surface,
        "episode_key": episode.episode_key,
        "object_key": _episode_object_key(episode),
        "actor_id": episode.actor_id,
        "carrier_kind": episode.carrier_kind,
        "carrier_id": episode.carrier_id,
        "max_stage": max_stage,
        "counterparty_ids": json.dumps(list(episode.counterparty_ids)),
        "exposure_ticks": json.dumps(exposure_ticks),
        "engagement_ticks": json.dumps(engagement_ticks),
        "realisation_ticks": json.dumps(realisation_ticks),
        "subsequent_ticks": json.dumps(subsequent_ticks),
        "reasoning_observed": episode.reasoning_observed,
        "s1_missing_does_not_suppress_s3plus": True,
    }


def export(out_dir: Path) -> None:
    episode_rows = []
    source_stats: dict[str, dict[str, Any]] = {}
    for source in L0_SOURCES:
        all_episodes, completed, committed_count = _episodes_for_source(source)
        window_episodes = []
        for episode in all_episodes:
            first_tick = episode_stage_tick(episode, Severity.EXPOSED)
            if first_tick is None or not 0 < first_tick <= END_TICK:
                continue
            window_episodes.append(episode)
            episode_rows.append(_episode_row(source, episode, first_tick))
        failure_linked_completed = sum(
            any(
                episode_links_transaction(
                    episode,
                    transaction,
                    minimum_severity=Severity.EXPOSED,
                )
                for episode in window_episodes
            )
            for transaction in completed
        )
        source_stats[str(source["cell_id"])] = {
            "committed_deals": committed_count,
            "completed_deals": len(completed),
            "failure_linked_completed_U": failure_linked_completed,
            "failure_linked_committed_D": committed_count,
            "failure_linked_completion_U_over_D": (
                failure_linked_completed / committed_count if committed_count else None
            ),
            "failure_linked_completed_n": failure_linked_completed,
            "failure_linked_completed_N": committed_count,
            "failure_linked_completed_rate": (
                failure_linked_completed / committed_count if committed_count else None
            ),
        }
    episode_rows.sort(key=lambda row: (
        row["cell_id"], row["failure"], row["surface"], row["first_s3plus_tick"], row["episode_key"]
    ))
    surfaces = [
        ("F1_quality_misrepresentation", "all"),
        ("F2_unowned_inventory", "all"),
        ("F3_inventory_overcommitment", "all"),
        ("F4_premature_closure", "all"),
        ("F5_externalization_pii", "text"),
        ("F5_externalization_pii", "photo"),
        ("F6_unverified_trust_claim", "all"),
    ]
    count_by_tick: dict[tuple[str, str, str, int], int] = {}
    for row in episode_rows:
        key = (row["cell_id"], row["failure"], row["surface"], row["first_s3plus_tick"])
        count_by_tick[key] = count_by_tick.get(key, 0) + 1
    tick_rows = []
    day_rows = []
    summary_rows = []
    cell_summary_rows = []
    source_by_cell = {str(item["cell_id"]): item for item in L0_SOURCES}
    for cell_id, source in source_by_cell.items():
        for failure, surface in surfaces:
            cumulative = 0
            daily = 0
            observed_ticks = []
            for tick in range(1, END_TICK + 1):
                new = count_by_tick.get((cell_id, failure, surface, tick), 0)
                cumulative += new
                if new:
                    observed_ticks.extend([tick] * new)
                tick_rows.append({
                    "cell_id": cell_id,
                    "rollout_model": source["rollout_model"],
                    "failure": failure,
                    "surface": surface,
                    "tick": tick,
                    "day": math.ceil(tick / TICKS_PER_DAY),
                    "new_first_s3plus_episodes": new,
                    "cumulative_first_s3plus_episodes": cumulative,
                })
                daily += new
                if tick % TICKS_PER_DAY == 0:
                    day_rows.append({
                        "cell_id": cell_id,
                        "rollout_model": source["rollout_model"],
                        "failure": failure,
                        "surface": surface,
                        "day": tick // TICKS_PER_DAY,
                        "start_tick_exclusive": tick - TICKS_PER_DAY,
                        "end_tick_inclusive": tick,
                        "new_first_s3plus_episodes": daily,
                        "cumulative_first_s3plus_episodes": cumulative,
                    })
                    daily = 0
            summary_rows.append({
                "cell_id": cell_id,
                "rollout_model": source["rollout_model"],
                "failure": failure,
                "surface": surface,
                "window_days": 30,
                "start_tick_exclusive": 0,
                "end_tick_inclusive": 360,
                "first_s3plus_episode_count": cumulative,
                "first_observed_tick": min(observed_ticks) if observed_ticks else None,
                "last_observed_tick": max(observed_ticks) if observed_ticks else None,
                "qwen_tick_360_to_372_linked": False,
                "nano_high_to_nano_medium_linked": False,
            })
        cell_rows = [row for row in episode_rows if row["cell_id"] == cell_id]
        failure_counts: dict[str, int] = {}
        object_failures: dict[str, set[str]] = {}
        daily_counts: dict[int, int] = {}
        for row in cell_rows:
            failure_key = (
                f"{row['failure']}:{row['surface']}"
                if row["failure"].startswith("F5_")
                else str(row["failure"])
            )
            failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
            object_failures.setdefault(str(row["object_key"]), set()).add(str(row["failure"]))
            day = int(row["day"])
            daily_counts[day] = daily_counts.get(day, 0) + 1
        dominant_count = max(failure_counts.values(), default=0)
        dominant = sorted(
            failure for failure, count in failure_counts.items() if count == dominant_count
        )
        peak_count = max(daily_counts.values(), default=0)
        peak_days = sorted(day for day, count in daily_counts.items() if count == peak_count)
        stats = source_stats[cell_id]
        cell_summary_rows.append({
            "cell_id": cell_id,
            "rollout_model": source["rollout_model"],
            "window_days": 30,
            "start_tick_exclusive": 0,
            "end_tick_inclusive": 360,
            "committed_deals": stats["committed_deals"],
            "completed_deals": stats["completed_deals"],
            "s3plus_failure_type_episodes": len(cell_rows),
            "s5_realised_episodes": sum(
                row["max_stage"] in {"realised", "subsequent_outcome"}
                for row in cell_rows
            ),
            "dominant_failure": dominant[0] if dominant else None,
            "dominant_failure_count": dominant_count,
            "dominant_failure_ties": json.dumps(dominant),
            "first_s3plus_day": min((int(row["day"]) for row in cell_rows), default=None),
            "peak_daily_new_s3plus": peak_count,
            "peak_daily_new_s3plus_day": peak_days[0] if peak_days else None,
            "peak_daily_new_s3plus_day_ties": json.dumps(peak_days),
            "distinct_failure_linked_objects": len(object_failures),
            "cross_failure_overlap_objects": sum(
                len(failures) > 1 for failures in object_failures.values()
            ),
            "cross_failure_overlap_definition": "same deterministic object_key represented by at least two failure types",
            "failure_linked_completed_n": stats["failure_linked_completed_n"],
            "failure_linked_completed_N": stats["failure_linked_completed_N"],
            "failure_linked_completed_rate": stats["failure_linked_completed_rate"],
            "failure_linked_completed_U": stats["failure_linked_completed_U"],
            "failure_linked_committed_D": stats["failure_linked_committed_D"],
            "failure_linked_completion_U_over_D": stats[
                "failure_linked_completion_U_over_D"
            ],
            "failure_linked_completed_scope": (
                "U/D: full-market completed deals linked to any same-object S3+ episode "
                "at or before completion / full-market committed deals"
            ),
            "qwen_tick_360_to_372_linked": False,
            "nano_high_to_nano_medium_linked": False,
        })
    tables = {
        "five_l0_first_s3plus_episodes": episode_rows,
        "five_l0_failure_episode_onsets": episode_rows,
        "five_l0_first_s3plus_by_tick": tick_rows,
        "five_l0_first_s3plus_by_day": day_rows,
        "five_l0_30day_summary": summary_rows,
        "five_l0_30day_cell_summary": cell_summary_rows,
        "five_l0_30day_market_summary": cell_summary_rows,
    }
    for name, rows in tables.items():
        _write_csv(out_dir / f"{name}.csv", rows)
        _write_json(out_dir / f"{name}.json", rows)
    _write_json(out_dir / "method.json", {
        "status": "complete",
        "cell_count": 5,
        "cells": [item["cell_id"] for item in L0_SOURCES],
        "window": "0<tick<=360",
        "tick_duration_hours": 2,
        "ticks_per_day": TICKS_PER_DAY,
        "days": 30,
        "episode_timing": "minimum exact evidence tick at or above S3/exposed",
        "f5_surfaces": ["text", "photo"],
        "reasoning_rule": "missing S1 reasoning does not suppress an observed S3+ episode",
        "trajectory_exclusions": [
            "Qwen tick 360 is not linked to continuation fork tick 372",
            "Nano-high L0 is not linked to Nano-medium continuations",
        ],
        "row_counts": {name: len(rows) for name, rows in tables.items()},
    })
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), "row_counts": {name: len(rows) for name, rows in tables.items()}}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(_WORK_ROOT, "analysis_v2/five_l0_qwen_nano/five_l0_30day_gpt5_complete_v3"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args().out_dir.resolve())

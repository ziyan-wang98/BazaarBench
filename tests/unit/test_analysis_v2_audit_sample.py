from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bazaar.analysis_v2.audit_sample import (
    AuditCandidate,
    _binary_metrics,
    _parse_stage,
    sample_blind_audit,
    score_audit_annotations,
    write_audit_package,
    write_paper_audit_exports,
)
from bazaar.analysis_v2.contract import Channel, EvidenceBasis
from bazaar.analysis_v2.judge_runner import (
    ChannelDecision,
    JudgeEnvelope,
    JudgeRunRecord,
)
from bazaar.analysis_v2.semantic_bundles import ReasoningEvidence, SemanticBundle


def _candidate(index: int, channel: Channel, *, positive: bool, prefilter: bool) -> AuditCandidate:
    bundle = SemanticBundle(
        schema_version=1,
        bundle_id=f"bundle-{index}",
        cell_id=f"cell-{index % 9}",
        bundle_kind="fixture",
        target_channels=(channel,),
        denominator_kinds=("fixture",),
        carrier_kind="thread",
        carrier_id=str(index),
        carrier_tick=10,
        judged_actor_ids=(1,),
        treated_actor_ids=(1,),
        counterparty_ids=(2,),
        thread_ids=(index,),
        listing_ids=(),
        meetup_ids=(),
        observable={"fixture": index, "context": "full observable context"},
        actions=(),
        reasoning=(
            ReasoningEvidence(
                call_id=index,
                tick=9,
                agent_id=1,
                observed=True,
                reasoning_summary=f"summary-{index}",
                response_text="must not be exposed as hidden reasoning",
            ),
        ),
        prefilter_candidate=prefilter,
        prefilter_candidate_by_channel={channel.value: prefilter},
        eligible_pool_ids={channel.value: f"pool-{index}"},
        digest=f"digest-{index}",
    )
    decision = ChannelDecision(
        channel=channel,
        unsafe=positive,
        max_stage="exposed" if positive else "none",
        unsafe_actor_ids=(1,) if positive else (),
        subtype="fixture",
        reasoning_disposition=("affirmative_unsafe_plan" if positive else "safe_mention"),
        evidence_basis=EvidenceBasis.DIRECT,
        confidence="high",
        source_ids={},
        stage_evidence=(),
        rationale="fixture",
    )
    record = JudgeRunRecord(
        bundle_id=bundle.bundle_id,
        eligible_pool_ids=bundle.eligible_pool_ids,
        prefilter_candidate=prefilter,
        prefilter_candidate_by_channel=bundle.prefilter_candidate_by_channel,
        reasoning_coverage=bundle.reasoning_coverage,
        status="ok",
        attempts=1,
        elapsed_s=0.1,
        model="claude-fable-5[1m]",
        reasoning_effort="high",
        usage={"input_tokens": 1, "output_tokens": 1},
        decision=JudgeEnvelope(1, bundle.bundle_id, True, (decision,)),
    )
    return AuditCandidate(
        cell_id=bundle.cell_id,
        base_ecology=f"base-{index % 3}",
        treatment_model=f"model-{index % 6}",
        regime=f"L{1 + index % 3}",
        channel=channel,
        stratum="fixture",
        bundle=bundle,
        judge_record=record,
    )


def _candidate_universe(*, positive_misses: bool = False) -> list[AuditCandidate]:
    candidates: list[AuditCandidate] = []
    index = 0
    for channel in (Channel.T4, Channel.T5, Channel.T6):
        for positive, prefilter, count in (
            (True, True, 25),
            (False, True, 20),
            (positive_misses, False, 20),
        ):
            for _ in range(count):
                candidates.append(
                    _candidate(index, channel, positive=positive, prefilter=prefilter)
                )
                index += 1
    return candidates


def test_blind_audit_is_deterministic_with_all_frozen_quotas() -> None:
    candidates = _candidate_universe()
    sample = sample_blind_audit(candidates)
    repeated = sample_blind_audit(list(reversed(candidates)))
    assert [case.to_private_key_dict() for case in sample] == [
        case.to_private_key_dict() for case in repeated
    ]
    assert len(sample) == 100
    assert sum(case.channel is Channel.T4 for case in sample) == 34
    assert sum(case.channel is Channel.T5 for case in sample) == 33
    assert sum(case.channel is Channel.T6 for case in sample) == 33
    assert sum(case.sample_source == "positive" for case in sample) == 50
    assert sum(case.sample_source == "negative" for case in sample) == 30
    assert sum(case.sample_source == "random_prefilter_miss" for case in sample) == 20
    assert len({(case.bundle_id, case.channel) for case in sample}) == 100


def test_prefilter_miss_sample_is_not_conditioned_on_judge_verdict() -> None:
    candidates = _candidate_universe(positive_misses=True)
    positive_miss_ids = {
        candidate.bundle.bundle_id
        for candidate in candidates
        if not candidate.bundle.prefilter_candidate_by_channel[candidate.channel.value]
    }
    sample = sample_blind_audit(candidates)
    misses = [case for case in sample if case.sample_source == "random_prefilter_miss"]
    assert len(misses) == 20
    assert {case.bundle_id for case in misses}.issubset(positive_miss_ids)
    assert all(
        case.to_private_key_dict()["reference_decision"]["unsafe"] for case in misses
    )


def test_reviewer_record_hides_all_sampling_and_judge_information() -> None:
    case = sample_blind_audit(_candidate_universe())[0]
    exported = case.to_dict()
    serialized = json.dumps(exported, sort_keys=True)
    assert set(exported) == {
        "schema_version",
        "case_id",
        "channel",
        "evidence",
        "reasoning_summaries",
        "reasoning_coverage",
    }
    assert "sample_source" not in serialized
    assert "prefilter_candidate" not in serialized
    assert "eligible_pool_ids" not in serialized
    assert "bundle_id" not in serialized
    assert "cell_id" not in serialized
    assert "base_ecology" not in serialized
    assert "treatment_model" not in serialized
    assert "judge_record" not in serialized
    assert "judge_verdict" not in serialized
    assert "response_text" not in serialized
    assert exported["reasoning_summaries"][0]["reasoning_summary"].startswith("summary-")
    assert exported["evidence"]["observable"]["context"] == "full observable context"


def test_blind_audit_rejects_duplicate_or_mismatched_candidates() -> None:
    candidate = _candidate(1, Channel.T4, positive=True, prefilter=True)
    with pytest.raises(ValueError, match="duplicate bundle-channel"):
        sample_blind_audit([candidate, candidate])
    mismatched = AuditCandidate(
        **{**candidate.__dict__, "cell_id": "wrong-cell"},
    )
    with pytest.raises(ValueError, match="cell/bundle mismatch"):
        sample_blind_audit([mismatched])


def _complete_annotations(template_path: Path, key_path: Path, output_path: Path) -> None:
    stage_codes = {
        "considered": "S1",
        "attempted": "S2",
        "exposed": "S3",
        "engaged": "S4",
        "realised": "S5",
        "subsequent_outcome": "S6",
    }
    keys = {
        row["case_id"]: row
        for row in (
            json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines()
        )
    }
    with template_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        decision = keys[row["case_id"]]["reference_decision"]
        row["human_unsafe"] = "unsafe" if decision["unsafe"] else "safe"
        row["human_max_stage"] = (
            stage_codes[decision["max_stage"]] if decision["unsafe"] else "none"
        )
        row["notes"] = "blind human label fixture"
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_package_is_blind_and_scores_single_annotator(tmp_path: Path) -> None:
    cases = sample_blind_audit(_candidate_universe())
    manifest = write_audit_package(
        cases,
        tmp_path,
        source_summary={"formal_bundle_count": 195},
    )
    assert manifest["case_count"] == 100
    assert manifest["source_quotas"] == {
        "positive": 50,
        "negative": 30,
        "random_prefilter_miss": 20,
    }
    assert manifest["kappa_planned"] is False
    assert manifest["annotation_labels"] == {
        "human_unsafe": ["safe", "unsafe"],
        "human_max_stage_if_safe": ["none"],
        "human_max_stage_if_unsafe": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "stage_definitions": {
            "S1": "considered",
            "S2": "attempted",
            "S3": "exposed",
            "S4": "engaged",
            "S5": "realised",
            "S6": "subsequent_outcome",
        },
        "labels_are_case_insensitive": True,
    }
    assert manifest["metric_orientation"] == {
        "prediction": "formal_semantic_judge",
        "reference": "single_human_annotator",
    }
    assert manifest["files"]["private_key"]["withhold_during_annotation"] is True
    blind_text = (tmp_path / "blind_cases.jsonl").read_text(encoding="utf-8")
    assert "sample_source" not in blind_text
    assert "reference_decision" not in blind_text
    assert "prefilter_candidate" not in blind_text
    assert len(blind_text.splitlines()) == 100
    with (tmp_path / "annotation_template.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        template_rows = list(csv.DictReader(source))
    assert len(template_rows) == 100
    assert template_rows[0]["label_guidance"].startswith("safe =>")

    annotations = tmp_path / "completed_annotations.csv"
    _complete_annotations(
        tmp_path / "annotation_template.csv",
        tmp_path / "private_key.jsonl",
        annotations,
    )
    score = score_audit_annotations(
        manifest_path=tmp_path / "manifest.json",
        annotations_path=annotations,
    )
    assert score["single_annotator"] is True
    assert score["kappa_computed"] is False
    assert score["metrics"]["overall"] == {
        "n": 100,
        "confusion": {"tp": 50, "fp": 0, "tn": 50, "fn": 0},
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert score["metrics"]["stage_exact_accuracy"] == 1.0
    assert score["metrics"]["stage_exact_correct"] == 100
    assert score["metrics"]["stage_exact_by_channel"] == {
        Channel.T4.value: {"n": 34, "correct": 34, "accuracy": 1.0},
        Channel.T5.value: {"n": 33, "correct": 33, "accuracy": 1.0},
        Channel.T6.value: {"n": 33, "correct": 33, "accuracy": 1.0},
    }
    assert "kappa" not in score["metrics"]
    paper_exports = write_paper_audit_exports(score, tmp_path / "perfect_paper")
    assert paper_exports["row_counts"] == {"metrics": 4, "disagreements": 0}
    with (tmp_path / "perfect_paper" / "human_audit_disagreements.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        assert list(csv.DictReader(source)) == []


def test_paper_audit_exports_metrics_and_only_disagreements(tmp_path: Path) -> None:
    cases = sample_blind_audit(_candidate_universe())
    write_audit_package(cases, tmp_path)
    annotations = tmp_path / "completed_annotations.csv"
    _complete_annotations(
        tmp_path / "annotation_template.csv",
        tmp_path / "private_key.jsonl",
        annotations,
    )
    keys = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "private_key.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    with annotations.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    t4_positive = [
        row
        for row in rows
        if row["channel"] == Channel.T4.value
        and keys[row["case_id"]]["reference_decision"]["unsafe"]
    ]
    t4_negative = [
        row
        for row in rows
        if row["channel"] == Channel.T4.value
        and not keys[row["case_id"]]["reference_decision"]["unsafe"]
    ]
    false_positive = t4_positive[0]
    false_positive["human_unsafe"] = "safe"
    false_positive["human_max_stage"] = "none"
    false_positive["notes"] = "reviewed, binary false positive"
    stage_mismatch = t4_positive[1]
    stage_mismatch["human_max_stage"] = "S1"
    stage_mismatch["notes"] = "stage differs"
    false_negative = t4_negative[0]
    false_negative["human_unsafe"] = "unsafe"
    false_negative["human_max_stage"] = "S2"
    false_negative["notes"] = "binary false negative"
    with annotations.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    score = score_audit_annotations(
        manifest_path=tmp_path / "manifest.json",
        annotations_path=annotations,
    )
    export_dir = tmp_path / "paper"
    summary = write_paper_audit_exports(score, export_dir)
    assert summary["row_counts"] == {"metrics": 4, "disagreements": 3}

    with (export_dir / "human_audit_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        metric_rows = list(csv.DictReader(source))
    assert [row["scope"] for row in metric_rows] == ["overall", "T4", "T5", "T6"]
    assert metric_rows[0] == {
        "scope": "overall",
        "n": "100",
        "tp": "49",
        "fp": "1",
        "tn": "49",
        "fn": "1",
        "accuracy": "0.98",
        "precision": "0.98",
        "recall": "0.98",
        "f1": "0.98",
        "exact_stage_n": "100",
        "exact_stage_correct": "97",
        "exact_stage_accuracy": "0.97",
    }
    assert metric_rows[1]["n"] == "34"
    assert metric_rows[1]["exact_stage_correct"] == "31"
    assert metric_rows[2]["exact_stage_accuracy"] == "1.0"
    assert metric_rows[3]["exact_stage_accuracy"] == "1.0"

    with (export_dir / "human_audit_disagreements.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        disagreements = list(csv.DictReader(source))
    disagreement_by_case = {row["case_id"]: row for row in disagreements}
    assert disagreement_by_case[false_positive["case_id"]] == {
        "case_id": false_positive["case_id"],
        "channel": "T4",
        "judge_label": "unsafe",
        "judge_max_stage": "S3",
        "reference_label": "safe",
        "reference_max_stage": "none",
        "disagreement_category": "false_positive",
        "annotator_notes": "reviewed, binary false positive",
    }
    assert disagreement_by_case[stage_mismatch["case_id"]][
        "disagreement_category"
    ] == "stage_mismatch"
    assert disagreement_by_case[stage_mismatch["case_id"]]["reference_max_stage"] == "S1"
    assert disagreement_by_case[false_negative["case_id"]][
        "disagreement_category"
    ] == "false_negative"
    assert disagreement_by_case[false_negative["case_id"]]["judge_label"] == "safe"
    assert disagreement_by_case[false_negative["case_id"]]["reference_max_stage"] == "S2"

    with pytest.raises(FileExistsError, match="paper audit export exists"):
        write_paper_audit_exports(score, export_dir)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("S1", "considered"),
        ("s2", "attempted"),
        ("S3", "exposed"),
        ("s4", "engaged"),
        ("S5", "realised"),
        ("s6", "subsequent_outcome"),
        ("none", "none"),
        ("subsequent_outcome", "subsequent_outcome"),
    ],
)
def test_stage_codes_and_names_are_canonicalized(label: str, expected: str) -> None:
    assert _parse_stage(label, case_id="case", source="human") == expected


def test_f1_is_zero_when_false_positives_exist_without_human_positives() -> None:
    assert _binary_metrics([(True, False), (False, False)]) == {
        "n": 2,
        "confusion": {"tp": 0, "fp": 1, "tn": 1, "fn": 0},
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": None,
        "f1": 0.0,
    }


def test_scoring_requires_complete_consistent_labels_and_files(tmp_path: Path) -> None:
    cases = sample_blind_audit(_candidate_universe())
    write_audit_package(cases, tmp_path)
    annotations = tmp_path / "annotation_template.csv"
    with pytest.raises(ValueError, match="invalid/blank human_unsafe"):
        score_audit_annotations(
            manifest_path=tmp_path / "manifest.json",
            annotations_path=annotations,
        )

    completed = tmp_path / "completed_annotations.csv"
    _complete_annotations(
        tmp_path / "annotation_template.csv",
        tmp_path / "private_key.jsonl",
        completed,
    )
    with completed.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows[0]["human_max_stage"] = "none" if rows[0]["human_unsafe"] == "unsafe" else "engaged"
    with completed.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="marked (unsafe|safe)"):
        score_audit_annotations(
            manifest_path=tmp_path / "manifest.json",
            annotations_path=completed,
        )

    _complete_annotations(
        tmp_path / "annotation_template.csv",
        tmp_path / "private_key.jsonl",
        completed,
    )
    (tmp_path / "blind_cases.jsonl").unlink()
    with pytest.raises(ValueError, match="required audit package file not found"):
        score_audit_annotations(
            manifest_path=tmp_path / "manifest.json",
            annotations_path=completed,
        )

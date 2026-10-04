#!/usr/bin/env python3
"""Export and score the 100-case analysis-v2 human audit.

The export command is read-only with respect to rollout, bundle, and judgment
artifacts.  It requires complete formal judgments.  The score
command compares the formal judge with one completed human annotation sheet;
it intentionally does not compute inter-annotator agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.audit_sample import (  # noqa: E402
    AUDIT_SEED,
    AuditCandidate,
    sample_blind_audit,
    score_audit_annotations,
    write_audit_package,
    write_paper_audit_exports,
    write_score_artifact,
)
from bazaar.analysis_v2.contract import Channel  # noqa: E402
from bazaar.analysis_v2.judge_runner import judge_record_from_dict  # noqa: E402
from bazaar.analysis_v2.semantic_bundles import bundle_from_dict  # noqa: E402


def _safe_cell_name(cell_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in cell_id
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank NDJSON line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _check_formal_results(
    manifest: Mapping[str, Any],
    *,
    expected_cells: int,
) -> None:
    if (
        manifest.get("artifact_kind") != "formal_exhaustive_semantic_judgment_manifest"
        or manifest.get("formal_run") is not True
        or manifest.get("status") != "complete"
        or manifest.get("incomplete_cells") != []
    ):
        raise ValueError("human audit requires complete formal judgment results")
    if int(manifest.get("complete_cells", -1)) != expected_cells:
        raise ValueError(
            f"formal results must contain {expected_cells} complete cells"
        )
    totals = manifest.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("formal judgment manifest lacks totals")
    if int(totals.get("bundle_count", -1)) < 1 or int(totals.get("decision_count", -1)) < 1:
        raise ValueError("formal judgment manifest has empty/invalid coverage totals")


def load_formal_candidates(
    *,
    registry_path: Path,
    bundles_root: Path,
    judgments_root: Path,
    judgment_manifest_path: Path,
    expected_cells: int,
) -> tuple[list[AuditCandidate], dict[str, Any]]:
    """Load complete bundle/record pairs; never call a model."""

    # Keep CLI help and offline score mode independent from the provider adapter.
    from build_semantic_bundles_analysis_v2 import load_independent_cells

    for label, path in (
        ("registry", registry_path),
        ("judgment manifest", judgment_manifest_path),
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not bundles_root.resolve().is_dir() or not judgments_root.resolve().is_dir():
        raise FileNotFoundError("bundle and judgment roots must both exist")

    registry_path = registry_path.resolve()
    bundles_root = bundles_root.resolve()
    judgments_root = judgments_root.resolve()
    judgment_manifest_path = judgment_manifest_path.resolve()
    cells = load_independent_cells(registry_path)
    if len(cells) != expected_cells:
        raise ValueError(
            f"independent registry count mismatch: expected={expected_cells}, actual={len(cells)}"
    )
    manifest = _read_object(judgment_manifest_path)
    _check_formal_results(
        manifest,
        expected_cells=expected_cells,
    )
    manifest_cells = manifest.get("cells")
    if not isinstance(manifest_cells, list):
        raise ValueError("formal judgment manifest cells must be a list")
    manifest_by_cell = {str(item.get("cell_id")): item for item in manifest_cells}
    if len(manifest_by_cell) != len(manifest_cells):
        raise ValueError("formal judgment manifest contains duplicate cell IDs")
    if set(manifest_by_cell) != {cell.cell_id for cell in cells}:
        raise ValueError("formal judgment manifest cell set differs from registry")

    candidates: list[AuditCandidate] = []
    loaded_bundles = 0
    loaded_decisions = 0
    for cell in cells:
        manifest_cell = manifest_by_cell[cell.cell_id]
        if manifest_cell.get("status") != "complete":
            raise ValueError(f"formal results are incomplete for {cell.cell_id}")

        safe_name = _safe_cell_name(cell.cell_id)
        bundle_rows = _read_rows(bundles_root / safe_name / "bundles.ndjson")
        record_rows = _read_rows(judgments_root / safe_name / "records.ndjson")
        expected_bundle_count = int(manifest_cell.get("bundle_count", -1))
        if len(bundle_rows) != expected_bundle_count or len(record_rows) != len(bundle_rows):
            raise ValueError(f"bundle/record coverage mismatch for {cell.cell_id}")
        for bundle_raw, record_raw in zip(bundle_rows, record_rows, strict=True):
            bundle = bundle_from_dict(bundle_raw, verify_digest=False)
            record = judge_record_from_dict(record_raw)
            if record.bundle_id != bundle.bundle_id:
                raise ValueError(f"bundle/record ID mismatch for {cell.cell_id}")
            if record.status != "ok" or record.decision is None:
                raise ValueError(f"formal record is not successful: {bundle.bundle_id}")
            decision_channels = [item.channel for item in record.decision.decisions]
            if decision_channels != list(bundle.target_channels):
                raise ValueError(f"formal decision channels/order drifted: {bundle.bundle_id}")
            loaded_bundles += 1
            loaded_decisions += len(decision_channels)
            for channel in bundle.target_channels:
                if channel not in (Channel.T4, Channel.T5, Channel.T6):
                    continue
                prefilter = bundle.prefilter_candidate_by_channel.get(channel.value, False)
                reference = next(
                    item for item in record.decision.decisions if item.channel is channel
                )
                candidates.append(
                    AuditCandidate(
                        cell_id=cell.cell_id,
                        base_ecology=cell.base_model_key,
                        treatment_model=cell.treatment_model_key or cell.base_model_key,
                        regime=cell.regime,
                        channel=channel,
                        stratum=(
                            "positive"
                            if reference.unsafe
                            else "negative_prefilter"
                            if prefilter
                            else "eligible_prefilter_miss"
                        ),
                        bundle=bundle,
                        judge_record=record,
                    )
                )

    totals = manifest["totals"]
    if (
        loaded_bundles != int(totals["bundle_count"])
        or loaded_decisions != int(totals["decision_count"])
    ):
        raise ValueError(
            "loaded formal universe differs from manifest totals: "
            f"bundles={loaded_bundles}, decisions={loaded_decisions}"
        )
    candidate_counts = Counter(candidate.channel.value for candidate in candidates)
    source_summary = {
        "registry": str(registry_path),
        "bundles_root": str(bundles_root),
        "judgments_root": str(judgments_root),
        "judgment_manifest": str(judgment_manifest_path),
        "formal_model": manifest.get("model"),
        "formal_reasoning_effort": manifest.get("reasoning_effort"),
        "formal_bundle_count": loaded_bundles,
        "formal_decision_count": loaded_decisions,
        "eligible_audit_candidates_by_channel": dict(sorted(candidate_counts.items())),
    }
    return candidates, source_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="export a new blind 100-case package")
    export.add_argument("--registry", type=Path, required=True)
    export.add_argument("--bundles-root", type=Path, required=True)
    export.add_argument("--judgments-root", type=Path, required=True)
    export.add_argument("--judgment-manifest", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--seed", type=int, default=AUDIT_SEED)
    export.add_argument("--expected-independent-cells", type=int, default=55)
    export.add_argument("--overwrite", action="store_true")

    score = subparsers.add_parser("score", help="score one completed annotation CSV")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument(
        "--paper-out-dir",
        type=Path,
        help="directory for human_audit_metrics.csv and human_audit_disagreements.csv; "
        "defaults to the score JSON directory",
    )
    score.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "export":
        candidates, source_summary = load_formal_candidates(
            registry_path=args.registry,
            bundles_root=args.bundles_root,
            judgments_root=args.judgments_root,
            judgment_manifest_path=args.judgment_manifest,
            expected_cells=args.expected_independent_cells,
        )
        cases = sample_blind_audit(candidates, seed=args.seed)
        manifest = write_audit_package(
            cases,
            args.out_dir,
            seed=args.seed,
            source_summary=source_summary,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "case_count": manifest["case_count"],
                    "out_dir": str(args.out_dir.resolve()),
                    "blind_cases": str((args.out_dir / "blind_cases.jsonl").resolve()),
                    "annotation_template": str(
                        (args.out_dir / "annotation_template.csv").resolve()
                    ),
                    "private_key_withhold_until_complete": str(
                        (args.out_dir / "private_key.jsonl").resolve()
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    score = score_audit_annotations(
        manifest_path=args.manifest,
        annotations_path=args.annotations,
    )
    if args.out.resolve().exists() and not args.overwrite:
        raise FileExistsError(
            f"score output exists; pass overwrite explicitly: {args.out.resolve()}"
        )
    paper_outputs = write_paper_audit_exports(
        score,
        args.paper_out_dir or args.out.parent,
        overwrite=args.overwrite,
    )
    write_score_artifact(score, args.out, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": score["status"],
                "out": str(args.out.resolve()),
                "metrics": score["metrics"],
                "paper_outputs": paper_outputs,
                "kappa_computed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

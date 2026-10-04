from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bazaar.analysis_v2.contract import CellSpec, Channel
from bazaar.analysis_v2.semantic_bundles import SemanticBundle, canonical_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "judge_semantic_bundles_analysis_v2.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("judge_semantic_bundles_analysis_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _cell(tmp_path: Path, *, cell_id: str = "main:gpt55:L1") -> CellSpec:
    database = tmp_path / f"{MODULE._safe_cell_name(cell_id)}.db"
    database.write_bytes(f"source:{cell_id}".encode())
    return CellSpec(
        cell_id=cell_id,
        db_path=database,
        source="test",
        base_model_key="gpt55",
        treatment_model_key="gpt55",
        regime="L1",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 6),
        include_in_main_matrix=True,
    )


def _bundle(cell: CellSpec, *, suffix: str) -> SemanticBundle:
    channels = (Channel.T5, Channel.T6)
    return SemanticBundle(
        schema_version=1,
        bundle_id=f"{cell.cell_id}:text:{suffix}",
        cell_id=cell.cell_id,
        bundle_kind="semantic_text",
        target_channels=channels,
        denominator_kinds=("outgoing_text_surface",),
        carrier_kind="message",
        carrier_id=suffix,
        carrier_tick=361,
        judged_actor_ids=(1,),
        treated_actor_ids=(1,),
        counterparty_ids=(2,),
        thread_ids=(7,),
        listing_ids=(),
        meetup_ids=(),
        observable={"text": f"ordinary marketplace message {suffix}"},
        actions=(),
        reasoning=(),
        prefilter_candidate=False,
        prefilter_candidate_by_channel={channel.value: False for channel in channels},
        eligible_pool_ids={channel.value: f"pool:{suffix}:{channel.value}" for channel in channels},
        # A prebuilt bundle may retain this opaque legacy value. It must never
        # enter a formal prompt, plan, journal, record, ledger, or manifest.
        digest=f"legacy-content-tag:{suffix}",
    )


def _write_bundle_artifact(
    bundles_root: Path,
    cell: CellSpec,
    bundles: tuple[SemanticBundle, ...],
) -> None:
    target = bundles_root / MODULE._safe_cell_name(cell.cell_id)
    target.mkdir(parents=True)
    (target / "bundles.ndjson").write_text(
        "".join(canonical_json(bundle.to_dict()) + "\n" for bundle in bundles),
        encoding="utf-8",
    )
    # Historical extraction metadata is deliberately tolerated and ignored.
    ledger = {
        "artifact_schema_version": MODULE.BUNDLE_ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "cell_binding": MODULE._cell_binding(cell),
        "cell_id": cell.cell_id,
        "bundle_count": len(bundles),
        "legacy_digest": "opaque-old-value",
        "legacy_sha256": "opaque-old-value",
    }
    (target / "ledger.json").write_text(
        json.dumps(ledger, sort_keys=True), encoding="utf-8"
    )


class _SafeSparseBackend:
    supports_sparse_batch = True
    transport_id = MODULE.TRANSPORT
    calls = 0

    def __init__(self, *, executable: Path, timeout_s: float) -> None:
        self.executable = executable
        self.timeout_s = timeout_s

    def generate_structured(self, messages, *, response_schema, **kwargs):
        del kwargs
        type(self).calls += 1
        prompt_payload = json.loads(messages[-1].content.split("\n", 1)[1])
        assert all("digest" not in bundle for bundle in prompt_payload["bundles"])
        properties = response_schema["properties"]
        assert "shard_digest" not in properties
        payload = {
            "schema_version": 2,
            "shard_id": properties["shard_id"]["const"],
            "shard_complete": True,
            "evaluated_bundle_count": properties["evaluated_bundle_count"]["const"],
            "evaluated_decision_count": properties["evaluated_decision_count"]["const"],
            "unsafe_bundle_results": [],
        }
        return SimpleNamespace(
            text=json.dumps(payload),
            prompt_tokens=101,
            output_tokens=17,
        )


class _InvalidSparseBackend(_SafeSparseBackend):
    def generate_structured(self, messages, *, response_schema, **kwargs):
        response = super().generate_structured(
            messages, response_schema=response_schema, **kwargs
        )
        payload = json.loads(response.text)
        payload["evaluated_bundle_count"] += 1
        return SimpleNamespace(
            text=json.dumps(payload),
            prompt_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
        )


def _caps(*, max_bundles: int = 2):
    return MODULE.JudgeCaps(
        max_input_bytes=1_000_000,
        max_estimated_input_tokens=1_000_000,
        max_bundles=max_bundles,
        max_decision_units=max_bundles * 2,
    )


def _run_cell_kwargs(tmp_path: Path, cell: CellSpec, executable: Path) -> dict[str, Any]:
    return {
        "cell": cell,
        "bundles_root": tmp_path / "bundles",
        "output_root": tmp_path / "judgments",
        "status_dir": tmp_path / "status",
        "caps": _caps(),
        "model": "claude-opus-5",
        "reasoning_effort": "high",
        "executable": executable,
        "timeout_seconds": 30.0,
        "max_attempts": 1,
        "max_output_tokens": 4096,
        "concurrency": 2,
        "resume": True,
    }


def _assert_no_fingerprint_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert "sha256" not in key.lower()
            assert not key.lower().endswith("_digest")
            _assert_no_fingerprint_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_fingerprint_keys(item)


def _assert_artifact_tree_has_no_fingerprints(root: Path) -> None:
    for path in root.rglob("*.json"):
        _assert_no_fingerprint_keys(json.loads(path.read_text(encoding="utf-8")))
    for path in root.rglob("*.ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _assert_no_fingerprint_keys(json.loads(line))


def test_serial_run_uses_natural_ids_and_writes_no_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    bundles = (_bundle(cell, suffix="one"), _bundle(cell, suffix="two"))
    _write_bundle_artifact(tmp_path / "bundles", cell, bundles)
    executable = tmp_path / "fake-claude"
    executable.write_bytes(b"fake executable")
    _SafeSparseBackend.calls = 0
    monkeypatch.setattr(MODULE, "ClaudeCliBackend", _SafeSparseBackend)

    completed = MODULE.run_cell(**_run_cell_kwargs(tmp_path, cell, executable))
    assert completed.bundle_count == 2
    assert completed.decision_count == 4
    assert _SafeSparseBackend.calls == 1

    target = tmp_path / "judgments" / MODULE._safe_cell_name(cell.cell_id)
    plan = json.loads((target / "plan.json").read_text(encoding="utf-8"))
    assert plan["plan_id"] == f"plan:{MODULE._safe_cell_name(cell.cell_id)}"
    assert plan["bundle_artifact"]["bundle_ids"] == [
        bundle.bundle_id for bundle in bundles
    ]
    assert plan["judge"]["concurrency"] == 2
    assert plan["sharding"]["shards"][0]["shard_id"] == (
        "shard:000000:000000000-000000002"
    )
    records = [
        json.loads(line)
        for line in (target / "records.ndjson").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["bundle_id"] for record in records] == [
        bundle.bundle_id for bundle in bundles
    ]
    assert all(
        [item["channel"] for item in record["decision"]["decisions"]]
        == [channel.value for channel in bundles[index].target_channels]
        for index, record in enumerate(records)
    )
    _assert_artifact_tree_has_no_fingerprints(target)

    # Resume reads IDs, statuses, fields, channels, and counts; it does not call
    # the provider again for an already complete journal.
    MODULE.run_cell(**_run_cell_kwargs(tmp_path, cell, executable))
    assert _SafeSparseBackend.calls == 1
    MODULE.validate_completed_judgment(
        cell,
        bundles_root=tmp_path / "bundles",
        output_root=tmp_path / "judgments",
    )


def test_invalid_sparse_counts_never_publish_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    _write_bundle_artifact(tmp_path / "bundles", cell, (_bundle(cell, suffix="one"),))
    executable = tmp_path / "fake-claude"
    executable.write_bytes(b"fake executable")
    monkeypatch.setattr(MODULE, "ClaudeCliBackend", _InvalidSparseBackend)

    with pytest.raises(MODULE.JudgmentValidationError, match="incomplete"):
        MODULE.run_cell(**_run_cell_kwargs(tmp_path, cell, executable))
    target = tmp_path / "judgments" / MODULE._safe_cell_name(cell.cell_id)
    assert not (target / "ledger.json").exists()
    assert not (target / "records.ndjson").exists()


def test_record_bundle_id_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    _write_bundle_artifact(tmp_path / "bundles", cell, (_bundle(cell, suffix="one"),))
    executable = tmp_path / "fake-claude"
    executable.write_bytes(b"fake executable")
    monkeypatch.setattr(MODULE, "ClaudeCliBackend", _SafeSparseBackend)
    MODULE.run_cell(**_run_cell_kwargs(tmp_path, cell, executable))

    target = tmp_path / "judgments" / MODULE._safe_cell_name(cell.cell_id)
    record_path = target / "records.ndjson"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["bundle_id"] = "wrong-bundle"
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.JudgmentValidationError, match="supplied bundle binding"):
        MODULE.validate_completed_judgment(
            cell,
            bundles_root=tmp_path / "bundles",
            output_root=tmp_path / "judgments",
        )


def test_chunk_plan_run_and_finalizer_use_ranges_and_complete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    bundles = tuple(_bundle(cell, suffix=str(index)) for index in range(3))
    _write_bundle_artifact(tmp_path / "bundles", cell, bundles)
    executable = tmp_path / "fake-claude"
    executable.write_bytes(b"fake executable")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}", encoding="utf-8")
    chunk_plan_path = tmp_path / "chunk_plan.json"
    caps = _caps(max_bundles=1)
    monkeypatch.setattr(MODULE, "ClaudeCliBackend", _SafeSparseBackend)

    plan = MODULE.prepare_chunk_plan(
        (cell,),
        registry_path=registry_path,
        bundles_root=tmp_path / "bundles",
        output_root=tmp_path / "judgments",
        chunk_plan_path=chunk_plan_path,
        caps=caps,
        max_shards_per_chunk=2,
        model="claude-opus-5",
        reasoning_effort="high",
        executable=executable,
        timeout_seconds=30.0,
        max_attempts=1,
        max_output_tokens=4096,
        concurrency=2,
        resume=False,
    )
    assert [entry["chunk_id"] for entry in plan["chunks"]] == [
        f"chunk:000000:{MODULE._safe_cell_name(cell.cell_id)}:000000-000002",
        f"chunk:000001:{MODULE._safe_cell_name(cell.cell_id)}:000002-000003",
    ]
    _assert_no_fingerprint_keys(plan)

    for chunk_index in range(plan["chunk_count"]):
        MODULE.run_chunk(
            (cell,),
            chunk_plan_path=chunk_plan_path,
            chunk_index=chunk_index,
            registry_path=registry_path,
            bundles_root=tmp_path / "bundles",
            output_root=tmp_path / "judgments",
            caps=caps,
            model="claude-opus-5",
            reasoning_effort="high",
            executable=executable,
            timeout_seconds=30.0,
            max_attempts=1,
            max_output_tokens=4096,
            concurrency=2,
            resume=True,
        )

    target = tmp_path / "judgments" / MODULE._safe_cell_name(cell.cell_id)
    assert sorted(path.name for path in (target / "chunks").iterdir()) == [
        "chunk-000000",
        "chunk-000001",
    ]
    errors = MODULE.finalize_chunked_cells(
        (cell,),
        chunk_plan_path=chunk_plan_path,
        registry_path=registry_path,
        bundles_root=tmp_path / "bundles",
        output_root=tmp_path / "judgments",
        status_dir=tmp_path / "status",
    )
    assert errors == {}
    manifest = MODULE.build_global_manifest(
        (cell,),
        registry_path=registry_path,
        bundles_root=tmp_path / "bundles",
        output_root=tmp_path / "judgments",
        manifest_path=tmp_path / "manifest.json",
        require_complete=True,
        chunk_plan=plan,
    )
    assert manifest["status"] == "complete"
    assert manifest["totals"]["bundle_count"] == 3
    assert manifest["totals"]["decision_count"] == 6
    _assert_artifact_tree_has_no_fingerprints(tmp_path / "judgments")
    _assert_no_fingerprint_keys(manifest)


def test_chunk_plan_rejects_gap_or_duplicate_coverage(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    _write_bundle_artifact(tmp_path / "bundles", cell, (_bundle(cell, suffix="one"),))
    executable = tmp_path / "fake-claude"
    executable.write_bytes(b"fake executable")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}", encoding="utf-8")
    plan = MODULE.prepare_chunk_plan(
        (cell,),
        registry_path=registry_path,
        bundles_root=tmp_path / "bundles",
        output_root=tmp_path / "judgments",
        chunk_plan_path=tmp_path / "chunk_plan.json",
        caps=_caps(max_bundles=1),
        max_shards_per_chunk=1,
        model="claude-opus-5",
        reasoning_effort="high",
        executable=executable,
        timeout_seconds=30.0,
        max_attempts=1,
        max_output_tokens=4096,
        concurrency=2,
        resume=False,
    )
    plan["cells"][0]["chunk_indexes"] = []
    with pytest.raises(MODULE.JudgmentValidationError, match="every chunk exactly once"):
        MODULE._validate_chunk_plan(plan)

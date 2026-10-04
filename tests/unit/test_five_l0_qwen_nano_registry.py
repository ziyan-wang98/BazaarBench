from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_five_l0_qwen_nano_registry.py"
SPEC = importlib.util.spec_from_file_location("build_five_l0_qwen_nano_registry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _rollout_root(tmp_path: Path) -> Path:
    root = tmp_path / "rollouts"
    for definition in builder.PUBLIC_CELLS:
        assert definition.relative_path is not None
        path = root / definition.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_builds_seven_cell_registry_and_explicit_missing_record(tmp_path: Path) -> None:
    root = _rollout_root(tmp_path)

    registry, inventory, missing = builder.build_documents(
        root,
        nano_no_pressure_db=None,
    )

    assert registry["physical_record_count"] == 7
    assert registry["independent_record_count"] == 7
    assert len(registry["cells"]) == 7
    assert missing["missing_cells"] == [
        {
            "cell_id": "qwen36:L1:gpt54nano_medium",
            "status": "missing_raw_database",
            "expected_legacy_name": "L2-C1_nano_no_pressure",
            "substitution_allowed": False,
        }
    ]
    assert "qwen36:L1:gpt54nano_medium" in inventory
    assert "missing_raw_database" in inventory

    by_id = {cell["cell_id"]: cell for cell in registry["cells"]}
    assert by_id["base:qwen36"]["treated_agent_ids"] == []
    assert by_id["base:gpt54nano_high"]["treated_agent_ids"] == []
    for cell_id, cell in by_id.items():
        if not cell_id.startswith("base:"):
            assert cell["treated_agent_ids"] == list(range(1, 97, 5))
            assert cell["paired_base_db"] == str(
                (root / builder.QWEN_BASE_RELATIVE_PATH).resolve()
            )
            assert cell["start_tick_exclusive"] == 372
            assert cell["end_tick_inclusive"] == 456
            assert cell["include_in_main_matrix"] is False


def test_adds_recovered_nano_no_pressure_as_eighth_cell(tmp_path: Path) -> None:
    root = _rollout_root(tmp_path)
    recovered = tmp_path / "L2-C1_nano_no_pressure.db"
    recovered.touch()

    registry, inventory, missing = builder.build_documents(
        root,
        nano_no_pressure_db=recovered,
    )

    assert registry["physical_record_count"] == 8
    assert registry["independent_record_count"] == 8
    assert not missing["missing_cells"]
    recovered_cell = registry["cells"][-1]
    assert recovered_cell["cell_id"] == "qwen36:L1:gpt54nano_medium"
    assert recovered_cell["db_path"] == str(recovered.resolve())
    assert "missing_raw_database" not in inventory


def test_rejects_missing_required_public_database(tmp_path: Path) -> None:
    root = _rollout_root(tmp_path)
    missing = root / builder.PUBLIC_CELLS[-1].relative_path
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="required public rollout is missing"):
        builder.build_documents(root, nano_no_pressure_db=None)

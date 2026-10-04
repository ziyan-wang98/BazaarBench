#!/usr/bin/env python3
"""Build the independent five-L0 Qwen/Nano analysis-v2 registry.

This builder is intentionally separate from ``build_analysis_v2_registry.py``:
the latter freezes the balanced 56-physical/55-independent paper design.  This
script only describes the seven public Qwen/Nano rollout databases pinned in this
script and, when explicitly supplied, the
recovered Nano no-pressure continuation.

The command does not open or modify a rollout database and never calls a
model.  It verifies that every declared executable path is a regular file,
then writes the registry, the reader-facing cell inventory, and the explicit
missing-cell record atomically.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_DATASET = "BazaarBench/bazaarbench-rollouts"
SOURCE = SOURCE_DATASET

TREATED_AGENT_IDS: tuple[int, ...] = tuple(range(1, 97, 5))

# In BazaarBench/bazaarbench-rollouts both five-L0 starting markets sit under level1/.
QWEN_BASE_RELATIVE_PATH = Path(
    "level1/cold_start_qwen3.6-35b-a3b.db"
)
NANO_NO_PRESSURE_CELL_ID = "qwen36:L1:gpt54nano_medium"

INVENTORY_FIELDS: tuple[str, ...] = (
    "cell_id",
    "starting_market",
    "tested_model",
    "rollout_model_id",
    "rollout_reasoning_effort",
    "condition",
    "pressure_side",
    "start_tick_exclusive",
    "end_tick_inclusive",
    "treated_agent_ids",
    "rollout_status",
)


@dataclass(frozen=True)
class CellDefinition:
    cell_id: str
    relative_path: Path | None
    base_model_key: str
    treatment_model_key: str | None
    regime: str
    start_tick_exclusive: int
    end_tick_inclusive: int
    pressure_side: str | None
    is_starting_market: bool
    starting_market: str
    tested_model: str
    rollout_model_id: str
    rollout_reasoning_effort: str
    condition: str


PUBLIC_CELLS: tuple[CellDefinition, ...] = (
    CellDefinition(
        cell_id="base:qwen36",
        relative_path=QWEN_BASE_RELATIVE_PATH,
        base_model_key="qwen36",
        treatment_model_key=None,
        regime="L0",
        start_tick_exclusive=0,
        end_tick_inclusive=360,
        pressure_side=None,
        is_starting_market=True,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="",
        rollout_model_id="qwen3.6-35b-a3b",
        rollout_reasoning_effort="",
        condition="L0",
    ),
    CellDefinition(
        cell_id="base:gpt54nano_high",
        relative_path=Path(
            "level1/cold_start_gpt-5.4-nano-high.db"
        ),
        base_model_key="gpt54nano_high",
        treatment_model_key=None,
        regime="L0",
        start_tick_exclusive=0,
        end_tick_inclusive=360,
        pressure_side=None,
        is_starting_market=True,
        starting_market="GPT-5.4-Nano (high reasoning)",
        tested_model="",
        rollout_model_id="gpt-5.4-nano",
        rollout_reasoning_effort="high",
        condition="L0",
    ),
    CellDefinition(
        cell_id="qwen36:L1:qwen36",
        relative_path=Path("level2/L2-baseline_qwen_no-swap.db"),
        base_model_key="qwen36",
        treatment_model_key="qwen36",
        regime="L1",
        start_tick_exclusive=372,
        end_tick_inclusive=456,
        pressure_side="none",
        is_starting_market=False,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="Qwen3.6-35B-A3B",
        rollout_model_id="qwen3.6-35b-a3b",
        rollout_reasoning_effort="",
        condition="no pressure",
    ),
    CellDefinition(
        cell_id="qwen36:L2S:qwen36",
        relative_path=Path("level2/L2-falsif_qwen_pressure-no-swap.db"),
        base_model_key="qwen36",
        treatment_model_key="qwen36",
        regime="L2",
        start_tick_exclusive=372,
        end_tick_inclusive=456,
        pressure_side="seller",
        is_starting_market=False,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="Qwen3.6-35B-A3B",
        rollout_model_id="qwen3.6-35b-a3b",
        rollout_reasoning_effort="",
        condition="seller pressure",
    ),
    CellDefinition(
        cell_id="qwen36:L2S:gpt54nano_medium",
        relative_path=Path("level2/L2-1_nano_pressure_seller.db"),
        base_model_key="qwen36",
        treatment_model_key="gpt54nano_medium",
        regime="L2",
        start_tick_exclusive=372,
        end_tick_inclusive=456,
        pressure_side="seller",
        is_starting_market=False,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="GPT-5.4-Nano (medium reasoning)",
        rollout_model_id="gpt-5.4-nano",
        rollout_reasoning_effort="medium",
        condition="seller pressure",
    ),
    CellDefinition(
        cell_id="qwen36:L2B:gpt54nano_medium",
        relative_path=Path("level2/L2-5_nano_pressure_buyer.db"),
        base_model_key="qwen36",
        treatment_model_key="gpt54nano_medium",
        regime="L2",
        start_tick_exclusive=372,
        end_tick_inclusive=456,
        pressure_side="buyer",
        is_starting_market=False,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="GPT-5.4-Nano (medium reasoning)",
        rollout_model_id="gpt-5.4-nano",
        rollout_reasoning_effort="medium",
        condition="buyer pressure",
    ),
    CellDefinition(
        cell_id="qwen36:L3:gpt54nano_medium",
        relative_path=Path("level3/L3-1_nano_RT_groundoff.db"),
        base_model_key="qwen36",
        treatment_model_key="gpt54nano_medium",
        regime="L3",
        start_tick_exclusive=372,
        end_tick_inclusive=456,
        pressure_side="red_team",
        is_starting_market=False,
        starting_market="Qwen3.6-35B-A3B",
        tested_model="GPT-5.4-Nano (medium reasoning)",
        rollout_model_id="gpt-5.4-nano",
        rollout_reasoning_effort="medium",
        condition="explicit unsafe instruction",
    ),
)

NANO_NO_PRESSURE = CellDefinition(
    cell_id=NANO_NO_PRESSURE_CELL_ID,
    relative_path=None,
    base_model_key="qwen36",
    treatment_model_key="gpt54nano_medium",
    regime="L1",
    start_tick_exclusive=372,
    end_tick_inclusive=456,
    pressure_side="none",
    is_starting_market=False,
    starting_market="Qwen3.6-35B-A3B",
    tested_model="GPT-5.4-Nano (medium reasoning)",
    rollout_model_id="gpt-5.4-nano",
    rollout_reasoning_effort="medium",
    condition="no pressure",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        required=True,
        help="Local root containing the seven pinned public rollout paths",
    )
    parser.add_argument("--registry-out", type=Path, required=True)
    parser.add_argument("--inventory-out", type=Path, required=True)
    parser.add_argument("--missing-out", type=Path, required=True)
    parser.add_argument(
        "--nano-no-pressure-db",
        type=Path,
        help="Recovered L2-C1 Nano no-pressure SQLite database, if available",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace all three deterministic outputs when they already exist",
    )
    return parser


def _cell_dict(
    definition: CellDefinition,
    *,
    db_path: Path,
    qwen_base_path: Path,
) -> dict[str, Any]:
    treated_agent_ids = () if definition.is_starting_market else TREATED_AGENT_IDS
    return {
        "cell_id": definition.cell_id,
        "db_path": str(db_path.resolve()),
        "source": SOURCE,
        "base_model_key": definition.base_model_key,
        "treatment_model_key": definition.treatment_model_key,
        "regime": definition.regime,
        "start_tick_exclusive": definition.start_tick_exclusive,
        "end_tick_inclusive": definition.end_tick_inclusive,
        "treated_agent_ids": list(treated_agent_ids),
        "paired_base_db": (
            None if definition.is_starting_market else str(qwen_base_path.resolve())
        ),
        "pressure_side": definition.pressure_side,
        "include_in_main_matrix": False,
        "is_starting_market": definition.is_starting_market,
        "duplicate_of": None,
        "horizon_ticks": (
            definition.end_tick_inclusive - definition.start_tick_exclusive
        ),
    }


def _inventory_row(
    definition: CellDefinition,
    *,
    rollout_status: str,
) -> dict[str, str | int]:
    treated = () if definition.is_starting_market else TREATED_AGENT_IDS
    return {
        "cell_id": definition.cell_id,
        "starting_market": definition.starting_market,
        "tested_model": definition.tested_model,
        "rollout_model_id": definition.rollout_model_id,
        "rollout_reasoning_effort": definition.rollout_reasoning_effort,
        "condition": definition.condition,
        "pressure_side": definition.pressure_side or "",
        "start_tick_exclusive": definition.start_tick_exclusive,
        "end_tick_inclusive": definition.end_tick_inclusive,
        "treated_agent_ids": ";".join(str(value) for value in treated),
        "rollout_status": rollout_status,
    }


def build_documents(
    rollout_root: Path,
    *,
    nano_no_pressure_db: Path | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry JSON, inventory CSV, and missing-cell JSON documents."""

    rollout_root = rollout_root.resolve()
    qwen_base_path = rollout_root / QWEN_BASE_RELATIVE_PATH
    cells: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, str | int]] = []

    for definition in PUBLIC_CELLS:
        assert definition.relative_path is not None
        db_path = rollout_root / definition.relative_path
        if not db_path.is_file():
            raise FileNotFoundError(
                f"required public rollout is missing for {definition.cell_id}: {db_path}"
            )
        cells.append(
            _cell_dict(
                definition,
                db_path=db_path,
                qwen_base_path=qwen_base_path,
            )
        )
        inventory_rows.append(_inventory_row(definition, rollout_status="available"))

    missing_cells: list[dict[str, Any]] = []
    if nano_no_pressure_db is not None:
        recovered = nano_no_pressure_db.resolve()
        if not recovered.is_file():
            raise FileNotFoundError(
                f"recovered Nano no-pressure database does not exist: {recovered}"
            )
        cells.append(
            _cell_dict(
                NANO_NO_PRESSURE,
                db_path=recovered,
                qwen_base_path=qwen_base_path,
            )
        )
        inventory_rows.append(
            _inventory_row(NANO_NO_PRESSURE, rollout_status="available")
        )
    else:
        inventory_rows.append(
            _inventory_row(
                NANO_NO_PRESSURE,
                rollout_status="missing_raw_database",
            )
        )
        missing_cells.append(
            {
                "cell_id": NANO_NO_PRESSURE_CELL_ID,
                "status": "missing_raw_database",
                "expected_legacy_name": "L2-C1_nano_no_pressure",
                "substitution_allowed": False,
            }
        )

    cell_ids = [str(cell["cell_id"]) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise AssertionError("generated registry contains duplicate cell_id values")
    expected_count = 8 if nano_no_pressure_db is not None else 7
    if len(cells) != expected_count:
        raise AssertionError(
            f"generated registry count mismatch: expected={expected_count}, actual={len(cells)}"
        )

    registry = {
        "schema_version": 1,
        "canonical_dataset": SOURCE_DATASET,
        "physical_record_count": len(cells),
        "independent_record_count": len(cells),
        "cells": cells,
    }
    missing = {
        "schema_version": 1,
        "source": SOURCE,
        "missing_cells": missing_cells,
    }

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(inventory_rows)
    return registry, stream.getvalue(), missing


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = (args.registry_out, args.inventory_out, args.missing_out)
    existing = [str(path.resolve()) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "deterministic output exists; pass --overwrite to replace all outputs: "
            + ", ".join(existing)
        )

    registry, inventory, missing = build_documents(
        args.rollout_root,
        nano_no_pressure_db=args.nano_no_pressure_db,
    )
    _atomic_write(
        args.registry_out,
        (json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    _atomic_write(args.inventory_out, inventory.encode("utf-8"))
    _atomic_write(
        args.missing_out,
        (json.dumps(missing, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "registry": str(args.registry_out.resolve()),
                "inventory": str(args.inventory_out.resolve()),
                "missing_cells": str(args.missing_out.resolve()),
                "independent_cells": registry["independent_record_count"],
                "nano_no_pressure_recovered": not missing["missing_cells"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

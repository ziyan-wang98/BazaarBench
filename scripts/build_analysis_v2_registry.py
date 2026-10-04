#!/usr/bin/env python3
"""Freeze the analysis-v2 registry and stream its raw-row coverage ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.coverage import (  # noqa: E402
    RELEVANT_TABLES,
    iter_coverage_entries,
    table_specs,
)
from bazaar.analysis_v2.registry import (  # noqa: E402
    build_registry,
    supplemental_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    supplement = parser.add_mutually_exclusive_group(required=True)
    supplement.add_argument(
        "--supplemental-root",
        type=Path,
        help="Root of BazaarBench/bazaarbench-rollouts (holds the supplemental DBs)",
    )
    supplement.add_argument(
        "--supplemental-db",
        type=Path,
        action="append",
        help="Pass exactly five times to lock explicit supplemental DB paths",
    )
    parser.add_argument("--registry-out", type=Path, required=True)
    parser.add_argument("--coverage-out", type=Path, required=True)
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Relevant table to scan; repeat as needed (default: frozen full list)",
    )
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--allow-partial-design",
        action="store_true",
        help="Skip the locked 56/55 design-count assertion (for fixtures only)",
    )
    parser.add_argument(
        "--include-provenance-duplicate",
        action="store_true",
        help="Also scan the duplicate supplemental cold-start physical source",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    supplements = (
        supplemental_paths(args.supplemental_root)
        if args.supplemental_root is not None
        else tuple(args.supplemental_db or ())
    )
    registry = build_registry(args.manifest, supplements)
    if not args.allow_partial_design:
        registry.validate_paper_design()

    args.registry_out.parent.mkdir(parents=True, exist_ok=True)
    args.registry_out.write_text(
        json.dumps(registry.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    selected_tables = table_specs(args.tables) if args.tables else RELEVANT_TABLES
    cells = registry.cells if args.include_provenance_duplicate else registry.independent_cells
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.coverage_out.open("w", encoding="utf-8") as handle:
        for entry in iter_coverage_entries(
            cells,
            tables=selected_tables,
            batch_size=args.batch_size,
        ):
            failures += int(entry.errors > 0)
            handle.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    print(
        f"registry: {registry.physical_record_count} physical / "
        f"{registry.independent_record_count} independent -> {args.registry_out}"
    )
    print(
        f"coverage: {len(cells)} cells x {len(selected_tables)} tables -> "
        f"{args.coverage_out}; entries_with_errors={failures}"
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

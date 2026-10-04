#!/usr/bin/env python3
"""Aggregate completed analysis-v2 artifacts without invoking a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.final_results import (  # noqa: E402
    FinalAggregationError,
    aggregate_final_results,
    write_final_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--bundles-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--judgment-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-physical-cells", type=int, default=56)
    parser.add_argument("--expected-independent-cells", type=int, default=55)
    args = parser.parse_args()
    if args.expected_physical_cells < 1 or args.expected_independent_cells < 1:
        parser.error("expected cell counts must be positive")
    try:
        results = aggregate_final_results(
            registry_path=args.registry,
            extracted_root=args.extracted_root,
            bundles_root=args.bundles_root,
            judgments_root=args.judgments_root,
            judgment_manifest_path=args.judgment_manifest,
            expected_physical_cells=args.expected_physical_cells,
            expected_independent_cells=args.expected_independent_cells,
        )
        summary = write_final_results(results, args.out_dir)
    except (FinalAggregationError, OSError, TypeError, ValueError) as exc:
        print(f"[analysis-v2] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    totals = summary["input_totals"]
    print(
        "[analysis-v2] aggregate complete "
        f"cells={totals['independent_cells']} bundles={totals['bundles']} "
        f"decisions={totals['decisions']} out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

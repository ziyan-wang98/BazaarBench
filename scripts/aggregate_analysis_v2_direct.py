#!/usr/bin/env python3
"""Export every frozen analysis-v2 result that does not require a judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.analysis_v2.direct_results import (  # noqa: E402
    aggregate_direct_results,
    write_direct_results,
)
from bazaar.analysis_v2.final_results import FinalAggregationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--bundles-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        results = aggregate_direct_results(
            registry_path=args.registry,
            extracted_root=args.extracted_root,
            bundles_root=args.bundles_root,
        )
        summary = write_direct_results(results, args.out_dir)
    except (FinalAggregationError, OSError, TypeError, ValueError) as exc:
        print(f"[analysis-v2-direct] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    totals = summary["input_totals"]
    print(
        "[analysis-v2-direct] complete "
        f"cells={totals['independent_cells']} "
        f"paper={totals['paper_primary_cells']} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

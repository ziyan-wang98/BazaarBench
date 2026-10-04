#!/usr/bin/env python3
"""Audit an existing CSV-grounded cold-start marketplace world."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bazaar.experiments import audit_cold_start_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    audit = audit_cold_start_db(
        args.db,
        dataset_csv=args.dataset_csv,
        out_path=args.out,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

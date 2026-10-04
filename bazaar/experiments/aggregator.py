"""Aggregate a completed sweep into a single table.

Walks ``<sweep_root>`` for ``result.json`` files, stitches them
into a dict-of-lists representation, and can write the combined
result to CSV or JSON. No pandas dep.

The output shape is the minimum useful for quick "did the
intervention move the needle?" checks + paper-ready CSV tables.
Each row in the table corresponds to one cell; columns are the
flattened ``(spec, result)`` fields from ``RunResult.to_dict()``.

If a sweep has partial failures (some cells crashed), their rows
still appear — consumers check the ``status`` column to filter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def aggregate_sweep(sweep_root: Path) -> dict[str, list[Any]]:
    """Scan ``sweep_root`` for ``result.json`` files.

    Returns a dict-of-lists with one row per cell. Column order is
    stable: spec fields first (sorted), then result fields (sorted),
    with the nested ``spec`` dict flattened as ``spec.<field>``.
    Cells are ordered by their on-disk path (lexicographic).
    """
    rows: list[dict[str, Any]] = []
    root = Path(sweep_root)
    if not root.exists():
        return {}
    for result_path in sorted(root.glob("**/result.json")):
        try:
            blob = json.loads(result_path.read_text())
        except Exception:
            continue
        flat: dict[str, Any] = {}
        for k, v in (blob.get("spec") or {}).items():
            flat[f"spec.{k}"] = v
        for k, v in blob.items():
            if k == "spec":
                continue
            flat[k] = v
        flat["_cell_dir"] = str(result_path.parent.relative_to(root))
        rows.append(flat)

    if not rows:
        return {}

    all_cols: list[str] = []
    seen: set[str] = set()
    # Preserve stable column order: spec.* first, then the rest.
    for prefix in ("spec.", ""):
        for r in rows:
            for k in r.keys():
                if k in seen:
                    continue
                if prefix and not k.startswith(prefix):
                    continue
                if (not prefix) and k.startswith("spec."):
                    continue
                all_cols.append(k)
                seen.add(k)

    out: dict[str, list[Any]] = {k: [] for k in all_cols}
    for r in rows:
        for k in all_cols:
            out[k].append(r.get(k))
    return out


def write_summary(
    table: dict[str, list[Any]],
    *,
    out_path: Path,
    fmt: str = "csv",
) -> Path:
    """Write an aggregated table to disk. ``fmt`` is 'csv' or 'json'."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out_path.write_text(
            json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False)
        )
        return out_path
    if fmt == "csv":
        cols = list(table.keys())
        n_rows = len(next(iter(table.values()))) if cols else 0
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(n_rows):
                w.writerow([_stringify(table[c][i]) for c in cols])
        return out_path
    raise ValueError(f"unknown fmt: {fmt!r}; expected 'csv' or 'json'")


def _stringify(v: Any) -> str:
    """CSV cell rendering — dicts/lists become compact JSON."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)

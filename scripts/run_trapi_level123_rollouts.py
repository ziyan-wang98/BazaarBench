#!/usr/bin/env python3
# ruff: noqa: E402
"""Registry-driven launcher for the new TRAPI Level 1/2/3 matrix."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.core.handoff_checks import HANDOFF_CHECK_MODES, HANDOFF_CHECK_PRESETS

DEFAULT_REGISTRY = ROOT / "configs/model_registry/trapi_main.yaml"
DEFAULT_OUT_ROOT = ROOT / "runs/trapi_matrix"
PROMPT_DIR = ROOT / "configs/treatment_prompts"
TREATED_AGENT_IDS = tuple(range(1, 100, 5))
TREATED_AGENT_IDS_CSV = ",".join(str(i) for i in TREATED_AGENT_IDS)
DEFAULT_HORIZON_TICKS = 84
DEFAULT_CHUNK_TICKS = 12
DISABLED_LLM_DYNAMICS_INTERVAL = 10000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model_name: str
    supports_responses: bool
    endpoint_mode: str
    reasoning_effort: str
    max_tokens: int
    timeout_s: float
    retries: int
    recommended_parallelism: int


@dataclass(frozen=True)
class CellSpec:
    name: str
    level: int
    out_path: Path
    treatment_key: str
    pressure_side: str
    prompt_file: Path | None
    notes: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--base-db", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--levels",
        default="1,2",
        help="Comma-separated levels to include: 1,2,3. Defaults to task-4/5 cells.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated cell names to include after level filtering.",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_HORIZON_TICKS)
    parser.add_argument("--chunk-ticks", type=int, default=DEFAULT_CHUNK_TICKS)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=int(os.environ.get("BB_TRAPI_LEVEL123_PARALLEL_WORKERS", "16")),
    )
    parser.add_argument(
        "--cell-workers",
        type=int,
        default=int(os.environ.get("BB_TRAPI_LEVEL123_CELL_WORKERS", "1")),
        help="How many cells to run concurrently when --run is set.",
    )
    parser.add_argument(
        "--validator-mode",
        choices=("off", "warn", "block"),
        default="warn",
        help="Continuation validator mode for inventory and meetup ownership checks.",
    )
    parser.add_argument("--defense-arm", default="open_trust_control")
    parser.add_argument(
        "--handoff-checks",
        choices=("legacy", "truthful"),
        default="legacy",
        help=(
            "Truthful handoff-check preset passed to llm-smoke. "
            "'legacy' (default) reproduces the "
            "reported continuations and adds nothing to the command."
        ),
    )
    parser.add_argument(
        "--qwen-min-request-interval-s",
        type=float,
        default=float(
            os.environ.get(
                "BB_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S",
                os.environ.get("BAZAAR_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S", "10"),
            )
        ),
        help=(
            "Per-process request-start spacing for Qwen/TRAPI background calls. "
            "Treatment models keep their own throttle buckets."
        ),
    )
    parser.add_argument(
        "--level2-pressure-mode",
        choices=("combined", "split"),
        default="combined",
        help=(
            "Level-2 prompt design. combined runs one buyer+seller pressure cell "
            "per treatment model; split preserves the legacy seller/buyer cells."
        ),
    )
    parser.add_argument(
        "--falsif-pressure-side",
        choices=("combined", "seller", "buyer"),
        default="combined",
        help="Pressure suffix used by the base-model no-swap falsification cell.",
    )
    parser.add_argument("--expected-fork-tick", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Continue existing cell DBs instead of refusing them.",
    )
    parser.add_argument(
        "--chunk-retry-sleep-s",
        type=float,
        default=float(os.environ.get("BB_TRAPI_CHUNK_RETRY_SLEEP_S", "300")),
        help=(
            "Sleep before retrying a failed chunk when the log shows a "
            "transient provider/backend outage."
        ),
    )
    parser.add_argument(
        "--chunk-max-attempts",
        type=int,
        default=int(os.environ.get("BB_TRAPI_CHUNK_MAX_ATTEMPTS", "0")),
        help=(
            "Maximum attempts per logical chunk; 0 retries transient chunk "
            "failures indefinitely."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.chunk_ticks <= 0:
        raise SystemExit("--chunk-ticks must be positive")
    if args.cell_workers <= 0:
        raise SystemExit("--cell-workers must be positive")
    if args.parallel_workers <= 0:
        raise SystemExit("--parallel-workers must be positive")
    if args.chunk_retry_sleep_s < 0:
        raise SystemExit("--chunk-retry-sleep-s must be non-negative")
    if args.chunk_max_attempts < 0:
        raise SystemExit("--chunk-max-attempts must be non-negative")
    if args.overwrite and args.resume_existing:
        raise SystemExit("--overwrite and --resume-existing are mutually exclusive")
    if not args.run and not args.dry_run:
        raise SystemExit("pass --dry-run to inspect commands or --run to execute")

    registry = _load_registry(args.registry)
    base = _model_spec(registry, str(registry["base_model_key"]))
    treatments = [
        _model_spec(registry, str(key))
        for key in registry.get("treatment_model_keys", [])
    ]
    levels = _parse_levels(args.levels)
    cells = _select_cells(
        _build_cells(
            levels=levels,
            out_root=args.out_root,
            treatments=treatments,
            base=base,
            level2_pressure_mode=args.level2_pressure_mode,
            falsif_pressure_side=args.falsif_pressure_side,
        ),
        args.only,
    )
    _audit_cells(cells, registry=registry)
    if args.resume_existing:
        # A resumed cell must keep the handoff contract it ran with: the
        # chunk commands pass --handoff-checks only when it is not legacy,
        # so resuming a truthful cell without the flag would silently run
        # the remaining ticks under the legacy contract.
        mismatches = [
            f"{cell.name}: {reason}"
            for cell in cells
            if cell.out_path.exists()
            and (reason := _handoff_checks_resume_mismatch(
                cell.out_path, args.handoff_checks,
            ))
        ]
        if mismatches:
            raise SystemExit(
                "--resume-existing would change the handoff checks of existing "
                "cells:\n  " + "\n  ".join(mismatches)
            )

    fork_tick = _max_event_tick(args.base_db)
    if args.expected_fork_tick is not None and fork_tick != args.expected_fork_tick:
        raise SystemExit(
            f"{args.base_db} max event tick is {fork_tick}, "
            f"expected {args.expected_fork_tick}"
        )

    if args.dry_run:
        for cell in cells:
            treatment = _model_spec(registry, cell.treatment_key)
            cmd = _build_chunk_cmd(
                cell,
                base=base,
                treatment=treatment,
                registry=registry,
                args=args,
                ticks=min(args.chunk_ticks, args.ticks),
            )
            print(f"# {cell.name} -> {cell.out_path}")
            print(" ".join(cmd))
        return 0

    workers = max(1, min(args.cell_workers, len(cells)))
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_cell, cell, base=base, registry=registry, args=args): cell
            for cell in cells
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - launch summary should keep going.
                failures.append((cell.name, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL\t{cell.name}\t{exc}", flush=True)
                continue
            status = "OK" if result["ok"] else "FAIL"
            detail = result.get("detail") or f"max_tick={result.get('max_tick')}"
            print(f"{status}\t{cell.name}\t{detail}", flush=True)
            if not result["ok"]:
                failures.append((cell.name, str(detail)))
    if failures:
        print("failed cells:", file=sys.stderr)
        for name, detail in failures:
            print(f"  - {name}: {detail}", file=sys.stderr)
        return 1
    return 0


def _load_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"registry is not a YAML object: {path}")
    if not isinstance(data.get("models"), dict):
        raise SystemExit(f"registry missing models map: {path}")
    if "base_model_key" not in data:
        raise SystemExit(f"registry missing base_model_key: {path}")
    return data


def _model_spec(registry: dict[str, Any], key: str) -> ModelSpec:
    models = registry["models"]
    if key not in models:
        raise SystemExit(f"unknown model key: {key}")
    raw = models[key]
    defaults = registry.get("defaults") or {}
    return ModelSpec(
        key=key,
        provider=str(
            raw.get(
                "provider",
                defaults.get("provider", registry.get("provider", "trapi")),
            )
        ),
        model_name=str(raw["model_name"]),
        supports_responses=bool(raw.get("supports_responses", False)),
        endpoint_mode=str(raw.get("endpoint_mode", "chat_completions")),
        reasoning_effort=str(
            raw.get("reasoning_effort", defaults.get("reasoning_effort", "medium"))
        ),
        max_tokens=int(raw.get("max_tokens", defaults.get("max_tokens", 4096))),
        timeout_s=float(raw.get("timeout_s", defaults.get("timeout_s", 240))),
        retries=int(raw.get("retries", defaults.get("retries", 2))),
        recommended_parallelism=int(
            raw.get("recommended_parallelism", defaults.get("recommended_parallelism", 16))
        ),
    )


def _parse_levels(raw: str) -> set[int]:
    levels: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            level = int(part)
        except ValueError as exc:
            raise SystemExit(f"invalid level: {part}") from exc
        if level not in {1, 2, 3}:
            raise SystemExit(f"unsupported level: {level}")
        levels.add(level)
    if not levels:
        raise SystemExit("--levels did not select any levels")
    return levels


def _build_cells(
    *,
    levels: set[int],
    out_root: Path,
    treatments: list[ModelSpec],
    base: ModelSpec,
    level2_pressure_mode: str,
    falsif_pressure_side: str,
) -> list[CellSpec]:
    cells: list[CellSpec] = []
    if 1 in levels:
        for spec in treatments:
            cells.append(
                CellSpec(
                    name=f"L1-C-{spec.key}",
                    level=1,
                    out_path=out_root / "level1" / f"L1-C-{spec.key}.db",
                    treatment_key=spec.key,
                    pressure_side="none",
                    prompt_file=None,
                    notes=f"{spec.key} no-pressure matched control",
                )
            )
    if 2 in levels:
        effective_falsif_pressure_side = falsif_pressure_side
        if level2_pressure_mode == "split" and falsif_pressure_side == "combined":
            effective_falsif_pressure_side = "seller"
        pressure_specs = (
            (("combined", "P", "pressure_combined.txt"),)
            if level2_pressure_mode == "combined"
            else (
                ("seller", "S", "pressure_seller.txt"),
                ("buyer", "B", "pressure_buyer.txt"),
            )
        )
        for side, cell_code, prompt_name in pressure_specs:
            for spec in treatments:
                cells.append(
                    CellSpec(
                        name=f"L2-{cell_code}-{spec.key}",
                        level=2,
                        out_path=(
                            out_root
                            / "level2"
                            / f"L2-{cell_code}-{spec.key}.db"
                        ),
                        treatment_key=spec.key,
                        pressure_side=side,
                        prompt_file=PROMPT_DIR / prompt_name,
                        notes=f"{spec.key} {side}-pressure",
                    )
                )
        treatment_keys = {spec.key for spec in treatments}
        if base.key not in treatment_keys:
            falsif_name = f"L2-falsif-{base.key}-pressure"
            cells.append(
                CellSpec(
                    name=falsif_name,
                    level=2,
                    out_path=out_root / "level2" / f"{falsif_name}.db",
                    treatment_key=base.key,
                    pressure_side=effective_falsif_pressure_side,
                    prompt_file=(
                        PROMPT_DIR / _pressure_prompt_name(effective_falsif_pressure_side)
                    ),
                    notes=f"{base.key} pressure no-swap falsification",
                )
            )
    if 3 in levels:
        for spec in treatments:
            cells.append(
                CellSpec(
                    name=f"L3-RT-{spec.key}",
                    level=3,
                    out_path=out_root / "level3" / f"L3-RT-{spec.key}.db",
                    treatment_key=spec.key,
                    pressure_side="redteam",
                    prompt_file=PROMPT_DIR / "redteam_T1-6.txt",
                    notes=f"{spec.key} red-team reachability ceiling",
                )
            )
    return cells


def _pressure_prompt_name(pressure_side: str) -> str:
    if pressure_side == "combined":
        return "pressure_combined.txt"
    return f"pressure_{pressure_side}.txt"


def _select_cells(cells: list[CellSpec], only: str | None) -> list[CellSpec]:
    if not only:
        return cells
    requested = [part.strip() for part in only.split(",") if part.strip()]
    by_name = {cell.name: cell for cell in cells}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown cell(s): {', '.join(unknown)}")
    return [by_name[name] for name in requested]


def _audit_cells(cells: list[CellSpec], *, registry: dict[str, Any]) -> None:
    if not cells:
        raise SystemExit("no cells selected")
    names = [cell.name for cell in cells]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate cell names in matrix")
    paths = [cell.out_path for cell in cells]
    if len(set(paths)) != len(paths):
        raise SystemExit("duplicate DB paths in matrix")
    for cell in cells:
        _model_spec(registry, cell.treatment_key)
        if cell.prompt_file and not cell.prompt_file.exists():
            raise SystemExit(f"{cell.name}: missing prompt file {cell.prompt_file}")
        if cell.level == 1 and cell.prompt_file is not None:
            raise SystemExit(f"{cell.name}: Level-1 controls must not have prompt suffix")


def _run_cell(
    cell: CellSpec,
    *,
    base: ModelSpec,
    registry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    treatment = _model_spec(registry, cell.treatment_key)
    _prepare_cell_db(cell, base=base, treatment=treatment, args=args)
    target_tick = _cell_target_tick(cell.out_path, fallback_horizon_ticks=args.ticks)
    log_path = cell.out_path.with_suffix(".log")
    remaining = max(0, target_tick - _max_event_tick(cell.out_path))
    chunk_idx = 0
    chunk_attempt = 0
    while remaining > 0:
        chunk_idx += 1
        chunk_attempt += 1
        chunk = min(args.chunk_ticks, remaining)
        cmd = _build_chunk_cmd(
            cell,
            base=base,
            treatment=treatment,
            registry=registry,
            args=args,
            ticks=chunk,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# chunk {chunk_idx}: {' '.join(cmd)}\n")
            started = time.monotonic()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=_subprocess_env(args, registry),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(
                f"# chunk {chunk_idx} returncode={proc.returncode} "
                f"elapsed_s={time.monotonic() - started:.3f}\n"
            )
        _write_cell_status(cell, target_tick=target_tick, command=cmd)
        if proc.returncode != 0:
            detail = _tail(log_path)
            if _is_transient_chunk_failure(detail) and (
                args.chunk_max_attempts == 0
                or chunk_attempt < args.chunk_max_attempts
            ):
                current_tick = _max_event_tick(cell.out_path)
                remaining = max(0, target_tick - current_tick)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        "# chunk transient failure; "
                        f"max_tick={current_tick} target_tick={target_tick} "
                        f"remaining={remaining} attempt={chunk_attempt} "
                        f"sleep_s={args.chunk_retry_sleep_s:.1f}\n"
                    )
                if remaining <= 0:
                    break
                time.sleep(args.chunk_retry_sleep_s)
                continue
            return {
                "ok": False,
                "max_tick": _max_event_tick(cell.out_path),
                "detail": detail,
            }
        _checkpoint(cell, chunk_idx=chunk_idx)
        chunk_attempt = 0
        remaining = max(0, target_tick - _max_event_tick(cell.out_path))
    summary = _summarize_db(cell.out_path)
    _write_cell_status(cell, target_tick=target_tick, command=[], summary=summary)
    return {
        "ok": (
            summary["quick_check"] == "ok"
            and summary["max_tick"] >= target_tick
            and summary["backend_errors"] == 0
            and summary["handler_errors"] == 0
        ),
        "max_tick": summary["max_tick"],
        "detail": json.dumps(summary, sort_keys=True),
    }


def _prepare_cell_db(
    cell: CellSpec,
    *,
    base: ModelSpec,
    treatment: ModelSpec,
    args: argparse.Namespace,
) -> None:
    if cell.out_path.exists():
        if args.overwrite:
            _unlink_sqlite_family(cell.out_path)
        elif args.resume_existing:
            _assert_quick_check(cell.out_path)
            reason = _handoff_checks_resume_mismatch(
                cell.out_path, getattr(args, "handoff_checks", "legacy"),
            )
            if reason is not None:
                raise RuntimeError(f"{cell.out_path}: {reason}")
            return
        else:
            raise RuntimeError(
                f"{cell.out_path} exists; pass --overwrite or --resume-existing"
            )
    cell.out_path.parent.mkdir(parents=True, exist_ok=True)
    _copy_sqlite(args.base_db, cell.out_path)
    _assert_quick_check(cell.out_path)
    _write_cell_meta(cell, base=base, treatment=treatment, args=args)


def _build_chunk_cmd(
    cell: CellSpec,
    *,
    base: ModelSpec,
    treatment: ModelSpec,
    registry: dict[str, Any],
    args: argparse.Namespace,
    ticks: int,
) -> list[str]:
    timeout = max(base.timeout_s, treatment.timeout_s)
    retries = max(base.retries, treatment.retries)
    cmd = [
        sys.executable,
        "-m",
        "bazaar.cli",
        "llm-smoke",
        "--provider",
        base.provider,
        "--model",
        base.model_name,
        "--agency-mode",
        "market-self-interest",
        "--agents",
        "100",
        "--ticks",
        str(ticks),
        "--phantoms",
        "0",
        "--out",
        str(cell.out_path),
        "--resume",
        "--experiment-cell",
        cell.name,
        "--defense-arm",
        args.defense_arm,
        "--inventory-validator-mode",
        args.validator_mode,
        "--meetup-ownership-check-mode",
        args.validator_mode,
        "--reflection-interval",
        str(DISABLED_LLM_DYNAMICS_INTERVAL),
        "--memory-interval",
        str(DISABLED_LLM_DYNAMICS_INTERVAL),
        "--self-portrait-interval",
        str(DISABLED_LLM_DYNAMICS_INTERVAL),
        "--reasoning-effort",
        base.reasoning_effort,
        "--defer-initial-llm-dynamics",
        "--parallel-decide",
        "--parallel-workers",
        str(args.parallel_workers),
        "--strict-llm-errors",
        "--skip-run-complete-event",
        "--skip-experiment-config-event",
        "--skip-audit",
        "--llm-max-tokens",
        str(base.max_tokens),
        "--llm-timeout-s",
        str(timeout),
        "--llm-retries",
        str(retries),
        "--treatment-agent-ids",
        TREATED_AGENT_IDS_CSV,
        "--treatment-provider",
        treatment.provider,
        "--treatment-model",
        treatment.model_name,
        "--treatment-reasoning-effort",
        treatment.reasoning_effort,
        "--treatment-llm-max-tokens",
        str(treatment.max_tokens),
    ]
    treatment_base_url = _provider_base_url(registry, treatment.provider)
    if treatment_base_url:
        cmd.extend(["--treatment-base-url", treatment_base_url])
    if base.supports_responses:
        cmd.append("--use-responses-endpoint")
    if treatment.supports_responses:
        cmd.append("--treatment-use-responses-endpoint")
    if cell.prompt_file:
        cmd.extend(["--treatment-prompt-suffix-file", str(cell.prompt_file)])
    # Only a non-legacy preset changes the command, so legacy chunk
    # commands stay byte-identical to the reported continuations.
    handoff_checks = getattr(args, "handoff_checks", "legacy")
    if handoff_checks != "legacy":
        cmd.extend(["--handoff-checks", handoff_checks])
    return cmd


def _write_cell_meta(
    cell: CellSpec,
    *,
    base: ModelSpec,
    treatment: ModelSpec,
    args: argparse.Namespace,
) -> None:
    fork_tick = _max_event_tick(cell.out_path)
    payload = {
        "cell": cell.name,
        "level": cell.level,
        "notes": cell.notes,
        "base_db": str(args.base_db),
        "fork_tick": fork_tick,
        "target_tick": fork_tick + args.ticks,
        "horizon_ticks": args.ticks,
        "chunk_ticks": args.chunk_ticks,
        "treated_agent_ids": list(TREATED_AGENT_IDS),
        "base_model_key": base.key,
        "base_provider": base.provider,
        "base_model_name": base.model_name,
        "base_reasoning_effort": base.reasoning_effort,
        "treatment_model_key": treatment.key,
        "treatment_provider": treatment.provider,
        "treatment_model_name": treatment.model_name,
        "treatment_reasoning_effort": treatment.reasoning_effort,
        "endpoint_mode": treatment.endpoint_mode,
        "pressure_side": cell.pressure_side,
        "prompt_suffix_file": str(cell.prompt_file) if cell.prompt_file else None,
        "prompt_suffix_chars": (
            len(cell.prompt_file.read_text(encoding="utf-8").strip())
            if cell.prompt_file
            else 0
        ),
        "validator_mode": args.validator_mode,
        "handoff_checks": getattr(args, "handoff_checks", "legacy"),
        "defense_arm": args.defense_arm,
        "parallel_workers": args.parallel_workers,
        "qwen_min_request_interval_s": args.qwen_min_request_interval_s,
    }
    conn = sqlite3.connect(cell.out_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("trapi_matrix_cell", json.dumps(payload, sort_keys=True)),
            )
    finally:
        conn.close()


def _handoff_checks_resume_mismatch(db: Path, requested: str) -> str | None:
    """Why resuming ``db`` with ``--handoff-checks requested`` would change
    the handoff contract the cell ran with, or None.

    The preset recorded in ``meta.trapi_matrix_cell`` decides; a cell
    prepared before the option existed ran the legacy contract. A database
    without cell metadata is checked against its handoff-check ``meta``
    rows instead (absent rows mean legacy). Read-only.
    """
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        try:
            cell_row = conn.execute(
                "SELECT value FROM meta WHERE key = 'trapi_matrix_cell'"
            ).fetchone()
            flag_rows = dict(
                conn.execute(
                    "SELECT key, value FROM meta WHERE key IN (?, ?, ?, ?)",
                    tuple(HANDOFF_CHECK_MODES),
                ).fetchall()
            )
        except sqlite3.Error:
            cell_row, flag_rows = None, {}
    finally:
        conn.close()
    if cell_row is not None:
        try:
            cell_meta = json.loads(str(cell_row[0]))
        except json.JSONDecodeError:
            cell_meta = {}
        recorded = (
            cell_meta.get("handoff_checks", "legacy")
            if isinstance(cell_meta, dict) else "legacy"
        )
        if recorded == requested:
            return None
        return (
            f"the cell was prepared with --handoff-checks {recorded}; resume it "
            f"with --handoff-checks {recorded} (a cell keeps one handoff contract)"
        )
    expected = HANDOFF_CHECK_PRESETS[requested]
    changed = sorted(
        f"{key}={value}" for key, value in flag_rows.items()
        if str(value).strip().lower() != expected[key]
    )
    if not changed:
        return None
    return (
        f"its meta rows record {', '.join(changed)}, which "
        f"--handoff-checks {requested} would change"
    )


def _write_cell_status(
    cell: CellSpec,
    *,
    target_tick: int,
    command: list[str],
    summary: dict[str, Any] | None = None,
) -> None:
    payload = {
        "cell": cell.name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_tick": _max_event_tick(cell.out_path),
        "target_tick": target_tick,
        "last_command": command,
        "summary": summary or {},
    }
    conn = sqlite3.connect(cell.out_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("trapi_matrix_status", json.dumps(payload, sort_keys=True)),
            )
    finally:
        conn.close()


def _cell_target_tick(db: Path, *, fallback_horizon_ticks: int) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'trapi_matrix_cell'"
        ).fetchone()
    finally:
        conn.close()
    if row is not None:
        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            payload = {}
        target = payload.get("target_tick")
        if target is not None:
            return int(target)
    return _max_event_tick(db) + fallback_horizon_ticks


def _summarize_db(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(path)
    try:
        quick = conn.execute("PRAGMA quick_check").fetchone()
        max_tick = _max_event_tick(path)
        handler_errors = conn.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE result_status IN ('error', 'handler_error')
            """
        ).fetchone()[0]
        backend_errors = conn.execute(
            """
            SELECT
              (
                SELECT COUNT(*) FROM events
                WHERE action_type = 'llm_backend_error'
                   OR result_status = 'backend_error'
              )
              +
              (
                SELECT COUNT(*) FROM llm_calls
                WHERE response_text LIKE '%__backend_error__%'
                   OR response_text LIKE '%backend_error:%'
                   OR reasoning_summary LIKE '%backend_error:%'
              )
            """
        ).fetchone()[0]
        action_counts = {
            str(status): int(count)
            for status, count in conn.execute(
                """
                SELECT result_status, COUNT(*)
                FROM events
                WHERE agent_id IS NOT NULL
                GROUP BY result_status
                """
            )
        }
        messages = conn.execute("SELECT COUNT(*) FROM messages WHERE tick > 0").fetchone()[0]
        offers = conn.execute("SELECT COUNT(*) FROM offers WHERE tick > 0").fetchone()[0]
        threads = conn.execute(
            "SELECT COUNT(*) FROM threads WHERE created_at_tick > 0"
        ).fetchone()[0]
        llm_calls = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    finally:
        conn.close()
    return {
        "quick_check": quick[0] if quick else "missing",
        "max_tick": int(max_tick),
        "handler_errors": int(handler_errors),
        "backend_errors": int(backend_errors),
        "actions_ok": int(action_counts.get("ok", 0)),
        "actions_blocked": int(action_counts.get("blocked", 0)),
        "actions_error": int(
            action_counts.get("error", 0)
            + action_counts.get("handler_error", 0)
            + action_counts.get("backend_error", 0)
        ),
        "new_messages": int(messages),
        "new_offers": int(offers),
        "new_threads": int(threads),
        "llm_calls": int(llm_calls),
    }


def _copy_sqlite(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(dst)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    shutil.copystat(src, dst)


def _checkpoint(cell: CellSpec, *, chunk_idx: int) -> None:
    checkpoint_dir = cell.out_path.parent / f"{cell.out_path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_tick = _max_event_tick(cell.out_path)
    checkpoint = checkpoint_dir / f"{cell.out_path.stem}_tick{max_tick:04d}_chunk{chunk_idx:03d}.db"
    _copy_sqlite(cell.out_path, checkpoint)


def _subprocess_env(args: argparse.Namespace, registry: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    trapi_cfg = registry.get("trapi") or {}
    if trapi_cfg.get("default_auth_mode"):
        env.setdefault("BAZAAR_TRAPI_AUTH_MODE", str(trapi_cfg["default_auth_mode"]))
    if trapi_cfg.get("default_instance"):
        env.setdefault("BAZAAR_TRAPI_INSTANCE", str(trapi_cfg["default_instance"]))
    trapi_base_urls = _trapi_base_urls(registry)
    if trapi_base_urls:
        env["BAZAAR_TRAPI_BASE_URLS"] = ",".join(trapi_base_urls)
    if trapi_cfg.get("default_base_url"):
        env.setdefault("BAZAAR_TRAPI_BASE_URL", str(trapi_cfg["default_base_url"]))
    foundry_cfg = registry.get("foundry") or {}
    if foundry_cfg.get("default_auth_mode"):
        env.setdefault("AZURE_FOUNDRY_AUTH_MODE", str(foundry_cfg["default_auth_mode"]))
    if foundry_cfg.get("default_base_url"):
        env.setdefault("AZURE_FOUNDRY_BASE_URL", str(foundry_cfg["default_base_url"]))
    env["BAZAAR_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S"] = str(
        args.qwen_min_request_interval_s
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _trapi_base_url(registry: dict[str, Any]) -> str:
    urls = _trapi_base_urls(registry)
    return (
        urls[0]
        if urls
        else "https://research-gateway.example.com/region-b/shared/openai/v1/"
    )


def _trapi_base_urls(registry: dict[str, Any]) -> list[str]:
    trapi_cfg = registry.get("trapi") or {}
    candidates: list[str] = []
    candidates.extend(_split_csv(os.environ.get("BAZAAR_TRAPI_BASE_URLS", "")))
    candidates.extend(_split_csv(os.environ.get("TRAPI_BASE_URLS", "")))
    raw_candidates = trapi_cfg.get("base_url_candidates") or []
    if isinstance(raw_candidates, str):
        candidates.extend(_split_csv(raw_candidates))
    else:
        candidates.extend(str(item) for item in raw_candidates if item)
    if not candidates:
        candidates.extend(
            _split_csv(os.environ.get("BAZAAR_TRAPI_BASE_URL", ""))
        )
        candidates.extend(_split_csv(os.environ.get("TRAPI_BASE_URL", "")))
    if not candidates and trapi_cfg.get("default_base_url"):
        candidates.append(str(trapi_cfg["default_base_url"]))
    return _dedupe_preserving_order(_normalise_base_url(url) for url in candidates)


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalise_base_url(url: str) -> str:
    clean = str(url).strip()
    if not clean:
        return clean
    return clean.rstrip("/") + "/"


def _dedupe_preserving_order(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _provider_base_url(registry: dict[str, Any], provider: str) -> str | None:
    p = (provider or "").lower()
    if p in ("trapi", "cloudgpt"):
        return _trapi_base_url(registry)
    if p in ("foundry", "azure_foundry", "ai_foundry"):
        foundry_cfg = registry.get("foundry") or {}
        return (
            os.environ.get("AZURE_FOUNDRY_BASE_URL")
            or foundry_cfg.get("default_base_url")
            or "https://models.example.azure.com/openai/v1"
        )
    if p in ("openai", "gpt"):
        return os.environ.get("OPENAI_BASE_URL")
    return None


def _max_event_tick(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(tick), -1) FROM events "
            "WHERE action_type NOT IN ('experiment_config', 'experiment_run_complete')"
        ).fetchone()
    finally:
        conn.close()
    return int(row[0] if row and row[0] is not None else -1)


def _assert_quick_check(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    finally:
        conn.close()
    if row is None or row[0] != "ok":
        raise RuntimeError(f"quick_check failed for {db}: {row}")


def _unlink_sqlite_family(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists():
            target.unlink()


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])[-4000:]


def _is_transient_chunk_failure(detail: str) -> bool:
    """Whether a failed subprocess should be retried from the last DB tick."""
    text = detail.lower()
    transient_markers = (
        "all-backends-unhealthy",
        "token limit is exceeded",
        "rate limit",
        "retry after",
        "backend_error: 429",
        "backend_error: 503",
        "backend_error: 504",
        "remote end closed connection",
        "connection reset",
        "remote reset",
        "ssl",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )
    return any(marker in text for marker in transient_markers)


if __name__ == "__main__":
    raise SystemExit(main())

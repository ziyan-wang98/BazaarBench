#!/usr/bin/env python3
"""Run a Level-0 TRAPI rollout with instance failover.

The runner keeps the rollout paper-clean by preserving one model per DB while
cycling TRAPI instances after no-progress or partial-progress failures.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.agents.llm_backends.trapi import trapi_supports_responses  # noqa: E402

DEFAULT_INSTANCES = "region-c/batch,region-c/shared,region-b/shared,region-a/interactive"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tick", type=int, default=360)
    parser.add_argument("--chunk-ticks", type=int, default=12)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--instances", default=DEFAULT_INSTANCES)
    parser.add_argument("--auth-mode", default=os.environ.get("BAZAAR_TRAPI_AUTH_MODE", "azure_cli"))
    parser.add_argument("--agency-mode", default="market-self-interest")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout-s", type=float, default=360.0)
    parser.add_argument("--llm-retries", type=int, default=12)
    parser.add_argument(
        "--provider-fail-fast-errors",
        type=int,
        default=8,
        help=(
            "Abort the current TRAPI instance after this many consecutive "
            "provider-level SSL/403/429/reset errors inside one chunk. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--provider-fail-fast-window-s",
        type=float,
        default=300.0,
        help="Consecutive provider errors must fall within this window.",
    )
    parser.add_argument(
        "--provider-circuit-open-s",
        type=float,
        default=600.0,
        help="How long a failed provider circuit stays open inside one subprocess.",
    )
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--reflection-interval", type=int, default=10000)
    parser.add_argument("--memory-interval", type=int, default=10000)
    parser.add_argument("--self-portrait-interval", type=int, default=10000)
    parser.add_argument("--gpt-min-request-interval-s", type=float, default=0.05)
    parser.add_argument("--sleep-after-no-progress-s", type=float, default=60.0)
    parser.add_argument("--max-no-progress-cycles", type=int, default=2)
    parser.add_argument(
        "--forever",
        action="store_true",
        help=(
            "Never give up before --target-tick. After max no-progress "
            "instance cycles, sleep and retry from the current DB tick."
        ),
    )
    parser.add_argument(
        "--forever-sleep-s",
        type=float,
        default=300.0,
        help="Sleep duration between retry cycles when --forever is set.",
    )
    parser.add_argument("--no-responses-endpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.chunk_ticks <= 0:
        raise SystemExit("--chunk-ticks must be positive")
    if args.target_tick <= 0:
        raise SystemExit("--target-tick must be positive")
    if args.max_no_progress_cycles <= 0:
        raise SystemExit("--max-no-progress-cycles must be positive")
    if args.provider_fail_fast_errors < 0:
        raise SystemExit("--provider-fail-fast-errors must be non-negative")
    if args.provider_fail_fast_window_s < 0:
        raise SystemExit("--provider-fail-fast-window-s must be non-negative")
    if args.provider_circuit_open_s < 0:
        raise SystemExit("--provider-circuit-open-s must be non-negative")
    if args.forever_sleep_s < 0:
        raise SystemExit("--forever-sleep-s must be non-negative")
    if not args.db.exists():
        raise SystemExit(f"--db not found: {args.db}")

    instances = [item.strip() for item in args.instances.split(",") if item.strip()]
    if not instances:
        raise SystemExit("--instances must contain at least one TRAPI instance")

    checkpoint_dir = args.checkpoint_dir or args.db.parent / "checkpoints"
    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    instance_idx = 0
    no_progress_cycles = 0
    while True:
        current_tick = _max_tick(args.db)
        if current_tick >= args.target_tick:
            print(f"[failover] reached target tick {current_tick}", flush=True)
            return 0

        span = min(args.chunk_ticks, args.target_tick - current_tick)
        instance = instances[instance_idx % len(instances)]
        cmd = _rollout_cmd(
            db=args.db,
            model=args.model,
            agency_mode=args.agency_mode,
            ticks=span,
            target_tick=args.target_tick,
            checkpoint_dir=checkpoint_dir,
            workers=args.workers,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout_s=args.llm_timeout_s,
            llm_retries=args.llm_retries,
            reasoning_effort=args.reasoning_effort,
            reflection_interval=args.reflection_interval,
            memory_interval=args.memory_interval,
            self_portrait_interval=args.self_portrait_interval,
            use_responses_endpoint=(
                not args.no_responses_endpoint and trapi_supports_responses(args.model)
            ),
        )
        print(
            "[failover] "
            f"tick={current_tick} target={args.target_tick} span={span} "
            f"instance={instance} workers={args.workers} model={args.model}",
            flush=True,
        )
        print(f"[failover] cmd={' '.join(cmd)}", flush=True)
        if args.dry_run:
            instance_idx += 1
            if instance_idx >= len(instances):
                return 0
            continue

        env = _subprocess_env(args, instance=instance)
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        next_tick = _max_tick(args.db)
        print(
            f"[failover] exit={proc.returncode} tick_before={current_tick} "
            f"tick_after={next_tick} instance={instance}",
            flush=True,
        )

        if next_tick > current_tick:
            _write_checkpoint(
                args.db,
                checkpoint_dir
                / f"{args.db.stem}_tick{next_tick:04d}_{_safe_instance(instance)}.db",
            )
            no_progress_cycles = 0
            if proc.returncode != 0:
                instance_idx += 1
            continue

        instance_idx += 1
        if instance_idx % len(instances) == 0:
            no_progress_cycles += 1
            print(
                "[failover] "
                f"completed a no-progress instance cycle "
                f"{no_progress_cycles}/{args.max_no_progress_cycles}",
                flush=True,
            )
            if no_progress_cycles >= args.max_no_progress_cycles:
                if args.forever:
                    print(
                        "[failover] "
                        f"no progress after {no_progress_cycles} full "
                        f"instance cycle(s); sleeping {args.forever_sleep_s:.1f}s "
                        "before retrying from current DB tick",
                        flush=True,
                    )
                    no_progress_cycles = 0
                    time.sleep(args.forever_sleep_s)
                    continue
                return proc.returncode or 1
            time.sleep(args.sleep_after_no_progress_s)


def _rollout_cmd(
    *,
    db: Path,
    model: str,
    agency_mode: str,
    ticks: int,
    target_tick: int,
    checkpoint_dir: Path,
    workers: int,
    llm_max_tokens: int,
    llm_timeout_s: float,
    llm_retries: int,
    reasoning_effort: str,
    reflection_interval: int,
    memory_interval: int,
    self_portrait_interval: int,
    use_responses_endpoint: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/level1/run_rollout.py",
        "--db",
        str(db),
        "--provider",
        "trapi",
        "--model",
        model,
        "--agency-mode",
        agency_mode,
        "--ticks",
        str(ticks),
        "--chunk-ticks",
        str(ticks),
        "--max-tick",
        str(target_tick),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--reflection-interval",
        str(reflection_interval),
        "--memory-interval",
        str(memory_interval),
        "--self-portrait-interval",
        str(self_portrait_interval),
        "--reasoning-effort",
        reasoning_effort,
        "--inventory-validator-mode",
        "block",
        "--defer-initial-llm-dynamics",
        "--parallel-decide",
        "--parallel-workers",
        str(workers),
        "--llm-max-tokens",
        str(llm_max_tokens),
        "--llm-timeout-s",
        str(llm_timeout_s),
        "--llm-retries",
        str(llm_retries),
        "--strict-llm-errors",
        "--skip-audit",
        "--skip-run-complete-event",
        "--skip-experiment-config-event",
    ]
    if use_responses_endpoint:
        cmd.append("--use-responses-endpoint")
    return cmd


def _subprocess_env(args: argparse.Namespace, *, instance: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["BAZAAR_TRAPI_INSTANCE"] = instance
    env["BAZAAR_TRAPI_AUTH_MODE"] = args.auth_mode
    env["BAZAAR_TRAPI_GPT_MIN_REQUEST_INTERVAL_S"] = str(args.gpt_min_request_interval_s)
    env["BAZAAR_TRAPI_PROVIDER_FAIL_FAST_ERRORS"] = str(args.provider_fail_fast_errors)
    env["BAZAAR_TRAPI_PROVIDER_FAIL_FAST_WINDOW_S"] = str(
        args.provider_fail_fast_window_s
    )
    env["BAZAAR_TRAPI_PROVIDER_CIRCUIT_OPEN_S"] = str(args.provider_circuit_open_s)
    env.pop("BAZAAR_TRAPI_BASE_URL", None)
    env.pop("TRAPI_BASE_URL", None)
    return env


def _max_tick(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT COALESCE(MAX(tick), 0) FROM events").fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def _write_checkpoint(db: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    shutil.copy2(db, tmp)
    os.replace(tmp, out)
    print(f"[failover] checkpoint {out}", flush=True)


def _safe_instance(instance: str) -> str:
    return instance.strip("/").replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())

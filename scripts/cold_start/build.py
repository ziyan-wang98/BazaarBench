#!/usr/bin/env python3
"""Build a CSV-grounded cold-start marketplace world (v3).

Cold-start v3 framework: each agent gets a tier (auto-assigned from the
brand+category+price-band cluster aggregate), an independent
``lifetime_days`` (sampled per tier), an LLM-generated journal (events
materialized at negative ticks), typed memories, and a self-summary.
LLM is mandatory; backend failures retry up to ``--llm-max-retries``
times before aborting.

DB suffix convention: pass ``--auto-suffix`` to append a timestamped
``_v<YYYYmmddHHMM>`` before the ``.db`` extension so each build lands
at its own path. Pick the base world for downstream case experiments
by passing the suffixed path as ``--base-db`` to inject scripts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from bazaar.experiments import ColdStartConfig, build_cold_start_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--auto-suffix",
        action="store_true",
        help="Append _v<UTC-timestamp> to --out so each build lands at a unique path.",
    )
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--item-sample-size", type=int, default=30_000)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional deterministic row sample fraction, e.g. 0.1.",
    )
    parser.add_argument("--min-inventory-items", type=int, default=3)
    parser.add_argument("--max-inventory-items", type=int, default=8)
    parser.add_argument("--history-events-per-agent", type=int, default=5)
    parser.add_argument("--initial-listings", type=int)
    parser.add_argument(
        "--agency-mode",
        default="market-self-interest",
        choices=("safe", "market-self-interest"),
    )
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-5.2")
    parser.add_argument("--llm-max-tokens", type=int, default=900)
    parser.add_argument("--llm-temperature", type=float, default=0.6)
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=10,
        help="Per-agent LLM enrichment retry budget. Build aborts on exhaustion.",
    )
    parser.add_argument(
        "--no-tool-schema",
        action="store_true",
        help="Skip the JSON-tool spec and rely on free-form JSON-in-text. "
             "Use for small local models that can't handle nested enum schemas.",
    )
    parser.add_argument(
        "--pressure-mode",
        default="baseline",
        choices=("baseline", "pressure"),
        help="Layer 0 baseline (no deadline / no financial stress / soft buyer "
             "goal — encourages active trade) vs Layer 2 pressure (deadline + "
             "financial stress + strict-substitute clause — measures objective "
             "shift under pressure).",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        default="medium",
        choices=("none", "low", "medium", "high", "xhigh"),
        help="OpenAI reasoning-family effort (gpt-5.x, o-series). Ignored elsewhere.",
    )
    parser.add_argument(
        "--llm-use-responses-endpoint",
        action="store_true",
        help="Route OpenAI calls through /v1/responses (recommended for gpt-5.x; "
             "surfaces reasoning summary, supports tool_choice='required').",
    )
    parser.add_argument(
        "--verifier-mode",
        default="rule",
        choices=("none", "rule", "llm", "both"),
        help="Cold-start audit mode. LLM modes add one aggregate verifier call.",
    )
    parser.add_argument(
        "--min-groundedness-score",
        type=float,
        default=0.85,
        help="Fail if the rule audit groundedness score falls below this value.",
    )
    parser.add_argument("--audit-out", type=Path)
    parser.add_argument("--seed-plan-out", type=Path)
    parser.add_argument("--profile-out", type=Path)
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=0,
        help="Optionally run a benign resume rollout after cold-start build.",
    )
    parser.add_argument("--warmup-provider", default="qwen")
    parser.add_argument("--warmup-model", default="qwen3.6-35b-a3b")
    parser.add_argument("--warmup-chunk-ticks", type=int, default=24)
    parser.add_argument("--warmup-agent-limit", type=int)
    parser.add_argument("--warmup-checkpoint-dir", type=Path)
    parser.add_argument("--warmup-reasoning-effort", default="medium")
    parser.add_argument(
        "--warmup-use-responses-endpoint",
        action="store_true",
    )
    parser.add_argument(
        "--warmup-inventory-validator-mode",
        choices=("off", "warn", "block"),
        default="block",
    )
    parser.add_argument(
        "--warmup-dry-run",
        action="store_true",
        help="Print the warmup command without running chunks.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out
    if args.auto_suffix:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
        out = out.with_name(f"{out.stem}_v{ts}{out.suffix or '.db'}")

    summary = build_cold_start_world(
        ColdStartConfig(
            db_path=out,
            dataset_csv=args.dataset_csv,
            n_agents=args.agents,
            days=args.days,
            seed=args.seed,
            item_sample_size=args.item_sample_size,
            item_sample_fraction=args.sample_fraction,
            min_inventory_items=args.min_inventory_items,
            max_inventory_items=args.max_inventory_items,
            history_events_per_agent=args.history_events_per_agent,
            initial_listings=args.initial_listings,
            agency_mode=args.agency_mode,
            use_llm=True,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_max_tokens=args.llm_max_tokens,
            llm_temperature=args.llm_temperature,
            llm_max_retries=args.llm_max_retries,
            use_tool_schema=not args.no_tool_schema,
            pressure_mode=args.pressure_mode,
            llm_reasoning_effort=args.llm_reasoning_effort,
            llm_use_responses_endpoint=args.llm_use_responses_endpoint,
            verifier_mode=args.verifier_mode,
            min_groundedness_score=args.min_groundedness_score,
            audit_out=args.audit_out,
            seed_plan_out=args.seed_plan_out,
            profile_out=args.profile_out,
            force=args.force,
        )
    )
    summary["resolved_db_path"] = str(out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.warmup_ticks > 0:
        cmd = _warmup_cmd(args, out)
        print(json.dumps({"warmup_command": cmd}, indent=2, sort_keys=True))
        _run_warmup_or_exit(cmd)


def _warmup_cmd(args: argparse.Namespace, db_path: Path) -> list[str]:
    script = Path(__file__).with_name("run_scaleup_rollout.py")
    cmd = [
        sys.executable,
        str(script),
        "--db",
        str(db_path),
        "--provider",
        args.warmup_provider,
        "--model",
        args.warmup_model,
        "--agency-mode",
        args.agency_mode,
        "--ticks",
        str(args.warmup_ticks),
        "--chunk-ticks",
        str(args.warmup_chunk_ticks),
        "--reasoning-effort",
        args.warmup_reasoning_effort,
        "--inventory-validator-mode",
        args.warmup_inventory_validator_mode,
    ]
    if args.warmup_agent_limit is not None:
        cmd.extend(["--agent-limit", str(args.warmup_agent_limit)])
    if args.warmup_checkpoint_dir is not None:
        cmd.extend(["--checkpoint-dir", str(args.warmup_checkpoint_dir)])
    if args.warmup_use_responses_endpoint:
        cmd.append("--use-responses-endpoint")
    if args.warmup_dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_warmup_or_exit(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "[cold-start-build] warmup command failed with exit code "
            f"{result.returncode}: {' '.join(cmd)}"
        )


if __name__ == "__main__":
    main()

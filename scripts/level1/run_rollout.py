#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a scale-up DB in checkpointed chunks through the existing CLI."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazaar.agents.persona import (
    MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    MARKETPLACE_AGENCY_MODES,
    normalize_marketplace_agency,
)
from scripts.level1.apply_strategy_adaptation import ADAPTATION_MODES, apply_adaptation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument(
        "--agency-mode",
        default=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
        choices=MARKETPLACE_AGENCY_MODES,
        help=(
            "Benign-agent marketplace agency layer. Applied to selected "
            "resumed benign personas before each chunk."
        ),
    )
    parser.add_argument("--ticks", type=int, default=360)
    parser.add_argument("--chunk-ticks", type=int, default=24)
    parser.add_argument(
        "--max-tick",
        type=int,
        default=None,
        help="Stop the rollout after the next chunk completes if the DB max_tick reaches this value.",
    )
    parser.add_argument("--agent-limit", type=int)
    parser.add_argument("--agent-ids")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--reflection-interval", type=int, default=24)
    parser.add_argument("--memory-interval", type=int, default=24)
    parser.add_argument("--self-portrait-interval", type=int, default=84)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--use-responses-endpoint", action="store_true")
    parser.add_argument("--enable-probes", action="store_true")
    parser.add_argument("--probe-model")
    parser.add_argument(
        "--adaptive-mode",
        choices=("none", *ADAPTATION_MODES),
        default="none",
        help=(
            "Inject train-free strategy adaptation into prompt-visible "
            "ledger/summary rows before each chunk."
        ),
    )
    parser.add_argument("--adaptation-history-window", type=int, default=72)
    parser.add_argument("--adaptation-seller-target", type=int, default=8)
    parser.add_argument("--adaptation-deadline-offset", type=int, default=72)
    parser.add_argument("--adaptation-public-limit", type=int, default=8)
    parser.add_argument("--adaptation-out-dir", type=Path)
    parser.add_argument(
        "--chunk-timeout-sec",
        type=int,
        help=(
            "Optional wall-clock timeout for each llm-smoke chunk. On timeout, "
            "copy a checkpoint and stop cleanly so partial results are preserved."
        ),
    )
    parser.add_argument("--no-r20-nudge", action="store_true")
    parser.add_argument("--no-seller-inventory-guard", action="store_true")
    parser.add_argument("--require-handoff-proof", action="store_true")
    parser.add_argument(
        "--inventory-validator-mode",
        choices=("off", "warn", "block"),
        default="off",
    )
    parser.add_argument(
        "--defer-initial-llm-dynamics",
        dest="defer_initial_llm_dynamics",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--run-initial-llm-dynamics",
        dest="defer_initial_llm_dynamics",
        action="store_false",
    )
    parser.add_argument(
        "--parallel-decide",
        action="store_true",
        help="Run agents' decide() concurrently per tick (cloud-LLM speedup).",
    )
    parser.add_argument("--parallel-workers", type=int, default=32)
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=4096,
        help="Per-decision max_tokens budget for LLMPolicy. With qwen3.6 "
             "thinking ON this includes reasoning + tool-call JSON. "
             "Bump to 8192 for complex multi-decision ticks.",
    )
    parser.add_argument(
        "--llm-timeout-s",
        type=float,
        default=300.0,
        help="Per-decision LLM backend timeout passed through to llm-smoke.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=5,
        help="Per-decision LLM backend retry count passed through to llm-smoke.",
    )
    parser.add_argument(
        "--strict-llm-errors",
        action="store_true",
        help="Fail chunks when llm-smoke records backend errors.",
    )
    parser.add_argument(
        "--continue-on-tick-advance-failure",
        action="store_true",
        help=(
            "Continue after a non-zero llm-smoke chunk if market ticks advanced. "
            "This is unsafe for paper-facing strict runs and is opt-in."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip cheap BazaarBench audit gates before/after chunks.",
    )
    parser.add_argument(
        "--skip-run-complete-event",
        action="store_true",
        help=(
            "Do not let each llm-smoke chunk append experiment_run_complete. "
            "Use this for checkpointed protocol rollouts whose fork ticks must "
            "stay on market action ticks."
        ),
    )
    parser.add_argument(
        "--skip-experiment-config-event",
        action="store_true",
        help=(
            "Do not let each llm-smoke chunk append experiment_config to events. "
            "The config is still written to meta by llm-smoke."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"--db not found: {args.db}")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.chunk_ticks <= 0:
        raise SystemExit("--chunk-ticks must be positive")

    agent_ids = _resolve_agent_ids(args.db, args.agent_ids, args.agent_limit)
    agency_mode = normalize_marketplace_agency(args.agency_mode)
    checkpoint_dir = args.checkpoint_dir or args.db.parent / f"{args.db.stem}_checkpoints"
    adaptation_out_dir = (
        args.adaptation_out_dir
        or checkpoint_dir / "strategy_adaptation"
    )
    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.adaptive_mode != "none":
            adaptation_out_dir.mkdir(parents=True, exist_ok=True)

    remaining = args.ticks
    chunk_idx = 0
    display_agents = (
        len(agent_ids.split(",")) if agent_ids else _count_active_benign_agents(args.db)
    )
    if args.dry_run and not args.skip_audit:
        print(f"[preflight] {' '.join(_audit_cmd(args.db))}")
    if not args.dry_run and not args.skip_audit:
        _run_audit(args.db, label="preflight")

    while remaining > 0:
        chunk_idx += 1
        chunk = min(args.chunk_ticks, remaining)
        cmd = _llm_smoke_cmd(
            db=args.db,
            provider=args.provider,
            model=args.model,
            agency_mode=agency_mode,
            ticks=chunk,
            display_agents=display_agents,
            agent_ids=agent_ids,
            reflection_interval=args.reflection_interval,
            memory_interval=args.memory_interval,
            self_portrait_interval=args.self_portrait_interval,
            reasoning_effort=args.reasoning_effort,
            use_responses_endpoint=args.use_responses_endpoint,
            enable_probes=args.enable_probes,
            probe_model=args.probe_model,
            disable_r20_nudge=args.no_r20_nudge,
            disable_seller_inventory_guard=args.no_seller_inventory_guard,
            require_handoff_proof=args.require_handoff_proof,
            inventory_validator_mode=args.inventory_validator_mode,
            defer_initial_llm_dynamics=args.defer_initial_llm_dynamics,
            parallel_decide=args.parallel_decide,
            parallel_workers=args.parallel_workers,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout_s=args.llm_timeout_s,
            llm_retries=args.llm_retries,
            strict_llm_errors=args.strict_llm_errors,
            skip_audit=args.skip_audit,
            skip_run_complete_event=args.skip_run_complete_event,
            skip_experiment_config_event=args.skip_experiment_config_event,
        )
        print(f"[chunk {chunk_idx}] {' '.join(cmd)}")
        if not args.dry_run:
            _apply_agency_mode_to_db(args.db, agency_mode=agency_mode, agent_ids=agent_ids)
            if args.adaptive_mode != "none":
                report = apply_adaptation(
                    args.db,
                    mode=args.adaptive_mode,
                    agent_ids=_agent_ids_list(agent_ids),
                    agent_limit=None if agent_ids else args.agent_limit,
                    tick=_max_tick(args.db) + 1,
                    history_window=args.adaptation_history_window,
                    seller_target=args.adaptation_seller_target,
                    deadline_offset=args.adaptation_deadline_offset,
                    public_limit=args.adaptation_public_limit,
                    dry_run=False,
                )
                report_path = (
                    adaptation_out_dir
                    / f"{args.db.stem}_chunk{chunk_idx:03d}.json"
                )
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[chunk {chunk_idx}] strategy adaptation "
                    f"{args.adaptive_mode}: {report['inserted_count']} agents "
                    f"-> {report_path}"
                )
            tick_before_chunk = _max_tick(args.db)
            try:
                subprocess.run(cmd, check=True, timeout=args.chunk_timeout_sec)
            except subprocess.TimeoutExpired:
                max_tick = _max_tick(args.db)
                checkpoint = (
                    checkpoint_dir
                    / f"{args.db.stem}_tick{max_tick:04d}_timeout_chunk{chunk_idx:03d}.db"
                )
                _copy_sqlite_checkpoint(args.db, checkpoint)
                print(
                    f"[chunk {chunk_idx}] timed out after "
                    f"{args.chunk_timeout_sec}s; checkpoint {checkpoint}"
                )
                break
            except subprocess.CalledProcessError as e:
                _handle_chunk_process_failure(
                    e,
                    db=args.db,
                    chunk_idx=chunk_idx,
                    tick_before_chunk=tick_before_chunk,
                    continue_on_tick_advance_failure=(
                        args.continue_on_tick_advance_failure
                    ),
                )
            max_tick = _max_tick(args.db)
            checkpoint = checkpoint_dir / f"{args.db.stem}_tick{max_tick:04d}.db"
            _copy_sqlite_checkpoint(args.db, checkpoint)
            print(f"[chunk {chunk_idx}] checkpoint {checkpoint}")
            if not args.skip_audit:
                _run_audit(args.db, label=f"chunk {chunk_idx} postflight")
        remaining -= chunk
        stop_sentinel = args.db.parent / f"STOP_{args.db.stem}"
        if stop_sentinel.exists():
            print(f"[chunk {chunk_idx}] sentinel {stop_sentinel} present; stopping at tick {_max_tick(args.db)}")
            break
        if args.max_tick and _max_tick(args.db) >= args.max_tick:
            print(f"[chunk {chunk_idx}] reached --max-tick {args.max_tick}; stopping")
            break


def _resolve_agent_ids(
    db: Path,
    raw_agent_ids: str | None,
    agent_limit: int | None,
) -> str | None:
    if raw_agent_ids:
        ids = sorted({int(part.strip()) for part in raw_agent_ids.split(",") if part.strip()})
        if not ids:
            raise SystemExit("--agent-ids did not contain any ids")
        return ",".join(str(i) for i in ids)
    if agent_limit is None:
        return None
    if agent_limit <= 0:
        raise SystemExit("--agent-limit must be positive")
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            """
            SELECT agent_id
            FROM agents
            WHERE is_seeded = 0 AND is_redteam = 0 AND status = 'active'
            ORDER BY agent_id
            LIMIT ?
            """,
            (agent_limit,),
        ).fetchall()
    finally:
        conn.close()
    ids = [int(row[0]) for row in rows]
    if len(ids) < agent_limit:
        raise SystemExit(f"only found {len(ids)} active benign agents")
    return ",".join(str(i) for i in ids)


def _copy_sqlite_checkpoint(source: Path, dest: Path) -> None:
    """Write a standalone SQLite checkpoint, including uncheckpointed WAL pages."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp")
    tmp.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        tmp.with_name(f"{tmp.name}{suffix}").unlink(missing_ok=True)
    src_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(tmp)
        try:
            src_conn.backup(dst_conn)
            dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            dst_conn.execute("PRAGMA journal_mode=DELETE")
            quick_check = dst_conn.execute("PRAGMA quick_check").fetchone()[0]
            if quick_check != "ok":
                raise sqlite3.DatabaseError(f"checkpoint quick_check failed: {quick_check}")
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    tmp.replace(dest)
    for suffix in ("-wal", "-shm"):
        tmp.with_name(f"{tmp.name}{suffix}").unlink(missing_ok=True)
        dest.with_name(f"{dest.name}{suffix}").unlink(missing_ok=True)


def _handle_chunk_process_failure(
    exc: subprocess.CalledProcessError,
    *,
    db: Path,
    chunk_idx: int,
    tick_before_chunk: int,
    continue_on_tick_advance_failure: bool = False,
) -> None:
    tick_after_chunk = _max_tick(db)
    if continue_on_tick_advance_failure and tick_after_chunk > tick_before_chunk:
        print(
            f"[chunk {chunk_idx}] subprocess exit {exc.returncode} "
            f"but tick advanced {tick_before_chunk}->{tick_after_chunk}; "
            f"treating as soft-fail and continuing"
        )
        return
    if tick_after_chunk > tick_before_chunk:
        raise SystemExit(
            f"[chunk {chunk_idx}] subprocess exit {exc.returncode} "
            f"after tick advanced {tick_before_chunk}->{tick_after_chunk}; "
            "aborting because --continue-on-tick-advance-failure was not set"
        )
    raise SystemExit(
        f"[chunk {chunk_idx}] subprocess exit {exc.returncode} "
        "with no tick advance; aborting"
    )


def _agent_ids_list(raw_agent_ids: str | None) -> list[int] | None:
    if not raw_agent_ids:
        return None
    return [int(part) for part in raw_agent_ids.split(",") if part]


def _llm_smoke_cmd(
    *,
    db: Path,
    provider: str,
    model: str,
    agency_mode: str = MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ticks: int,
    display_agents: int,
    agent_ids: str | None,
    reflection_interval: int,
    memory_interval: int,
    self_portrait_interval: int,
    reasoning_effort: str,
    use_responses_endpoint: bool,
    enable_probes: bool,
    probe_model: str | None,
    disable_r20_nudge: bool,
    disable_seller_inventory_guard: bool,
    require_handoff_proof: bool,
    inventory_validator_mode: str,
    defer_initial_llm_dynamics: bool,
    parallel_decide: bool = False,
    parallel_workers: int = 32,
    llm_max_tokens: int = 4096,
    llm_timeout_s: float = 300.0,
    llm_retries: int = 5,
    strict_llm_errors: bool = False,
    skip_audit: bool = False,
    skip_run_complete_event: bool = False,
    skip_experiment_config_event: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "bazaar.cli",
        "llm-smoke",
        "--provider",
        provider,
        "--model",
        model,
        "--agency-mode",
        agency_mode,
        "--agents",
        str(display_agents),
        "--ticks",
        str(ticks),
        "--phantoms",
        "0",
        "--out",
        str(db),
        "--resume",
        "--reflection-interval",
        str(reflection_interval),
        "--memory-interval",
        str(memory_interval),
        "--self-portrait-interval",
        str(self_portrait_interval),
        "--reasoning-effort",
        reasoning_effort,
    ]
    if use_responses_endpoint:
        cmd.append("--use-responses-endpoint")
    if enable_probes:
        cmd.append("--enable-probes")
    if probe_model:
        cmd.extend(["--probe-model", probe_model])
    if disable_r20_nudge:
        cmd.append("--no-r20-nudge")
    if disable_seller_inventory_guard:
        cmd.append("--no-seller-inventory-guard")
    if require_handoff_proof:
        cmd.append("--require-handoff-proof")
    if inventory_validator_mode != "off":
        cmd.extend(["--inventory-validator-mode", inventory_validator_mode])
    if defer_initial_llm_dynamics:
        cmd.append("--defer-initial-llm-dynamics")
    else:
        cmd.append("--run-initial-llm-dynamics")
    if agent_ids:
        cmd.extend(["--resume-agent-ids", agent_ids])
    if parallel_decide:
        cmd.append("--parallel-decide")
        cmd.extend(["--parallel-workers", str(parallel_workers)])
    cmd.extend(["--llm-max-tokens", str(llm_max_tokens)])
    cmd.extend(["--llm-timeout-s", str(llm_timeout_s)])
    cmd.extend(["--llm-retries", str(llm_retries)])
    if strict_llm_errors:
        cmd.append("--strict-llm-errors")
    if skip_audit:
        cmd.append("--skip-audit")
    if skip_run_complete_event:
        cmd.append("--skip-run-complete-event")
    if skip_experiment_config_event:
        cmd.append("--skip-experiment-config-event")
    return cmd


def _audit_cmd(db: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bazaar.cli",
        "audit",
        str(db),
        "--allow-legacy-seeds",
        "--json",
    ]


def _run_audit(db: Path, *, label: str) -> None:
    result = subprocess.run(
        _audit_cmd(db),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"[{label}] BazaarBench audit failed for {db}")
    print(f"[{label}] BazaarBench audit ok")


def _apply_agency_mode_to_db(
    db: Path,
    *,
    agency_mode: str,
    agent_ids: str | None,
) -> None:
    clauses = ["COALESCE(is_seeded, 0) = 0", "COALESCE(is_redteam, 0) = 0"]
    params: list[int] = []
    if agent_ids:
        ids = [int(part) for part in agent_ids.split(",") if part]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        clauses.append(f"agent_id IN ({placeholders})")
        params.extend(ids)
    where = " AND ".join(clauses)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            f"SELECT agent_id, persona_json FROM agents WHERE {where}",
            tuple(params),
        ).fetchall()
        with conn:
            for agent_id, raw in rows:
                try:
                    payload = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    payload = {}
                payload["agency_mode"] = agency_mode
                conn.execute(
                    "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                    (json.dumps(payload, sort_keys=True), int(agent_id)),
                )
            conn.execute(
                """
                INSERT INTO meta (key, value)
                VALUES ('agency_mode', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (agency_mode,),
            )
    finally:
        conn.close()


def _max_tick(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(tick), -1)
            FROM events
            WHERE action_type NOT IN ('experiment_config', 'experiment_run_complete')
            """
        ).fetchone()
    finally:
        conn.close()
    return int(row[0] if row is not None else -1)


def _count_active_benign_agents(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM agents
            WHERE is_seeded = 0 AND is_redteam = 0 AND status = 'active'
            """
        ).fetchone()
    finally:
        conn.close()
    return int(row[0] if row is not None else 0)


if __name__ == "__main__":
    main()

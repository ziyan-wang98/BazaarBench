from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from scripts import run_trapi_level123_rollouts as runner


def _write_min_db(path: Path, tick: int = 372) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE events (tick INTEGER, action_type TEXT)")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO events (tick, action_type) VALUES (?, ?)",
            (tick, "agent_action"),
        )
        conn.commit()
    finally:
        conn.close()


def test_trapi_level12_cells_match_goal_matrix() -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    base = runner._model_spec(registry, str(registry["base_model_key"]))
    treatments = [
        runner._model_spec(registry, str(key))
        for key in registry["treatment_model_keys"]
    ]

    cells = runner._build_cells(
        levels={1, 2},
        out_root=Path("runs/trapi_matrix"),
        treatments=treatments,
        base=base,
        level2_pressure_mode="combined",
        falsif_pressure_side="combined",
    )

    names = {cell.name for cell in cells}
    treatment_keys = {spec.key for spec in treatments}
    assert len(cells) == (12 if base.key in treatment_keys else 13)
    assert "L1-C-gpt54mini" in names
    assert "L1-C-gpt55" in names
    assert "L1-C-gpt54nano" in names
    assert "L1-C-deepseekv32" in names
    assert "L1-C-llama33" in names
    assert "L1-C-gptoss120b" in names
    assert "L2-P-gpt54mini" in names
    assert "L2-P-gptoss120b" in names
    assert "L2-S-gpt54mini" not in names
    assert "L2-B-gptoss120b" not in names
    if base.key in treatment_keys:
        assert f"L2-falsif-{base.key}-pressure" not in names
    else:
        assert f"L2-falsif-{base.key}-pressure" in names
    assert all(cell.prompt_file is None for cell in cells if cell.level == 1)
    assert all(
        cell.prompt_file and cell.prompt_file.name == "pressure_combined.txt"
        for cell in cells
        if cell.level == 2
    )
    assert runner.TREATED_AGENT_IDS == tuple(range(1, 100, 5))


def test_trapi_level2_split_pressure_cells_remain_available() -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    base = runner._model_spec(registry, str(registry["base_model_key"]))
    treatments = [
        runner._model_spec(registry, str(key))
        for key in registry["treatment_model_keys"]
    ]

    cells = runner._build_cells(
        levels={2},
        out_root=Path("runs/trapi_matrix"),
        treatments=treatments,
        base=base,
        level2_pressure_mode="split",
        falsif_pressure_side="seller",
    )

    by_name = {cell.name: cell for cell in cells}
    assert "L2-S-gpt54mini" in by_name
    assert "L2-B-gpt54mini" in by_name
    assert by_name["L2-S-gpt54mini"].prompt_file
    assert by_name["L2-S-gpt54mini"].prompt_file.name == "pressure_seller.txt"
    assert by_name["L2-B-gpt54mini"].prompt_file
    assert by_name["L2-B-gpt54mini"].prompt_file.name == "pressure_buyer.txt"


def test_trapi_level12_adds_base_falsification_when_base_is_not_treatment() -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    base = runner._model_spec(registry, "qwen35_397b")
    treatments = [
        runner._model_spec(registry, str(key))
        for key in registry["treatment_model_keys"]
    ]

    cells = runner._build_cells(
        levels={2},
        out_root=Path("runs/trapi_matrix"),
        treatments=treatments,
        base=base,
        level2_pressure_mode="combined",
        falsif_pressure_side="combined",
    )

    by_name = {cell.name: cell for cell in cells}
    assert "L2-falsif-qwen35_397b-pressure" in by_name
    assert by_name["L2-falsif-qwen35_397b-pressure"].treatment_key == "qwen35_397b"
    assert by_name["L2-falsif-qwen35_397b-pressure"].prompt_file
    assert (
        by_name["L2-falsif-qwen35_397b-pressure"].prompt_file.name
        == "pressure_combined.txt"
    )


def test_trapi_level2_split_falsification_defaults_to_legacy_seller_prompt() -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    base = runner._model_spec(registry, "qwen35_397b")
    treatments = [
        runner._model_spec(registry, str(key))
        for key in registry["treatment_model_keys"]
    ]

    cells = runner._build_cells(
        levels={2},
        out_root=Path("runs/trapi_matrix"),
        treatments=treatments,
        base=base,
        level2_pressure_mode="split",
        falsif_pressure_side="combined",
    )

    by_name = {cell.name: cell for cell in cells}
    falsif = by_name["L2-falsif-qwen35_397b-pressure"]
    assert falsif.pressure_side == "seller"
    assert falsif.prompt_file
    assert falsif.prompt_file.name == "pressure_seller.txt"


def test_trapi_level3_cells_are_available_when_requested() -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    treatments = [
        runner._model_spec(registry, str(key))
        for key in registry["treatment_model_keys"]
    ]
    base = runner._model_spec(registry, str(registry["base_model_key"]))

    cells = runner._build_cells(
        levels={3},
        out_root=Path("runs/trapi_matrix"),
        treatments=treatments,
        base=base,
        level2_pressure_mode="combined",
        falsif_pressure_side="seller",
    )

    assert [cell.name for cell in cells] == [
        "L3-RT-gpt54mini",
        "L3-RT-gpt55",
        "L3-RT-gpt54nano",
        "L3-RT-deepseekv32",
        "L3-RT-llama33",
        "L3-RT-gptoss120b",
    ]
    assert all(cell.prompt_file and cell.prompt_file.name == "redteam_T1-6.txt" for cell in cells)


def test_trapi_level123_dry_run_uses_registry_models(tmp_path, capsys) -> None:
    base_db = tmp_path / "base.db"
    _write_min_db(base_db)

    rc = runner.main([
        "--base-db",
        str(base_db),
        "--out-root",
        str(tmp_path / "matrix"),
            "--levels",
            "1",
            "--only",
            "L1-C-gpt55",
            "--dry-run",
        ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "--provider trapi" in out
    assert "--use-responses-endpoint" in out
    assert "--model gpt-5.4-mini_2026-03-17" in out
    assert "--treatment-provider trapi" in out
    assert "--treatment-model gpt-5.5_2026-04-24" in out
    assert "--treatment-base-url https://research-gateway.example.com/region-a/interactive/openai/v1/" in out
    assert "--treatment-use-responses-endpoint" in out
    assert "--treatment-prompt-suffix-file" not in out


def test_trapi_level123_chunk_cmd_uses_treatment_token_budget(tmp_path) -> None:
    registry = runner._load_registry(Path("configs/model_registry/foundry_level1.yaml"))
    base = runner._model_spec(registry, "gpt55")
    treatment = runner._model_spec(registry, "gpt54")
    cell = runner.CellSpec(
        name="L1-C-gpt54",
        level=1,
        out_path=tmp_path / "L1-C-gpt54.db",
        treatment_key="gpt54",
        pressure_side="none",
        prompt_file=None,
        notes="test",
    )
    args = argparse.Namespace(
        defense_arm="open_trust_control",
        validator_mode="warn",
        parallel_workers=1,
    )

    cmd = runner._build_chunk_cmd(
        cell,
        base=base,
        treatment=treatment,
        registry=registry,
        args=args,
        ticks=12,
    )

    assert cmd[cmd.index("--llm-max-tokens") + 1] == "8192"
    assert cmd[cmd.index("--treatment-llm-max-tokens") + 1] == "12288"


def test_trapi_level123_unknown_cell_is_rejected(tmp_path) -> None:
    base_db = tmp_path / "base.db"
    _write_min_db(base_db)

    with pytest.raises(SystemExit, match="unknown cell"):
        runner.main([
            "--base-db",
            str(base_db),
            "--only",
            "L2-S-missing",
            "--dry-run",
        ])


def test_trapi_level123_refuses_existing_db_without_overwrite(tmp_path) -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    base = runner._model_spec(registry, str(registry["base_model_key"]))
    treatment = runner._model_spec(registry, "gpt54mini")
    base_db = tmp_path / "base.db"
    out_db = tmp_path / "cell.db"
    _write_min_db(base_db)
    _write_min_db(out_db)
    cell = runner.CellSpec(
        name="L1-C-gpt54mini",
        level=1,
        out_path=out_db,
        treatment_key="gpt54mini",
        pressure_side="none",
        prompt_file=None,
        notes="test",
    )
    args = argparse.Namespace(
        base_db=base_db,
        ticks=84,
        chunk_ticks=12,
        validator_mode="warn",
        defense_arm="open_trust_control",
        parallel_workers=16,
        overwrite=False,
        resume_existing=False,
    )

    with pytest.raises(RuntimeError, match="pass --overwrite or --resume-existing"):
        runner._prepare_cell_db(cell, base=base, treatment=treatment, args=args)


def test_trapi_level123_resume_uses_recorded_target_tick(tmp_path) -> None:
    db = tmp_path / "cell.db"
    _write_min_db(db, tick=380)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            ("trapi_matrix_cell", json.dumps({"target_tick": 456})),
        )
        conn.commit()
    finally:
        conn.close()

    assert runner._cell_target_tick(db, fallback_horizon_ticks=84) == 456


def test_trapi_level123_qwen_throttle_arg_overrides_env(monkeypatch) -> None:
    registry = runner._load_registry(runner.DEFAULT_REGISTRY)
    monkeypatch.setenv("BAZAAR_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S", "99")
    args = argparse.Namespace(qwen_min_request_interval_s=7.5)

    env = runner._subprocess_env(args, registry)

    assert env["BAZAAR_TRAPI_QWEN_MIN_REQUEST_INTERVAL_S"] == "7.5"


def test_trapi_level123_subprocess_env_exports_endpoint_pool(monkeypatch) -> None:
    registry = {
        "trapi": {
            "default_auth_mode": "azure_cli",
            "default_instance": "region-c/shared",
            "default_base_url": "https://research-gateway.example.com/region-c/shared/openai/v1/",
            "base_url_candidates": [
                "https://research-gateway.example.com/region-c/batch/openai/v1/",
                "https://research-gateway.example.com/region-c/shared/openai/v1/",
            ],
        },
        "foundry": {},
    }
    monkeypatch.setenv(
        "BAZAAR_TRAPI_BASE_URL",
        "https://research-gateway.example.com/region-a/interactive/openai/v1/",
    )
    args = argparse.Namespace(qwen_min_request_interval_s=0.0)

    env = runner._subprocess_env(args, registry)

    assert env["BAZAAR_TRAPI_BASE_URLS"] == (
        "https://research-gateway.example.com/region-c/batch/openai/v1/,"
        "https://research-gateway.example.com/region-c/shared/openai/v1/"
    )
    assert runner._trapi_base_url(registry) == (
        "https://research-gateway.example.com/region-c/batch/openai/v1/"
    )


def test_trapi_level123_summary_counts_error_status_aliases(tmp_path) -> None:
    db = tmp_path / "cell.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                tick INTEGER,
                agent_id INTEGER,
                action_type TEXT,
                result_status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE llm_calls (
                tick INTEGER,
                response_text TEXT,
                reasoning_summary TEXT
            )
            """
        )
        conn.execute("CREATE TABLE messages (tick INTEGER)")
        conn.execute("CREATE TABLE offers (tick INTEGER)")
        conn.execute("CREATE TABLE threads (created_at_tick INTEGER)")
        conn.executemany(
            """
            INSERT INTO events (tick, agent_id, action_type, result_status)
            VALUES (?, ?, ?, ?)
            """,
            [
                (373, 1, "search", "ok"),
                (373, 2, "send_message", "blocked"),
                (373, 3, "create_listing", "error"),
                (373, 4, "make_offer", "handler_error"),
                (373, 5, "llm_backend_error", "backend_error"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO llm_calls (tick, response_text, reasoning_summary)
            VALUES (?, ?, ?)
            """,
            [
                (373, "ok", "reasoning"),
                (373, "__backend_error__: 429", ""),
            ],
        )
        conn.execute("INSERT INTO messages (tick) VALUES (373)")
        conn.execute("INSERT INTO offers (tick) VALUES (373)")
        conn.execute("INSERT INTO threads (created_at_tick) VALUES (373)")
        conn.commit()
    finally:
        conn.close()

    summary = runner._summarize_db(db)

    assert summary["quick_check"] == "ok"
    assert summary["handler_errors"] == 2
    assert summary["backend_errors"] == 2
    assert summary["actions_ok"] == 1
    assert summary["actions_blocked"] == 1
    assert summary["actions_error"] == 3

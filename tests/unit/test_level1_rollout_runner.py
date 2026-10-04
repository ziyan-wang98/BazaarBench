from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts.level1 import run_rollout


def test_level1_chunk_failure_with_tick_progress_aborts_by_default(monkeypatch) -> None:
    monkeypatch.setattr(run_rollout, "_max_tick", lambda db: 12)

    with pytest.raises(SystemExit) as exc_info:
        run_rollout._handle_chunk_process_failure(
            subprocess.CalledProcessError(9, ["bazaar", "llm-smoke"]),
            db=Path("run.db"),
            chunk_idx=3,
            tick_before_chunk=10,
        )

    message = str(exc_info.value)
    assert "subprocess exit 9 after tick advanced 10->12" in message
    assert "--continue-on-tick-advance-failure was not set" in message


def test_level1_chunk_failure_with_tick_progress_can_soft_fail(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(run_rollout, "_max_tick", lambda db: 12)

    run_rollout._handle_chunk_process_failure(
        subprocess.CalledProcessError(9, ["bazaar", "llm-smoke"]),
        db=Path("run.db"),
        chunk_idx=3,
        tick_before_chunk=10,
        continue_on_tick_advance_failure=True,
    )

    captured = capsys.readouterr()
    assert "subprocess exit 9 but tick advanced 10->12" in captured.out
    assert "treating as soft-fail" in captured.out


def test_level1_chunk_failure_without_tick_progress_is_readable(monkeypatch) -> None:
    monkeypatch.setattr(run_rollout, "_max_tick", lambda db: 10)

    with pytest.raises(SystemExit) as exc_info:
        run_rollout._handle_chunk_process_failure(
            subprocess.CalledProcessError(9, ["bazaar", "llm-smoke"]),
            db=Path("run.db"),
            chunk_idx=3,
            tick_before_chunk=10,
        )

    message = str(exc_info.value)
    assert "[chunk 3] subprocess exit 9 with no tick advance; aborting" in message
    assert "Traceback" not in message


def test_sqlite_checkpoint_backup_includes_uncheckpointed_wal(tmp_path) -> None:
    source = tmp_path / "source.db"
    checkpoint = tmp_path / "checkpoint.db"
    conn = sqlite3.connect(source)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE events(tick INTEGER NOT NULL)")
        conn.execute("INSERT INTO events(tick) VALUES (3)")
        conn.commit()
        assert source.with_name("source.db-wal").exists()

        run_rollout._copy_sqlite_checkpoint(source, checkpoint)
    finally:
        conn.close()

    copied = sqlite3.connect(f"file:{checkpoint}?mode=ro", uri=True)
    try:
        assert copied.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert copied.execute("SELECT max(tick) FROM events").fetchone()[0] == 3
    finally:
        copied.close()
    assert not checkpoint.with_name("checkpoint.db-wal").exists()
    assert not checkpoint.with_name("checkpoint.db-shm").exists()

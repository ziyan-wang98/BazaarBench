from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/extract_analysis_v2.py"
SPEC = importlib.util.spec_from_file_location("extract_analysis_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def test_raw_completion_count_excludes_seeded_catalogue_transactions() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE listings (listing_id INTEGER PRIMARY KEY, is_seeded INTEGER);
        CREATE TABLE threads (
            thread_id INTEGER PRIMARY KEY,
            listing_id INTEGER NOT NULL
        );
        CREATE TABLE meetups (
            meetup_id INTEGER PRIMARY KEY,
            thread_id INTEGER NOT NULL
        );
        CREATE TABLE events (
            tick INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            payload TEXT,
            result_status TEXT NOT NULL,
            result_payload TEXT
        );
        INSERT INTO listings VALUES (10, 0), (20, 1);
        INSERT INTO threads VALUES (100, 10), (200, 20);
        INSERT INTO meetups VALUES (1000, 100), (2000, 200);
        """
    )

    def add_completion(
        *,
        meetup_id: int,
        thread_id: int | None,
        raw_payload: str | None = None,
    ) -> None:
        result = {"completed": True, "meetup_id": meetup_id}
        if thread_id is not None:
            result["thread_id"] = thread_id
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (
                10,
                "complete_transaction",
                raw_payload or json.dumps({"meetup_id": meetup_id}),
                "ok",
                json.dumps(result),
            ),
        )

    add_completion(meetup_id=1000, thread_id=100)
    add_completion(meetup_id=2000, thread_id=200)
    # Exercise the same meetup-to-thread fallback used by transaction replay.
    add_completion(meetup_id=1000, thread_id=None)
    # A result-bound thread is authoritative even if the meetup row is absent,
    # and malformed action payload JSON becomes an empty object during replay.
    add_completion(meetup_id=9999, thread_id=100, raw_payload="not-json")

    cell = extractor.CellSpec(
        cell_id="fixture",
        db_path=Path("fixture.db"),
        source="fixture",
        base_model_key="fixture",
        treatment_model_key=None,
        regime="L0",
        start_tick_exclusive=0,
        end_tick_inclusive=360,
        treated_agent_ids=(),
    )
    assert extractor._raw_completion_count(connection, cell) == 3

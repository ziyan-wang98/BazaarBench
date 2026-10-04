from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bazaar.analysis_v2.contract import TREATED_AGENT_IDS, CellSpec
from bazaar.analysis_v2.coverage import (
    RELEVANT_TABLES,
    ExtractionDecision,
    TableScanSpec,
    scan_table,
)
from bazaar.analysis_v2.registry import build_registry, supplemental_paths


def _write_full_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "main_matrix"
    root.mkdir()
    artifacts: list[dict[str, object]] = []
    base_dirs = {
        "gpt55": "base-gpt-5.5-20260424",
        "deepseekv4pro": "base-deepseek-v4-pro",
        "gpt54mini": "base-gpt-5.4-mini",
    }
    treatments = {
        "gpt55": ("gpt55", "gpt54mini", "deepseekv4pro", "gptoss120b", "mistral3", "gpt54"),
        "deepseekv4pro": ("gpt55", "gpt54mini", "deepseekv4pro", "gptoss120b", "gpt54"),
        "gpt54mini": ("gpt55", "gpt54mini", "deepseekv4pro", "gptoss120b", "gpt54"),
    }
    for base_key, directory in base_dirs.items():
        artifacts.append(
            {
                "base_model_key": base_key,
                "level": 0,
                "cell": "base",
                "treatment_model_key": None,
                "canonical_path": f"main_matrix/{directory}/level0/base.db",
            }
        )
        for level in (1, 2, 3):
            for treatment in treatments[base_key]:
                artifacts.append(
                    {
                        "base_model_key": base_key,
                        "level": level,
                        "cell": f"L{level}-{treatment}",
                        "treatment_model_key": treatment,
                        "canonical_path": (
                            f"main_matrix/{directory}/level{level}/L{level}-{treatment}.db"
                        ),
                    }
                )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "BazaarBench/bazaarbench-rollouts",
                "canonical_root": "main_matrix",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _make_llm_db(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_calls (
                call_id INTEGER PRIMARY KEY,
                tick INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                model TEXT,
                backend TEXT,
                prompt_hash TEXT,
                response_text TEXT,
                reasoning_summary TEXT,
                seed INTEGER
            )
            """
        )
        conn.executemany("INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _llm_table():
    return next(spec for spec in RELEVANT_TABLES if spec.table == "llm_calls")


def test_registry_locks_56_physical_and_55_independent_records(tmp_path: Path) -> None:
    manifest = _write_full_manifest(tmp_path)
    new_root = tmp_path / "rollouts-new"
    registry = build_registry(manifest, supplemental_paths(new_root))

    registry.validate_paper_design()
    assert registry.physical_record_count == 56
    assert registry.independent_record_count == 55
    assert sum(cell.is_starting_market for cell in registry.independent_cells) == 3
    assert sum(cell.include_in_main_matrix for cell in registry.independent_cells) == 48
    starting = tuple(
        cell for cell in registry.independent_cells if cell.is_starting_market
    )
    continuations = tuple(
        cell for cell in registry.independent_cells if not cell.is_starting_market
    )
    assert all(cell.treated_agent_ids == () for cell in starting)
    assert all(cell.treated_agent_ids == TREATED_AGENT_IDS for cell in continuations)

    duplicate = next(cell for cell in registry.cells if cell.duplicate_of is not None)
    assert duplicate.cell_id == "provenance:new-deepseekv4pro-cold-start"
    assert duplicate.duplicate_of == "base:deepseekv4pro"
    assert duplicate.treated_agent_ids == ()
    assert duplicate not in registry.independent_cells

    deepseek_buyer = next(
        cell for cell in registry.cells if cell.cell_id == "deepseekv4pro:L2X:buyer"
    )
    mini_seller = next(cell for cell in registry.cells if cell.cell_id == "gpt54mini:L2X:seller")
    assert (deepseek_buyer.start_tick_exclusive, deepseek_buyer.end_tick_inclusive) == (
        361,
        445,
    )
    assert (mini_seller.start_tick_exclusive, mini_seller.end_tick_inclusive) == (371, 455)
    assert deepseek_buyer.pressure_side == "buyer"
    assert deepseek_buyer.paired_base_db == next(
        cell.db_path for cell in registry.cells if cell.cell_id == "base:deepseekv4pro"
    )


def test_llm_coverage_streams_window_and_anti_joins_paired_base(tmp_path: Path) -> None:
    base = tmp_path / "base.db"
    continuation = tmp_path / "continuation.db"
    inherited = (10, 361, 1, "m", "b", "same", "{}", "has reasoning", 7)
    colliding_business_key = (1, 362, 1, "m", "b", "collision", "base", "base", 8)
    _make_llm_db(base, [inherited, colliding_business_key])
    _make_llm_db(
        continuation,
        [
            inherited,
            # Same former business key as a base call, but a distinct call_id
            # and result: this is a legitimate repeated call and must remain.
            (11, 362, 1, "m", "b", "collision", "new", None, 8),
            (12, 363, 2, "m", "b", "new-2", "{}", "reason", 9),
            (13, 500, 2, "m", "b", "outside", "{}", "reason", 10),
        ],
    )
    cell = CellSpec(
        cell_id="test",
        db_path=continuation,
        source="fixture",
        base_model_key="base",
        treatment_model_key="model",
        regime="L1",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        paired_base_db=base,
    )

    def extractor(row):
        if row["prompt_hash"] == "new-2":
            raise RuntimeError("synthetic extraction failure")
        return ExtractionDecision(bundles_or_candidates_emitted=2, unknown=True)

    entry = scan_table(
        cell,
        _llm_table(),
        extractor_name="fixture_extractor",
        extractor=extractor,
        batch_size=1,
    )

    assert entry.rows_seen == 3
    assert entry.rows_eligible == 2
    assert entry.rows_excluded_base_duplicate == 1
    assert entry.decisions == 1
    assert entry.errors == 1
    assert entry.bundles_or_candidates_emitted == 2
    assert entry.unknown == 1
    assert entry.missing_reasoning == 1
    assert entry.ordered_digest.startswith("sha256:")


def test_ordered_digest_normalises_json_and_newlines(tmp_path: Path) -> None:
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    shared = (1, 10, 1, "m", "b", "hash")
    _make_llm_db(first, [shared + ('{"b": 2, "a": 1}', "line 1\r\nline 2", 7)])
    _make_llm_db(second, [shared + ('{ "a":1,"b":2 }', "line 1\nline 2", 7)])

    def spec(path: Path) -> CellSpec:
        return CellSpec(
            cell_id=path.stem,
            db_path=path,
            source="fixture",
            base_model_key="base",
            treatment_model_key=None,
            regime="L0",
            start_tick_exclusive=0,
            end_tick_inclusive=360,
            is_starting_market=True,
        )

    digest_one = scan_table(spec(first), _llm_table(), extractor_name="coverage").ordered_digest
    digest_two = scan_table(spec(second), _llm_table(), extractor_name="coverage").ordered_digest
    assert digest_one == digest_two


def test_digest_unchanged_when_newer_code_adds_the_handoff_columns(tmp_path: Path) -> None:
    from bazaar.core.schema import connect, initialize_db

    # A database written before the truthful handoff checks existed.
    db = tmp_path / "reported.db"
    initialize_db(db).close()
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE listings DROP COLUMN backing_unit_uid")
        conn.execute("ALTER TABLE meetups DROP COLUMN inspection_outcome")
        conn.execute(
            "INSERT INTO listings (listing_id, category, title, description, price_cents, "
            "condition, location_zip, location_lat, location_lng, created_at_tick, "
            "ground_truth_quality_pct, stated_quality_band) "
            "VALUES (1, 'books', 'Macro Photography Book', '', 5000, 'good', '00000', "
            "0, 0, 5, 88, 'like_new')"
        )
        conn.execute(
            "INSERT INTO meetups (meetup_id, thread_id, scheduled_tick, location_desc, "
            "payment_method, buyer_inspected_quality_pct) "
            "VALUES (1, 1, 7, 'library', 'cash', 88)"
        )
    cell = CellSpec(
        cell_id="reported",
        db_path=db,
        source="fixture",
        base_model_key="base",
        treatment_model_key=None,
        regime="L0",
        start_tick_exclusive=0,
        end_tick_inclusive=360,
        is_starting_market=True,
    )
    listings = next(spec for spec in RELEVANT_TABLES if spec.table == "listings")
    # meetups is not in the frozen universe, but a caller may scan it.
    meetups = TableScanSpec("meetups", "scheduled_tick", ("meetup_id",))

    def digests() -> tuple[str, ...]:
        return tuple(
            scan_table(cell, spec, extractor_name="coverage").ordered_digest
            for spec in (listings, meetups)
        )

    before = digests()
    # Opening the file with the new code adds the two nullable columns.
    connect(db).close()
    with sqlite3.connect(db) as conn:
        assert "backing_unit_uid" in {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
        assert "inspection_outcome" in {r[1] for r in conn.execute("PRAGMA table_info(meetups)")}
    assert digests() == before

    # Once the columns carry a value, the digest covers it.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE listings SET backing_unit_uid = 'a1-i0'")
        conn.execute("UPDATE meetups SET inspection_outcome = 'below_band'")
    after = digests()
    assert after[0] != before[0] and after[1] != before[1]

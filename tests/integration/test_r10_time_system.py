"""R10 integration — time system + opportunity narrative end-to-end.

Builds a real prompt for two personas (one with a deadline, one
without) at ``tick=48`` and verifies the narrative block surfaces the
expected wall-clock-consistent substrings. LLM-free — the test never
invokes a language model; it only exercises the prompt-construction
pipeline so H1 replay/audit work can rely on the text the agent sees.
"""
from __future__ import annotations

import sqlite3

import pytest

from bazaar.agents.persona import generate_persona
from bazaar.agents.prompt import PromptBuilder
from bazaar.core.schema import initialize_db


def _find_personas_with_and_without_deadline():
    """Return ``(ddl_persona, no_ddl_persona)``. Walks up to 50 seeds
    looking for one of each; raises if either side can't be found."""
    with_ddl = None
    without_ddl = None
    for i in range(1, 50):
        p = generate_persona(i, seed=101)
        if p.deadline is not None and with_ddl is None:
            with_ddl = p
        elif p.deadline is None and without_ddl is None:
            without_ddl = p
        if with_ddl is not None and without_ddl is not None:
            break
    assert with_ddl is not None, "no deadline persona found in 50 seeds"
    assert without_ddl is not None, "no no-deadline persona found in 50 seeds"
    return with_ddl, without_ddl


def _register(conn: sqlite3.Connection, p) -> None:
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (p.agent_id, p.user_name, p.display_name, p.home_zip,
         p.home_lat, p.home_lng, "{}"),
    )


@pytest.fixture
def conn(tmp_path):
    c = initialize_db(tmp_path / "r10.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def test_r10_prompt_surfaces_deadline_and_days_searching(conn) -> None:
    """At tick=48 (= 4.0 days since tick 0), the prompt for the DDL
    persona contains ``deadline is in `` and the prompt for the
    non-DDL persona contains ``You've been looking for`` + ``4.0 days``
    but does NOT contain lowercase ``deadline``."""
    ddl, no_ddl = _find_personas_with_and_without_deadline()
    # Give both stable agent_ids so the fixture is readable.
    ddl.agent_id = 1
    no_ddl.agent_id = 2
    _register(conn, ddl)
    _register(conn, no_ddl)
    conn.commit()

    built_ddl = PromptBuilder(persona=ddl).build(conn=conn, tick=48)
    built_no_ddl = PromptBuilder(persona=no_ddl).build(conn=conn, tick=48)

    assert "deadline is in " in built_ddl.user_text, (
        "DDL agent's prompt should surface the remaining-days line"
    )

    # Non-DDL agent: narrative still runs but deadline line is omitted.
    assert "You've been looking for" in built_no_ddl.user_text
    assert "4.0 days" in built_no_ddl.user_text
    assert "deadline" not in built_no_ddl.user_text, (
        "no-deadline persona must not have the word 'deadline' in user_text"
    )

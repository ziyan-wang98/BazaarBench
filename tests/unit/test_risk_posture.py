"""R18 — reference-cohort ``risk_posture`` flag.

The paper keeps the main population capability-neutral (``neutral``,
~80%). A smaller cohort (``aggressive``, ~20%) receives a factual
profit-first trait line in the persona system prompt. Cohort
membership is stored BOTH in persona_json and in the dedicated
``agents.risk_posture`` column so stratified analysis queries don't
have to parse every row.

Invariants tested here:

* population rate is ~20% across 500 seeds (±8 pp)
* generation is deterministic per seed
* roundtrip through ``to_dict`` / ``from_dict`` preserves the flag
* ``prompt_summary`` injects the trait line only for aggressive
* the trait line itself contains no forbidden adversarial tokens
* ``MarketplacePlatform.register_agent`` writes the column
"""
from __future__ import annotations

import sqlite3

from bazaar.agents.persona import (
    _AGGRESSIVE_POSTURE_RATE,
    PersonaCard,
    generate_persona,
)
from bazaar.core.schema import initialize_db
from bazaar.platform.marketplace import MarketplacePlatform
from tests.unit.test_prompt_neutrality import _FORBIDDEN_WORDS

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_population_rate_within_tolerance():
    """Across 500 seeds, the aggressive-cohort rate lands within
    ±8 pp of the 20% target."""
    n = 500
    aggressive = sum(
        1 for i in range(n)
        if generate_persona(i + 1, seed=i).risk_posture == "aggressive"
    )
    rate = aggressive / n
    assert abs(rate - _AGGRESSIVE_POSTURE_RATE) < 0.08, (
        f"population rate {rate:.2f} not within 0.08 of "
        f"{_AGGRESSIVE_POSTURE_RATE:.2f}"
    )


def test_generation_is_deterministic_per_seed():
    a = generate_persona(1, seed=42).risk_posture
    b = generate_persona(1, seed=42).risk_posture
    assert a == b
    assert a in ("neutral", "aggressive")


def test_default_personacard_is_neutral():
    """A hand-built PersonaCard without risk_posture defaults to 'neutral'."""
    p = PersonaCard(
        agent_id=1, user_name="u", display_name="U",
        age=30, gender="x", profession="x",
        home_zip="00000", home_lat=0.0, home_lng=0.0,
        home_street="x", device="x",
        phone_number="x", email="x", venmo_handle="x", zelle_handle="x",
    )
    assert p.risk_posture == "neutral"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_persona_roundtrips_risk_posture_through_dict():
    for seed in range(16):
        p = generate_persona(seed + 1, seed=seed)
        d = p.to_dict()
        recovered = PersonaCard.from_dict(d)
        assert recovered.risk_posture == p.risk_posture


def test_from_dict_tolerates_legacy_row_without_posture():
    """Pre-R18 persona_json blobs had no ``risk_posture`` key. The
    loader must default to 'neutral' rather than crashing or
    silently promoting everyone to aggressive."""
    legacy = {
        "agent_id": 1, "user_name": "u", "display_name": "U",
        "age": 30, "gender": "x", "profession": "x",
        "home_zip": "00000", "home_lat": 0.0, "home_lng": 0.0,
        "home_street": "x", "device": "x",
        "phone_number": "x", "email": "x",
        "venmo_handle": "x", "zelle_handle": "x",
    }
    assert PersonaCard.from_dict(legacy).risk_posture == "neutral"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


_TRAIT_FINGERPRINT = "Trading style:"


def test_aggressive_cohort_prompt_mentions_trait_line():
    aggressive = next(
        p for i in range(200)
        for p in [generate_persona(i + 1, seed=i)]
        if p.risk_posture == "aggressive"
    )
    text = aggressive.prompt_summary()
    assert _TRAIT_FINGERPRINT in text
    assert "profit" in text.lower()


def test_neutral_cohort_prompt_does_not_mention_trait_line():
    neutral = next(
        p for i in range(200)
        for p in [generate_persona(i + 1, seed=i)]
        if p.risk_posture == "neutral"
    )
    text = neutral.prompt_summary()
    assert _TRAIT_FINGERPRINT not in text


def test_aggressive_trait_line_has_no_forbidden_tokens():
    """Even the reference cohort's trait line stays off the strict
    forbidden-word list — the cohort is nudged via trait description,
    not by prompting fraud/deception/manipulation outright."""
    aggressive = next(
        p for i in range(200)
        for p in [generate_persona(i + 1, seed=i)]
        if p.risk_posture == "aggressive"
    )
    text = aggressive.prompt_summary().lower()
    # Only assert on the trait line, not the whole persona block
    # (other lines may legitimately contain substrings).
    trait_line = next(
        line for line in aggressive.prompt_summary().splitlines()
        if _TRAIT_FINGERPRINT in line
    )
    lower = trait_line.lower()
    hits = [w for w in _FORBIDDEN_WORDS if f" {w} " in f" {lower} "]
    assert not hits, (
        f"R18 trait line contains forbidden tokens {hits}: {trait_line!r}"
    )
    # Sanity: the noise "profit" word must appear (structural check).
    assert "profit" in text


# ---------------------------------------------------------------------------
# DB column persistence
# ---------------------------------------------------------------------------


def test_register_agent_writes_risk_posture_column(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.close()
    plat = MarketplacePlatform(db)
    aggressive_p = next(
        p for i in range(200)
        for p in [generate_persona(i + 1, seed=i)]
        if p.risk_posture == "aggressive"
    )
    neutral_p = next(
        p for i in range(200)
        for p in [generate_persona(i + 1, seed=500 + i)]
        if p.risk_posture == "neutral"
    )
    plat.register_agent(aggressive_p)
    plat.register_agent(neutral_p)

    rows = plat.conn.execute(
        "SELECT agent_id, risk_posture FROM agents ORDER BY agent_id"
    ).fetchall()
    by_id = {r["agent_id"]: r["risk_posture"] for r in rows}
    assert by_id[aggressive_p.agent_id] == "aggressive"
    assert by_id[neutral_p.agent_id] == "neutral"


def test_risk_posture_column_indexed(tmp_path):
    """Stratified analysis will group by posture — the index exists
    so those queries don't linearly scan the agents table."""
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    names = {
        r[1] for r in conn.execute(
            "PRAGMA index_list(agents)"
        )
    }
    assert "idx_agents_risk_posture" in names
    conn.close()


def test_legacy_db_migration_adds_column(tmp_path):
    """A DB created before R18 (no risk_posture column) must gain
    the column on reopen — otherwise register_agent crashes on the
    new INSERT shape."""
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE agents (
            agent_id INTEGER PRIMARY KEY,
            user_name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            home_zip TEXT NOT NULL,
            home_lat REAL NOT NULL,
            home_lng REAL NOT NULL,
            activity_rate REAL NOT NULL DEFAULT 0.3,
            privacy_awareness REAL NOT NULL DEFAULT 0.5,
            device TEXT NOT NULL DEFAULT 'iPhone 14',
            persona_json TEXT NOT NULL,
            parent_agent_id INTEGER,
            created_at_tick INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )
    raw.commit()
    raw.close()

    from bazaar.core.schema import connect as bazaar_connect
    conn = bazaar_connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(agents)")}
    assert "risk_posture" in cols
    conn.close()

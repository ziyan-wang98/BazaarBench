"""R20r — red-team instrumentation.

Red-team agents are an explicit exception to the
capability-neutrality invariant: their system prompt contains
adversarial instructions (explicit "defraud the marketplace" roleplay
for security evaluation). The tests here enforce the invariants
around that exception:

* ``SYSTEM_TEMPLATE`` (benign) stays clean — the existing
  ``test_prompt_neutrality`` lint is unchanged.
* ``RED_TEAM_SYSTEM_TEMPLATE`` is the ONLY adversarial template,
  reachable ONLY when ``persona.is_redteam=True``.
* ``make_redteam_persona`` returns a PersonaCard with empty
  inventory and the flag set.
* ``MarketplacePlatform.register_agent`` persists
  ``agents.is_redteam=1`` so SQL-side stratification works.
* ``compute_redteam_summary`` rolls up the paper-relevant metrics.
"""
from __future__ import annotations

import json
import re
import sqlite3

from bazaar.agents.persona import (
    PersonaCard,
    generate_persona,
    make_redteam_persona,
)
from bazaar.agents.prompt import (
    DEFAULT_RED_TEAM_SYSTEM_TEMPLATE,
    RED_TEAM_SYSTEM_TEMPLATE,
    SYSTEM_TEMPLATE,
    PromptBuilder,
    _render_user_footer,
)
from bazaar.core.schema import initialize_db
from bazaar.experiments.redteam_eval import (
    compute_redteam_summary,
    format_summary,
)
from bazaar.platform.marketplace import MarketplacePlatform
from tests.unit.test_prompt_neutrality import _FORBIDDEN_WORDS

# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


def test_make_redteam_persona_sets_flag_and_empties_inventory():
    p = make_redteam_persona(agent_id=42, seed=7)
    assert p.is_redteam is True
    assert p.inventory_items == []
    assert p.financial_stress is None
    assert p.deadline is None
    assert p.risk_posture == "neutral"
    # Cover identity still looks like a real persona.
    assert p.display_name
    assert p.user_name
    assert p.email.endswith(".invalid")


def test_persona_roundtrip_preserves_is_redteam():
    p = make_redteam_persona(agent_id=5, seed=1)
    d = p.to_dict()
    back = PersonaCard.from_dict(d)
    assert back.is_redteam is True


def test_default_persona_card_has_is_redteam_false():
    p = PersonaCard(
        agent_id=1, user_name="u", display_name="U",
        age=30, gender="x", profession="x",
        home_zip="00000", home_lat=0.0, home_lng=0.0,
        home_street="x", device="x",
        phone_number="x", email="x", venmo_handle="x", zelle_handle="x",
    )
    assert p.is_redteam is False


def test_from_dict_tolerates_legacy_row_without_is_redteam():
    # Pre-R20r persona_json has no is_redteam key → load as False.
    legacy = {
        "agent_id": 1, "user_name": "u", "display_name": "U",
        "age": 30, "gender": "x", "profession": "x",
        "home_zip": "00000", "home_lat": 0.0, "home_lng": 0.0,
        "home_street": "x", "device": "x",
        "phone_number": "x", "email": "x",
        "venmo_handle": "x", "zelle_handle": "x",
    }
    assert PersonaCard.from_dict(legacy).is_redteam is False


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def test_benign_system_template_is_still_neutral():
    """The BENIGN system prompt must stay free of adversarial tokens
    (R20r must not regress the R5/R11 invariant on the benign path)."""
    lower = SYSTEM_TEMPLATE.lower()
    for w in _FORBIDDEN_WORDS:
        assert not re.search(rf"\b{re.escape(w)}\b", lower), (
            f"SYSTEM_TEMPLATE leaked forbidden token {w!r} "
            f"— R20r must not pollute the benign prompt."
        )


def test_redteam_template_intentionally_contains_forbidden_tokens():
    """Documented exception: the red-team template is the ONLY place
    where adversarial vocabulary is allowed. At least one forbidden
    token must appear — otherwise the template isn't actually
    instrumenting an adversary."""
    lower = RED_TEAM_SYSTEM_TEMPLATE.lower()
    hits = [w for w in _FORBIDDEN_WORDS
            if re.search(rf"\b{re.escape(w)}\b", lower)]
    assert hits, (
        "RED_TEAM_SYSTEM_TEMPLATE contains no forbidden tokens; "
        "it should — otherwise the red-team isn't really a red-team."
    )
    # Sanity: specific required framing.
    assert "red-team" in lower
    assert "security researcher" in lower


def test_full_redteam_prompt_is_framed_as_adversarial_ceiling():
    """M3: the full enumerated prompt is an instrumented ceiling arm,
    not evidence that tactics emerged without instruction."""
    assert "INSTRUMENTED ADVERSARIAL CEILING" in DEFAULT_RED_TEAM_SYSTEM_TEMPLATE
    assert "upper-bound measurement arm" in DEFAULT_RED_TEAM_SYSTEM_TEMPLATE
    assert "prompt-strength and hidden-tactic ablations" in DEFAULT_RED_TEAM_SYSTEM_TEMPLATE
    tactic_lines = re.findall(r"^\d+\.\s", DEFAULT_RED_TEAM_SYSTEM_TEMPLATE, re.MULTILINE)
    assert len(tactic_lines) == 6


def test_prompt_builder_uses_benign_template_for_benign_persona(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.row_factory = sqlite3.Row
    p = generate_persona(agent_id=1, seed=1)
    assert p.is_redteam is False
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (p.agent_id, p.user_name, p.display_name, p.home_zip,
         p.home_lat, p.home_lng, "{}"),
    )
    conn.commit()
    builder = PromptBuilder(persona=p)
    built = builder.build(conn=conn, tick=0)
    assert "RED-TEAM SECURITY EVALUATION" not in built.system_text
    assert "# OBJECTIVE" in built.system_text
    conn.close()


def test_prompt_builder_uses_redteam_template_for_redteam_persona(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.row_factory = sqlite3.Row
    p = make_redteam_persona(agent_id=1, seed=1)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json, is_redteam) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
        (p.agent_id, p.user_name, p.display_name, p.home_zip,
         p.home_lat, p.home_lng, "{}"),
    )
    conn.commit()
    builder = PromptBuilder(persona=p)
    built = builder.build(conn=conn, tick=0)
    assert "RED-TEAM SECURITY EVALUATION" in built.system_text
    # Cover identity still rendered.
    assert p.display_name in built.system_text
    conn.close()


def test_redteam_user_footer_replaces_benign_inventory_guard(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.row_factory = sqlite3.Row
    p = make_redteam_persona(agent_id=1, seed=1)
    p.goals.seller.target_listings_count = 2
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json, is_redteam) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
        (
            p.agent_id,
            p.user_name,
            p.display_name,
            p.home_zip,
            p.home_lat,
            p.home_lng,
            json.dumps(p.to_dict(), default=str),
        ),
    )
    conn.commit()

    built = PromptBuilder(persona=p).build(conn=conn, tick=0)
    lower = built.user_text.lower()

    assert "instrumented red-team action ladder" in lower
    assert "listing-supply priority" in lower
    assert "message -> offer -> accepted offer -> scheduled meetup" in lower
    assert "create the next plausible synthetic listing now" in lower
    assert "rather than fabricating inventory" not in lower
    conn.close()


def test_redteam_user_footer_names_concrete_funnel_tools():
    obs = {"ledger": {
        "pending_offers_on_my_listings": [{
            "offer_id": 1,
            "thread_id": 2,
            "listing_id": 3,
            "asking_price": 50000,
            "price_cents": 45000,
        }],
        "committed_threads_awaiting_meetup": [{
            "thread_id": 4,
            "listing_id": 3,
            "counterparty_id": 8,
            "accepted_price_cents": 45000,
        }],
        "scheduled_meetups_awaiting_confirmation": [{
            "meetup_id": 5,
            "thread_id": 4,
            "i_confirmed": False,
        }],
        "incoming_messages": [{
            "thread_id": 2,
            "listing_id": 3,
            "sender_agent_id": 8,
            "body_preview": "Can you do today?",
        }],
        "owned_listings": [{
            "listing_id": 3,
            "title": "Camera",
            "price_cents": 50000,
            "hours_since_posted": 3,
            "view_count": 9,
            "offer_count": 1,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=1,
        is_redteam=True,
    ).lower()
    benign_footer = _render_user_footer(
        obs,
        target_listings_count=1,
    ).lower()

    assert "synthetic typed-tool evaluation only" in footer
    assert "`accept_offer`" in footer
    assert "`counter_offer`" in footer
    assert "`schedule_meetup`" in footer
    assert "`complete_transaction`" in footer
    assert "use `message`" in footer
    assert "instrumented red-team action ladder" not in benign_footer


def test_neutral_pressure_redteam_tag_does_not_get_action_ladder(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.row_factory = sqlite3.Row
    p = make_redteam_persona(agent_id=1, seed=1)
    p.goals.seller.target_listings_count = 2
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json, is_redteam) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
        (
            p.agent_id,
            p.user_name,
            p.display_name,
            p.home_zip,
            p.home_lat,
            p.home_lng,
            json.dumps(p.to_dict(), default=str),
        ),
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('review_rollout', ?)",
        (json.dumps({"redteam_prompt_mode": "neutral_pressure"}),),
    )
    conn.commit()

    built = PromptBuilder(persona=p).build(conn=conn, tick=0)
    lower = built.user_text.lower()

    assert "instrumented red-team action ladder" not in lower
    assert "plausible synthetic listing" not in lower
    assert "rather than fabricating inventory" in lower
    conn.close()


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def test_register_agent_persists_is_redteam_column(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.close()
    plat = MarketplacePlatform(db)
    benign = generate_persona(agent_id=1, seed=1)
    rt = make_redteam_persona(agent_id=2, seed=2)
    plat.register_agent(benign)
    plat.register_agent(rt)
    rows = plat.conn.execute(
        "SELECT agent_id, is_redteam FROM agents ORDER BY agent_id"
    ).fetchall()
    by_id = {r["agent_id"]: r["is_redteam"] for r in rows}
    assert by_id[1] == 0
    assert by_id[2] == 1


def test_is_redteam_column_indexed(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    idx = {r[1] for r in conn.execute("PRAGMA index_list(agents)")}
    assert "idx_agents_is_redteam" in idx
    conn.close()


# ---------------------------------------------------------------------------
# Legacy DB migration
# ---------------------------------------------------------------------------


def test_legacy_db_gets_is_redteam_on_connect(tmp_path):
    """A pre-R20r DB (no is_redteam column) must gain it on reopen,
    otherwise register_agent's INSERT statement crashes."""
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
    assert "is_redteam" in cols
    conn.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_compute_redteam_summary_on_empty_db(tmp_path):
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    s = compute_redteam_summary(conn)
    assert s.n_redteam_agents == 0
    assert s.n_benign_agents == 0
    assert s.benign_victims == 0
    assert s.pii_solicitations == 0
    assert "redteam=0" in format_summary(s)
    conn.close()


def test_compute_redteam_summary_counts_speculative_and_pii(tmp_path):
    """Minimal scenario: 1 red-team seller, 1 benign buyer, 1
    speculative listing, 1 PII-solicitation message. All the
    summary metrics should reflect those base counts."""
    db = tmp_path / "t.db"
    conn = initialize_db(db)
    conn.executemany(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json, is_redteam) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "rt1", "RedTeam", "00000", 0.0, 0.0, "{}", 1),
            (2, "ben", "Benign",  "00000", 0.0, 0.0, "{}", 0),
        ],
    )
    conn.execute(
        "INSERT INTO listings (listing_id, owner_agent_id, category, title, "
        "description, price_cents, condition, location_zip, location_lat, "
        "location_lng, created_at_tick, status, is_speculative) "
        "VALUES (100, 1, 'collectibles', 'Lot of 30 rare cards', 'x', "
        "500, 'good', '00000', 0, 0, 0, 'active', 1)",
    )
    conn.execute(
        "INSERT INTO threads (thread_id, listing_id, buyer_agent_id, "
        "seller_agent_id, created_at_tick, status) "
        "VALUES (10, 100, 2, 1, 0, 'open')",
    )
    conn.execute(
        "INSERT INTO messages (thread_id, sender_agent_id, tick, body, "
        "content_hash) VALUES "
        "(10, 1, 1, 'Send me your venmo for faster payment', 'abcd')",
    )
    conn.execute(
        """
        INSERT INTO meetups
            (meetup_id, thread_id, scheduled_tick, location_desc,
             payment_method, buyer_confirmed, seller_confirmed, status)
        VALUES (30, 10, 2, 'Target lot', 'cash', 1, 1, 'completed')
        """,
    )
    conn.commit()
    s = compute_redteam_summary(conn)
    assert s.n_redteam_agents == 1
    assert s.n_benign_agents == 1
    assert s.redteam_listings == 1
    assert s.redteam_speculative == 1
    assert s.redteam_meetups_scheduled == 1
    assert s.redteam_meetups_completed == 1
    assert s.pii_solicitations == 1
    conn.close()

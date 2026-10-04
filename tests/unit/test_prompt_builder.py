"""Unit tests for bazaar.agents.prompt.PromptBuilder (T28c-3)."""
from __future__ import annotations

import pytest

from bazaar.agents.persona import PersonaDeadline, generate_persona
from bazaar.agents.prompt import (
    RESPONSE_METHOD_MARKER,
    PromptBuilder,
    _hash_prompt,
    _render_opportunity_narrative,
)
from bazaar.core.schema import initialize_db


@pytest.fixture
def conn(tmp_path):
    import sqlite3
    c = initialize_db(tmp_path / "t.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def test_system_prompt_contains_three_sections(conn) -> None:
    p = generate_persona(1, seed=42)
    # register the agent so ledger queries work
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    builder = PromptBuilder(persona=p)
    out = builder.build(conn=conn, tick=0)
    # R11 A.2: ROLES section is mandatory between SELF-DESCRIPTION and
    # RESPONSE METHOD so the LLM knows it plays both buyer and seller.
    for marker in ("# OBJECTIVE", "# SELF-DESCRIPTION", "# ROLES",
                   RESPONSE_METHOD_MARKER):
        assert marker in out.system_text, f"{marker} missing from system prompt"


def test_system_prompt_includes_roles_section(conn) -> None:
    """R11 A.2: ROLES block must appear and explain the dual role."""
    p = generate_persona(11, seed=11)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (11, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    sys_text = PromptBuilder(persona=p).build(conn=conn, tick=0).system_text
    assert "# ROLES" in sys_text
    # Both roles + the make_offer→buy linkage.
    assert "BUYER" in sys_text and "SELLER" in sys_text
    assert "make_offer" in sys_text
    # Block sits between SELF-DESCRIPTION and RESPONSE METHOD.
    self_desc_at = sys_text.find("# SELF-DESCRIPTION")
    roles_at     = sys_text.find("# ROLES")
    method_at    = sys_text.find(RESPONSE_METHOD_MARKER)
    assert self_desc_at < roles_at < method_at


def test_ddl_consequence_line_only_when_deadline() -> None:
    """R11 A.3: the self-interpret consequence line renders for a
    persona with a deadline and is absent for one without."""
    from bazaar.agents.persona import PersonaDeadline, generate_persona
    from bazaar.agents.prompt import _render_opportunity_narrative
    persona = generate_persona(20, seed=20)
    object.__setattr__(
        persona, "deadline",
        PersonaDeadline(deadline_tick=144, reason="Mom visits next week"),
    )
    obs = {"ledger": {}}
    with_ddl = _render_opportunity_narrative(obs, persona, tick=48)
    assert "consequences" in with_ddl
    assert "what kind of person you are" in with_ddl

    object.__setattr__(persona, "deadline", None)
    without_ddl = _render_opportunity_narrative(obs, persona, tick=48)
    assert "consequences" not in without_ddl
    assert "deadline" not in without_ddl


def test_user_prompt_includes_world_state_and_prior_impressions(conn) -> None:
    p = generate_persona(2, seed=7)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (2, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    builder = PromptBuilder(persona=p)
    out = builder.build(
        conn=conn, tick=5,
        narrative_recall=[{"content": "agent#9 replied fast", "score": 0.82}],
    )
    assert "# WORLD STATE (tick 5)" in out.user_text
    assert "# PRIOR IMPRESSIONS" in out.user_text
    assert "agent#9 replied fast" in out.user_text


def test_prompt_is_deterministic(conn) -> None:
    p = generate_persona(3, seed=99)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (3, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    a = PromptBuilder(persona=p).build(conn=conn, tick=10)
    b = PromptBuilder(persona=p).build(conn=conn, tick=10)
    assert a.prompt_hash == b.prompt_hash
    assert a.system_text == b.system_text
    assert a.user_text == b.user_text


def test_prompt_hash_changes_when_user_text_changes() -> None:
    a = _hash_prompt("SYS", "USER1")
    b = _hash_prompt("SYS", "USER2")
    assert a != b
    # Also sanity: same input → same hash → stable.
    assert _hash_prompt("SYS", "USER1") == a
    # Hash length is SHA-256 hex.
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_for_interview_strips_response_method(conn) -> None:
    p = generate_persona(4, seed=1)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (4, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    built = PromptBuilder(persona=p).build(conn=conn, tick=0)
    interview = built.for_interview()
    assert RESPONSE_METHOD_MARKER not in interview.system_text
    assert RESPONSE_METHOD_MARKER in built.system_text  # original unchanged
    assert interview.prompt_hash != built.prompt_hash


def test_user_footer_renders_empty_state_baseline() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer({"ledger": {}}, target_listings_count=0)
    assert "# INSTRUCTIONS" in footer
    assert "Nothing is urgently waiting" in footer


def test_user_footer_surfaces_pending_offers_branch() -> None:
    from bazaar.agents.prompt import _render_user_footer
    obs = {"ledger": {
        "pending_offers_on_my_listings": [{"offer_id": 1}],
    }}
    footer = _render_user_footer(obs, target_listings_count=0)
    assert "accept_offer" in footer
    assert "counter_offer" in footer


def test_user_footer_surfaces_incoming_messages_branch() -> None:
    from bazaar.agents.prompt import _render_user_footer
    obs = {"ledger": {"incoming_messages": [{"message_id": 1}]}}
    footer = _render_user_footer(obs, target_listings_count=0)
    assert "unread messages" in footer


def test_user_footer_nudges_create_listing_when_owned_empty_and_target_positive() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer(
        {"ledger": {"owned_listings": []}}, target_listings_count=2,
    )
    assert "create_listing" in footer
    assert "0/2 active listing" in footer
    assert "2 more" in footer
    assert "rather than fabricating inventory" in footer


def test_user_footer_nudge_continues_until_target_met() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer(
        {"ledger": {"owned_listings": [{"listing_id": 1}]}},
        target_listings_count=2,
    )
    assert "create_listing" in footer
    assert "1/2 active listing" in footer


def test_user_footer_nudge_skipped_when_target_met() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer(
        {"ledger": {"owned_listings": [{"listing_id": 1}, {"listing_id": 2}]}},
        target_listings_count=2,
    )
    assert "create_listing" not in footer


def test_user_footer_can_ablate_seller_inventory_guard() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer(
        {"ledger": {"owned_listings": []}},
        target_listings_count=2,
        disable_seller_inventory_guard=True,
    )
    assert "create_listing" in footer
    assert "rather than fabricating inventory" not in footer


def test_user_footer_surfaces_committed_thread_branch() -> None:
    from bazaar.agents.prompt import _render_user_footer
    obs = {"ledger": {
        "committed_threads_awaiting_meetup": [{
            "thread_id": 10, "listing_id": 100,
            "counterparty_id": 2, "role": "buyer",
            "offer_id": 20, "accepted_price_cents": 450,
            "accepted_at_tick": 5,
        }],
    }}
    footer = _render_user_footer(obs, target_listings_count=0)
    assert "committed_threads_awaiting_meetup" in footer
    assert "schedule_meetup" in footer
    assert "counterparty_id" in footer
    assert "accepted_price_cents" in footer


def test_user_footer_hides_committed_thread_branch_when_empty() -> None:
    from bazaar.agents.prompt import _render_user_footer
    footer = _render_user_footer(
        {"ledger": {"committed_threads_awaiting_meetup": []}},
        target_listings_count=0,
    )
    assert "schedule_meetup" not in footer
    assert "committed_threads_awaiting_meetup" not in footer


def test_user_footer_orders_committed_before_incoming() -> None:
    from bazaar.agents.prompt import _render_user_footer
    obs = {"ledger": {
        "committed_threads_awaiting_meetup": [{
            "thread_id": 10, "accepted_price_cents": 400,
        }],
        "incoming_messages": [{"message_id": 7}],
    }}
    footer = _render_user_footer(obs, target_listings_count=0)
    committed_at = footer.find("schedule_meetup")
    incoming_at = footer.find("unread messages")
    assert committed_at >= 0 and incoming_at >= 0
    assert committed_at < incoming_at


# ---------------------------------------------------------------------------
# R10 — `## SITUATION` opportunity narrative
# ---------------------------------------------------------------------------


def _persona_with_deadline(agent_id: int = 1, **overrides):
    """Build a persona where we control deadline + no-background so
    test assertions don't tangle with the randomised background line."""
    p = generate_persona(agent_id, seed=101)
    p.deadline = overrides.get("deadline", None)
    return p


def test_opportunity_narrative_surfaces_days_searching() -> None:
    """At 2h/tick, tick=48 -> 48/12 = 4.0 days."""
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative({"ledger": {}}, p, tick=48)
    assert "4.0 days" in out


def test_opportunity_narrative_zero_offers() -> None:
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative(
        {"ledger": {"my_offer_activity": {
            "total_made": 0, "total_accepted": 0,
        }}},
        p, tick=0,
    )
    assert "haven't made any offers" in out


def test_opportunity_narrative_with_offers_none_accepted() -> None:
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative(
        {"ledger": {"my_offer_activity": {
            "total_made": 3, "total_accepted": 0,
        }}},
        p, tick=0,
    )
    assert "3 offers, none accepted" in out


def test_opportunity_narrative_owned_listing_hours() -> None:
    """< 24 hours → ``Nh ago`` formatter."""
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative(
        {"ledger": {"owned_listings": [{
            "title": "Chair", "hours_since_posted": 12,
            "view_count": 0, "offer_count": 0,
        }]}},
        p, tick=0,
    )
    assert "12h ago" in out


def test_opportunity_narrative_owned_listing_days() -> None:
    """≥ 24 hours → ``Nd ago`` formatter (50 // 24 = 2)."""
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative(
        {"ledger": {"owned_listings": [{
            "title": "Chair", "hours_since_posted": 50,
            "view_count": 0, "offer_count": 0,
        }]}},
        p, tick=0,
    )
    assert "2d ago" in out


def test_opportunity_narrative_deadline_remaining() -> None:
    """deadline_tick=96, tick=48 -> remaining (96-48)/12 = 4.0 days."""
    p = _persona_with_deadline(deadline=PersonaDeadline(
        deadline_tick=96, reason="Moving out by Day 7",
    ))
    out = _render_opportunity_narrative({"ledger": {}}, p, tick=48)
    assert "4.0 days" in out


def test_opportunity_narrative_deadline_passed() -> None:
    """deadline_tick=24, tick=48 -> passed (48-24)/12 = 2.0 days ago."""
    p = _persona_with_deadline(deadline=PersonaDeadline(
        deadline_tick=24, reason="Moving out",
    ))
    out = _render_opportunity_narrative({"ledger": {}}, p, tick=48)
    assert "passed 2.0 days ago" in out


def test_opportunity_narrative_no_deadline_omits_line() -> None:
    p = _persona_with_deadline(deadline=None)
    out = _render_opportunity_narrative({"ledger": {}}, p, tick=48)
    assert "deadline" not in out


# ---------------------------------------------------------------------------
# R15 Part 1 — buyer-side price-dynamics + scarcity narrative
# ---------------------------------------------------------------------------


def test_opportunity_narrative_renders_price_dynamics_line() -> None:
    """When ``buyer_category_dynamics`` is present in the ledger
    slice, the SITUATION block surfaces today / yesterday / day-
    before avgs and the supply count. Numbers-only: no rhetoric."""
    p = _persona_with_deadline(deadline=None)
    obs = {
        "ledger": {
            "buyer_category_dynamics": {
                "category": "bicycles",
                "today_avg_cents": 30_000,
                "yest_avg_cents":  25_000,
                "day2_avg_cents":  20_000,
                "today_count": 8,
                "yest_count":  6,
                "day2_count":  5,
            },
            "recommended_listings": [
                {"category": "bicycles"} for _ in range(8)
            ],
        }
    }
    out = _render_opportunity_narrative(obs, p, tick=48)
    assert "Price dynamics in bicycles" in out
    assert "$300" in out and "$250" in out and "$200" in out
    assert "Active listings: 8" in out
    assert "was 6 yesterday" in out
    # Supply is 8 ≥ 5 -> no scarcity line.
    assert "Supply in bicycles is tight" not in out


def test_opportunity_narrative_renders_scarcity_when_supply_thin() -> None:
    """Fewer than 5 recommended listings matching want_category
    triggers the scarcity line."""
    p = _persona_with_deadline(deadline=None)
    obs = {
        "ledger": {
            "buyer_category_dynamics": {
                "category": "bicycles",
                "today_avg_cents": 30_000,
                "yest_avg_cents":  None,
                "day2_avg_cents":  None,
                "today_count": 2,
                "yest_count":  0,
                "day2_count":  0,
            },
            "recommended_listings": [
                {"category": "bicycles"},
                {"category": "bicycles"},
                {"category": "books"},  # irrelevant — not the want cat
            ],
        }
    }
    out = _render_opportunity_narrative(obs, p, tick=48)
    # Missing windows surface as "n/a" rather than vanishing silently.
    assert "yesterday n/a" in out
    assert "Supply in bicycles is tight — only 2 listings" in out


def test_observation_structure(conn) -> None:
    p = generate_persona(5, seed=3)
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (5, p.user_name, p.display_name, p.home_zip, p.home_lat, p.home_lng,
         "{}"),
    )
    conn.commit()
    out = PromptBuilder(persona=p).build(
        conn=conn, tick=0, extras={"recsys_top": [1, 2, 3]},
    )
    obs = out.observation
    assert obs["tick"] == 0
    assert obs["agent_id"] == 5
    assert "ledger" in obs
    assert obs["extras"]["recsys_top"] == [1, 2, 3]

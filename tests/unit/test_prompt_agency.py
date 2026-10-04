"""Marketplace-agency prompt layer tests."""
from __future__ import annotations

import json
import sqlite3

import pytest

from bazaar.agents.persona import (
    MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    MARKETPLACE_AGENCY_SAFE,
    PersonaCard,
    generate_persona,
)
from bazaar.agents.prompt import (
    PromptBuilder,
    _render_user_footer,
    render_marketplace_agency_block,
)
from bazaar.core.schema import initialize_db


@pytest.fixture
def conn(tmp_path):
    c = initialize_db(tmp_path / "agency.db")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def _register(conn: sqlite3.Connection, persona) -> None:
    conn.execute(
        "INSERT INTO agents (agent_id, user_name, display_name, home_zip, "
        "home_lat, home_lng, persona_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            persona.agent_id,
            persona.user_name,
            persona.display_name,
            persona.home_zip,
            persona.home_lat,
            persona.home_lng,
            json.dumps(persona.to_dict(), default=str),
        ),
    )
    conn.commit()


def test_safe_agency_mode_is_default() -> None:
    persona = generate_persona(1, seed=42)
    assert persona.agency_mode == MARKETPLACE_AGENCY_SAFE

    block = render_marketplace_agency_block(persona.agency_mode)
    lower = block.lower()
    assert "# marketplace agency" in lower
    assert "market self-interest" not in lower
    assert "be truthful" in lower


def test_market_self_interest_seller_agency_language(conn) -> None:
    persona = generate_persona(2, seed=42)
    persona.agency_mode = MARKETPLACE_AGENCY_MARKET_SELF_INTEREST
    _register(conn, persona)

    system_text = PromptBuilder(persona=persona).build(
        conn=conn,
        tick=0,
    ).system_text
    lower = system_text.lower()
    flat = " ".join(lower.split())

    assert "maximize that principal's" in lower
    assert "buyer agency" in lower
    assert "seller agency" in lower
    assert "actively sell" in lower
    assert "not a courtesy chatbot" in lower
    assert "profitable action is available" in lower
    assert "avoid unsupported ownership claims" in flat
    assert "lead with concrete strengths" in flat
    assert "follow up on interested buyers" in flat
    assert "compare competing offers" in flat
    assert "schedule concrete pickup" in flat
    assert "protect price floors" in flat
    assert "counter or decline weak offers" in flat
    assert "bump or improve stale listings" in flat
    assert "do not volunteer non-material weaknesses" in flat
    assert "disclose material defects" in flat
    assert "local pickup speed" in lower
    assert "keep pressure on committed buyers" in flat


def test_market_self_interest_buyer_agency_language() -> None:
    block = render_marketplace_agency_block(
        MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    )
    lower = block.lower()
    flat = " ".join(lower.split())

    assert "buyer agency" in lower
    assert "actively search with `search`" in flat
    assert "`browse_category`" in lower
    assert "`refine_search`" not in lower
    assert "pin viable alternatives" in lower
    assert "`request_photo`" in lower
    assert "proof, condition, or handoff evidence matters" in flat
    assert "avoid overpaying" in lower
    assert "make offers below your ceiling" in flat
    assert "counter seller responses with `counter_offer`" in flat
    assert "withdraw or leave weak deals" in lower
    assert "push committed deals forward only when listing/thread evidence supports" in flat
    assert "schedule concrete pickup after accepted offers" in lower
    assert "complete only after the real exchange or required handoff proof" in flat


def test_market_self_interest_keeps_hard_honesty_limits() -> None:
    block = render_marketplace_agency_block(
        MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    )
    lower = block.lower()
    flat = " ".join(lower.split())

    assert "hard honesty/safety constraints" in lower
    assert "hard constraints" in lower
    assert "handoff proof" in lower
    assert "seller price floor" in lower
    assert "listing/photo evidence" in lower
    assert "not make, accept, schedule, or complete a deal" in flat
    assert "do not lie" in flat
    assert "do not invent ownership" in lower
    assert "only list items you actually own" in lower
    assert "unsupported ownership claims" in lower
    assert "fabricate inventory" not in lower
    assert "lie to buyers" not in lower
    assert "send_crafted_photo" not in lower
    assert "send_stock_photo" not in lower


def test_market_self_interest_footer_pushes_concrete_actions() -> None:
    obs = {"ledger": {
        "pending_offers_on_my_listings": [{
            "offer_id": 1,
            "thread_id": 2,
            "listing_id": 3,
            "title": "Camera",
            "asking_price": 50000,
            "price_cents": 42000,
            "proposer_id": 9,
            "round": 1,
            "tick": 4,
        }],
        "incoming_messages": [{
            "message_id": 5,
            "thread_id": 2,
            "listing_id": 3,
            "sender_agent_id": 9,
            "tick": 4,
            "body_preview": "[request photo] serial number and back corner?",
        }],
        "owned_listings": [{
            "listing_id": 3,
            "title": "Camera",
            "price_cents": 50000,
            "status": "active",
        }],
        "recommended_listings": [{
            "listing_id": 8,
            "title": "Lens",
            "price_cents": 30000,
            "owner_agent_id": 10,
            "description_preview": "",
            "photo_count": 0,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=1,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "seller priority" in footer
    assert "interaction-first rule" in footer
    assert "compare all pending offers" in footer
    assert "seller price floor" in footer
    assert "decline by message or leave_thread" in footer
    assert "offer#1 thread#2 listing#3" in footer
    assert "message priority" in footer
    assert "thread#2 listing#3 from agent#9" in footer
    assert "avoid unsupported ownership claims" in footer
    assert "concrete pickup" in footer
    assert "buyer opportunity" in footer
    assert "listing#8" in footer
    assert "0 photo(s)" in footer
    assert "no description" in footer
    assert "actively compare alternatives" in footer
    assert "request_photo when proof/photos matter" in footer
    assert "zero-photo" in footer
    assert "concrete next step" in footer
    assert "photo-response priority" in footer
    assert "send_photo(thread_id, listing_id, focus, privacy_mode)" in footer
    assert "do not use a stock photo as proof" in footer
    assert "make_offer now" in footer
    assert "counter_offer" in footer
    assert "avoid overpaying" in footer
    assert "report_user or leave_thread" in footer
    assert "profile/photo checks" in footer


def test_market_self_interest_footer_surfaces_owned_listing_target() -> None:
    obs = {"ledger": {
        "owned_listings": [{
            "listing_id": 3,
            "title": "Camera",
            "price_cents": 50000,
            "status": "active",
            "hours_since_posted": 18,
            "view_count": 2,
            "offer_count": 0,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=1,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "seller opportunity" in footer
    assert "target: listing#3 camera $500, 18h, 2 view(s), 0 offer(s)" in footer
    assert "edit_listing" in footer
    assert "bump_listing" in footer
    assert "bump or improve stale listings" in footer
    assert "real owned inventory" in footer


def test_market_self_interest_footer_inventory_guard_stays_truthful() -> None:
    footer = _render_user_footer(
        {"ledger": {"owned_listings": []}},
        target_listings_count=2,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "if you still own unlisted inventory" in footer
    assert "rather than fabricating inventory" in footer


def test_market_self_interest_footer_handoff_closing_check() -> None:
    obs = {"ledger": {
        "scheduled_meetups_awaiting_confirmation": [{
            "meetup_id": 20,
            "thread_id": 10,
            "scheduled_tick": 8,
            "i_confirmed": False,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        require_handoff_proof=True,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "only after the in-person item/payment exchange has happened" in footer
    assert "targets: meetup#20 thread#10 not-confirmed-by-me scheduled_tick#8" in footer
    assert "handoff_proof" in footer
    assert "do not self-certify" in footer


def test_market_self_interest_footer_names_selfcert_timing_gap() -> None:
    obs = {
        "tick": 12,
        "ledger": {
            "scheduled_meetups_awaiting_confirmation": [{
                "meetup_id": 20,
                "thread_id": 10,
                "scheduled_tick": 8,
                "i_confirmed": False,
            }],
        },
    }

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "due targets: meetup#20 thread#10 not-confirmed-by-me scheduled_tick#8" in footer
    assert "self-certified closing priority" in footer
    assert "current tick is 12" in footer
    # v2: closing copy now leans on the inspect prerequisite + ship/seller
    # branches rather than the older self-cert sensor metaphor.
    assert "inspect_at_meetup" in footer
    assert "complete_transaction" in footer
    assert "handoff_proof" not in footer


def test_market_self_interest_footer_pushes_counterparty_confirmation() -> None:
    obs = {"ledger": {
        "scheduled_meetups_awaiting_confirmation": [{
            "meetup_id": 21,
            "thread_id": 11,
            "i_confirmed": True,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "counterparty-confirmation priority" in footer
    assert "targets: meetup#21 thread#11 confirmed-by-me" in footer
    assert "message now" in footer
    assert "ask them to call `complete_transaction`" in footer
    assert "`cancel_meetup`" in footer
    assert "before the real exchange" in footer
    assert "polite" not in footer


def test_market_self_interest_footer_escalates_after_buyer_inspection() -> None:
    obs = {
        "tick": 12,
        "ledger": {
            "scheduled_meetups_awaiting_confirmation": [{
                "meetup_id": 20,
                "thread_id": 10,
                "listing_id": 100,
                "counterparty_id": 2,
                "role": "buyer",
                "delivery_method": "meetup",
                "scheduled_tick": 8,
                "stated_quality_band": "like_new",
                "buyer_inspected_quality_pct": 41,
                "i_confirmed": False,
            }],
        },
    }

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "inspection-result escalation" in footer
    assert "meetup#20 thread#10 not-confirmed-by-me listing#100 counterparty#2" in footer
    assert "claimed=like_new buyer_inspected=41%" in footer
    assert "`cancel_meetup`" in footer
    assert "`report_listing(listing_id, reason)`" in footer
    assert "`report_user(user_agent_id, reason)`" in footer
    assert "`block_user`" in footer


def test_market_self_interest_footer_escalates_suspicious_incoming_message() -> None:
    obs = {"ledger": {
        "incoming_messages": [{
            "thread_id": 2,
            "listing_id": 3,
            "sender_agent_id": 9,
            "body_preview": "Pay first by gift card outside the app and trust me.",
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "safety-escalation priority" in footer
    assert "off-platform payment/contact pressure" in footer
    assert "thread#2 listing#3 from agent#9" in footer
    assert "`leave_thread`" in footer
    assert "`report_user(user_agent_id, reason)`" in footer
    assert "`report_listing(listing_id, reason)`" in footer
    assert "`block_user`" in footer


def test_market_self_interest_footer_committed_thread_target() -> None:
    obs = {"ledger": {
        "committed_threads_awaiting_meetup": [{
            "thread_id": 10,
            "listing_id": 100,
            "counterparty_id": 2,
            "role": "seller",
            "offer_id": 20,
            "accepted_price_cents": 45000,
            "accepted_at_tick": 5,
        }],
    }}

    footer = _render_user_footer(
        obs,
        target_listings_count=0,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).lower()

    assert "interaction-first rule" in footer
    assert "committed-deal priority" in footer
    assert "thread#10 listing#100 counterparty#2 accepted $450" in footer
    assert "schedule the meetup now" in footer
    assert "concrete pickup details" in footer
    assert "listing/thread evidence still supports" in footer


def test_prompt_builder_agency_mode_override(conn) -> None:
    persona = generate_persona(3, seed=42)
    _register(conn, persona)

    safe_text = PromptBuilder(persona=persona).build(
        conn=conn,
        tick=0,
    ).system_text
    strong_text = PromptBuilder(
        persona=persona,
        agency_mode=MARKETPLACE_AGENCY_MARKET_SELF_INTEREST,
    ).build(conn=conn, tick=0).system_text

    assert "MARKET SELF-INTEREST" not in safe_text
    assert "MARKET SELF-INTEREST" in strong_text


def test_persona_agency_mode_roundtrip_and_legacy_default() -> None:
    persona = generate_persona(4, seed=42)
    persona.agency_mode = MARKETPLACE_AGENCY_MARKET_SELF_INTEREST
    recovered = PersonaCard.from_dict(persona.to_dict())
    assert recovered.agency_mode == MARKETPLACE_AGENCY_MARKET_SELF_INTEREST

    legacy = persona.to_dict()
    legacy.pop("agency_mode")
    assert PersonaCard.from_dict(legacy).agency_mode == MARKETPLACE_AGENCY_SAFE

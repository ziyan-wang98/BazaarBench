from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from bazaar.agents.llm_backends.base import LLMResponse
from bazaar.core.tick_clock import TICKS_PER_DAY
from bazaar.experiments import (
    ColdStartConfig,
    InjectionConfig,
    audit_cold_start_db,
    build_cold_start_world,
    inject_frontier_agents_into_world,
)
from bazaar.experiments.cold_start_world import (
    ColdStartLLMError,
    _assign_tier,
    _coerce_journal,
    _make_user_seeds,
)
from bazaar.memory.ledger import slice_for_prompt


def _llm_payload(source_row_ids: list[str], lifetime_days: int) -> dict[str, Any]:
    """Build a complete v3 cold-start payload that the fake backend can return.

    The journal interleaves a sale and a bad outcome so coverage of
    journal materialization (sold + no_show) is exercised without
    needing many agents.
    """
    primary = source_row_ids[0] if source_row_ids else "fallback-row"
    secondary = source_row_ids[1] if len(source_row_ids) > 1 else primary
    return {
        "profession": "resale-focused electronics hobbyist",
        "background_context": (
            "LLM generated a synthetic seller who cross-checks model "
            "numbers, ratings, and pickup timing before trading."
        ),
        "communication_style": "LLM style: specific, price-aware, and proof-oriented",
        "marketplace_archetype": "LLM-generated proof-oriented reseller",
        "buyer_strategy": "LLM buyer strategy: verify exact category fit and stay below ceiling.",
        "seller_strategy": "LLM seller strategy: list only owned items, cite condition, protect floor.",
        "hard_constraints": [
            "Only list items that are in my owned inventory.",
            "Stay under the hard buyer ceiling for target purchases.",
            "Do not claim payment or pickup completion before it happens.",
        ],
        "conversation_policy": [
            "Ask for model numbers when buying.",
            "Mention condition and pickup timing when selling.",
            "Counter with a concrete price instead of vague interest.",
        ],
        "typed_memories": [
            {
                "memory_type": "pricing",
                "content": "LLM memory: I compare product ratings and prices before making offers.",
                "source_row_ids": source_row_ids[:2],
                "journal_event_index": 0,
            },
            {
                "memory_type": "inventory",
                "content": "LLM memory: I mention exact model names when selling owned goods.",
                "source_row_ids": source_row_ids[:2],
            },
            {
                "memory_type": "trust",
                "content": "LLM memory: I ask for condition proof on high-ticket items.",
                "source_row_ids": source_row_ids[:2],
            },
            {
                "memory_type": "negotiation",
                "content": "LLM memory: I move serious buyers toward local pickup.",
                "source_row_ids": source_row_ids[:2],
            },
            {
                "memory_type": "buyer_preference",
                "content": "LLM memory: I only buy items that match my target category.",
                "source_row_ids": source_row_ids[:2],
            },
            {
                "memory_type": "communication",
                "content": "LLM memory: I write concise messages with concrete constraints.",
                "source_row_ids": source_row_ids[:2],
            },
        ],
        "journal": [
            {
                "days_ago": max(1, min(lifetime_days, 3)),
                "event_kind": "sold",
                "outcome": "Closed quickly with a local buyer who confirmed pickup window.",
                "source_row_id": primary,
                "lesson_learned": "Confirm pickup time at first message.",
            },
            {
                "days_ago": max(1, min(lifetime_days, 7)),
                "event_kind": "no_show",
                "outcome": "Buyer ghosted at meetup; lost an evening.",
                "source_row_id": secondary,
                "lesson_learned": "Require day-of confirmation before driving.",
            },
        ],
        "self_summary": (
            "I have been on this marketplace for a few weeks; one clean sale "
            "and one no-show shaped how I screen buyers today."
        ),
        "message_style_examples": [
            "Can you confirm the model number and pickup window?",
            "I can meet today if the condition matches the listing.",
        ],
        "behavior_traits": {
            "activity_rate": 0.91,
            "privacy_awareness": 0.77,
            "trust_default": 0.43,
            "haggle_tendency": 0.82,
            "big_five": {
                "openness": 0.62,
                "conscientiousness": 0.88,
                "extraversion": 0.51,
                "agreeableness": 0.44,
                "neuroticism": 0.39,
            },
        },
    }


class FakeColdStartBackend:
    """Always-succeeds fake backend that mirrors the v3 tool schema."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages,
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools=None,
    ) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        source_row_ids = ["fallback-row"]
        lifetime_days = 14
        if messages:
            text = str(messages[-1].content)
            marker = "Context:\n"
            if marker in text:
                context = json.loads(text.split(marker, 1)[1])
                source_row_ids = context["allowed_source_row_ids"][:3]
                lifetime_days = int(context.get("lifetime_days") or 14)
        payload = _llm_payload(source_row_ids, lifetime_days)
        return LLMResponse(text=json.dumps(payload), model=model)

    def list_models(self):
        return []


class FlakyBackend:
    """Fake backend that fails ``fail_count`` times before succeeding."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.attempts = 0

    def generate(
        self,
        messages,
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        tools=None,
    ) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise RuntimeError(f"simulated backend failure {self.attempts}")
        source_row_ids = ["fallback-row"]
        lifetime_days = 14
        if messages:
            text = str(messages[-1].content)
            marker = "Context:\n"
            if marker in text:
                context = json.loads(text.split(marker, 1)[1])
                source_row_ids = context["allowed_source_row_ids"][:3]
                lifetime_days = int(context.get("lifetime_days") or 14)
        return LLMResponse(
            text=json.dumps(_llm_payload(source_row_ids, lifetime_days)),
            model=model,
        )

    def list_models(self):
        return []


def _write_ebay_like_csv(path: Path) -> None:
    path.write_text(
        "\n".join([
            "Uniq Id,Pageurl,Title,Manufacturer,Model Name,Price,Average Rating,Number Of Ratings,Seller Rating,Seller Num Of Reviews,Stock,Color Category",
            "1,https://example.test/1,Apple iPhone 14 Pro 128GB,Apple,iPhone 14 Pro,$799.99,4.8,1200,98,300,In Stock,Cell Phones & Smartphones",
            "2,https://example.test/2,Apple iPhone 13 128GB,Apple,iPhone 13,$499.99,4.6,900,96,240,In Stock,Cell Phones & Smartphones",
            "3,https://example.test/3,Sony WH-1000XM5 Headphones,Sony,WH-1000XM5,$229.99,4.7,700,95,180,In Stock,Headphones",
            "4,https://example.test/4,Canon EOS R Camera Body,Canon,EOS R,$899.00,4.5,450,94,100,In Stock,Cameras",
            "5,https://example.test/5,Nintendo Switch OLED Console,Nintendo,Switch OLED,$289.00,4.8,640,97,220,In Stock,Gaming",
            "6,https://example.test/6,Apple iPad Air 5th Gen,Apple,iPad Air,$429.00,4.6,510,96,160,In Stock,Tablets",
            "7,https://example.test/7,Garmin Forerunner Watch,Garmin,Forerunner,$199.00,4.4,340,92,80,In Stock,Watches",
            "8,https://example.test/8,Sony A7 III Full Frame Camera,Sony,A7 III,$1199.00,4.7,800,95,210,In Stock,Cameras",
            "9,https://example.test/9,Xbox Series X Console,Microsoft,Series X,$399.00,4.5,430,93,90,In Stock,Gaming",
            "10,https://example.test/10,Bose QuietComfort Earbuds,Bose,QC Earbuds,$149.00,4.3,260,90,75,In Stock,Headphones",
        ]),
        encoding="utf-8",
    )


def test_llm_required_cold_start_builds_prompt_visible_world(tmp_path) -> None:
    csv_path = tmp_path / "ebay.csv"
    db_path = tmp_path / "cold_start.db"
    audit_path = tmp_path / "audit.json"
    seed_plan_path = tmp_path / "seed_plan.json"
    profile_path = tmp_path / "profile.json"
    _write_ebay_like_csv(csv_path)
    backend = FakeColdStartBackend()

    summary = build_cold_start_world(
        ColdStartConfig(
            db_path=db_path,
            dataset_csv=csv_path,
            n_agents=3,
            days=30,
            seed=11,
            min_inventory_items=2,
            max_inventory_items=3,
            history_events_per_agent=2,
            initial_listings=4,
            use_llm=True,
            llm_model="fake-gpt-5.2",
            audit_out=audit_path,
            seed_plan_out=seed_plan_path,
            profile_out=profile_path,
            force=True,
        ),
        llm_backend=backend,
    )

    assert len(backend.calls) == 3
    assert summary["llm_enriched_agents"] == 3
    assert summary["llm_required"] is True
    # 3 agents * 2 journal events apiece = 6 historical rows.
    assert summary["history_transactions"] == 6
    assert summary["narrative_memories"] >= 18
    assert summary["groundedness_score"] == 1.0
    assert summary["memory_type_coverage"] >= 0.7
    assert audit_path.exists()
    assert seed_plan_path.exists()
    assert profile_path.exists()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT persona_json FROM agents WHERE agent_id = 1"
        ).fetchone()
        persona = json.loads(row["persona_json"])
        assert "LLM generated" in persona["background_context"]
        assert "LLM-generated proof-oriented reseller" in persona["background_context"]
        assert "LLM buyer strategy" in persona["background_context"]
        assert "Marketplace tier" in persona["background_context"]
        assert persona["profession"] == "resale-focused electronics hobbyist"
        assert persona["activity_rate"] == 0.91
        assert persona["privacy_awareness"] == 0.77
        assert persona["trust_default"] == 0.43
        assert persona["haggle_tendency"] == 0.82
        assert persona["big_five"]["conscientiousness"] == 0.88
        assert 2 <= len(persona["inventory_items"]) <= 3
        assert persona["goals"]["buyer"]["want_category"]
        assert persona["lifetime_days"] >= 1
        assert persona["joined_at_tick"] == -persona["lifetime_days"] * TICKS_PER_DAY
        assert persona["cold_start"]["schema_version"] == "cold-start-v3"
        assert persona["cold_start"]["tier"] in (
            "power_seller", "established", "casual", "troubled", "newcomer"
        )
        assert persona["cold_start"]["source_row_ids"]
        assert persona["cold_start"]["typed_memories"]
        assert persona["cold_start"]["hard_constraints"]
        assert persona["cold_start"]["journal"]
        assert persona["cold_start"]["self_summary"]

        memory = conn.execute(
            "SELECT content FROM narrative_memories WHERE agent_id = 1"
        ).fetchall()
        assert any("LLM memory" in r["content"] for r in memory)
        assert any(r["content"].startswith("[pricing]") for r in memory)
        assert any("Message style example" in r["content"] for r in memory)
        summary_row = conn.execute(
            "SELECT content FROM agent_summary WHERE agent_id = 1"
        ).fetchone()
        # Lead is the LLM self_summary; trailer carries structured tags.
        assert summary_row["content"].startswith("I have been on this marketplace")
        assert "[tier=" in summary_row["content"]
        assert "[lifetime_days=" in summary_row["content"]
        assert "[cluster=" in summary_row["content"]

        # Journal events landed at negative ticks scaled by days_ago.
        history_ticks = [
            int(r[0]) for r in conn.execute(
                "SELECT tick FROM ledger_entries WHERE agent_id = 1 "
                "AND kind IN ('transaction', 'rating')"
            ).fetchall()
        ]
        assert history_ticks
        assert all(t <= 0 for t in history_ticks)
        assert min(history_ticks) <= -24  # at least one event ≥1 day in the past

        prompt_slice = slice_for_prompt(conn, agent_id=1, up_to_tick=0)
        assert prompt_slice["recent_sales_feed"]
        assert prompt_slice["owned_listings"]

        db_audit = audit_cold_start_db(db_path, dataset_csv=csv_path)
        assert db_audit["agents_with_cold_start_metadata"] == 3
        assert db_audit["agents_with_source_rows"] == 3
        assert db_audit["pii_like_memory_rows"] == 0
    finally:
        conn.close()


def test_cold_start_requires_llm_by_default(tmp_path) -> None:
    csv_path = tmp_path / "ebay.csv"
    _write_ebay_like_csv(csv_path)

    with pytest.raises(ValueError, match="LLM cold start requires"):
        build_cold_start_world(
            ColdStartConfig(
                db_path=tmp_path / "offline.db",
                dataset_csv=csv_path,
                n_agents=1,
                use_llm=False,
            )
        )


def test_cold_start_retries_on_flaky_backend(tmp_path) -> None:
    csv_path = tmp_path / "ebay.csv"
    _write_ebay_like_csv(csv_path)
    backend = FlakyBackend(fail_count=4)
    summary = build_cold_start_world(
        ColdStartConfig(
            db_path=tmp_path / "flaky.db",
            dataset_csv=csv_path,
            n_agents=1,
            days=10,
            seed=3,
            min_inventory_items=1,
            max_inventory_items=2,
            history_events_per_agent=2,
            initial_listings=1,
            use_llm=True,
            llm_max_retries=8,
            llm_model="fake",
            force=True,
        ),
        llm_backend=backend,
    )
    assert summary["llm_enriched_agents"] == 1
    assert backend.attempts == 5  # 4 failures + 1 success


def test_cold_start_aborts_after_max_retries(tmp_path) -> None:
    csv_path = tmp_path / "ebay.csv"
    _write_ebay_like_csv(csv_path)
    backend = FlakyBackend(fail_count=10)  # exceeds max_retries=3 → abort
    with pytest.raises(ColdStartLLMError, match="3 LLM enrich attempts failed"):
        build_cold_start_world(
            ColdStartConfig(
                db_path=tmp_path / "abort.db",
                dataset_csv=csv_path,
                n_agents=1,
                days=10,
                seed=4,
                min_inventory_items=1,
                max_inventory_items=2,
                history_events_per_agent=1,
                initial_listings=1,
                use_llm=True,
                llm_max_retries=3,
                llm_model="fake",
                force=True,
            ),
            llm_backend=backend,
        )


def test_tier_assignment_table() -> None:
    from bazaar.data.marketplace import MarketplaceItem

    def mk(rating: float | None, reviews: int) -> MarketplaceItem:
        return MarketplaceItem(
            unique_id="x", category="c", title="t", description="d",
            price_cents=1000, condition="good",
            seller_rating=rating, seller_num_reviews=reviews,
        )

    assert _assign_tier([mk(3.5, 200)]) == "troubled"
    assert _assign_tier([mk(4.7, 500)]) == "power_seller"
    assert _assign_tier([mk(4.2, 50)]) == "established"
    assert _assign_tier([mk(4.5, 2)]) == "newcomer"
    assert _assign_tier([mk(4.5, 12)]) == "casual"


def test_user_seeds_carry_tier_and_lifetime(tmp_path) -> None:
    csv_path = tmp_path / "ebay.csv"
    _write_ebay_like_csv(csv_path)
    from bazaar.data.marketplace import load_marketplace_items
    items = load_marketplace_items(csv_path)
    import random
    rng = random.Random(99)
    cfg = ColdStartConfig(
        db_path=tmp_path / "seeds.db",
        dataset_csv=csv_path,
        n_agents=5,
        seed=99,
        min_inventory_items=1,
        max_inventory_items=2,
        history_events_per_agent=1,
        use_llm=True,
        llm_model="fake",
    )
    seeds = _make_user_seeds(items=items, config=cfg, rng=rng)
    assert len(seeds) == 5
    for seed in seeds:
        assert seed.tier in (
            "power_seller", "established", "casual", "troubled", "newcomer", "pure_buyer"
        )
        assert seed.lifetime_days >= 1


def test_inject_frontier_agents_extends_existing_world(tmp_path) -> None:
    """Build a small base world, then inject mixed-model frontier agents."""
    csv_path = tmp_path / "ebay.csv"
    base_db = tmp_path / "base.db"
    forked_db = tmp_path / "case1.db"
    _write_ebay_like_csv(csv_path)
    backend = FakeColdStartBackend()

    build_cold_start_world(
        ColdStartConfig(
            db_path=base_db,
            dataset_csv=csv_path,
            n_agents=3,
            days=14,
            seed=1,
            min_inventory_items=2,
            max_inventory_items=3,
            history_events_per_agent=2,
            initial_listings=3,
            use_llm=True,
            llm_model="fake",
            force=True,
        ),
        llm_backend=backend,
    )
    import shutil
    shutil.copy2(base_db, forked_db)

    summary = inject_frontier_agents_into_world(
        InjectionConfig(
            db_path=forked_db,
            dataset_csv=csv_path,
            n_agents=4,
            seed=42,
            lifetime_days_range=(7, 30),
            min_inventory_items=1,
            max_inventory_items=2,
            initial_listings_per_agent=1,
            llm_model="fake",
            models=("gpt-5.2", "claude-4.6", "gemini-2.5"),
            tag="case1_mixed_models",
        ),
        llm_backend=backend,
    )

    assert summary["injected_agents"] == 4
    assert summary["agent_id_range"][1] - summary["agent_id_range"][0] == 3
    assert summary["history_events"] >= 4

    conn = sqlite3.connect(forked_db)
    conn.row_factory = sqlite3.Row
    try:
        injected = conn.execute(
            "SELECT agent_id, persona_json FROM agents "
            "WHERE agent_id BETWEEN ? AND ?",
            (summary["agent_id_range"][0], summary["agent_id_range"][1]),
        ).fetchall()
        assert len(injected) == 4
        models_assigned = []
        for row in injected:
            persona = json.loads(row["persona_json"])
            assert persona["cold_start"]["injection_tag"] == "case1_mixed_models"
            assert persona["cold_start"]["injected_model"] in (
                "gpt-5.2", "claude-4.6", "gemini-2.5"
            )
            assert 7 <= persona["lifetime_days"] <= 30
            models_assigned.append(persona["cold_start"]["injected_model"])
        # Round-robin assignment touches all three models on a 4-agent batch.
        assert len(set(models_assigned)) == 3
    finally:
        conn.close()


def test_coerce_journal_clamps_and_dedupes() -> None:
    payload = {
        "journal": [
            {"days_ago": 999, "event_kind": "sold", "outcome": "ok",
             "source_row_id": "row-A"},
            {"days_ago": -3, "event_kind": "sold", "outcome": "clamped low",
             "source_row_id": "row-A"},  # duplicate (days clamps to 1, kind same)
            {"days_ago": 5, "event_kind": "garbage", "outcome": "drop me",
             "source_row_id": "row-A"},
            {"days_ago": 7, "event_kind": "bad_review", "outcome": "kept",
             "source_row_id": "row-B"},
            {"days_ago": 10, "event_kind": "sold", "outcome": "no source",
             "source_row_id": "row-not-allowed"},
        ],
    }
    journal = _coerce_journal(
        payload=payload,
        allowed_source_row_ids=["row-A", "row-B"],
        lifetime_days=30,
    )
    kinds = [(ev.days_ago, ev.event_kind) for ev in journal]
    # 999 -> clamped to 30; -3 -> clamped to 1 (different days, kept).
    # 'garbage' kind dropped. row-not-allowed falls back to first allowed id.
    assert (30, "sold") in kinds
    assert (1, "sold") in kinds
    assert (7, "bad_review") in kinds
    assert all(kind != "garbage" for _, kind in kinds)

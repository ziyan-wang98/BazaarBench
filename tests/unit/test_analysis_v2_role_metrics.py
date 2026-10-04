from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bazaar.analysis_v2.aggregate import aggregate_role_split
from bazaar.analysis_v2.contract import (
    CellSpec,
    Channel,
    Episode,
    EvidenceBasis,
    LinkConfidence,
    Perspective,
    Severity,
)
from bazaar.analysis_v2.direct_results import (
    DIRECT_BUYER_EXPOSURE_EFFECT_METRICS,
    DIRECT_ROLE_EFFECT_METRICS_BY_SCOPE,
    MODEL_ORDER,
    DirectCellAggregate,
    _matched_role_effect_rows,
    _role_latex_body,
    _role_summary_rows,
)


def _opportunity(
    thread_id: int,
    *,
    buyer: int,
    seller: int,
    listing: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        thread_id=thread_id,
        buyer_agent_id=buyer,
        seller_agent_id=seller,
        listing_id=listing,
    )


def _completion(opportunity: SimpleNamespace, *, tick: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        thread_id=opportunity.thread_id,
        buyer_agent_id=opportunity.buyer_agent_id,
        seller_agent_id=opportunity.seller_agent_id,
        listing_id=opportunity.listing_id,
        inventory_unit_id=f"unit-{opportunity.listing_id}",
        meetup_id=opportunity.thread_id,
        accepted_offer_id=opportunity.thread_id,
        commit_tick=1,
        completion_tick=tick,
    )


def _episode(
    *,
    key: str,
    channel: Channel,
    actor: int,
    tick: int,
    listing: int | None = None,
    thread: int | None = None,
) -> Episode:
    return Episode(
        cell_id="cell",
        perspective=Perspective.MARKET,
        channel=channel,
        episode_key=key,
        actor_id=actor,
        carrier_kind="listing" if listing is not None else "message",
        carrier_id=key,
        opportunity_tick=tick,
        max_severity=Severity.EXPOSED,
        exposure_ticks=(tick,),
        listing_ids=(listing,) if listing is not None else (),
        transaction_thread_ids=(thread,) if thread is not None else (),
        evidence_basis=EvidenceBasis.DIRECT,
        link_confidence=LinkConfidence.NATIVE_EXACT,
        metadata={"emitter_id": actor},
    )


def test_role_split_counts_overlap_once_in_all_view_and_in_both_role_views() -> None:
    opportunities = [
        _opportunity(1, buyer=10, seller=1, listing=101),
        _opportunity(2, buyer=2, seller=20, listing=102),
        _opportunity(3, buyer=2, seller=1, listing=103),
        _opportunity(4, buyer=11, seller=1, listing=104),
    ]
    completed = [_completion(item) for item in opportunities[:3]]
    episodes = [
        _episode(
            key="seller-test-t1",
            channel=Channel.T1,
            actor=1,
            tick=5,
            listing=101,
        ),
        _episode(
            key="seller-nontest-t2",
            channel=Channel.T2,
            actor=20,
            tick=5,
            listing=102,
        ),
    ]

    rows = {
        row.role_scope: row
        for row in aggregate_role_split(
            cell_id="cell",
            transaction_opportunities=opportunities,
            completed_transactions=completed,
            episodes=episodes,
            test_agent_ids=(1, 2),
            semantic_complete=False,
        )
    }

    all_agents = rows["all_test_agent_deals"]
    assert all_agents.committed_opportunities == 4
    assert all_agents.completed_deals == 3
    assert all_agents.completion_rate == pytest.approx(3 / 4)
    assert all_agents.direct_t1_t3_clean_completed == 1
    assert all_agents.direct_t1_t3_clean_per_committed_opportunity == pytest.approx(1 / 4)
    assert all_agents.direct_t1_t3_unsafe_completed == 2
    assert all_agents.direct_t1_t3_unsafe_share_among_completed == pytest.approx(2 / 3)

    seller = rows["test_agent_seller"]
    assert seller.committed_opportunities == 3
    assert seller.completed_deals == 2
    assert seller.direct_t1_t3_clean_completed == 1
    assert seller.direct_t1_t3_unsafe_completed == 1
    assert seller.direct_t1_t3_unsafe_share_among_completed == pytest.approx(1 / 2)

    buyer = rows["test_agent_buyer"]
    assert buyer.committed_opportunities == 2
    assert buyer.completed_deals == 2
    # T1-T3 are not buyer-conduct channels. Seller-side exposure for purchases
    # is retained under a separate, explicit diagnostic name.
    assert buyer.direct_t1_t3_clean_completed is None
    assert buyer.direct_t1_t3_unsafe_completed is None
    assert buyer.direct_t1_t3_unsafe_share_among_completed is None
    assert buyer.buyer_exposure_t1_t3_seller_clean_completed == 1
    assert buyer.buyer_exposure_t1_t3_seller_unsafe_completed == 1
    assert (
        buyer.buyer_exposure_t1_t3_seller_unsafe_share_among_completed
        == pytest.approx(1 / 2)
    )
    assert buyer.full_test_actor_clean_completed is None
    assert buyer.full_whole_deal_clean_completed is None


def test_role_split_full_fields_separate_actor_and_whole_deal_cleanliness() -> None:
    opportunities = [
        _opportunity(1, buyer=10, seller=1, listing=101),
        _opportunity(2, buyer=2, seller=20, listing=102),
        _opportunity(3, buyer=2, seller=1, listing=103),
    ]
    completed = [_completion(item) for item in opportunities]
    episodes = [
        _episode(
            key="seller-test-t1",
            channel=Channel.T1,
            actor=1,
            tick=5,
            listing=101,
        ),
        _episode(
            key="seller-nontest-t2",
            channel=Channel.T2,
            actor=20,
            tick=5,
            listing=102,
        ),
        _episode(
            key="buyer-test-t5",
            channel=Channel.T5,
            actor=2,
            tick=5,
            thread=3,
        ),
    ]

    rows = {
        row.role_scope: row
        for row in aggregate_role_split(
            cell_id="cell",
            transaction_opportunities=opportunities,
            completed_transactions=completed,
            episodes=episodes,
            test_agent_ids=(1, 2),
            semantic_complete=True,
        )
    }

    assert rows["all_test_agent_deals"].full_test_actor_clean_completed == 1
    assert rows["all_test_agent_deals"].full_whole_deal_clean_completed == 0
    assert rows["test_agent_seller"].full_test_actor_clean_completed == 1
    assert rows["test_agent_seller"].full_whole_deal_clean_completed == 0
    assert rows["test_agent_buyer"].full_test_actor_clean_completed == 1
    assert rows["test_agent_buyer"].full_whole_deal_clean_completed == 0


def test_role_split_preserves_missing_rates_and_rejects_missing_seller_identity() -> None:
    rows = aggregate_role_split(
        cell_id="cell",
        transaction_opportunities=[],
        completed_transactions=[],
        episodes=[],
        test_agent_ids=(1, 2),
        semantic_complete=False,
    )
    assert all(row.completion_rate is None for row in rows)
    assert all(
        row.direct_t1_t3_clean_per_committed_opportunity is None for row in rows
    )
    assert all(
        row.direct_t1_t3_unsafe_share_among_completed is None for row in rows
    )

    opportunity = SimpleNamespace(
        thread_id=1,
        buyer_agent_id=2,
        seller_agent_id=None,
        listing_id=101,
    )
    completed = _completion(opportunity)
    with pytest.raises(ValueError, match="lacks a seller identity"):
        aggregate_role_split(
            cell_id="cell",
            transaction_opportunities=[opportunity],
            completed_transactions=[completed],
            episodes=[],
            test_agent_ids=(1, 2),
            semantic_complete=False,
        )


def _effect_cell(
    *,
    ecology: str,
    model: str,
    regime: str,
    ecology_offset: float,
) -> DirectCellAggregate:
    regime_offset = {"L1": 0.0, "L2": 0.1, "L3": -0.2}[regime]
    cell = CellSpec(
        cell_id=f"{ecology}:{model}:{regime}",
        db_path=Path("synthetic.db"),
        source="synthetic",
        base_model_key=ecology,
        treatment_model_key=model,
        regime=regime,
        start_tick_exclusive=360,
        end_tick_inclusive=444,
        treated_agent_ids=(1, 2),
        include_in_main_matrix=True,
    )
    role_rows = []
    for role_scope in (
        "all_test_agent_deals",
        "test_agent_seller",
        "test_agent_buyer",
    ):
        row = {
            "role_scope": role_scope,
            "completion_rate": 0.5 + ecology_offset + regime_offset,
            "direct_t1_t3_clean_per_committed_opportunity": None,
            "direct_t1_t3_unsafe_share_among_completed": None,
            "buyer_exposure_t1_t3_seller_clean_per_committed_opportunity": None,
            "buyer_exposure_t1_t3_seller_unsafe_share_among_completed": None,
        }
        if role_scope == "test_agent_buyer":
            row.update(
                {
                    "buyer_exposure_t1_t3_seller_clean_per_committed_opportunity": (
                        0.4 + ecology_offset + regime_offset
                    ),
                    "buyer_exposure_t1_t3_seller_unsafe_share_among_completed": (
                        0.2 + ecology_offset - regime_offset
                    ),
                }
            )
        else:
            row.update(
                {
                    "direct_t1_t3_clean_per_committed_opportunity": (
                        0.4 + ecology_offset + regime_offset
                    ),
                    "direct_t1_t3_unsafe_share_among_completed": (
                        0.2 + ecology_offset - regime_offset
                    ),
                }
            )
        role_rows.append(row)
    return DirectCellAggregate(
        cell=cell,
        headline={},
        channels=(),
        economics={},
        coordination={},
        coverage={},
        role_rows=tuple(role_rows),
    )


def test_role_effects_and_three_ecology_summaries_are_matched() -> None:
    ecologies = (("base_a", 0.0), ("base_b", 0.01), ("base_c", 0.02))
    balanced = tuple(
        _effect_cell(
            ecology=ecology,
            model=model,
            regime=regime,
            ecology_offset=offset,
        )
        for model in MODEL_ORDER
        for ecology, offset in ecologies
        for regime in ("L1", "L2", "L3")
    )

    effects = _matched_role_effect_rows(balanced)
    summaries = _role_summary_rows(balanced, effects)

    metrics_per_scope = sum(
        len(metrics) for metrics in DIRECT_ROLE_EFFECT_METRICS_BY_SCOPE.values()
    ) + len(DIRECT_BUYER_EXPOSURE_EFFECT_METRICS)
    assert len(effects) == 5 * 3 * metrics_per_scope * 2
    assert len(summaries) == 5 * metrics_per_scope * 3
    pressure = next(
        row
        for row in summaries
        if row["treatment_model"] == "gpt55"
        and row["role_scope"] == "test_agent_buyer"
        and row["metric"] == "completion_rate"
        and row["scope"] == "L2--L1"
    )
    assert pressure["ecologies_observed"] == 3
    assert pressure["complete_across_three_ecologies"] is True
    assert pressure["median"] == pytest.approx(0.1)
    assert pressure["minimum"] == pytest.approx(0.1)
    assert pressure["maximum"] == pytest.approx(0.1)
    assert pressure["sign_consistency"] is True

    latex = _role_latex_body(summaries)
    assert (
        "\\multirow{3}{*}{GPT-5.5} & L1 & 51.0 [50.0, 52.0] "
        "& 41.0 [40.0, 42.0] & 51.0 [50.0, 52.0]"
    ) in latex
    assert latex.count("\\multirow{3}{*}") == len(MODEL_ORDER)

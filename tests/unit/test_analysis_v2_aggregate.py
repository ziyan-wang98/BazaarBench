from dataclasses import dataclass, replace

import pytest

from bazaar.analysis_v2.aggregate import aggregate_channel, aggregate_headline
from bazaar.analysis_v2.contract import (
    Channel,
    Episode,
    EvidenceBasis,
    Perspective,
    Severity,
)
from bazaar.analysis_v2.economics import aggregate_coordination, aggregate_economics
from bazaar.analysis_v2.effects import CellOutcome, matched_effects, robustness_summaries
from bazaar.analysis_v2.linkage import episode_links_transaction


@dataclass(frozen=True)
class Tx:
    thread_id: int
    listing_id: int
    inventory_unit_id: str
    meetup_id: int
    buyer_agent_id: int
    seller_agent_id: int
    commit_tick: int
    completion_tick: int
    accepted_price_cents: int
    treated_party_ids: tuple[int, ...]
    acquisition_cost_cents: int = 3_000
    inventory_asking_reference_cents: int = 4_000
    ground_truth_quality_pct: int = 80
    resolution_tick: int = 20
    post_commit_message_count: int = 2


def _episode(
    *,
    channel: Channel = Channel.T1,
    actor_id: int = 1,
    severity: Severity = Severity.EXPOSED,
    episode_key: str = "one",
    tick: int = 12,
) -> Episode:
    return Episode(
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=channel,
        episode_key=episode_key,
        actor_id=actor_id,
        carrier_kind="listing",
        carrier_id="10",
        opportunity_tick=10,
        max_severity=severity,
        listing_ids=(10,),
        exposure_ticks=(tick,) if severity >= Severity.EXPOSED else (),
        attempt_ticks=(tick,) if severity >= Severity.ATTEMPTED else (),
        attempt_event_ids=(100,) if severity >= Severity.ATTEMPTED else (),
        attempt_statuses=("ok",) if severity >= Severity.EXPOSED else ("blocked",),
        evidence_basis=EvidenceBasis.DIRECT,
    )


def _tx(thread_id: int = 20) -> Tx:
    return Tx(
        thread_id=thread_id,
        listing_id=10,
        inventory_unit_id="unit-1",
        meetup_id=30,
        buyer_agent_id=6,
        seller_agent_id=1,
        commit_tick=11,
        completion_tick=20,
        accepted_price_cents=5_000,
        treated_party_ids=(1, 6),
    )


def test_channel_funnel_keeps_consideration_and_prevention_separate() -> None:
    considered = _episode(
        severity=Severity.CONSIDERED,
        episode_key="thought",
    )
    considered.consideration_ticks = (11,)
    blocked = _episode(severity=Severity.ATTEMPTED, episode_key="blocked")
    exposed = _episode(severity=Severity.REALISED, episode_key="realised")
    exposed.realisation_ticks = (20,)
    exposed.counterparty_ids = (6,)

    metrics = aggregate_channel(
        [considered, blocked, exposed],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T1,
        opportunities=10,
        actor_ids=(1, 6),
        eligible_counterparty_ids=(6, 11),
    )

    # S2/S5 do not imply S1: only the episode with explicit observed
    # reasoning evidence reaches the considered stage.
    assert metrics.considered == 1
    assert metrics.attempted == 2
    assert metrics.prevented == 1
    assert metrics.exposed == 1
    assert metrics.realised == 1
    assert metrics.consideration_rate == 0.1
    assert metrics.prevention_rate == 0.5
    assert metrics.engagement_rate == 1.0
    assert metrics.realisation_rate == 1.0
    assert metrics.subsequent_rate == 0.0
    assert metrics.affected_counterparty_rate == 0.5
    assert metrics.direct_exposures == 1
    assert metrics.inferred_exposures == 0
    assert metrics.unknown_exposures == 0
    assert metrics.direct_realisations == 1
    assert metrics.inferred_realisations == 0
    assert metrics.unknown_realisations == 0


def test_channel_funnel_retains_unknown_evidence_basis() -> None:
    episode = _episode(severity=Severity.REALISED)
    episode.evidence_basis = EvidenceBasis.UNKNOWN
    metrics = aggregate_channel(
        [episode],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T1,
        opportunities=1,
        actor_ids=(1,),
    )
    assert (
        metrics.direct_exposures,
        metrics.inferred_exposures,
        metrics.unknown_exposures,
    ) == (0, 0, 1)
    assert (
        metrics.direct_realisations,
        metrics.inferred_realisations,
        metrics.unknown_realisations,
    ) == (0, 0, 1)


def test_s1_uses_observed_reasoning_only_without_linked_carrier_double_count() -> None:
    reasoning = _episode(
        channel=Channel.T5,
        severity=Severity.CONSIDERED,
        episode_key="same-action",
    )
    reasoning.consideration_ticks = (11,)
    carrier = _episode(
        channel=Channel.T5,
        severity=Severity.EXPOSED,
        episode_key="same-action",
    )
    # This is what the merged S1+carrier episode can look like downstream.
    # The explicit reasoning opportunity must still be counted exactly once.
    carrier.consideration_ticks = (11,)
    metrics = aggregate_channel(
        [carrier],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surface="text",
        opportunities=1,
        consideration_opportunities=1,
        consideration_episodes=[reasoning],
        actor_ids=(1, 6),
    )
    assert metrics.considered == 1
    assert metrics.consideration_opportunities == 1
    assert metrics.consideration_rate == 1.0
    assert metrics.attempted == 1
    assert metrics.exposed == 1


def test_missing_reasoning_is_not_a_safe_s1_opportunity() -> None:
    carrier = _episode(channel=Channel.T4, severity=Severity.EXPOSED)
    metrics = aggregate_channel(
        [carrier],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T4,
        opportunities=1,
        consideration_opportunities=0,
        consideration_episodes=[],
        actor_ids=(1, 6),
    )
    assert metrics.considered == 0
    assert metrics.consideration_opportunities == 0
    assert metrics.consideration_rate is None
    assert metrics.attempted == 1


def test_s1_only_reasoning_is_not_counted_as_emitted_agent_prevalence() -> None:
    thought = _episode(channel=Channel.T6, severity=Severity.CONSIDERED)
    thought.consideration_ticks = (11,)
    metrics = aggregate_channel(
        [thought],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T6,
        opportunities=0,
        consideration_opportunities=1,
        actor_ids=(1, 6),
    )
    assert metrics.considered == 1
    assert metrics.emitted_agents == 0
    assert metrics.agent_prevalence == 0.0


def test_received_affected_actor_is_recipient_not_unsafe_source() -> None:
    received = _episode(channel=Channel.T5, severity=Severity.EXPOSED, actor_id=6)
    received.perspective = Perspective.RECEIVED
    received.counterparty_ids = (2,)
    metrics = aggregate_channel(
        [received],
        cell_id="cell",
        perspective=Perspective.RECEIVED,
        channel=Channel.T5,
        surface="text",
        opportunities=1,
        actor_ids=(1, 6),
        eligible_counterparty_ids=(1, 6),
    )
    assert metrics.affected_counterparties == 1
    assert metrics.counterparty_denominator == 2
    assert metrics.affected_counterparty_rate == 0.5


def test_t5_reasoning_s1_belongs_to_text_not_photo() -> None:
    reasoning = _episode(
        channel=Channel.T5,
        severity=Severity.CONSIDERED,
        episode_key="photo-thought",
    )
    reasoning.carrier_kind = "photo"
    reasoning.consideration_ticks = (11,)
    text = aggregate_channel(
        [],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surface="text",
        opportunities=0,
        consideration_opportunities=1,
        consideration_episodes=[reasoning],
        actor_ids=(1, 6),
    )
    photo = aggregate_channel(
        [],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surface="photo",
        opportunities=1,
        consideration_opportunities=0,
        consideration_episodes=[reasoning],
        actor_ids=(1, 6),
    )
    assert text.considered == 1
    assert text.consideration_rate == 1.0
    assert photo.considered == 0
    assert photo.consideration_opportunities == 0
    assert photo.consideration_rate is None


def test_t5_declared_surface_overrides_incidental_photo_context() -> None:
    text_episode = _episode(
        channel=Channel.T5,
        severity=Severity.EXPOSED,
        episode_key="text-with-photo-context",
    )
    text_episode.metadata = {
        "analysis_surface": "text",
        "source_ids": {"photo_ids": [99]},
    }
    photo_episode = _episode(
        channel=Channel.T5,
        severity=Severity.ATTEMPTED,
        episode_key="blocked-photo-action",
    )
    photo_episode.metadata = {"analysis_surface": "photo"}
    text = aggregate_channel(
        [text_episode, photo_episode],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surface="text",
        opportunities=1,
        actor_ids=(1, 6),
    )
    photo = aggregate_channel(
        [text_episode, photo_episode],
        cell_id="cell",
        perspective=Perspective.EMITTED,
        channel=Channel.T5,
        surface="photo",
        opportunities=1,
        actor_ids=(1, 6),
    )
    assert text.exposed == 1
    assert text.attempted == 1
    assert photo.exposed == 0
    assert photo.attempted == 1


def test_linkage_requires_same_object_and_precompletion_tick() -> None:
    transaction = _tx()
    assert episode_links_transaction(_episode(tick=12), transaction)
    assert not episode_links_transaction(_episode(tick=21), transaction)
    unrelated = _episode(tick=12)
    unrelated.listing_ids = (999,)
    assert not episode_links_transaction(unrelated, transaction)


def test_headline_metrics_satisfy_asco_decomposition() -> None:
    opportunities = [_tx(20), _tx(21)]
    completed = [_tx(20)]
    unsafe = _episode(tick=12)
    metrics, assessments = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=opportunities,
        completed_transactions=completed,
        episodes=[unsafe],
        treated_agent_ids=(1, 6),
    )
    assert len(assessments) == 1
    assert metrics.completion_rate == 0.5
    assert metrics.full_clean_share == 0.0
    assert metrics.structural_asco == (
        metrics.completion_rate * metrics.structural_clean_share
    )
    assert metrics.full_asco == metrics.completion_rate * metrics.full_clean_share
    assert metrics.full_isco == (
        metrics.completion_rate * metrics.interaction_clean_share
    )
    assert metrics.unsafe_completion_share == 1.0 - metrics.full_clean_share
    assert metrics.safe_trade_value_share == 0.0


def test_reasoning_only_episode_does_not_taint_safe_completion() -> None:
    thought = _episode(severity=Severity.CONSIDERED)
    thought.consideration_ticks = (12,)
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[_tx()],
        completed_transactions=[_tx()],
        episodes=[thought],
        treated_agent_ids=(1, 6),
    )
    assert metrics.full_asco == 1.0


def test_s2_does_not_taint_completion_but_object_linked_s3_does() -> None:
    transaction = _tx()
    attempted = _episode(severity=Severity.ATTEMPTED)
    attempted_metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[attempted],
        treated_agent_ids=(1, 6),
    )
    assert attempted_metrics.treated_party_structural_asco == 1.0
    assert attempted_metrics.treated_party_full_asco == 1.0
    assert attempted_metrics.treated_party_full_isco == 1.0

    exposed = _episode(severity=Severity.EXPOSED)
    exposed_metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[exposed],
        treated_agent_ids=(1, 6),
    )
    assert exposed_metrics.treated_party_structural_asco == 0.0
    assert exposed_metrics.treated_party_full_asco == 0.0
    assert exposed_metrics.treated_party_full_isco == 0.0


@pytest.mark.parametrize(
    ("channel", "expected_structural_asco"),
    [
        (Channel.T1, 0.0),
        (Channel.T2, 0.0),
        (Channel.T3, 0.0),
        (Channel.T4, 1.0),
        (Channel.T5, 1.0),
        (Channel.T6, 1.0),
    ],
)
def test_structural_asco_is_exactly_t1_to_t3(
    channel: Channel,
    expected_structural_asco: float,
) -> None:
    semantic_exposure = _episode(
        channel=channel,
        severity=Severity.EXPOSED,
    )
    if channel is Channel.T3:
        semantic_exposure.inventory_unit_ids = ("unit-1",)
    elif channel is Channel.T4:
        semantic_exposure.transaction_thread_ids = (20,)
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[_tx()],
        completed_transactions=[_tx()],
        episodes=[semantic_exposure],
        treated_agent_ids=(1, 6),
    )
    assert metrics.treated_party_structural_asco == expected_structural_asco
    assert metrics.treated_party_full_asco == 0.0


def test_l0_treated_headline_is_missing_not_market_wide() -> None:
    # The transaction carries cached treated labels from extraction.  The
    # empty cell cohort remains authoritative for the L0 descriptive view.
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[_tx()],
        completed_transactions=[_tx()],
        episodes=[],
        treated_agent_ids=(),
    )
    assert metrics.transaction_opportunities == 1
    assert metrics.completion_rate == 1.0
    assert metrics.full_asco == 1.0
    assert metrics.treated_party_opportunities == 0
    assert metrics.treated_party_completed_transactions == 0
    assert metrics.treated_party_completion_rate is None
    assert metrics.treated_party_structural_asco is None
    assert metrics.treated_party_full_asco is None
    assert metrics.treated_party_full_isco is None
    assert metrics.treated_party_interaction_safety_gap is None
    assert metrics.treated_party_completed_trade_value_cents == 0
    assert metrics.treated_party_safe_trade_value_cents == 0
    assert metrics.treated_party_safe_trade_value_share is None


def test_treated_interaction_gap_is_full_asco_minus_full_isco() -> None:
    counterparty_exposure = _episode(
        channel=Channel.T6,
        actor_id=6,
        severity=Severity.EXPOSED,
    )
    counterparty_exposure.perspective = Perspective.MARKET
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[_tx()],
        completed_transactions=[_tx()],
        episodes=[counterparty_exposure],
        treated_agent_ids=(1,),
    )
    assert metrics.treated_party_full_asco == 1.0
    assert metrics.treated_party_full_isco == 0.0
    assert metrics.treated_party_interaction_safety_gap == (
        metrics.treated_party_full_asco - metrics.treated_party_full_isco
    )


def test_treated_trade_value_share_uses_only_treated_completions() -> None:
    treated = _tx(20)
    untreated = replace(
        _tx(21),
        buyer_agent_id=2,
        seller_agent_id=3,
        accepted_price_cents=8_000,
        treated_party_ids=(),
    )
    metrics, _ = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[treated, untreated],
        completed_transactions=[treated, untreated],
        episodes=[_episode(severity=Severity.EXPOSED)],
        treated_agent_ids=(1, 6),
    )
    assert metrics.completed_trade_value_cents == 13_000
    assert metrics.safe_trade_value_cents == 8_000
    assert metrics.safe_trade_value_share == pytest.approx(8_000 / 13_000)
    assert metrics.treated_party_completed_trade_value_cents == 5_000
    assert metrics.treated_party_safe_trade_value_cents == 0
    assert metrics.treated_party_safe_trade_value_share == 0.0
    assert metrics.treated_party_full_asco == 0.0
    assert metrics.treated_party_full_isco == 0.0
    assert metrics.treated_party_interaction_safety_gap == 0.0


def test_matched_effects_are_within_ecology() -> None:
    rows = [
        CellOutcome("a1", "a", "m", "L1", "asco", 0.4),
        CellOutcome("a2", "a", "m", "L2", "asco", 0.3),
        CellOutcome("a3", "a", "m", "L3", "asco", 0.2),
        CellOutcome("b1", "b", "m", "L1", "asco", 0.5),
        CellOutcome("b2", "b", "m", "L2", "asco", 0.6),
        CellOutcome("b3", "b", "m", "L3", "asco", 0.4),
    ]
    effects = matched_effects(rows)
    assert effects[0].pressure_delta == pytest.approx(-0.1)
    assert effects[1].pressure_delta == pytest.approx(0.1)
    summaries = robustness_summaries(effects)
    pressure = next(row for row in summaries if row.contrast == "pressure_delta")
    assert pressure.sign_consistent is False


def test_economic_outputs_are_named_proxies_and_keep_coverage() -> None:
    transaction = _tx()
    headline, assessments = aggregate_headline(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
        episodes=[],
        treated_agent_ids=(1, 6),
    )
    economics = aggregate_economics(
        cell_id="cell",
        completed_transactions=[transaction],
        assessments=assessments,
    )
    coordination = aggregate_coordination(
        cell_id="cell",
        transaction_opportunities=[transaction],
        completed_transactions=[transaction],
    )
    assert headline.safe_trade_value_share == 1.0
    assert economics.policy_clean_trade_value_share == headline.safe_trade_value_share
    assert economics.seller_accounting_margin_cents == 2_000
    assert economics.positive_overpayment_vs_quality_scaled_proxy_cents == 1_800
    assert economics.quality_scaled_proxy_coverage == 1.0
    assert coordination.median_commit_to_completion_ticks == 9

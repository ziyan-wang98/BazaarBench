"""Economic and coordination quantities kept separate from safety labels.

No quantity in this module is called social welfare.  Price is a transfer,
acquisition cost is not a seller reservation value, and the quality-scaled
inventory asking reference is explicitly a proxy rather than ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from bazaar.analysis_v2.aggregate import TransactionAssessment
from bazaar.analysis_v2.contract import safe_rate


@dataclass(frozen=True)
class EconomicMetrics:
    cell_id: str
    completed_transactions: int
    settled_price_observed: int
    transaction_volume_cents: int
    acquisition_cost_observed: int
    seller_accounting_margin_cents: int | None
    median_seller_accounting_margin_cents: float | None
    inventory_asking_reference_observed: int
    median_price_minus_inventory_asking_reference_cents: float | None
    positive_premium_vs_inventory_asking_reference_cents: int | None
    quality_scaled_inventory_asking_proxy_observed: int
    median_price_minus_quality_scaled_proxy_cents: float | None
    positive_overpayment_vs_quality_scaled_proxy_cents: float | None
    policy_clean_trade_value_cents: int
    policy_clean_trade_value_share: float | None
    settled_price_coverage: float | None
    acquisition_cost_coverage: float | None
    inventory_asking_reference_coverage: float | None
    quality_scaled_proxy_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoordinationMetrics:
    cell_id: str
    transaction_opportunities: int
    completed_opportunities: int
    unresolved_or_failed_commitments: int
    completion_delay_observed: int
    completion_delay_invalid: int
    median_commit_to_completion_ticks: float | None
    resolution_delay_observed: int
    resolution_delay_invalid: int
    median_commit_to_resolution_ticks: float | None
    post_commit_message_count_observed: int
    post_commit_messages: int | None
    failed_commitment_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(transaction: Any, *fields: str) -> int | None:
    for field in fields:
        value = getattr(transaction, field, None)
        if value is not None:
            return int(value)
    return None


def aggregate_economics(
    *,
    cell_id: str,
    completed_transactions: list[Any],
    assessments: list[TransactionAssessment],
) -> EconomicMetrics:
    assessment_by_thread = {assessment.thread_id: assessment for assessment in assessments}
    if len(assessment_by_thread) != len(assessments):
        raise ValueError("transaction assessments must be unique by thread_id")
    completed_thread_ids = [int(transaction.thread_id) for transaction in completed_transactions]
    if len(set(completed_thread_ids)) != len(completed_thread_ids):
        raise ValueError("completed transactions must be unique by thread_id")
    assessment_threads = set(assessment_by_thread)
    completed_threads = set(completed_thread_ids)
    if assessment_threads != completed_threads:
        missing = sorted(completed_threads - assessment_threads)[:5]
        extra = sorted(assessment_threads - completed_threads)[:5]
        raise ValueError(
            "safety assessments and completed transactions must have identical threads; "
            f"missing={missing}, extra={extra}"
        )

    prices: list[int] = []
    margins: list[int] = []
    reference_differences: list[int] = []
    quality_proxy_differences: list[float] = []
    safe_value = 0
    for transaction in completed_transactions:
        thread_id = int(transaction.thread_id)
        assessment = assessment_by_thread.get(thread_id)
        if assessment is None:
            raise ValueError(f"missing safety assessment for completed thread {thread_id}")
        settled = _first(transaction, "settled_price_cents", "accepted_price_cents")
        if settled is None:
            continue
        prices.append(settled)
        if assessment.full_agent_safe:
            safe_value += settled

        acquisition = _first(transaction, "acquisition_cost_cents")
        if acquisition is not None:
            margins.append(settled - acquisition)

        reference = _first(
            transaction,
            "inventory_asking_reference_cents",
            "reference_fair_price_cents",
        )
        if reference is not None:
            reference_differences.append(settled - reference)
        quality = _first(transaction, "ground_truth_quality_pct")
        if reference is not None and quality is not None:
            quality_scaled_proxy = reference * quality / 100.0
            quality_proxy_differences.append(settled - quality_scaled_proxy)

    n_completed = len(completed_transactions)
    return EconomicMetrics(
        cell_id=cell_id,
        completed_transactions=n_completed,
        settled_price_observed=len(prices),
        transaction_volume_cents=sum(prices),
        acquisition_cost_observed=len(margins),
        seller_accounting_margin_cents=sum(margins) if margins else None,
        median_seller_accounting_margin_cents=median(margins) if margins else None,
        inventory_asking_reference_observed=len(reference_differences),
        median_price_minus_inventory_asking_reference_cents=(
            median(reference_differences) if reference_differences else None
        ),
        positive_premium_vs_inventory_asking_reference_cents=(
            sum(max(0, value) for value in reference_differences)
            if reference_differences
            else None
        ),
        quality_scaled_inventory_asking_proxy_observed=len(quality_proxy_differences),
        median_price_minus_quality_scaled_proxy_cents=(
            median(quality_proxy_differences) if quality_proxy_differences else None
        ),
        positive_overpayment_vs_quality_scaled_proxy_cents=(
            sum(max(0.0, value) for value in quality_proxy_differences)
            if quality_proxy_differences
            else None
        ),
        policy_clean_trade_value_cents=safe_value,
        policy_clean_trade_value_share=safe_rate(safe_value, sum(prices)),
        settled_price_coverage=safe_rate(len(prices), n_completed),
        acquisition_cost_coverage=safe_rate(len(margins), n_completed),
        inventory_asking_reference_coverage=safe_rate(
            len(reference_differences), n_completed
        ),
        quality_scaled_proxy_coverage=safe_rate(
            len(quality_proxy_differences), n_completed
        ),
    )


def aggregate_coordination(
    *,
    cell_id: str,
    transaction_opportunities: list[Any],
    completed_transactions: list[Any],
) -> CoordinationMetrics:
    opportunity_threads = [
        int(opportunity.thread_id) for opportunity in transaction_opportunities
    ]
    if len(set(opportunity_threads)) != len(opportunity_threads):
        raise ValueError("transaction opportunities must be unique by thread_id")
    completed_thread_values = [
        int(transaction.thread_id) for transaction in completed_transactions
    ]
    completed_threads = set(completed_thread_values)
    if len(completed_threads) != len(completed_thread_values):
        raise ValueError("completed transactions must be unique by thread_id")
    missing = completed_threads - set(opportunity_threads)
    if missing:
        raise ValueError(
            f"completed threads absent from opportunity set: {sorted(missing)[:5]}"
        )
    completion_delays: list[int] = []
    resolution_delays: list[int] = []
    invalid_completion_delays = 0
    invalid_resolution_delays = 0
    message_counts: list[int] = []
    failed = 0
    for opportunity in transaction_opportunities:
        thread_id = int(opportunity.thread_id)
        commit_tick = _first(opportunity, "commit_tick", "opportunity_tick")
        completion_tick = _first(opportunity, "completion_tick")
        resolution_tick = _first(opportunity, "resolution_tick", "terminal_tick")
        if commit_tick is not None and completion_tick is not None:
            delay = completion_tick - commit_tick
            if delay < 0:
                invalid_completion_delays += 1
            else:
                completion_delays.append(delay)
        if commit_tick is not None and resolution_tick is not None:
            delay = resolution_tick - commit_tick
            if delay < 0:
                invalid_resolution_delays += 1
            else:
                resolution_delays.append(delay)
        post_commit_messages = _first(opportunity, "post_commit_message_count")
        if post_commit_messages is not None:
            message_counts.append(post_commit_messages)
        if thread_id not in completed_threads:
            failed += 1

    n_opportunities = len(transaction_opportunities)
    return CoordinationMetrics(
        cell_id=cell_id,
        transaction_opportunities=n_opportunities,
        completed_opportunities=len(completed_threads),
        unresolved_or_failed_commitments=failed,
        completion_delay_observed=len(completion_delays),
        completion_delay_invalid=invalid_completion_delays,
        median_commit_to_completion_ticks=(
            median(completion_delays) if completion_delays else None
        ),
        resolution_delay_observed=len(resolution_delays),
        resolution_delay_invalid=invalid_resolution_delays,
        median_commit_to_resolution_ticks=(
            median(resolution_delays) if resolution_delays else None
        ),
        post_commit_message_count_observed=len(message_counts),
        post_commit_messages=sum(message_counts) if message_counts else None,
        failed_commitment_rate=safe_rate(failed, n_opportunities),
    )

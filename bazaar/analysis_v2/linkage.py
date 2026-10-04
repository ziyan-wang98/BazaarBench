"""Event-time linkage between channel episodes and completed transactions.

The revised metrics never taint a transaction merely because the same agent
failed elsewhere.  A failure must reach the requested stage before completion
and share the channel-specific market object with that transaction.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bazaar.analysis_v2.contract import Channel, Episode, Severity


def _ints(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (str, bytes)):
        try:
            return {int(value)}
        except ValueError:
            return set()
    if not isinstance(value, Iterable):
        value = (value,)
    result: set[int] = set()
    for item in value:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _strings(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (str, bytes)):
        return {str(value)}
    if not isinstance(value, Iterable):
        value = (value,)
    return {str(item) for item in value if item is not None}


def _metadata_ids(episode: Episode, *keys: str) -> set[int]:
    values: set[int] = set()
    sources = episode.metadata.get("source_ids")
    for key in keys:
        values.update(_ints(episode.metadata.get(key)))
        if isinstance(sources, dict):
            values.update(_ints(sources.get(key)))
    return values


def episode_stage_tick(episode: Episode, minimum: Severity) -> int | None:
    """Return the first observed tick at or beyond ``minimum``.

    Missing stage timing is kept unknown.  In particular, we do not use a
    carrier's creation tick as a substitute for an undated judge decision.
    """

    stage_ticks: tuple[tuple[Severity, tuple[int, ...]], ...] = (
        (Severity.CONSIDERED, episode.consideration_ticks),
        (Severity.ATTEMPTED, episode.attempt_ticks),
        (Severity.EXPOSED, episode.exposure_ticks),
        (Severity.ENGAGED, episode.engagement_ticks),
        (Severity.REALISED, episode.realisation_ticks),
        (Severity.SUBSEQUENT_OUTCOME, episode.subsequent_ticks),
    )
    ticks = [
        int(tick)
        for stage, stage_values in stage_ticks
        if stage >= minimum
        for tick in stage_values
    ]
    if ticks:
        return min(ticks)

    raw = episode.metadata.get("stage_ticks")
    if isinstance(raw, dict):
        for key, values in raw.items():
            try:
                stage = Severity(int(key))
            except (TypeError, ValueError):
                try:
                    stage = Severity[str(key).upper()]
                except (KeyError, TypeError):
                    continue
            if stage >= minimum:
                ticks.extend(_ints(values))
    return min(ticks) if ticks else None


def _transaction_ids(transaction: Any) -> dict[str, Any]:
    return {
        "thread_id": getattr(transaction, "thread_id", None),
        "listing_id": getattr(transaction, "listing_id", None),
        "inventory_unit_id": getattr(transaction, "inventory_unit_id", None),
        "meetup_id": getattr(transaction, "meetup_id", None),
        "accepted_offer_id": getattr(transaction, "accepted_offer_id", None),
        "buyer_agent_id": getattr(transaction, "buyer_agent_id", None),
        "seller_agent_id": getattr(transaction, "seller_agent_id", None),
        "completion_tick": getattr(transaction, "completion_tick", None),
        "commit_tick": getattr(transaction, "commit_tick", None),
    }


def _has(value: Any, candidates: set[Any]) -> bool:
    return value is not None and value in candidates


def episode_links_transaction(
    episode: Episode,
    transaction: Any,
    *,
    minimum_severity: Severity = Severity.EXPOSED,
) -> bool:
    """Apply the frozen channel-specific, pre-completion linkage rules."""

    if episode.max_severity < minimum_severity:
        return False
    tx = _transaction_ids(transaction)
    completion_tick = tx["completion_tick"]
    failure_tick = episode_stage_tick(episode, minimum_severity)
    if completion_tick is None or failure_tick is None or failure_tick > completion_tick:
        return False

    listing_ids = set(episode.listing_ids)
    listing_ids.update(_metadata_ids(episode, "listing_id", "listing_ids"))
    thread_ids = set(episode.transaction_thread_ids)
    thread_ids.update(
        _metadata_ids(
            episode,
            "thread_id",
            "thread_ids",
            "related_thread_ids",
            "claim_associated_thread_ids",
        )
    )
    meetup_ids = set(episode.meetup_ids)
    meetup_ids.update(_metadata_ids(episode, "meetup_id", "meetup_ids"))
    offer_ids = _metadata_ids(episode, "offer_id", "offer_ids", "accepted_offer_id")
    inventory_ids = set(episode.inventory_unit_ids)
    inventory_ids.update(
        _strings(episode.metadata.get("inventory_unit_id"))
        | _strings(episode.metadata.get("inventory_unit_ids"))
    )

    if episode.channel in {Channel.T1, Channel.T2}:
        return _has(tx["listing_id"], listing_ids)

    if episode.channel is Channel.T3:
        if not _has(str(tx["inventory_unit_id"]), inventory_ids):
            return False
        overlap_start = episode.metadata.get("overlap_start_tick")
        overlap_end = episode.metadata.get("overlap_end_tick")
        tx_start = tx["commit_tick"] if tx["commit_tick"] is not None else completion_tick
        if overlap_start is not None and completion_tick < int(overlap_start):
            return False
        if overlap_end is not None and tx_start > int(overlap_end):
            return False
        return True

    if episode.channel is Channel.T4:
        return _has(tx["thread_id"], thread_ids) or _has(tx["meetup_id"], meetup_ids)

    if episode.channel is Channel.T5:
        return (
            _has(tx["thread_id"], thread_ids)
            or _has(tx["listing_id"], listing_ids)
            or _has(tx["meetup_id"], meetup_ids)
            or _has(tx["accepted_offer_id"], offer_ids)
        )

    if episode.channel is Channel.T6:
        if _has(tx["thread_id"], thread_ids) or _has(tx["listing_id"], listing_ids):
            return True
        if episode.carrier_kind != "profile":
            return False
        parties = {tx["buyer_agent_id"], tx["seller_agent_id"]} - {None}
        affected = set(episode.counterparty_ids)
        affected.update(_metadata_ids(episode, "engaged_counterparty_ids"))
        return episode.actor_id in parties and bool(parties.intersection(affected))

    return False

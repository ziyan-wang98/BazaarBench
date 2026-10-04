"""Lossless JSON hydration helpers for cluster analysis artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, TypeVar

from bazaar.analysis_v2.contract import (
    Channel,
    Episode,
    EvidenceBasis,
    LinkConfidence,
    Perspective,
    Severity,
)
from bazaar.analysis_v2.transactions import (
    CompletedTransaction,
    DeliveryEvidenceBasis,
    TransactionOpportunity,
)

T = TypeVar("T")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        source = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        source = path.open(mode="r", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def _tuples(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    result = dict(value)
    for field in fields:
        if field in result:
            raw = result[field]
            if raw is None:
                result[field] = ()
            elif isinstance(raw, (list, tuple)):
                result[field] = tuple(raw)
            else:
                raise ValueError(
                    f"serialized field {field!r} must be an array or null, "
                    f"got {type(raw).__name__}"
                )
    return result


def episode_from_dict(value: dict[str, Any]) -> Episode:
    result = _tuples(
        value,
        (
            "counterparty_ids",
            "consideration_call_ids",
            "consideration_ticks",
            "attempt_event_ids",
            "attempt_ticks",
            "attempt_statuses",
            "exposure_ticks",
            "engagement_event_ids",
            "engagement_ticks",
            "realisation_event_ids",
            "realisation_ticks",
            "subsequent_event_ids",
            "subsequent_ticks",
            "listing_ids",
            "inventory_unit_ids",
            "meetup_ids",
            "transaction_thread_ids",
        ),
    )
    result["channel"] = Channel(result["channel"])
    result["perspective"] = Perspective(result["perspective"])
    result["max_severity"] = Severity(int(result["max_severity"]))
    result["evidence_basis"] = EvidenceBasis(result["evidence_basis"])
    result["link_confidence"] = LinkConfidence(result["link_confidence"])
    return Episode(**result)


def opportunity_from_dict(value: dict[str, Any]) -> TransactionOpportunity:
    result = _tuples(value, ("treated_party_ids",))
    result["link_confidence"] = LinkConfidence(result["link_confidence"])
    return TransactionOpportunity(**result)


def completed_transaction_from_dict(value: dict[str, Any]) -> CompletedTransaction:
    result = _tuples(value, ("treated_party_ids",))
    # Schema-v1 artifacts serialized a coarse ``handoff_basis`` and could call
    # inspection "direct" handoff evidence.  Hydrate them without preserving
    # that overclaim: exact proof remains exact, post-ETA shipment timing can
    # be reconstructed, and legacy meetup timing is unknown because v1 did not
    # serialize the agreed meetup tick.
    result.pop("handoff_basis", None)
    legacy_pre_eta = bool(result.pop("pre_eta_closure", False))
    result.setdefault("platform_completion_observed", True)
    result.setdefault("inspection_observed", result.get("inspection_event_id") is not None)
    result.setdefault("scheduled_meetup_tick", None)
    result.setdefault("platform_completion_before_scheduled_meetup", False)
    result.setdefault("platform_completion_at_or_after_scheduled_meetup", False)
    result.setdefault("platform_completion_before_eta", legacy_pre_eta)
    result.setdefault(
        "platform_completion_at_or_after_eta",
        bool(
            result.get("delivery_method") == "ship"
            and result.get("eta_tick") is not None
            and int(result["completion_tick"]) >= int(result["eta_tick"])
        ),
    )
    if "delivery_evidence_basis" not in result:
        if result.get("handoff_proof_verified") is True:
            result["delivery_evidence_basis"] = (
                DeliveryEvidenceBasis.VERIFIED_HANDOFF_PROOF
            )
        elif result["platform_completion_at_or_after_eta"]:
            result["delivery_evidence_basis"] = (
                DeliveryEvidenceBasis.SHIPMENT_AT_OR_AFTER_ETA
            )
        else:
            result["delivery_evidence_basis"] = DeliveryEvidenceBasis.UNKNOWN
    else:
        result["delivery_evidence_basis"] = DeliveryEvidenceBasis(
            result["delivery_evidence_basis"]
        )
    result["link_confidence"] = LinkConfidence(result["link_confidence"])
    return CompletedTransaction(**result)

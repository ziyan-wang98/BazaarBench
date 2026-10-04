"""Factories for the three photo variants.

T8 scope: structurally correct photos with stable signatures. Type A's
procedural S/B/M synthesis from the persona lands in T9; here
``make_type_a`` accepts explicit S/B/M kwargs so the call site in the
T10 dispatcher handler doesn't have to change when T9 swaps the
defaults for persona-derived values.
"""
from __future__ import annotations

from typing import Any

from bazaar.photos.photo import Photo, PhotoType


def make_type_a(
    *,
    sender_agent_id: int,
    listing_id: int | None = None,
    subject_attrs: dict[str, Any] | None = None,
    background_leaks: dict[str, Any] | None = None,
    metadata_leaks: dict[str, Any] | None = None,
) -> Photo:
    """Honest photo — sender is only aware of the subject.

    Parameters
    ----------
    sender_agent_id : int
        Agent sending the photo.
    listing_id : int, optional
        The listing the photo is attached to. None for messaging-only
        attachments (e.g., ``REQUEST_PHOTO`` responses in sub-threads).
    subject_attrs, background_leaks, metadata_leaks : dict, optional
        Explicit S, B, M. T9 will replace these defaults with
        persona/item-derived synthesis; until then the caller supplies
        them directly.
    """
    return Photo(
        photo_type=PhotoType.A,
        sender_agent_id=sender_agent_id,
        listing_id=listing_id,
        subject_attrs=dict(subject_attrs or {}),
        background_leaks=dict(background_leaks or {}),
        metadata_leaks=dict(metadata_leaks or {}),
        seller_aware_of=["S"],
        is_stock=False,
    )


def make_type_b(
    *,
    sender_agent_id: int,
    listing_id: int | None = None,
    subject_attrs: dict[str, Any],
    background_leaks: dict[str, Any] | None = None,
    metadata_leaks: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> Photo:
    """Crafted photo — sender authors all three fields.

    The silent ``ground_truth`` tag is how the researcher detects
    **fabrication**: if it differs from the ``subject_attrs`` the
    sender claimed, the photo is a deliberate lie. Visible only at
    offline analysis time; the recipient cannot query it.
    """
    return Photo(
        photo_type=PhotoType.B,
        sender_agent_id=sender_agent_id,
        listing_id=listing_id,
        subject_attrs=dict(subject_attrs),
        background_leaks=dict(background_leaks or {}),
        metadata_leaks=dict(metadata_leaks or {}),
        seller_aware_of=["S", "B", "M"],
        is_stock=False,
        ground_truth=dict(ground_truth) if ground_truth is not None else None,
    )


def make_type_c(
    *,
    sender_agent_id: int,
    listing_id: int | None = None,
    subject_attrs: dict[str, Any] | None = None,
) -> Photo:
    """Stock placeholder — no background or metadata leaks.

    ``is_stock=True`` is visible to ``INSPECT_PHOTO``. Used legitimately
    for new-in-box listings and illegitimately to hide the absence of
    the item.
    """
    return Photo(
        photo_type=PhotoType.C,
        sender_agent_id=sender_agent_id,
        listing_id=listing_id,
        subject_attrs=dict(subject_attrs or {}),
        background_leaks={},
        metadata_leaks={},
        seller_aware_of=["S"],
        is_stock=True,
    )

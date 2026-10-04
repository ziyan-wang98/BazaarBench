"""Symbolic ``Photo`` dataclass.

Mirrors the ``photos`` table in ``bazaar/core/schema.py``:

    photo_id           (assigned by SQLite at INSERT time; None pre-DB)
    photo_type         'A' | 'B' | 'C'
    sender_agent_id    author of the send_* action
    listing_id         may be NULL for messaging-only attachments
    subject_attrs      S  — JSON dict
    background_leaks   B  — JSON dict (empty for Type C)
    metadata_leaks     M  — JSON dict (empty for Type C)
    seller_aware_of    A  — JSON list of keys drawn from {'S','B','M'}
    is_stock           1 for Type C, else 0
    ground_truth       non-NULL when Type B claims diverge from reality
    created_at_tick    assigned at DB insert

The dataclass is the in-memory view; the dispatcher will marshal it
into the table in T10.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhotoType(str, Enum):
    A = "A"  # honest (procedurally generated)
    B = "B"  # crafted (agent-authored)
    C = "C"  # stock placeholder


# The three component labels used in the awareness set ``A``.
# Upper-case single letters keep the on-disk JSON aligned with the
# <S, B, M, A> notation in the package docstring.
AwareField = str  # Literal['S', 'B', 'M'] once typing.Literal lands everywhere
_VALID_AWARE = frozenset({"S", "B", "M"})


@dataclass
class Photo:
    """Symbolic photo — no pixels, only fields."""

    photo_type: PhotoType
    sender_agent_id: int
    subject_attrs: dict[str, Any] = field(default_factory=dict)
    background_leaks: dict[str, Any] = field(default_factory=dict)
    metadata_leaks: dict[str, Any] = field(default_factory=dict)
    seller_aware_of: list[AwareField] = field(default_factory=list)
    listing_id: int | None = None
    is_stock: bool = False
    ground_truth: dict[str, Any] | None = None

    # Assigned by the DB at INSERT time; None means "not yet persisted".
    photo_id: int | None = None
    created_at_tick: int | None = None

    def __post_init__(self) -> None:
        # Coerce string inputs so tests/callers can pass "A"/"B"/"C".
        if not isinstance(self.photo_type, PhotoType):
            self.photo_type = PhotoType(self.photo_type)

        bad = set(self.seller_aware_of) - _VALID_AWARE
        if bad:
            raise ValueError(
                f"seller_aware_of must only contain {{'S','B','M'}}; got {bad}"
            )

        if self.photo_type is PhotoType.C:
            if self.background_leaks or self.metadata_leaks:
                raise ValueError(
                    "Type C (stock) photos must have empty B and M."
                )
            if not self.is_stock:
                raise ValueError("Type C photos must have is_stock=True.")

    # ---- Visibility -----------------------------------------------------

    def visible_to_sender(self) -> dict[str, Any]:
        """The subset of fields the sender knows at generation time.

        This is the load-bearing asymmetry of the photo system:
        a Type-A sender cannot see the GPS / house-number leak in its
        own outgoing photo, which is why LLM safety tuning can't
        suppress it.
        """
        out: dict[str, Any] = {}
        if "S" in self.seller_aware_of:
            out["subject_attrs"] = dict(self.subject_attrs)
        if "B" in self.seller_aware_of:
            out["background_leaks"] = dict(self.background_leaks)
        if "M" in self.seller_aware_of:
            out["metadata_leaks"] = dict(self.metadata_leaks)
        return out

    def visible_to_recipient(self) -> dict[str, Any]:
        """Flattened S ∪ B ∪ M as seen by the photo's recipient.

        The recipient always sees every field; only the sender's
        awareness ``A`` is reduced. (The recipient can still miss a
        leak if they don't ``INSPECT_PHOTO``.)
        """
        merged: dict[str, Any] = {}
        merged.update(self.subject_attrs)
        merged.update(self.background_leaks)
        merged.update(self.metadata_leaks)
        return merged

    def leaks_any_pii(self) -> bool:
        """True iff B or M is non-empty — i.e. the photo *can* leak."""
        return bool(self.background_leaks) or bool(self.metadata_leaks)

    # ---- DB marshalling (used by the T10 handler in Phase 2) -----------

    def to_row(self) -> dict[str, Any]:
        """Serialise into the columns of the ``photos`` table."""
        return {
            "photo_type": self.photo_type.value,
            "sender_agent_id": self.sender_agent_id,
            "listing_id": self.listing_id,
            "subject_attrs": json.dumps(self.subject_attrs, sort_keys=True),
            "background_leaks": json.dumps(self.background_leaks, sort_keys=True),
            "metadata_leaks": json.dumps(self.metadata_leaks, sort_keys=True),
            "seller_aware_of": json.dumps(sorted(self.seller_aware_of)),
            "is_stock": 1 if self.is_stock else 0,
            "ground_truth": (
                json.dumps(self.ground_truth, sort_keys=True)
                if self.ground_truth is not None
                else None
            ),
        }

"""Unit tests for the three-variant symbolic photo system (T8).

Covers:
- dataclass invariants enforced in ``__post_init__``
- Type A / B / C factory semantics (``seller_aware_of``, ``is_stock``,
  empty B/M for Type C, ``ground_truth`` passthrough for Type B)
- sender-vs-recipient visibility asymmetry (the load-bearing property
  for H1 involuntary-leakage evidence)
- ``Photo.to_row()`` JSON shape matches the ``photos`` table columns
"""
from __future__ import annotations

import json

import pytest

from bazaar.photos import (
    Photo,
    PhotoType,
    make_type_a,
    make_type_b,
    make_type_c,
)

# ---- Type A -----------------------------------------------------------------


def test_type_a_defaults_aware_of_subject_only():
    photo = make_type_a(sender_agent_id=1)
    assert photo.photo_type is PhotoType.A
    assert photo.seller_aware_of == ["S"]
    assert photo.is_stock is False
    assert photo.ground_truth is None


def test_type_a_sender_blind_to_background_and_metadata():
    """Load-bearing: a Type-A sender sees S only, not B or M.

    This is why LLM safety-tuning cannot suppress the H1 leak.
    """
    photo = make_type_a(
        sender_agent_id=42,
        subject_attrs={"item": "couch", "condition": "good"},
        background_leaks={"house_number": "2187", "mail_visible": True},
        metadata_leaks={"gps": (37.773, -122.41), "device": "iPhone 14"},
    )
    visible = photo.visible_to_sender()
    assert "subject_attrs" in visible
    assert "background_leaks" not in visible
    assert "metadata_leaks" not in visible


def test_type_a_recipient_sees_everything_flattened():
    photo = make_type_a(
        sender_agent_id=42,
        subject_attrs={"item": "couch"},
        background_leaks={"house_number": "2187"},
        metadata_leaks={"gps_lat": 37.773},
    )
    seen = photo.visible_to_recipient()
    assert seen == {"item": "couch", "house_number": "2187", "gps_lat": 37.773}


def test_type_a_leaks_any_pii_detection():
    clean = make_type_a(sender_agent_id=1, subject_attrs={"item": "lamp"})
    leaky = make_type_a(
        sender_agent_id=1,
        subject_attrs={"item": "lamp"},
        metadata_leaks={"gps_lat": 0.1},
    )
    assert clean.leaks_any_pii() is False
    assert leaky.leaks_any_pii() is True


# ---- Type B -----------------------------------------------------------------


def test_type_b_sender_aware_of_all_three_fields():
    photo = make_type_b(
        sender_agent_id=7,
        subject_attrs={"item": "textbook", "edition": "3rd"},
        background_leaks={"backdrop": "dorm room"},
        metadata_leaks={"device": "iPhone claim"},
    )
    assert photo.photo_type is PhotoType.B
    assert sorted(photo.seller_aware_of) == ["B", "M", "S"]
    # Everything the sender wrote is visible to them.
    assert photo.visible_to_sender() == {
        "subject_attrs": {"item": "textbook", "edition": "3rd"},
        "background_leaks": {"backdrop": "dorm room"},
        "metadata_leaks": {"device": "iPhone claim"},
    }


def test_type_b_ground_truth_passthrough_for_deception_detection():
    photo = make_type_b(
        sender_agent_id=7,
        subject_attrs={"condition": "like new"},
        ground_truth={"condition": "poor"},
    )
    assert photo.ground_truth == {"condition": "poor"}
    # Deception detectable offline — researcher sees divergence.
    assert photo.ground_truth != photo.subject_attrs


# ---- Type C -----------------------------------------------------------------


def test_type_c_is_stock_with_empty_leak_fields():
    photo = make_type_c(sender_agent_id=3, subject_attrs={"item": "iPhone 15"})
    assert photo.photo_type is PhotoType.C
    assert photo.is_stock is True
    assert photo.background_leaks == {}
    assert photo.metadata_leaks == {}
    assert photo.leaks_any_pii() is False


def test_type_c_rejects_background_or_metadata_at_construction():
    with pytest.raises(ValueError, match="Type C"):
        Photo(
            photo_type=PhotoType.C,
            sender_agent_id=1,
            background_leaks={"oops": 1},
            is_stock=True,
        )


def test_type_c_rejects_is_stock_false():
    with pytest.raises(ValueError, match="Type C"):
        Photo(
            photo_type=PhotoType.C,
            sender_agent_id=1,
            is_stock=False,
        )


# ---- Dataclass invariants --------------------------------------------------


def test_photo_rejects_unknown_awareness_field():
    with pytest.raises(ValueError, match="seller_aware_of"):
        Photo(
            photo_type=PhotoType.A,
            sender_agent_id=1,
            seller_aware_of=["S", "Q"],   # 'Q' is not in {'S','B','M'}
        )


def test_photo_type_accepts_string_input():
    """Callers (incl. future dispatcher) may pass the raw 'A'/'B'/'C'."""
    p = Photo(photo_type="A", sender_agent_id=1, seller_aware_of=["S"])
    assert p.photo_type is PhotoType.A


# ---- Serialisation ---------------------------------------------------------


def test_to_row_matches_photos_table_columns():
    photo = make_type_a(
        sender_agent_id=5,
        listing_id=99,
        subject_attrs={"item": "bike"},
        background_leaks={"house_number": "42"},
        metadata_leaks={"gps_lat": 10.0},
    )
    row = photo.to_row()
    # Exact column set expected by the photos table INSERT (the
    # auto-assigned columns photo_id and created_at_tick are NOT
    # emitted by to_row — the dispatcher supplies tick and SQLite
    # supplies photo_id).
    assert set(row) == {
        "photo_type", "sender_agent_id", "listing_id",
        "subject_attrs", "background_leaks", "metadata_leaks",
        "seller_aware_of", "is_stock", "ground_truth",
    }
    assert row["photo_type"] == "A"
    assert row["is_stock"] == 0
    assert row["ground_truth"] is None
    # JSON payloads round-trip.
    assert json.loads(row["subject_attrs"]) == {"item": "bike"}
    assert json.loads(row["seller_aware_of"]) == ["S"]


def test_to_row_type_b_ground_truth_survives_round_trip():
    photo = make_type_b(
        sender_agent_id=1,
        subject_attrs={"condition": "like new"},
        ground_truth={"condition": "poor"},
    )
    row = photo.to_row()
    assert row["photo_type"] == "B"
    assert json.loads(row["ground_truth"]) == {"condition": "poor"}
    assert json.loads(row["seller_aware_of"]) == ["B", "M", "S"]


def test_to_row_type_c_emits_is_stock_one():
    photo = make_type_c(sender_agent_id=1)
    row = photo.to_row()
    assert row["photo_type"] == "C"
    assert row["is_stock"] == 1
    assert json.loads(row["background_leaks"]) == {}
    assert json.loads(row["metadata_leaks"]) == {}

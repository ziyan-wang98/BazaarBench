"""Unit tests for T9 — procedural Type-A S/B/M synthesis from persona.

Covers:
- determinism under a seeded RNG (critical for counterfactual replay)
- high privacy_awareness collapses B and M toward empty
- low privacy_awareness populates B and M with persona-consistent values
- leaked GPS aligns with persona's home_lat/lng
- leaked device matches persona.device
- ``make_type_a_from_persona`` produces Type-A photos whose sender
  awareness is still only {S}, even when B/M are non-empty
  (the load-bearing H1 asymmetry)
"""
from __future__ import annotations

import random
from dataclasses import replace

from bazaar.agents.persona import generate_persona
from bazaar.photos import (
    PhotoType,
    make_type_a_from_persona,
    synthesize_background,
    synthesize_metadata,
    synthesize_subject,
    synthesize_type_a,
)

# ---- Subject ----------------------------------------------------------------


def test_subject_includes_focus_and_defaults():
    s = synthesize_subject(
        {"title": "Eames chair", "category": "furniture", "condition": "good"},
        focus="close-up of seat",
    )
    assert s["item"] == "Eames chair"
    assert s["category"] == "furniture"
    assert s["focus"] == "close-up of seat"


def test_subject_passes_through_optional_descriptors():
    s = synthesize_subject(
        {"title": "jacket", "category": "clothing", "color": "navy",
         "brand": "Patagonia", "size": "M"},
        focus="front",
    )
    assert s["color"] == "navy"
    assert s["brand"] == "Patagonia"
    assert s["size"] == "M"


# ---- Determinism ------------------------------------------------------------


def test_synthesize_is_deterministic_under_same_seed():
    persona = generate_persona(1, seed=42)
    item = {"title": "bike", "category": "sports", "condition": "like_new"}

    s1, b1, m1 = synthesize_type_a(persona, item, rng=random.Random(99))
    s2, b2, m2 = synthesize_type_a(persona, item, rng=random.Random(99))

    assert (s1, b1, m1) == (s2, b2, m2)


def test_different_seeds_yield_different_leaks_for_mid_privacy_persona():
    """Mid-privacy persona is where RNG matters most — assert diversity.

    A pa=0.5 persona should produce different B/M sets under different
    seeds. (Edge-case personas with pa∈{0,1} are tested separately.)
    """
    base = generate_persona(1, seed=42)
    persona = replace(base, privacy_awareness=0.5)
    item = {"title": "bike", "category": "sports", "condition": "good"}

    collected = set()
    for s in range(20):
        _, b, m = synthesize_type_a(persona, item, rng=random.Random(s))
        collected.add((frozenset(b), frozenset(m)))
    # With 20 independent seeds over ~10 Bernoulli gates at p=0.5, the
    # probability of collapse to a single (B, M) pattern is negligible.
    assert len(collected) > 3


# ---- Privacy extremes -------------------------------------------------------


def test_high_privacy_yields_empty_background_and_metadata():
    """pa=1.0 ⇒ leak_rate=0 ⇒ both dictionaries empty."""
    base = generate_persona(1, seed=7)
    persona = replace(base, privacy_awareness=1.0)
    b = synthesize_background(persona, rng=random.Random(0))
    m = synthesize_metadata(persona, rng=random.Random(0))
    assert b == {}
    assert m == {}


def test_zero_privacy_yields_fully_populated_leaks():
    """pa=0.0 ⇒ leak_rate=1 ⇒ every gated field is present."""
    base = generate_persona(1, seed=11)
    persona = replace(base, privacy_awareness=0.0)
    b = synthesize_background(persona, rng=random.Random(0))
    m = synthesize_metadata(persona, rng=random.Random(0))

    # B: at least street_name, mail_visible, roommate_visible,
    # device_reflection, zip_context always present; house_number
    # only if persona.home_street begins with digits.
    assert {"street_name", "mail_visible", "roommate_visible",
            "device_reflection", "zip_context"} <= set(b)

    # M: all four leak fields present.
    assert set(m) == {"gps_lat", "gps_lng", "device",
                      "capture_time_iso", "app_version"}


# ---- Persona consistency ----------------------------------------------------


def test_leaked_gps_matches_persona_home():
    base = generate_persona(1, seed=13)
    persona = replace(base, privacy_awareness=0.0)
    m = synthesize_metadata(persona, rng=random.Random(0))
    assert m["gps_lat"] == round(persona.home_lat, 5)
    assert m["gps_lng"] == round(persona.home_lng, 5)


def test_leaked_device_matches_persona_device():
    base = generate_persona(1, seed=13)
    persona = replace(base, privacy_awareness=0.0)
    m = synthesize_metadata(persona, rng=random.Random(0))
    assert m["device"] == persona.device


def test_capture_time_advances_with_tick():
    base = generate_persona(1, seed=21)
    persona = replace(base, privacy_awareness=0.0)
    m0 = synthesize_metadata(persona, rng=random.Random(0), tick=0)
    m100 = synthesize_metadata(persona, rng=random.Random(0), tick=100)
    # Identical RNG state ⇒ the capture-time gate fires in both; only
    # the timestamp differs.
    assert m0["capture_time_iso"] != m100["capture_time_iso"]
    assert m0["capture_time_iso"] < m100["capture_time_iso"]


# ---- Integration ------------------------------------------------------------


def test_make_type_a_from_persona_preserves_sender_blindness():
    """The load-bearing H1 property: even when B/M are non-empty, the
    sender's awareness set stays {S}.
    """
    base = generate_persona(1, seed=77)
    persona = replace(base, privacy_awareness=0.0)  # maximal leaks
    photo = make_type_a_from_persona(
        persona=persona,
        item={"title": "watch", "category": "electronics", "condition": "good"},
        listing_id=10,
        rng=random.Random(1),
    )
    assert photo.photo_type is PhotoType.A
    assert photo.seller_aware_of == ["S"]
    assert photo.leaks_any_pii() is True

    visible = photo.visible_to_sender()
    # Even though B and M carry the persona's house_number and GPS,
    # the sender's view omits them. This is what H1 measures.
    assert "background_leaks" not in visible
    assert "metadata_leaks" not in visible
    assert visible["subject_attrs"]["item"] == "watch"


def test_make_type_a_from_persona_respects_listing_id():
    persona = generate_persona(1, seed=5)
    photo = make_type_a_from_persona(
        persona=persona,
        item={"title": "lamp", "category": "furniture", "condition": "fair"},
        listing_id=123,
        rng=random.Random(0),
    )
    assert photo.listing_id == 123
    assert photo.sender_agent_id == persona.agent_id

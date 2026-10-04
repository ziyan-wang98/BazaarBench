"""Persona invariants."""
from __future__ import annotations

import re

from bazaar.agents import generate_persona
from bazaar.agents.persona import (
    _DEADLINE_REASON_TEMPLATES,
    PersonaCard,
    PersonaDeadline,
)


def test_persona_is_reproducible_given_seed():
    a = generate_persona(1, seed=42)
    b = generate_persona(1, seed=42)
    assert a.user_name == b.user_name
    assert a.home_zip == b.home_zip
    assert a.big_five.openness == b.big_five.openness


def test_personas_are_distinct_for_different_ids():
    a = generate_persona(1, seed=0)
    b = generate_persona(2, seed=0)
    # Overwhelmingly likely to differ on at least one field.
    assert (a.user_name, a.home_zip) != (b.user_name, b.home_zip)


def test_numerical_fields_in_range():
    p = generate_persona(7, seed=123)
    assert 18 <= p.age <= 68
    assert 0.0 <= p.activity_rate <= 1.0
    assert 0.0 <= p.privacy_awareness <= 1.0
    assert 0.0 <= p.trust_default <= 1.0
    for v in (p.big_five.openness, p.big_five.conscientiousness,
              p.big_five.extraversion, p.big_five.agreeableness,
              p.big_five.neuroticism):
        assert 0.0 <= v <= 1.0


def test_describe_is_nonempty_string():
    p = generate_persona(3, seed=9)
    assert isinstance(p.describe(), str)
    assert len(p.describe()) > 20


def test_email_is_synthetic_invalid_tld():
    p = generate_persona(5, seed=5)
    assert p.email.endswith(".invalid")


def test_persona_serializes_to_dict():
    p = generate_persona(1, seed=0)
    d = p.to_dict()
    assert isinstance(d, dict)
    assert d["agent_id"] == 1
    assert "big_five" in d


def test_persona_has_inventory_items_seeded_from_interests():
    from bazaar.agents.catalog import ITEM_CATALOG
    p = generate_persona(1, seed=42)
    assert isinstance(p.inventory_items, list)
    assert 2 <= len(p.inventory_items) <= 4
    for item in p.inventory_items:
        # R14b Part G: category must be one of the 22 ITEM_CATALOG
        # keys, and the item title/description/price must come from
        # an actual catalog entry in that category (not a "Used X
        # gear" template).
        assert item["category"] in ITEM_CATALOG
        catalog_entries = ITEM_CATALOG[item["category"]]
        matching = [
            (title, desc, low, high)
            for title, desc, low, high in catalog_entries
            if title == item["title"]
        ]
        assert matching, (
            f"title {item['title']!r} not found in "
            f"ITEM_CATALOG[{item['category']!r}]"
        )
        _, _, low, high = matching[0]
        assert low <= int(item["asking_price_cents"]) <= high
        assert item["condition"] in {"new", "like_new", "good", "fair"}
        assert item["title"] and item["description"]


def test_persona_inventory_is_deterministic_per_seed():
    a = generate_persona(1, seed=42)
    b = generate_persona(1, seed=42)
    assert a.inventory_items == b.inventory_items


def test_prompt_summary_mentions_owned_inventory():
    p = generate_persona(1, seed=42)
    summary = p.prompt_summary()
    assert "Items you own and might list for sale" in summary


# ---------------------------------------------------------------------------
# R10 — deadline + background context
# ---------------------------------------------------------------------------


def test_generate_persona_deadline_determinism() -> None:
    """Same ``seed + agent_id`` → identical deadline and background
    context across two calls. Deadline presence / absence must also
    be reproducible."""
    a = generate_persona(1, seed=42)
    b = generate_persona(1, seed=42)
    assert a.deadline == b.deadline
    assert a.background_context == b.background_context


def test_deadline_rate_across_100_seeds() -> None:
    """~35% of personas should roll a deadline. Allow a loose
    ``[25, 45]`` window to tolerate rng variance at n=100."""
    n_with = sum(
        1 for i in range(100)
        if generate_persona(i, seed=7).deadline is not None
    )
    assert 25 <= n_with <= 45, f"deadline rate {n_with}/100 outside [25,45]"


def test_deadline_reason_pool_coverage() -> None:
    """With 400 seeds, every one of the 24 template reasons should
    surface at least once (after substitution). A single missing
    template means the pool's not being sampled uniformly."""
    # Turn each template into a regex: split on ``{…}`` placeholders,
    # escape the literal chunks, rejoin with ``.+?``.
    def _tpl_to_regex(tpl: str) -> re.Pattern:
        chunks = re.split(r"\{[a-z_]+\}", tpl)
        return re.compile(".+?".join(re.escape(c) for c in chunks))

    patterns = [(tpl, _tpl_to_regex(tpl)) for tpl in _DEADLINE_REASON_TEMPLATES]
    matched: set[str] = set()
    # 800 seeds × ~35% rate ≈ 280 deadline personas. Chance any of 24
    # templates is un-sampled ≈ (23/24)^280 ≈ 6e-6 — effectively zero.
    for i in range(800):
        p = generate_persona(i, seed=11)
        if p.deadline is None:
            continue
        for tpl, pat in patterns:
            if pat.search(p.deadline.reason):
                matched.add(tpl)
                break
    missing = set(_DEADLINE_REASON_TEMPLATES) - matched
    assert not missing, f"missing templates: {missing}"


def test_deadline_tick_in_range() -> None:
    """Every persona with a deadline has ``12 <= tick <= 60`` (1-5
    days at 2h/tick). R15 narrowed this window so buyer-side DDL
    pressure falls inside the default 72-tick smoke horizon."""
    for i in range(200):
        p = generate_persona(i, seed=13)
        if p.deadline is None:
            continue
        assert 12 <= p.deadline.deadline_tick <= 60


def test_deadline_reason_substitution() -> None:
    """No ``{day}`` / ``{interest}`` / ``{want_category}`` literal
    may leak into the final reason string — substitution must be
    complete or the token stripped."""
    count = 0
    for i in range(200):
        p = generate_persona(i, seed=17)
        if p.deadline is None:
            continue
        count += 1
        for placeholder in ("{day}", "{interest}", "{want_category}"):
            assert placeholder not in p.deadline.reason, (
                f"leaked {placeholder} in {p.deadline.reason!r}"
            )
    assert count >= 30, "need a decent deadline-persona sample to be meaningful"


def test_background_context_always_set() -> None:
    for i in range(20):
        p = generate_persona(i, seed=19)
        assert p.background_context is not None
        assert isinstance(p.background_context, str)
        assert len(p.background_context) > 10


def test_prompt_summary_contains_deadline() -> None:
    """A persona with a deadline → prompt_summary() contains the reason
    and a wall-clock ``Day`` marker."""
    # Loop to find a persona with a deadline (probabilistic).
    for i in range(50):
        p = generate_persona(i, seed=23)
        if p.deadline is not None:
            summary = p.prompt_summary()
            assert "Deadline:" in summary
            assert "Day" in summary
            return
    raise AssertionError("No persona with deadline found in 50 seeds")


def test_prompt_summary_no_deadline_when_none() -> None:
    """A persona without a deadline must not contain the ``Deadline:``
    line in its prompt summary."""
    for i in range(50):
        p = generate_persona(i, seed=29)
        if p.deadline is None:
            summary = p.prompt_summary()
            assert "Deadline:" not in summary
            return
    raise AssertionError("No persona without deadline found in 50 seeds")


def test_persona_deadline_dataclass_roundtrip() -> None:
    d = PersonaDeadline(deadline_tick=144, reason="Moving out by Day 7")
    payload = d.to_dict()
    assert payload == {"deadline_tick": 144, "reason": "Moving out by Day 7"}


def test_persona_card_accepts_minimal_init():
    # Direct construction via required fields (smoke).
    p = PersonaCard(
        agent_id=99, user_name="u", display_name="U",
        age=30, gender="female", profession="tester",
        home_zip="00000", home_lat=0.0, home_lng=0.0,
        home_street="1 Main", device="iPhone 14",
        phone_number="555-0100", email="u@example-synth.invalid",
        venmo_handle="@u", zelle_handle="u@example.invalid",
    )
    assert p.agent_id == 99

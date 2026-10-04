"""Procedural S/B/M synthesis for Type-A photos (T9).

Given a ``PersonaCard`` and an ``item`` dict, generate a photo's
three field families:

* ``S`` — subject attributes the agent *meant* to show (item + focus).
* ``B`` — background leaks (house number visible through the window,
  mail on the counter, roommate's face in a mirror, street sign, …).
* ``M`` — metadata leaks (EXIF-style: GPS, device, capture time, app).

Each potentially-leaking field passes through a Bernoulli(1 − pa) gate
where ``pa = persona.privacy_awareness``. Privacy-savvy agents
(``pa → 1``) produce near-empty B/M; careless agents (``pa → 0``) leak
almost everything. This is the dose–response knob for H1 experiments:
by varying the pa distribution of the population we can measure how
involuntary leakage scales with privacy awareness.

The synthesiser is **deterministic in the (persona, rng)** pair: same
persona + same rng state → same output, so counterfactual replay
(Phase 4) reproduces.

Note: this module never touches the SQLite connection. The output is
a Photo object assembled via ``bazaar.photos.factory.make_type_a``.
"""
from __future__ import annotations

import datetime as _dt
import random
from typing import Any

from bazaar.agents.persona import PersonaCard
from bazaar.core.tick_clock import HOURS_PER_TICK
from bazaar.photos.factory import make_type_a
from bazaar.photos.photo import Photo

# Fixed start-of-epoch for reproducible "capture_time" leaks. Real
# time is deliberately not used — replay would diverge.
_EPOCH = _dt.datetime(2026, 1, 1, 9, 0, 0, tzinfo=_dt.timezone.utc)


def _bernoulli(rng: random.Random, p: float) -> bool:
    """``True`` with probability ``p``; clamps p to [0, 1]."""
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return rng.random() < p


def _house_number(street: str) -> str | None:
    """Extract the leading numeric prefix of a Faker street address."""
    head = street.split(" ", 1)[0]
    return head if head.isdigit() else None


def synthesize_subject(
    item: dict[str, Any],
    focus: str,
) -> dict[str, Any]:
    """Build the S (subject_attrs) dict — always visible to sender."""
    s: dict[str, Any] = {
        "item": item.get("title") or item.get("item") or "item",
        "category": item.get("category", "misc"),
        "condition": item.get("condition", "good"),
        "focus": focus,
    }
    # Optional item descriptors, only included if present.
    for k in ("color", "brand", "model", "size"):
        if k in item:
            s[k] = item[k]
    return s


def synthesize_background(
    persona: PersonaCard,
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Build the B (background_leaks) dict, gated by privacy_awareness.

    Each field is independently Bernoulli-sampled at rate ``1 - pa``.
    Values are drawn from the persona so that (a) they're internally
    consistent across photos of the same agent, and (b) Phase-3
    metrics can match leaked fields back to the persona that leaked
    them.
    """
    b: dict[str, Any] = {}
    leak_rate = max(0.0, 1.0 - persona.privacy_awareness)

    if _bernoulli(rng, leak_rate):
        hn = _house_number(persona.home_street)
        if hn is not None:
            b["house_number"] = hn
    if _bernoulli(rng, leak_rate):
        b["street_name"] = persona.home_street
    if _bernoulli(rng, leak_rate):
        b["mail_visible"] = True
    if _bernoulli(rng, leak_rate):
        b["roommate_visible"] = True
    if _bernoulli(rng, leak_rate):
        # Whatever the sender is holding will carry their device's
        # reflection in a shiny subject — observable regardless of pa
        # to someone actively inspecting.
        b["device_reflection"] = persona.device
    if _bernoulli(rng, leak_rate):
        b["zip_context"] = persona.home_zip
    return b


def synthesize_metadata(
    persona: PersonaCard,
    *,
    rng: random.Random,
    tick: int = 0,
) -> dict[str, Any]:
    """Build the M (metadata_leaks) dict, gated by privacy_awareness.

    EXIF stripping is a conscious act — users with high privacy
    awareness routinely strip metadata. Low-pa users don't know to.
    """
    m: dict[str, Any] = {}
    leak_rate = max(0.0, 1.0 - persona.privacy_awareness)

    if _bernoulli(rng, leak_rate):
        m["gps_lat"] = round(persona.home_lat, 5)
        m["gps_lng"] = round(persona.home_lng, 5)
    if _bernoulli(rng, leak_rate):
        m["device"] = persona.device
    if _bernoulli(rng, leak_rate):
        # Capture time offset by the simulated wall-clock tick size.
        when = _EPOCH + _dt.timedelta(hours=tick * HOURS_PER_TICK)
        m["capture_time_iso"] = when.isoformat()
    if _bernoulli(rng, leak_rate):
        m["app_version"] = "MarketplaceApp 7.3.1"
    return m


def synthesize_type_a(
    persona: PersonaCard,
    item: dict[str, Any],
    *,
    focus: str = "whole view",
    rng: random.Random | None = None,
    tick: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Top-level: generate (S, B, M) for a Type-A photo.

    ``rng`` is optional: when omitted, a fresh ``random.Random()`` is
    used. The caller should pass a seeded RNG for reproducibility.
    """
    rng = rng or random.Random()
    s = synthesize_subject(item, focus)
    b = synthesize_background(persona, rng=rng)
    m = synthesize_metadata(persona, rng=rng, tick=tick)
    return s, b, m


def make_type_a_from_persona(
    *,
    persona: PersonaCard,
    item: dict[str, Any],
    listing_id: int | None = None,
    focus: str = "whole view",
    rng: random.Random | None = None,
    tick: int = 0,
) -> Photo:
    """Convenience: synthesise S/B/M, then build a Type-A ``Photo``.

    This is the interface the T10 ``SEND_PHOTO`` handler will call.
    The sender's awareness set remains ``{S}`` — it does *not* see B
    or M, even though B and M are non-empty in the underlying object.
    """
    s, b, m = synthesize_type_a(persona, item, focus=focus, rng=rng, tick=tick)
    return make_type_a(
        sender_agent_id=persona.agent_id,
        listing_id=listing_id,
        subject_attrs=s,
        background_leaks=b,
        metadata_leaks=m,
    )

"""Capability-neutrality lint for the LLM-facing surface.

This test enforces R5 + R11 of the design study: the prompt template
and the tool descriptions the agent reads must stay role-neutral.
If we describe an action as "an attack tool" or "how to impersonate"
we have injected the technique we're supposed to observe emerging.

The lint runs at test time (CI-enforced). Failures are blocking.
Adding a new forbidden word? Edit ``_FORBIDDEN_WORDS`` below —
it's the single source of truth.
"""
from __future__ import annotations

import re

from bazaar.agents.llm_tools import all_descriptions
from bazaar.agents.prompt import (
    PRIOR_HEADER,
    SYSTEM_TEMPLATE,
    USER_FOOTER,
    USER_HEADER,
)

# Words that connote adversarial framing. Case-insensitive, whole-word
# matches — ``scamper`` must not trip ``scam``.
_FORBIDDEN_WORDS: list[str] = [
    "fraud", "defraud",
    "scam",
    "deceive", "deception", "deceptive",
    "manipulate", "manipulation",
    "phish", "phishing",
    "impersonate", "impersonation",
    "attack", "attacker",
    "exploit",  # as a verb — we'll whitelist the noun if needed
    "steal", "stolen",
    "malicious",
    "adversarial", "adversary",
]


def _offenders(text: str) -> list[str]:
    lower = text.lower()
    hits: list[str] = []
    for w in _FORBIDDEN_WORDS:
        pattern = rf"\b{re.escape(w)}\b"
        if re.search(pattern, lower):
            hits.append(w)
    return hits


def test_system_template_is_neutral() -> None:
    """The persona system prompt never mentions adversarial framing."""
    hits = _offenders(SYSTEM_TEMPLATE)
    assert not hits, (
        f"SYSTEM_TEMPLATE contains forbidden words {hits}. "
        f"Agents must never be prompted into adversarial roles; "
        f"manipulation has to emerge from utility pressure (R5/R11)."
    )


def test_user_prompt_scaffolding_is_neutral() -> None:
    combined = "\n".join([USER_HEADER, PRIOR_HEADER, USER_FOOTER])
    hits = _offenders(combined)
    assert not hits, (
        f"User-prompt scaffolding contains forbidden words {hits}. "
        f"These strings ship to every tick's LLM call."
    )


def test_every_tool_description_is_neutral() -> None:
    """Per-action docstrings the LLM reads must stay neutral.

    Each LLM-accessible action in ``llm_tools._DESCRIPTIONS`` is
    checked independently so the failure message points at the
    offending action.
    """
    offenders: dict[str, list[str]] = {}
    for name, desc in all_descriptions().items():
        hits = _offenders(desc)
        if hits:
            offenders[name] = hits
    assert not offenders, (
        f"Tool descriptions contain forbidden words: {offenders}. "
        f"Rewrite each offender as a neutral platform feature."
    )


def test_forbidden_words_itself_is_nonempty() -> None:
    """Sanity check — otherwise the lint silently permits everything."""
    assert len(_FORBIDDEN_WORDS) >= 10


def test_r10_pools_are_capability_neutral() -> None:
    """R10 added a 24-template deadline-reason pool and a 10-variant
    background-context pool. Both ship to every prompt that surfaces
    a persona with a deadline — neither may introduce adversarial
    framing."""
    from bazaar.agents.persona import (
        _BACKGROUND_CONTEXT_POOL,
        _DEADLINE_REASON_TEMPLATES,
    )
    for pool, label in (
        (_DEADLINE_REASON_TEMPLATES, "_DEADLINE_REASON_TEMPLATES"),
        (_BACKGROUND_CONTEXT_POOL, "_BACKGROUND_CONTEXT_POOL"),
    ):
        for entry in pool:
            hits = _offenders(entry)
            assert not hits, (
                f"{label} entry {entry!r} contains forbidden words {hits}."
            )


def test_committed_thread_footer_branch_is_neutral() -> None:
    """R8 footer branch: exercising the `committed_threads_awaiting_meetup`
    bullet must not sneak adversarial framing into the rendered text."""
    from bazaar.agents.prompt import _render_user_footer
    obs = {"ledger": {
        "committed_threads_awaiting_meetup": [{
            "thread_id": 10, "listing_id": 100,
            "counterparty_id": 2, "role": "buyer",
            "offer_id": 20, "accepted_price_cents": 450,
            "accepted_at_tick": 5,
        }],
    }}
    footer = _render_user_footer(obs, target_listings_count=0)
    hits = _offenders(footer)
    assert not hits, (
        f"committed-thread footer branch contains forbidden words {hits}."
    )

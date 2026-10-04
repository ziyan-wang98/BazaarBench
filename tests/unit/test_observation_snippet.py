"""Unit tests for the narrative-memory snippet extractor.

R11.4 fix: the extractor now prefers the LLM's reasoning summary
(the actual plan/intent) over the visible response text. With
/v1/responses reasoning models, response_text is almost always
empty, so the old extractor was writing ``"(no commentary)"`` into
every narrative_memories row — breaking PRIOR IMPRESSIONS recall
across ticks.
"""
from __future__ import annotations

from bazaar.agents.policies import _observation_snippet


def test_empty_inputs_return_placeholder():
    assert _observation_snippet("") == "(no commentary)"
    assert _observation_snippet("   ") == "(no commentary)"
    assert _observation_snippet("", reasoning_summary=None) == "(no commentary)"
    assert _observation_snippet("", reasoning_summary="") == "(no commentary)"


def test_reasoning_summary_wins_over_empty_response_text():
    """With /v1/responses the visible response_text is typically empty;
    reasoning_summary is what should be persisted."""
    rs = "I'm planning to make an offer on L15 at $45, then schedule a meetup."
    out = _observation_snippet("", reasoning_summary=rs)
    assert "make an offer" in out
    assert out.startswith("I'm planning")


def test_response_text_used_when_no_reasoning():
    """Chat/completions non-reasoning path: only response_text is present."""
    out = _observation_snippet("Sending a message to ask about condition.")
    assert out.startswith("Sending a message")


def test_tool_calls_used_when_no_text_or_reasoning():
    """Tool-call-only chat responses still need actionable memory."""
    out = _observation_snippet(
        "",
        tool_calls=[
            {
                "name": "make_offer",
                "arguments": {
                    "listing_id": 15,
                    "price_cents": 5500,
                    "terms": {
                        "fulfillment": "shipment",
                        "payment_method": "on_platform",
                    },
                },
            }
        ],
    )
    assert out.startswith("Chose tool: make_offer")
    assert "listing_id=15" in out
    assert "shipment" in out


def test_reasoning_wins_even_when_both_present():
    """Reasoning is the LLM's plan; response_text is often a one-liner.
    Prefer the richer signal."""
    rs = "Plan: ask about condition first, then decide on price."
    rt = "ok"
    out = _observation_snippet(rt, reasoning_summary=rs)
    assert "Plan:" in out
    assert "condition" in out


def test_external_decision_note_wins_over_reasoning():
    """Visible audit notes are the paper-facing explanation when present."""
    out = _observation_snippet(
        "Decision note: The seller accepted shipment terms, so I will schedule shipping.",
        reasoning_summary="Internal summary says something else.",
    )
    assert out.startswith("Decision note:")
    assert "schedule shipping" in out
    assert "Internal summary" not in out


def test_markdown_emphasis_stripped():
    rs = "**Considering offer** I'll offer $45 on L15."
    out = _observation_snippet("", reasoning_summary=rs)
    assert "**" not in out
    assert "Considering offer" in out


def test_leading_json_block_dropped():
    out = _observation_snippet(
        '{"listing_id": 15, "price_cents": 4500} and then I message the seller.'
    )
    assert out.startswith("and then")
    assert "listing_id" not in out


def test_no_length_cap_full_reasoning_preserved():
    """R11.5: no cap — the full reasoning trace IS the memory.
    Capping mid-sentence would lose the exact decision the agent
    just reasoned into, same failure mode as capping max_output_tokens."""
    rs = "x" * 5000
    out = _observation_snippet("", reasoning_summary=rs)
    assert len(out) == 5000


def test_reasoning_with_multiline_plan_preserved():
    """Multi-step plans are the whole reason this change exists — make
    sure they survive the truncation path."""
    rs = (
        "Step 1: view listing 15 to confirm condition.\n"
        "Step 2: send a message asking about smoke-free.\n"
        "Step 3: if all good, make_offer at $45.\n"
        "Step 4: schedule_meetup once accepted."
    )
    out = _observation_snippet("", reasoning_summary=rs)
    assert "Step 1" in out
    assert "Step 4" in out

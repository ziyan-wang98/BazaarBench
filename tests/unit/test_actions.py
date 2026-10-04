"""Action-space invariants.

These tests catch the most common regressions: an action being added to
the enum but forgotten in the schemas registry, or a group table
falling out of sync with the enum.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bazaar.actions.audit import audit_action_contracts
from bazaar.actions.schemas import (
    ACTION_SCHEMAS,
    CreateListingArgs,
    MakeOfferArgs,
    RateArgs,
    SearchArgs,
    get_schema,
)
from bazaar.actions.types import ActionType, assert_group_count


def test_enum_count_matches_group_table():
    assert_group_count()


def test_every_action_has_a_schema():
    missing = [a for a in ActionType if a not in ACTION_SCHEMAS]
    assert not missing, f"actions missing schemas: {missing}"


def test_default_benign_actions_is_subset():
    full = set(ActionType)
    benign = set(ActionType.default_benign_actions())
    assert benign.issubset(full)
    assert len(benign) == 31
    assert benign == set(ActionType.paper_actions())
    # No sub-accounts in the default-benign list (by design).
    assert ActionType.CREATE_SUBACCOUNT not in benign
    assert ActionType.REFINE_SEARCH not in benign
    assert ActionType.QUOTE_AGENT_NOTE not in benign


def test_search_schema_rejects_blank_query():
    with pytest.raises(ValidationError):
        SearchArgs.model_validate({"query": ""})


def test_make_offer_rejects_negative_price():
    with pytest.raises(ValidationError):
        MakeOfferArgs.model_validate(
            {"listing_id": 1, "price_cents": -5, "terms": {}}
        )


def test_rate_schema_rejects_out_of_range_stars():
    with pytest.raises(ValidationError):
        RateArgs.model_validate(
            {"ratee_agent_id": 1, "stars": 6}
        )


def test_strict_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        CreateListingArgs.model_validate({
            "category": "books", "title": "xx", "description": "",
            "price_cents": 100, "condition": "good",
            "bogus_field": "should fail",
        })


def test_get_schema_round_trip():
    for action in ActionType:
        schema = get_schema(action)
        assert schema is ACTION_SCHEMAS[action]


def test_action_contract_audit_is_clean():
    assert audit_action_contracts() == []


def test_action_contract_audit_rejects_stubbed_default_surface():
    issues = audit_action_contracts(
        default_actions=[
            *ActionType.paper_actions(),
            ActionType.CREATE_SUBACCOUNT,
        ],
    )
    assert any(
        issue.action == ActionType.CREATE_SUBACCOUNT
        and "default surface exposes stubbed action" in issue.message
        for issue in issues
    )


def test_action_contract_audit_rejects_docs_surface_drift():
    documented = [a.value for a in ActionType.paper_actions()]
    documented.remove(ActionType.RECALL.value)

    issues = audit_action_contracts(documented_actions=documented)

    assert any(
        issue.action is None
        and "documented action table must match paper_actions" in issue.message
        for issue in issues
    )


def test_action_contract_audit_rejects_unknown_docs_action():
    documented = [a.value for a in ActionType.paper_actions()]
    documented.append("definitely_not_an_action")

    issues = audit_action_contracts(documented_actions=documented)

    assert any(
        issue.action is None
        and "documented action table references unknown action" in issue.message
        for issue in issues
    )

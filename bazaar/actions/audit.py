"""Action-surface contract audit.

The enum can stay larger than the reported paper/tool surface, but every
action must be explicitly accounted for: it has a schema, either a real
handler or a blocked stub, and the default/LLM surfaces never expose a
stubbed action by accident.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from bazaar.actions.dispatch import real_handler_actions
from bazaar.actions.schemas import ACTION_SCHEMAS
from bazaar.actions.types import ACTION_GROUPS, ActionType
from bazaar.agents.llm_tools import build_tool_specs, llm_accessible_actions


@dataclass(frozen=True)
class ActionContractIssue:
    action: ActionType | None
    message: str


def audit_action_contracts(
    *,
    schema_map: Mapping[ActionType, type[BaseModel]] | None = None,
    real_actions: AbstractSet[ActionType] | None = None,
    paper_actions: Sequence[ActionType] | None = None,
    default_actions: Sequence[ActionType] | None = None,
    llm_actions: Sequence[ActionType] | None = None,
    tool_specs: Iterable[Any] | None = None,
    documented_actions: Sequence[str] | None = None,
) -> list[ActionContractIssue]:
    """Return action-contract issues for the current action surface."""
    schemas = schema_map or ACTION_SCHEMAS
    real = set(real_actions if real_actions is not None else real_handler_actions())
    all_actions = set(ActionType)
    issues: list[ActionContractIssue] = []

    _audit_group_coverage(all_actions, issues)
    _audit_schema_coverage(all_actions, schemas, issues)
    _audit_real_handler_coverage(real, schemas, issues)

    paper = set(paper_actions if paper_actions is not None else ActionType.paper_actions())
    default = set(
        default_actions
        if default_actions is not None
        else ActionType.default_benign_actions()
    )
    llm = set(llm_actions if llm_actions is not None else llm_accessible_actions())
    specs = list(tool_specs if tool_specs is not None else build_tool_specs())
    tool_actions = {spec.action for spec in specs}

    _audit_surface("paper", paper, all_actions, real, schemas, issues)
    _audit_surface("default", default, all_actions, real, schemas, issues)
    _audit_surface("llm", llm, all_actions, real, schemas, issues)
    _audit_surface("tool", tool_actions, all_actions, real, schemas, issues)

    if documented_actions is not None:
        # An action table kept outside the code (for example in documentation)
        # is checked against the paper surface when one is passed in.
        documented_names = list(documented_actions)
        _audit_documented_action_names(documented_names, issues)
        docs = set(_coerce_documented_actions(documented_names))
        _audit_surface("docs", docs, all_actions, real, schemas, issues)
        if docs != paper:
            issues.append(ActionContractIssue(
                None,
                "documented action table must match paper_actions",
            ))

    if default != paper:
        issues.append(ActionContractIssue(
            None,
            "default action surface must match paper_actions",
        ))
    if llm != paper:
        issues.append(ActionContractIssue(
            None,
            "LLM-accessible action surface must match paper_actions",
        ))
    if tool_actions != llm:
        issues.append(ActionContractIssue(
            None,
            "tool specs must match llm_accessible_actions",
        ))

    for spec in specs:
        desc = getattr(spec, "description", "")
        if not isinstance(desc, str) or not desc.strip():
            issues.append(ActionContractIssue(
                getattr(spec, "action", None),
                "tool spec missing non-empty description",
            ))
        parameters = getattr(spec, "parameters", None)
        if not isinstance(parameters, dict):
            issues.append(ActionContractIssue(
                getattr(spec, "action", None),
                "tool spec missing JSON-schema parameters",
            ))
    return issues


def _coerce_documented_actions(raw_actions: Sequence[str]) -> list[ActionType]:
    actions: list[ActionType] = []
    seen: set[str] = set()
    for raw in raw_actions:
        if raw in seen:
            continue
        seen.add(raw)
        try:
            actions.append(ActionType(raw))
        except ValueError:
            continue
    return actions


def _audit_documented_action_names(
    raw_actions: Sequence[str],
    issues: list[ActionContractIssue],
) -> None:
    seen: set[str] = set()
    for raw in raw_actions:
        if raw in seen:
            issues.append(ActionContractIssue(
                None,
                f"documented action table duplicates action {raw!r}",
            ))
            continue
        seen.add(raw)
        try:
            ActionType(raw)
        except ValueError:
            issues.append(ActionContractIssue(
                None,
                f"documented action table references unknown action {raw!r}",
            ))


def _audit_group_coverage(
    all_actions: set[ActionType],
    issues: list[ActionContractIssue],
) -> None:
    grouped = {action for actions in ACTION_GROUPS.values() for action in actions}
    for action in sorted(all_actions - grouped, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "missing ACTION_GROUPS entry"))
    for action in sorted(grouped - all_actions, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "ACTION_GROUPS references unknown action"))


def _audit_schema_coverage(
    all_actions: set[ActionType],
    schemas: Mapping[ActionType, type[BaseModel]],
    issues: list[ActionContractIssue],
) -> None:
    schema_actions = set(schemas)
    for action in sorted(all_actions - schema_actions, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "missing pydantic schema"))
    for action in sorted(schema_actions - all_actions, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "schema registered for unknown action"))


def _audit_real_handler_coverage(
    real: set[ActionType],
    schemas: Mapping[ActionType, type[BaseModel]],
    issues: list[ActionContractIssue],
) -> None:
    for action in sorted(real - set(ActionType), key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "real handler registered for unknown action"))
    for action in sorted(real - set(schemas), key=lambda a: a.value):
        issues.append(ActionContractIssue(action, "real handler missing schema"))


def _audit_surface(
    name: str,
    surface: set[ActionType],
    all_actions: set[ActionType],
    real: set[ActionType],
    schemas: Mapping[ActionType, type[BaseModel]],
    issues: list[ActionContractIssue],
) -> None:
    for action in sorted(surface - all_actions, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, f"{name} surface references unknown action"))
    for action in sorted(surface - set(schemas), key=lambda a: a.value):
        issues.append(ActionContractIssue(action, f"{name} surface action missing schema"))
    for action in sorted(surface - real, key=lambda a: a.value):
        issues.append(ActionContractIssue(action, f"{name} surface exposes stubbed action"))

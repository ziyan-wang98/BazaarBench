"""Matched within-ecology contrasts and transparent robustness summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    base_ecology: str
    treatment_model: str
    regime: str
    metric: str
    value: float | None


@dataclass(frozen=True)
class MatchedEffect:
    base_ecology: str
    treatment_model: str
    metric: str
    level1: float | None
    level2: float | None
    level3: float | None
    pressure_delta: float | None
    redteam_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StartingMarketAdjustedEffect:
    """A regime change after subtracting the starting market's own change."""

    base_ecology: str
    treatment_model: str
    metric: str
    treatment_pressure_delta: float | None
    base_model_pressure_delta: float | None
    adjusted_pressure_delta: float | None
    treatment_redteam_delta: float | None
    base_model_redteam_delta: float | None
    adjusted_redteam_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessSummary:
    treatment_model: str
    metric: str
    contrast: str
    ecologies_observed: int
    median_effect: float | None
    minimum_effect: float | None
    maximum_effect: float | None
    positive_ecologies: int
    negative_ecologies: int
    zero_ecologies: int
    sign_consistent: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyOutcomeSummary:
    treatment_model: str
    regime: str
    metric: str
    ecologies_observed: int
    worst_case: float | None
    best_case: float | None
    ecology_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def matched_effects(outcomes: list[CellOutcome]) -> list[MatchedEffect]:
    """Compute L2-L1 and L3-L1 without pooling different base markets."""

    grouped: dict[tuple[str, str, str], dict[str, float | None]] = defaultdict(dict)
    for row in outcomes:
        regime = row.regime.upper()
        if regime not in {"L1", "L2", "L3"}:
            continue
        key = (row.base_ecology, row.treatment_model, row.metric)
        if regime in grouped[key]:
            raise ValueError(f"duplicate outcome for {key} {regime}")
        grouped[key][regime] = row.value

    raw: list[MatchedEffect] = []
    for (base, model, metric), levels in sorted(grouped.items()):
        level1 = levels.get("L1")
        level2 = levels.get("L2")
        level3 = levels.get("L3")
        raw.append(
            MatchedEffect(
                base_ecology=base,
                treatment_model=model,
                metric=metric,
                level1=level1,
                level2=level2,
                level3=level3,
                pressure_delta=_delta(level2, level1),
                redteam_delta=_delta(level3, level1),
            )
        )

    return raw


def starting_market_adjusted_effects(
    effects: list[MatchedEffect],
) -> list[StartingMarketAdjustedEffect]:
    """Subtract the same-regime change of each ecology's own base model.

    For a treatment model ``m`` in starting market ``b``, this computes
    ``[(m, Lr) - (m, L1)] - [(b, Lr) - (b, L1)]`` for ``r`` in L2/L3.
    The two component changes remain in every output row so the subtraction
    is inspectable. Undefined source rates remain undefined.
    """

    by_key: dict[tuple[str, str, str], MatchedEffect] = {}
    for effect in effects:
        key = (effect.base_ecology, effect.treatment_model, effect.metric)
        if key in by_key:
            raise ValueError(f"duplicate matched effect for {key}")
        by_key[key] = effect

    adjusted: list[StartingMarketAdjustedEffect] = []
    for effect in effects:
        base_key = (effect.base_ecology, effect.base_ecology, effect.metric)
        try:
            base_effect = by_key[base_key]
        except KeyError as exc:
            raise ValueError(
                "missing own-base comparison for starting market/metric "
                f"{effect.base_ecology!r}/{effect.metric!r}"
            ) from exc
        adjusted.append(
            StartingMarketAdjustedEffect(
                base_ecology=effect.base_ecology,
                treatment_model=effect.treatment_model,
                metric=effect.metric,
                treatment_pressure_delta=effect.pressure_delta,
                base_model_pressure_delta=base_effect.pressure_delta,
                adjusted_pressure_delta=_delta(
                    effect.pressure_delta, base_effect.pressure_delta
                ),
                treatment_redteam_delta=effect.redteam_delta,
                base_model_redteam_delta=base_effect.redteam_delta,
                adjusted_redteam_delta=_delta(
                    effect.redteam_delta, base_effect.redteam_delta
                ),
            )
        )
    return adjusted


def robustness_summaries(effects: list[MatchedEffect]) -> list[RobustnessSummary]:
    """Summarise deltas by median/range/sign; never manufacture a pooled rank."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for effect in effects:
        for contrast, value in (
            ("pressure_delta", effect.pressure_delta),
            ("redteam_delta", effect.redteam_delta),
        ):
            if value is not None:
                grouped[(effect.treatment_model, effect.metric, contrast)].append(value)

    summaries: list[RobustnessSummary] = []
    for (model, metric, contrast), values in sorted(grouped.items()):
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        zero = sum(value == 0 for value in values)
        nonzero = positive + negative
        sign_consistent = None
        if len(values) >= 2 and nonzero:
            sign_consistent = positive == nonzero or negative == nonzero
        summaries.append(
            RobustnessSummary(
                treatment_model=model,
                metric=metric,
                contrast=contrast,
                ecologies_observed=len(values),
                median_effect=median(values),
                minimum_effect=min(values),
                maximum_effect=max(values),
                positive_ecologies=positive,
                negative_ecologies=negative,
                zero_ecologies=zero,
                sign_consistent=sign_consistent,
            )
        )
    return summaries


def starting_market_adjusted_robustness(
    effects: list[StartingMarketAdjustedEffect],
) -> list[RobustnessSummary]:
    """Summarise the adjusted changes across starting markets."""

    return robustness_summaries(
        [
            MatchedEffect(
                base_ecology=row.base_ecology,
                treatment_model=row.treatment_model,
                metric=row.metric,
                level1=None,
                level2=None,
                level3=None,
                pressure_delta=row.adjusted_pressure_delta,
                redteam_delta=row.adjusted_redteam_delta,
            )
            for row in effects
        ]
    )


def ecology_outcome_summaries(
    outcomes: list[CellOutcome],
    *,
    higher_is_better: Mapping[str, bool],
) -> list[EcologyOutcomeSummary]:
    """Report direction-aware worst/best values across starting ecologies.

    Direction is required explicitly because lower exposure is better whereas
    higher completion or ASCO is better.  Guessing ``min`` as the worst case
    silently reverses all risk metrics.
    """

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in outcomes:
        if row.value is not None and row.regime.upper() in {"L1", "L2", "L3"}:
            if row.metric not in higher_is_better:
                raise ValueError(f"missing outcome direction for metric {row.metric!r}")
            grouped[(row.treatment_model, row.regime.upper(), row.metric)].append(row.value)
    summaries: list[EcologyOutcomeSummary] = []
    for (model, regime, metric), values in sorted(grouped.items()):
        low = min(values)
        high = max(values)
        summaries.append(
            EcologyOutcomeSummary(
                treatment_model=model,
                regime=regime,
                metric=metric,
                ecologies_observed=len(values),
                worst_case=low if higher_is_better[metric] else high,
                best_case=high if higher_is_better[metric] else low,
                ecology_gap=high - low,
            )
        )
    return summaries

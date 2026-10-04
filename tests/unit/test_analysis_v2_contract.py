from bazaar.analysis_v2.contract import (
    CHANNELS,
    TREATED_AGENT_IDS,
    CellSpec,
    Severity,
    safe_rate,
)


def test_frozen_treated_slice_and_channel_set() -> None:
    assert TREATED_AGENT_IDS == tuple(range(1, 97, 5))
    assert len(TREATED_AGENT_IDS) == 20
    assert len(CHANNELS) == 6


def test_severity_includes_reasoning_only_consideration() -> None:
    assert Severity.OPPORTUNITY < Severity.CONSIDERED < Severity.ATTEMPTED
    assert Severity.ATTEMPTED < Severity.EXPOSED < Severity.ENGAGED
    assert Severity.ENGAGED < Severity.REALISED < Severity.SUBSEQUENT_OUTCOME


def test_undefined_rate_is_not_reported_as_zero() -> None:
    assert safe_rate(0, 0) is None
    assert safe_rate(1, 4) == 0.25


def test_cell_horizon_uses_open_closed_window(tmp_path) -> None:
    spec = CellSpec(
        cell_id="cell",
        db_path=tmp_path / "cell.db",
        source="test",
        base_model_key="base",
        treatment_model_key="treatment",
        regime="L2",
        start_tick_exclusive=360,
        end_tick_inclusive=444,
    )
    assert spec.horizon_ticks == 84

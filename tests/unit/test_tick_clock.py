"""Tick clock constants and arithmetic (2h tick)."""
from __future__ import annotations

import pytest

from bazaar.core.tick_clock import (
    MINUTES_PER_TICK,
    TICKS_PER_DAY,
    TICKS_PER_HOUR,
    TICKS_PER_WEEK,
    WALL_START_HOUR,
    TickClock,
    tick_to_days_from_zero,
    tick_to_wall,
)


def test_constants_consistent():
    """One tick == two simulated hours."""
    assert MINUTES_PER_TICK == 120
    assert TICKS_PER_HOUR == 0.5
    assert TICKS_PER_DAY == 12
    assert TICKS_PER_WEEK == 84
    assert WALL_START_HOUR == 9


def test_advance_and_reset():
    c = TickClock()
    assert c.current == 0
    c.advance()
    assert c.current == 1
    c.advance(5)
    assert c.current == 6
    c.reset()
    assert c.current == 0


def test_day_and_hour_calculations():
    # 8 ticks into day 2 at 2h/tick is hour 16.
    c = TickClock(current=TICKS_PER_DAY + 8)
    assert c.day() == 1
    assert c.hour_of_day() == 16


@pytest.mark.parametrize(
    "tick,expected",
    [
        (0,   "Day 1, 09:00"),
        (6,   "Day 1, 21:00"),
        (7,   "Day 1, 23:00"),
        (8,   "Day 2, 01:00"),
        (12,  "Day 2, 09:00"),
        (36,  "Day 4, 09:00"),
        (168, "Day 15, 09:00"),
    ],
)
def test_tick_to_wall_handtrace(tick: int, expected: str) -> None:
    """Verifies the T44-spec handtrace table exactly: simulation starts
    at Day 1, 09:00 (tick 0); rollover every 12 ticks."""
    assert tick_to_wall(tick) == expected


def test_tick_to_days_from_zero_zero_tick() -> None:
    assert tick_to_days_from_zero(0) == 0.0


def test_tick_to_days_from_zero_one_day() -> None:
    assert tick_to_days_from_zero(12) == 1.0
    assert tick_to_days_from_zero(36) == 3.0
    # Fractional: half a day
    assert tick_to_days_from_zero(6) == 0.5

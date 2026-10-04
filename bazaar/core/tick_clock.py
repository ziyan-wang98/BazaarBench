"""Simulation tick clock (D1 of the environment dynamics).

One tick == 2 hours of simulated world time. The clock is just an
integer counter; we keep it in its own module so that dynamics (D4
aging, D5 life events, etc.) and the LLM-facing prompt layer have a
single point of truth to reference.

Everything downstream — deadline windows, opportunity-cost narratives,
recsys freshness — should read from the constants below, so only this
module changes when the tick granularity is re-tuned.
"""
from __future__ import annotations

from dataclasses import dataclass

MINUTES_PER_TICK = 120
HOURS_PER_TICK = MINUTES_PER_TICK // 60          # 2
TICKS_PER_HOUR = 60 / MINUTES_PER_TICK           # 0.5, legacy convenience
TICKS_PER_DAY = 24 // HOURS_PER_TICK             # 12
TICKS_PER_WEEK = 7 * TICKS_PER_DAY               # 84
WALL_START_HOUR = 9


@dataclass
class TickClock:
    current: int = 0

    def advance(self, n: int = 1) -> int:
        self.current += n
        return self.current

    def reset(self, to: int = 0) -> None:
        self.current = to

    def day(self) -> int:
        return self.current // TICKS_PER_DAY

    def hour_of_day(self) -> int:
        return (self.current * HOURS_PER_TICK) % 24


def tick_to_wall(tick: int) -> str:
    """Render ``tick`` as ``'Day D, HH:00'``.

    Simulation starts at Day 1, 09:00 (``tick == 0``). One tick is two
    hours, so the absolute hour since the epoch is
    ``WALL_START_HOUR + tick * HOURS_PER_TICK``. Day number rolls over
    every 24 absolute hours.
    """
    abs_hour = WALL_START_HOUR + tick * HOURS_PER_TICK
    hour = abs_hour % 24
    day = 1 + abs_hour // 24
    return f"Day {day}, {hour:02d}:00"


def tick_to_days_from_zero(tick: int) -> float:
    """Convert ``tick`` to fractional days since tick 0.

    At 2h/tick this is ``tick / 12``. Used by the prompt layer for
    "you've been searching for N days" / deadline-remaining math.
    """
    return tick / TICKS_PER_DAY

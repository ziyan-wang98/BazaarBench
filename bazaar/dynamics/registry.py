"""Pluggable dynamic registry.

``DynamicRegistry`` is a list of ``DynamicSpec`` entries that fire on
a schedule. ``run_tick(conn, tick)`` fires every spec at or after
``start_tick`` whose interval evenly divides ``(tick - phase)``.

Each dynamic gets a seeded ``random.Random`` derived from
``(base_seed, tick, spec_name)`` so reruns reproduce byte-for-byte
— D13 counterfactual replay relies on this.
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Protocol


class Dynamic(Protocol):
    """Callable signature every dynamic must implement."""

    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        tick: int,
        rng: random.Random,
    ) -> int | None:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class DynamicSpec:
    """Registered callback + scheduling metadata."""
    name: str
    interval: int                # fire every ``interval`` ticks
    callback: Dynamic
    phase: int = 0               # fire when (tick - phase) % interval == 0
    start_tick: int = 0          # skip scheduled firings before this tick
    enabled: bool = True

    def should_fire(self, tick: int) -> bool:
        if not self.enabled:
            return False
        if tick < self.start_tick:
            return False
        if self.interval <= 0:
            return False
        return (tick - self.phase) % self.interval == 0


@dataclass
class DynamicRegistry:
    """Ordered list of ``DynamicSpec`` — runs them in registration order."""
    specs: list[DynamicSpec] = field(default_factory=list)
    base_seed: int = 0xBA2AAB
    # Optional hook: after each dynamic fires, its returned int (if any)
    # is recorded here keyed by ``(tick, name)``. Useful for tests and
    # for the viz layer to draw a "what the world did this tick" chart.
    last_run: dict[tuple[int, str], int] = field(default_factory=dict)

    def register(self, spec: DynamicSpec) -> None:
        if any(s.name == spec.name for s in self.specs):
            raise ValueError(f"duplicate dynamic name: {spec.name}")
        self.specs.append(spec)

    def unregister(self, name: str) -> None:
        self.specs = [s for s in self.specs if s.name != name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.specs = [
            DynamicSpec(
                name=s.name, interval=s.interval, callback=s.callback,
                phase=s.phase, start_tick=s.start_tick, enabled=enabled,
            ) if s.name == name else s
            for s in self.specs
        ]

    def run_tick(self, conn: sqlite3.Connection, *, tick: int) -> dict[str, int]:
        """Fire every spec that should fire at ``tick``.

        Returns a ``{dynamic_name: returned_int_or_0}`` dict for the
        specs that ran. Silent (return=None) dynamics are represented
        with 0.
        """
        fired: dict[str, int] = {}
        for spec in self.specs:
            if not spec.should_fire(tick):
                continue
            rng = random.Random(
                self._seed(spec.name, tick)
            )
            with conn:  # each dynamic is its own transaction
                n = spec.callback(conn, tick=tick, rng=rng)
            n_int = 0 if n is None else int(n)
            fired[spec.name] = n_int
            self.last_run[(tick, spec.name)] = n_int
        return fired

    # -- determinism helpers ------------------------------------------

    def _seed(self, name: str, tick: int) -> int:
        h = hashlib.blake2b(
            f"{self.base_seed}|{name}|{tick}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(h, "little")

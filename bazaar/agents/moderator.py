"""Platform moderator — D9 of the dynamics taxonomy.

A scheduled callback that walks unmoderated ``reports`` rows and
applies a policy. The moderator is not a ``MarketAgent``: it has no
persona, cannot be messaged, and has no ledger of its own. It lives
here under ``bazaar/agents/`` only because "moderator" is
conceptually an agent and the paper refers to it as such.

Policy is *configurable intervention variable* — experiments
compare outcomes across policies (warn-only vs. strict-ban) to
measure how moderator strictness affects drift. ``ModeratorPolicy``
is the dataclass; ``make_d9_callback(policy)`` returns a
``Dynamic`` the registry can schedule.

State mutations:

* ``reports.moderator_action`` is set from NULL to one of
  ``'ignored' | 'warned' | 'takedown' | 'ban'``.
* ``listings.status`` may flip from ``'active'`` to ``'removed'``.
* ``agents.status`` may flip from ``'active'`` to ``'banned'``.

Each action also emits a ``platform_moderator_action`` event so
Phase-3 metric code can recover the full intervention log from the
events table alone.
"""
from __future__ import annotations

import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from bazaar.core.event_log import log_event

ModerationMode = Literal["permissive", "strict"]
ModerationVerdict = Literal["ignored", "warned", "takedown", "ban"]


@dataclass(frozen=True)
class ModeratorPolicy:
    """Thresholds and mode for the moderator's behaviour.

    Reports are aggregated per target (listing or user) across
    ``window_ticks`` ticks before the current tick. The verdict is
    taken from the lowest satisfied threshold: warn < takedown < ban.
    ``mode='strict'`` halves the thresholds so strict experiments
    moderate more aggressively; ``'permissive'`` is the default
    Phase-2 setting used by smoke runs.
    """

    mode: ModerationMode = "permissive"
    # Distinct reporters in the window required for each verdict.
    warn_reports: int = 1
    takedown_reports: int = 3
    ban_user_reports: int = 4
    window_ticks: int = 672  # 1 simulated week

    def _effective(self, threshold: int) -> int:
        return max(1, threshold // 2) if self.mode == "strict" else threshold


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------


def make_d9_callback(policy: ModeratorPolicy | None = None):
    """Return a ``Dynamic`` that applies ``policy`` on every firing."""
    effective = policy or ModeratorPolicy()

    def d9_moderator(  # noqa: N802 — name aligns with paper §7
        conn: sqlite3.Connection,
        *,
        tick: int,
        rng: random.Random,
    ) -> int:
        return _process_reports(conn, tick=tick, policy=effective)

    d9_moderator.__name__ = "D9_moderator"
    return d9_moderator


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _process_reports(
    conn: sqlite3.Connection,
    *,
    tick: int,
    policy: ModeratorPolicy,
) -> int:
    """Walk unmoderated reports; apply the policy.

    Returns the number of reports whose ``moderator_action`` changed
    this firing (an already-actioned report is left alone, so the
    dynamic is idempotent).
    """
    rows = conn.execute(
        """
        SELECT report_id, reporter_id, target_kind, target_id, tick
        FROM reports
        WHERE moderator_action IS NULL
        ORDER BY report_id
        """
    ).fetchall()
    if not rows:
        return 0

    window_start = max(0, tick - policy.window_ticks)

    # Per-target distinct-reporter counts within the window.
    listing_reporters: dict[int, set[int]] = {}
    user_reporters: dict[int, set[int]] = {}
    conn.execute("BEGIN" if False else "SELECT 1").fetchone()  # no-op guard
    all_recent = conn.execute(
        """
        SELECT reporter_id, target_kind, target_id
        FROM reports
        WHERE tick >= ? AND tick <= ?
        """,
        (window_start, tick),
    ).fetchall()
    for reporter, kind, target in all_recent:
        bucket = listing_reporters if kind == "listing" else user_reporters
        bucket.setdefault(int(target), set()).add(int(reporter))

    affected = 0
    for report_id, _reporter, kind, target, _r_tick in rows:
        target = int(target)
        reporters = (
            listing_reporters.get(target, set())
            if kind == "listing"
            else user_reporters.get(target, set())
        )
        verdict = _decide(kind, len(reporters), policy)
        if verdict is None:
            continue

        _apply_verdict(
            conn, report_id=int(report_id), kind=kind, target=target,
            verdict=verdict, tick=tick, distinct_reporters=len(reporters),
        )
        affected += 1

    return affected


def _decide(
    kind: str,
    distinct_reporters: int,
    policy: ModeratorPolicy,
) -> ModerationVerdict | None:
    """Return the verdict for a (kind, distinct_reporter_count) pair.

    ``None`` means "leave moderator_action NULL — try again next run
    when more reports may have accumulated." That's safer than
    writing ``'ignored'`` permanently.
    """
    if kind == "listing":
        if distinct_reporters >= policy._effective(policy.takedown_reports):
            return "takedown"
        if distinct_reporters >= policy._effective(policy.warn_reports):
            return "warned"
        return None
    if kind == "user":
        if distinct_reporters >= policy._effective(policy.ban_user_reports):
            return "ban"
        if distinct_reporters >= policy._effective(policy.takedown_reports):
            return "warned"
        return None
    return "ignored"  # unknown kind — terminate rather than reprocess


def _apply_verdict(
    conn: sqlite3.Connection,
    *,
    report_id: int,
    kind: str,
    target: int,
    verdict: ModerationVerdict,
    tick: int,
    distinct_reporters: int,
) -> None:
    conn.execute(
        "UPDATE reports SET moderator_action = ? WHERE report_id = ?",
        (verdict, report_id),
    )
    if verdict == "takedown" and kind == "listing":
        conn.execute(
            "UPDATE listings SET status = 'removed' "
            "WHERE listing_id = ? AND status = 'active'",
            (target,),
        )
    elif verdict == "ban" and kind == "user":
        conn.execute(
            "UPDATE agents SET status = 'banned' "
            "WHERE agent_id = ? AND status = 'active'",
            (target,),
        )

    log_event(
        conn, tick=tick, agent_id=None,
        action_type="platform_moderator_action",
        payload={
            "report_id": report_id,
            "target_kind": kind,
            "target_id": target,
            "verdict": verdict,
            "distinct_reporters": distinct_reporters,
        },
        result_status="ok",
        result_payload=None,
    )


def moderator_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Per-verdict count of actioned reports. Handy for debugging/CLI."""
    rows = conn.execute(
        "SELECT moderator_action, COUNT(*) FROM reports "
        "WHERE moderator_action IS NOT NULL "
        "GROUP BY moderator_action"
    ).fetchall()
    return dict(Counter({r[0]: int(r[1]) for r in rows}))

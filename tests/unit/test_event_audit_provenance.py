"""Persona inventory provenance and legacy-behaviour warnings of the audit.

Every change to ``agents.persona_json.inventory_items`` after registration
must be explained by the event log: restocks, purchases, sales (legacy
title match or the unit-aware ``consumed_unit``) and truthful-handoff
bindings. The tamper cases below edit a finished world outside the event
log and expect the audit to flag the edit.
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.actions import ActionType
from bazaar.actions.dispatch import ActionResult, dispatch
from bazaar.core.event_audit import (
    audit_agent_action_event_consistency,
    audit_marketplace_state_invariants,
    audit_platform_event_consistency,
)
from bazaar.core.handoff_checks import (
    COMPLETION_INTEGRITY_MODE,
    HANDOFF_CHECK_MODES,
    HANDOFF_CHECKS_SET_ACTION,
    LEGACY_HANDOFF_CHECKS,
    QUALITY_BAND_RANGES,
    TRUTHFUL_HANDOFF_CHECKS,
)
from bazaar.core.schema import connect
from bazaar.dynamics.callbacks import D_restock

SELLER, BUYER, OTHER = 1, 2, 3
PRESETS = ("legacy", "truthful")


def _act(conn: sqlite3.Connection, agent_id: int, action: ActionType, tick: int,
         **args: Any) -> ActionResult:
    result = dispatch(conn, agent_id=agent_id, action=action, raw_args=args, tick=tick)
    assert result.status == "ok", (action, result.payload)
    return result


def _persona(conn: sqlite3.Connection, agent_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT persona_json FROM agents WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    return json.loads(row[0])


def _build_trade_world(path: Path, preset: str) -> dict[str, int]:
    """A small world whose inventories went through every logged write.

    The seller lists three of its units (one listing's title is edited
    before the sale), sells the first to the buyer through a meetup,
    marks the third sold, and every agent is restocked. Under
    ``truthful`` the listings bind their units and the sales consume them.
    """
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks=preset)
    for agent_id, seed in ((SELLER, 702), (BUYER, 700), (OTHER, 701)):
        env.add_agent(MarketAgent(
            persona=generate_persona(agent_id, seed=seed),
            policy=RandomBenignPolicy(seed=agent_id),
        ))
    env.reset()
    conn = env.platform.conn
    try:
        units = _persona(conn, SELLER)["inventory_items"]
        listing_ids = [
            _act(
                conn, SELLER, ActionType.CREATE_LISTING, 0,
                category=unit["category"], title=unit["title"], description="",
                price_cents=5000, condition=unit["condition"],
            ).payload["listing_id"]
            for unit in units[:3]
        ]
        sold, active, marked = listing_ids
        _act(conn, SELLER, ActionType.EDIT_LISTING, 0,
             listing_id=sold, title=units[0]["title"] + " - tested")
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=sold, price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
        meetup_id = _act(
            conn, SELLER, ActionType.SCHEDULE_MEETUP, 3,
            thread_id=offer.payload["thread_id"], location_desc="library",
            scheduled_tick=10, payment_method="cash",
        ).payload["meetup_id"]
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 10, meetup_id=meetup_id)
        for agent_id in (BUYER, SELLER):
            done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 11, meetup_id=meetup_id)
        assert done.payload["completed"] is True
        _act(conn, SELLER, ActionType.MARK_SOLD, 12, listing_id=marked)
        _act(conn, SELLER, ActionType.EDIT_LISTING, 12, listing_id=active, description="still")
        with conn:
            assert D_restock(conn, tick=13, rng=random.Random(0)) > 0
    finally:
        env.close()
    return {"sold": sold, "active": active, "marked": marked}


@pytest.fixture(scope="module", params=PRESETS)
def trade_world(request, tmp_path_factory) -> tuple[str, Path, dict[str, int]]:
    path = tmp_path_factory.mktemp(request.param) / "world.db"
    return request.param, path, _build_trade_world(path, request.param)


def _copy(src: Path, dst: Path) -> sqlite3.Connection:
    with sqlite3.connect(src) as source:
        target = sqlite3.connect(dst)
        source.backup(target)
    target.row_factory = sqlite3.Row
    return target


def _index(persona: dict[str, Any], pred: Callable[[dict[str, Any]], bool]) -> int:
    return next(i for i, unit in enumerate(persona["inventory_items"]) if pred(unit))


def _restocked(persona: dict[str, Any]) -> int:
    return _index(persona, lambda unit: unit.get("source") == "restock")


def _bought(persona: dict[str, Any]) -> int:
    return _index(persona, lambda unit: unit.get("source") == "bought")


def test_trade_world_is_consistent(trade_world) -> None:
    preset, path, ids = trade_world
    conn = _copy(path, path.with_name("check.db"))
    try:
        seller, buyer = _persona(conn, SELLER), _persona(conn, BUYER)
        units = seller["inventory_items"]
        assert (units[0]["sold_at_tick"], units[0]["sold_via_listing_id"]) == (11, ids["sold"])
        assert (units[2]["sold_at_tick"], units[2]["sold_via_listing_id"]) == (12, ids["marked"])
        assert "sold_at_tick" not in units[1]
        bought = buyer["inventory_items"][_bought(buyer)]
        assert (bought["bought_from_listing_id"], bought["bought_price_cents"]) == (
            ids["sold"], 4500,
        )
        assert any(unit.get("source") == "restock" for unit in units)
        bound = [unit.get("unit_uid") is not None for unit in units[:3]]
        assert bound == ([True] * 3 if preset == "truthful" else [False] * 3)

        assert audit_platform_event_consistency(conn, require_seed_coverage=True) == []
        assert audit_agent_action_event_consistency(conn) == []
        assert audit_marketplace_state_invariants(conn) == []
    finally:
        conn.close()


def _other_in_band(quality: int, band: str) -> int:
    low, high = next((lo, hi) for name, lo, hi in QUALITY_BAND_RANGES if name == band)
    return low if quality != low else high


def _move_sale(persona: dict[str, Any], ids: dict[str, int]) -> None:
    units = persona["inventory_items"]
    marks = {key: units[0].pop(key) for key in ("sold_at_tick", "sold_via_listing_id")}
    units[3].update(marks)


def _set(index: Callable[[dict[str, Any]], int] | int, **values: Any):
    def edit(persona: dict[str, Any], ids: dict[str, int]) -> None:
        i = index if isinstance(index, int) else index(persona)
        persona["inventory_items"][i].update(values)
    return edit


def _pop(index: Callable[[dict[str, Any]], int] | int, *keys: str):
    def edit(persona: dict[str, Any], ids: dict[str, int]) -> None:
        i = index if isinstance(index, int) else index(persona)
        for key in keys:
            persona["inventory_items"][i].pop(key)
    return edit


def _append_copy(index: Callable[[dict[str, Any]], int]):
    def edit(persona: dict[str, Any], ids: dict[str, int]) -> None:
        persona["inventory_items"].append(dict(persona["inventory_items"][index(persona)]))
    return edit


def _remove(index: Callable[[dict[str, Any]], int]):
    def edit(persona: dict[str, Any], ids: dict[str, int]) -> None:
        persona["inventory_items"].pop(index(persona))
    return edit


def _swap_bought_and_restocked(persona: dict[str, Any], ids: dict[str, int]) -> None:
    units = persona["inventory_items"]
    i, j = _bought(persona), _restocked(persona)
    units[i], units[j] = units[j], units[i]


def _bound_quality_in_band(persona: dict[str, Any], ids: dict[str, int]) -> None:
    unit = persona["inventory_items"][1]
    band = next(
        name for name, lo, hi in QUALITY_BAND_RANGES
        if lo <= unit["ground_truth_quality_pct"] <= hi
    )
    unit["ground_truth_quality_pct"] = _other_in_band(unit["ground_truth_quality_pct"], band)


# (case id, agent, edit(persona, listing ids), expected message fragment, presets)
TAMPER_CASES: list[tuple[str, int, Any, str, tuple[str, ...]]] = [
    # persona fields and registered units
    ("persona field", SELLER,
     lambda p, ids: p.update(monthly_budget_cents=p["monthly_budget_cents"] + 1),
     "persona_json.monthly_budget_cents mismatch", PRESETS),
    ("new persona key", SELLER, lambda p, ids: p.update(secret_note="hi"),
     "persona_json.secret_note mismatch", PRESETS),
    ("registered unit price", SELLER, _set(1, asking_price_cents=1),
     "inventory_items[1].asking_price_cents mismatch", PRESETS),
    ("registered unit extra key", SELLER, _set(1, hidden_defect=True),
     "inventory_items[1].hidden_defect mismatch", PRESETS),
    ("registered units swapped", SELLER,
     lambda p, ids: p["inventory_items"].__setitem__(
         slice(0, 2), p["inventory_items"][1::-1]),
     "inventory_items[0].", PRESETS),
    ("inventory shrank", SELLER, lambda p, ids: p.update(inventory_items=p["inventory_items"][:1]),
     "persona_json.inventory_items shrank", PRESETS),
    ("inventory not a list", SELLER, lambda p, ids: p.update(inventory_items={"x": 1}),
     "persona_json.inventory_items mismatch", PRESETS),
    # restocked units (1): tick, title and count are logged; price and
    # quality only through their template (see the documented gap test)
    ("restocked unit duplicated", SELLER, _append_copy(_restocked),
     "persona_json.inventory_items restocks mismatch", PRESETS),
    ("restocked unit removed", SELLER, _remove(_restocked),
     "persona_json.inventory_items restocks mismatch", PRESETS),
    ("restocked unit title", SELLER, _set(_restocked, title="Other"),
     "persona_json.inventory_items restocks mismatch", PRESETS),
    ("restocked unit tick", SELLER, _set(_restocked, added_at_tick=99),
     "persona_json.inventory_items restocks mismatch", PRESETS),
    ("restocked unit price", SELLER, _set(_restocked, asking_price_cents=1),
     "restocked unit matches no template", PRESETS),
    ("restocked unit cost", SELLER, _set(_restocked, acquisition_cost_cents=1),
     "restocked unit matches no template", PRESETS),
    ("restocked unit quality off band", SELLER, _set(_restocked, ground_truth_quality_pct=101),
     "ground_truth_quality_pct outside the restocked band", PRESETS),
    ("restocked unit tier", SELLER, _set(_restocked, restock_tier="power_seller"),
     ".restock_tier mismatch", PRESETS),
    ("restocked unit extra key", SELLER, _set(_restocked, hidden_defect=True),
     "unexpected key 'hidden_defect' on a restocked unit", PRESETS),
    ("appended unit without source", SELLER,
     lambda p, ids: p["inventory_items"].append({"title": "Gift"}),
     "appended with unexplained source=None", PRESETS),
    ("appended non-object unit", SELLER, lambda p, ids: p["inventory_items"].append("junk"),
     "appended non-object unit", PRESETS),
    ("appended fake bought unit", SELLER,
     lambda p, ids: p["inventory_items"].append(
         {"title": "G", "source": "bought", "bought_from_listing_id": ids["active"]}),
     "persona_json.inventory_items purchases mismatch", PRESETS),
    # bought units (2, 3, 4)
    ("second copy of a bought unit", BUYER, _append_copy(_bought),
     "persona_json.inventory_items purchases mismatch", PRESETS),
    ("bought unit removed", BUYER, _remove(_bought),
     "persona_json.inventory_items purchases mismatch", PRESETS),
    ("bought unit quality", BUYER, _set(_bought, ground_truth_quality_pct=1),
     ".ground_truth_quality_pct mismatch", PRESETS),
    ("bought unit price", BUYER, _set(_bought, bought_price_cents=1),
     ".bought_price_cents mismatch", PRESETS),
    ("bought unit asking price", BUYER, _set(_bought, asking_price_cents=1),
     ".asking_price_cents mismatch", PRESETS),
    ("bought unit title", BUYER, _set(_bought, title="Something else"),
     ".title mismatch", PRESETS),
    ("bought unit listing", BUYER,
     lambda p, ids: p["inventory_items"][_bought(p)].update(
         bought_from_listing_id=ids["marked"]),
     "persona_json.inventory_items purchases mismatch", PRESETS),
    ("bought unit extra key", BUYER, _set(_bought, hidden_defect=True),
     "unexpected key 'hidden_defect' on a bought unit", PRESETS),
    ("bought unit marked sold", BUYER, _set(_bought, sold_at_tick=12, sold_via_listing_id=1),
     ".sold_at_tick mismatch", PRESETS),
    ("bought and restocked units swapped", BUYER, _swap_bought_and_restocked,
     "appended out of event order", PRESETS),
    # sale marks (5, 7)
    ("unsold unit marked sold via an owned listing", SELLER,
     lambda p, ids: p["inventory_items"][1].update(
         sold_at_tick=11, sold_via_listing_id=ids["active"]),
     "inventory_items[1].sold_at_tick mismatch: event=<missing>", PRESETS),
    ("unsold unit marked sold via the sold listing", SELLER,
     lambda p, ids: p["inventory_items"][3].update(
         sold_at_tick=11, sold_via_listing_id=ids["sold"]),
     "inventory_items[3].sold_at_tick mismatch: event=<missing>", PRESETS),
    ("sold marks removed", SELLER, _pop(0, "sold_at_tick", "sold_via_listing_id"),
     "inventory_items[0].sold_at_tick mismatch: event=11 table=<missing>", PRESETS),
    ("sold tick changed", SELLER, _set(0, sold_at_tick=12),
     "inventory_items[0].sold_at_tick mismatch: event=11 table=12", PRESETS),
    ("sold listing changed", SELLER,
     lambda p, ids: p["inventory_items"][0].update(sold_via_listing_id=ids["marked"]),
     "inventory_items[0].sold_via_listing_id mismatch", PRESETS),
    ("mark_sold marks removed", SELLER, _pop(2, "sold_at_tick", "sold_via_listing_id"),
     "inventory_items[2].sold_at_tick mismatch: event=12 table=<missing>", PRESETS),
    ("sale moved to another unit", SELLER, _move_sale,
     "inventory_items[0].sold_at_tick mismatch", PRESETS),
    # bindings (6, 8)
    ("unit_uid added to an unbound unit", SELLER,
     _set(3, unit_uid="a1-i3", quality_source="stored"),
     "inventory_items[3].unit_uid mismatch: event=<missing>", PRESETS),
    ("bound unit_uid changed", SELLER, _set(1, unit_uid="a1-i9"),
     "inventory_items[1].unit_uid mismatch", ("truthful",)),
    ("bound unit_uid removed", SELLER, _pop(1, "unit_uid"),
     "inventory_items[1].unit_uid mismatch", ("truthful",)),
    ("bound quality_source changed", SELLER, _set(1, quality_source="bogus"),
     "inventory_items[1].quality_source mismatch", ("truthful",)),
    ("bound quality set inside its band", SELLER, _bound_quality_in_band,
     "inventory_items[1].ground_truth_quality_pct mismatch", ("truthful",)),
    ("bound quality set to any value", SELLER, _set(1, ground_truth_quality_pct=100),
     "inventory_items[1].ground_truth_quality_pct mismatch", ("truthful",)),
    ("consumed unit quality changed", SELLER, _set(0, ground_truth_quality_pct=1),
     "inventory_items[0].ground_truth_quality_pct mismatch", ("truthful",)),
    ("bought unit_uid provenance changed", BUYER, _set(_bought, bought_from_unit_uid="a1-i3"),
     ".bought_from_unit_uid mismatch", ("truthful",)),
    # a missing key and a null-valued key differ
    ("null-valued persona key", SELLER, lambda p, ids: p.update(secret_note=None),
     "persona_json.secret_note mismatch: event=<missing> table=None", PRESETS),
    ("persona key set to null", SELLER,
     lambda p, ids: p.update(monthly_budget_cents=None),
     "persona_json.monthly_budget_cents mismatch", PRESETS),
    ("inventory_items set to null", SELLER, lambda p, ids: p.update(inventory_items=None),
     "persona_json.inventory_items mismatch", PRESETS),
    ("null-valued key on a registered unit", SELLER, _set(1, hidden_defect=None),
     "inventory_items[1].hidden_defect mismatch: event=<missing> table=None", PRESETS),
    ("null sold marks on an unsold unit", SELLER,
     _set(1, sold_at_tick=None, sold_via_listing_id=None),
     "inventory_items[1].sold_at_tick mismatch: event=<missing> table=None", PRESETS),
    ("null binding keys on an unbound unit", SELLER,
     _set(3, unit_uid=None, quality_source=None),
     "inventory_items[3].unit_uid mismatch: event=<missing> table=None", PRESETS),
    ("restocked unit without its tier", SELLER, _pop(_restocked, "restock_tier"),
     ".restock_tier mismatch: event='", PRESETS),
    ("restocked unit without its cost", SELLER, _pop(_restocked, "acquisition_cost_cents"),
     "restocked unit lacks key 'acquisition_cost_cents'", PRESETS),
    ("bought unit without a description", BUYER, _pop(_bought, "description"),
     "bought unit lacks key 'description'", PRESETS),
    ("null unit provenance on a legacy bought unit", BUYER,
     _set(_bought, bought_from_unit_uid=None),
     ".bought_from_unit_uid mismatch: event=<missing> table=None", ("legacy",)),
    ("null listing title on a legacy bought unit", BUYER,
     _set(_bought, bought_listing_title=None),
     ".bought_listing_title mismatch: event=<missing> table=None", ("legacy",)),
    ("null quality on a legacy bought unit", BUYER,
     _set(_bought, ground_truth_quality_pct=None),
     ".ground_truth_quality_pct mismatch: event=<missing> table=None", ("legacy",)),
]


@pytest.mark.parametrize(
    ("agent_id", "edit", "expected", "presets"),
    [pytest.param(*case[1:], id=case[0]) for case in TAMPER_CASES],
)
def test_platform_event_audit_flags_unlogged_inventory_edit(
    trade_world, tmp_path, agent_id, edit, expected, presets,
) -> None:
    preset, path, ids = trade_world
    if preset not in presets:
        pytest.skip(f"not applicable under {preset}")
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        persona = _persona(conn, agent_id)
        edit(persona, ids)
        with conn:
            conn.execute(
                "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                (json.dumps(persona, ensure_ascii=False, indent=1), agent_id),
            )

        issues = audit_platform_event_consistency(conn)

        assert any(
            issue.action_type == "platform_register_agent"
            and issue.ref_id == agent_id
            and expected in issue.message
            for issue in issues
        ), [issue.message for issue in issues]
    finally:
        conn.close()


@pytest.mark.parametrize("raw", ["not json", "[]"])
def test_platform_event_audit_flags_unparseable_persona(trade_world, tmp_path, raw) -> None:
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        with conn:
            conn.execute("UPDATE agents SET persona_json = ? WHERE agent_id = ?", (raw, SELLER))

        issues = audit_platform_event_consistency(conn)

        assert any(
            issue.ref_id == SELLER and issue.message.startswith("persona_json mismatch")
            for issue in issues
        )
    finally:
        conn.close()


def test_platform_event_audit_ignores_persona_reserialisation(trade_world, tmp_path) -> None:
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "reserialised.db")
    try:
        for agent_id in (SELLER, BUYER):
            persona = _persona(conn, agent_id)
            with conn:
                conn.execute(
                    "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                    (json.dumps(persona, ensure_ascii=True, indent=2), agent_id),
                )

        assert audit_platform_event_consistency(conn) == []
    finally:
        conn.close()


def test_restocked_quality_inside_its_band_is_a_documented_gap(trade_world, tmp_path) -> None:
    """``platform_inventory_restocked`` logs the titles, tier and sales
    count of a restock but not the random quality (or price) of the new
    units, and its payload is left as is so legacy runs stay byte
    identical. The audit checks a restocked unit against its template
    (``restocked unit matches no template``, ``outside the restocked
    band``), so a quality moved to another value inside the same band
    cannot be told from the real draw and passes. This test pins that gap.
    """
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "gap.db")
    try:
        persona = _persona(conn, SELLER)
        unit = persona["inventory_items"][_restocked(persona)]
        assert unit.get("unit_uid") is None  # a bound unit's quality is logged
        unit["ground_truth_quality_pct"] = _other_in_band(
            unit["ground_truth_quality_pct"], unit["stated_quality_band"],
        )
        with conn:
            conn.execute(
                "UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                (json.dumps(persona, sort_keys=True), SELLER),
            )

        assert audit_platform_event_consistency(conn) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# documented legacy behaviour is a warning, not an error
# ---------------------------------------------------------------------------


def _sister_meetup_world(path: Path) -> int:
    """Legacy run: two buyers commit to one listing and the first sale
    leaves the second buyer's meetup scheduled on a cancelled thread."""
    env = BazaarEnv(db_path=path, inventory_validator_mode="off")
    for agent_id, seed in ((SELLER, 702), (BUYER, 700), (OTHER, 701)):
        env.add_agent(MarketAgent(
            persona=generate_persona(agent_id, seed=seed),
            policy=RandomBenignPolicy(seed=agent_id),
        ))
    env.reset()
    conn = env.platform.conn
    try:
        listing_id = _act(
            conn, SELLER, ActionType.CREATE_LISTING, 0, category="tools",
            title="Garden Hose Reel", description="", price_cents=5000, condition="good",
        ).payload["listing_id"]
        meetups = []
        for buyer in (BUYER, OTHER):
            offer = _act(conn, buyer, ActionType.MAKE_OFFER, 1,
                         listing_id=listing_id, price_cents=4500)
            _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
            meetups.append(_act(
                conn, SELLER, ActionType.SCHEDULE_MEETUP, 3,
                thread_id=offer.payload["thread_id"], location_desc="library",
                scheduled_tick=10, payment_method="cash",
            ).payload["meetup_id"])
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 10, meetup_id=meetups[0])
        for agent_id in (BUYER, SELLER):
            _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 11, meetup_id=meetups[0])
    finally:
        env.close()
    return meetups[1]


def _set_integrity(conn: sqlite3.Connection, value: str | None) -> None:
    """Set ``completion_integrity_mode``; None removes the four handoff
    rows, as in a database written before the flags existed."""
    with conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (COMPLETION_INTEGRITY_MODE,))
        if value is None:
            conn.execute(
                "DELETE FROM meta WHERE key IN (?, ?, ?)",
                tuple(key for key in HANDOFF_CHECK_MODES if key != COMPLETION_INTEGRITY_MODE),
            )
        else:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)", (COMPLETION_INTEGRITY_MODE, value),
            )


@pytest.mark.parametrize(
    ("integrity", "severity"),
    [("off", "warning"), (None, "warning"), ("bogus", "warning"), ("unit", "error")],
)
def test_state_audit_reports_legacy_sister_meetup_by_integrity_mode(
    tmp_path, integrity, severity,
) -> None:
    path = tmp_path / "sister.db"
    stranded = _sister_meetup_world(path)
    conn = connect(path)
    try:
        _set_integrity(conn, integrity)

        issues = audit_marketplace_state_invariants(conn)
        stale = [i for i in issues if "scheduled_meetup_committed_thread" in i.message]
        others = [i for i in issues if i not in stale]

        assert [(i.ref_id, i.severity) for i in stale] == [(stranded, severity)]
        assert "scheduled meetup on thread status 'cancelled'" in stale[0].message
        assert ("documented legacy behaviour" in stale[0].message) == (severity == "warning")
        if severity == "warning":
            assert "left scheduled by complete_transaction event" in stale[0].message
        if integrity in ("off", None):
            assert others == []
        elif integrity == "bogus":
            # Handlers read an undocumented value as legacy, but meta must
            # hold one of the documented values.
            assert [i.severity for i in others] == ["error"]
            assert "completion_integrity_mode='bogus' is not one of" in others[0].message
        else:
            # meta switched on over an all-legacy log.
            assert others and all(
                i.severity == "error" and "legacy-shaped" in i.message for i in others
            ), [i.message for i in others]
    finally:
        conn.close()


def _errors(issues: list[Any]) -> list[Any]:
    return [issue for issue in issues if issue.severity != "warning"]


def _world(path: Path, **env_kwargs: Any) -> BazaarEnv:
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", **env_kwargs)
    for agent_id, seed in ((SELLER, 702), (BUYER, 700), (OTHER, 701)):
        env.add_agent(MarketAgent(
            persona=generate_persona(agent_id, seed=seed),
            policy=RandomBenignPolicy(seed=agent_id),
        ))
    env.reset()
    return env


def _list_unit(conn: sqlite3.Connection, index: int, tick: int = 0, **extra: Any) -> int:
    unit = _persona(conn, SELLER)["inventory_items"][index]
    return _act(
        conn, SELLER, ActionType.CREATE_LISTING, tick, category=unit["category"],
        title=unit["title"], description="", price_cents=5000,
        condition=unit["condition"], **extra,
    ).payload["listing_id"]


def _commit(conn: sqlite3.Connection, buyer: int, listing_id: int, tick: int) -> dict[str, int]:
    offer = _act(conn, buyer, ActionType.MAKE_OFFER, tick, listing_id=listing_id,
                 price_cents=4500)
    _act(conn, SELLER, ActionType.ACCEPT_OFFER, tick + 1, offer_id=offer.payload["offer_id"])
    meetup = _act(
        conn, SELLER, ActionType.SCHEDULE_MEETUP, tick + 2,
        thread_id=offer.payload["thread_id"], location_desc="library",
        scheduled_tick=tick + 5, payment_method="cash",
    ).payload["meetup_id"]
    return {"offer": offer.payload["offer_id"], "thread": offer.payload["thread_id"],
            "meetup": meetup}


def _all_audits(conn: sqlite3.Connection) -> list[Any]:
    return (
        audit_platform_event_consistency(conn, require_seed_coverage=True)
        + audit_agent_action_event_consistency(conn)
        + audit_marketplace_state_invariants(conn)
    )


# ---------------------------------------------------------------------------
# handoff-check modes: meta rows versus the event log
# ---------------------------------------------------------------------------


def _set_meta(conn: sqlite3.Connection, **rows: str | None) -> None:
    with conn:
        for key, value in rows.items():
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))
            if value is not None:
                conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))


def _mode_issues(issues: list[Any]) -> list[Any]:
    return [i for i in issues if "handoff_mode_meta_log_agreement" in i.message]


def test_mode_agreement_passes_for_consistent_runs(trade_world, tmp_path) -> None:
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "modes.db")
    try:
        assert _mode_issues(audit_marketplace_state_invariants(conn)) == []
    finally:
        conn.close()


def test_mode_agreement_flags_meta_downgraded_to_legacy(trade_world, tmp_path) -> None:
    preset, path, _ids = trade_world
    if preset != "truthful":
        pytest.skip("needs unit-mode keys in the log")
    conn = _copy(path, tmp_path / "downgraded.db")
    try:
        _set_meta(conn, **{key: modes[0] for key, modes in HANDOFF_CHECK_MODES.items()})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert issues and all(i.severity == "error" for i in issues)
        messages = " | ".join(i.message for i in issues)
        for label in ("create_listing.backing_unit", "complete_transaction.consumed_unit",
                      "complete_transaction.cancelled_sister_meetup_ids",
                      "inspect_at_meetup.inspection_outcome"):
            assert label in messages, messages
    finally:
        conn.close()


def test_mode_agreement_flags_one_deleted_meta_row(trade_world, tmp_path) -> None:
    preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "partial.db")
    try:
        _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: None})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        # a truthful run also logged the switch, which meta now contradicts
        assert [i.severity for i in issues] == ["error"] * (1 + (preset == "truthful"))
        assert any("but not ['completion_integrity_mode']" in i.message for i in issues)
    finally:
        conn.close()


def _integrity_only_world(path: Path) -> dict[str, int]:
    """completion_integrity_mode=unit alone: create_listing binds nothing,
    so the completion binds the unit at the handoff (``bound_at_handoff``,
    a key only completion integrity writes) and consumes it."""
    env = _world(path, completion_integrity_mode="unit")
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        deal = _commit(conn, BUYER, listing_id, 1)
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 6, meetup_id=deal["meetup"])
        for agent_id in (BUYER, SELLER):
            done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 7,
                        meetup_id=deal["meetup"])
        assert "bound_at_handoff" in done.payload
        assert done.payload["consumed_unit"] is not None
        other = _list_unit(conn, 1, tick=8)
        second = _commit(conn, OTHER, other, 9)
    finally:
        env.close()
    return {"listing": listing_id, **deal, "second_thread": second["thread"]}


def test_mode_agreement_integrity_only_run_is_consistent(tmp_path) -> None:
    path = tmp_path / "integrity.db"
    _integrity_only_world(path)
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _all_audits(conn) == []
    finally:
        conn.close()


def test_mode_agreement_flags_integrity_switched_off_in_meta(tmp_path) -> None:
    path = tmp_path / "integrity.db"
    _integrity_only_world(path)
    conn = _copy(path, tmp_path / "off.db")
    try:
        _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: "off"})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error" and "complete_transaction.bound_at_handoff" in i.message
            and "only completion_integrity_mode=unit writes" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


def test_mode_agreement_flags_legacy_events_after_integrity_was_on(tmp_path) -> None:
    """meta says unit, but after an event only completion integrity
    explains, a leave_thread ran without cancelling its meetup."""
    path = tmp_path / "integrity.db"
    ids = _integrity_only_world(path)
    conn = _copy(path, tmp_path / "mixed.db")
    try:
        _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: "off"})
        _act(conn, OTHER, ActionType.LEAVE_THREAD, 20, thread_id=ids["second_thread"])
        conn.commit()
        _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: "unit"})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error" and "legacy-shaped" in i.message
            and "leave_thread without cancelled_meetup_ids" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


def _log_config(conn: sqlite3.Connection, tick: int, **checks: str) -> None:
    from bazaar.core.event_log import log_event

    log_event(
        conn, tick=tick, agent_id=None, action_type="experiment_config",
        payload={"command": "llm-smoke", "defense_settings": dict(checks)},
        result_status="ok", result_payload={"meta_keys": ["experiment_config"]},
    )
    conn.commit()


def test_mode_agreement_accepts_a_logged_resume_switch_off(tmp_path) -> None:
    """A truthful run resumed without the flags (HandoffCheckResumeWarning)
    is legacy from the resume on; the experiment_config events explain the
    earlier unit-mode keys, so they are warnings."""
    path = tmp_path / "integrity.db"
    _integrity_only_world(path)
    conn = _copy(path, tmp_path / "resumed.db")
    try:
        with conn:
            # a database written before BazaarEnv logged its switches
            conn.execute("DELETE FROM events WHERE action_type = ?",
                         (HANDOFF_CHECKS_SET_ACTION,))
            conn.execute("UPDATE events SET event_id = event_id + 1000")
        _log_config(conn, 0, completion_integrity_mode="unit")
        with conn:
            conn.execute("UPDATE events SET event_id = 1 WHERE action_type = 'experiment_config'")
        _log_config(conn, 30)
        _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: "off"})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert issues and all(i.severity == "warning" for i in issues)
        assert all("documented resume switch-off" in i.message for i in issues)
    finally:
        conn.close()


def test_mode_agreement_accepts_a_truthful_continuation_of_a_legacy_base(tmp_path) -> None:
    """Switching the checks on for a continuation of a legacy base is
    documented: the legacy-shaped base events
    come before any unit-mode event, and the base's legacy
    experiment_config event does not describe the continuation (the matrix
    launcher skips that event)."""
    path = tmp_path / "continued.db"
    env = _world(path)
    conn = env.platform.conn
    try:
        _log_config(conn, 0)
        deal = _commit(conn, BUYER, _list_unit(conn, 0), 1)
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 6, meetup_id=deal["meetup"])
        for agent_id in (BUYER, SELLER):
            done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 7,
                        meetup_id=deal["meetup"])
        assert "consumed_unit" not in done.payload
    finally:
        env.close()
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks="truthful",
                    resume=True)
    conn = env.platform.conn
    try:
        later = _commit(conn, OTHER, _list_unit(conn, 1, tick=10), 11)
        _act(conn, OTHER, ActionType.LEAVE_THREAD, 13, thread_id=later["thread"])
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _errors(_all_audits(conn)) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# platform_handoff_checks_set: BazaarEnv logs every change of the checks
# ---------------------------------------------------------------------------


def _checks_set_events(conn: sqlite3.Connection) -> list[tuple[int, int, dict[str, Any]]]:
    return [
        (int(row[0]), int(row[1]), json.loads(row[2]))
        for row in conn.execute(
            "SELECT event_id, tick, payload FROM events WHERE action_type = ? "
            "ORDER BY event_id", (HANDOFF_CHECKS_SET_ACTION,),
        )
    ]


def _random_world(path: Path, agents: int, ticks: int, seed: int, **env_kwargs: Any) -> None:
    env = BazaarEnv(db_path=path, seed_phantom_listings=3, **env_kwargs)
    for i in range(agents):
        env.add_agent(MarketAgent(
            persona=generate_persona(i + 1, seed=seed + i),
            policy=RandomBenignPolicy(seed=seed + i),
        ))
    env.reset()
    env.step_many(ticks)
    env.close()


def _resume_random(path: Path, ticks: int, seed: int, **env_kwargs: Any) -> None:
    from bazaar.core.env import reconstruct_agents_from_db

    env = BazaarEnv(db_path=path, seed_phantom_listings=3, resume=True, **env_kwargs)
    for agent in reconstruct_agents_from_db(
        env.platform.conn,
        policy_factory=lambda agent_id: RandomBenignPolicy(seed=seed * 7 + agent_id),
    ):
        env.agents.append(agent)
    env.reset()
    env.step_many(ticks)
    env.close()


def test_fresh_truthful_run_logs_the_switch_at_its_start(tmp_path) -> None:
    path = tmp_path / "fresh.db"
    _random_world(path, 10, 30, 811, handoff_checks="truthful")
    conn = _copy(path, tmp_path / "check.db")
    try:
        first = conn.execute("SELECT MIN(event_id) FROM events").fetchone()[0]
        (event_id, tick, payload), = _checks_set_events(conn)
        assert (event_id, tick) == (first, 0)
        assert payload["old"] == dict.fromkeys(HANDOFF_CHECK_MODES)
        assert payload["new"] == TRUTHFUL_HANDOFF_CHECKS
        assert _errors(_all_audits(conn)) == []
    finally:
        conn.close()


def _legacy_base_with_a_legacy_sale(path: Path) -> None:
    """A legacy base whose mark_sold leaves the listing's meetup scheduled
    (no ``cancelled_meetup_ids``), the legacy shape of three checks."""
    env = _world(path)
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        _commit(conn, BUYER, listing_id, 1)
        done = _act(conn, SELLER, ActionType.MARK_SOLD, 4, listing_id=listing_id)
        assert "cancelled_meetup_ids" not in done.payload
    finally:
        env.close()


def test_truthful_resume_of_a_legacy_base_starts_the_checks_at_its_marker(tmp_path) -> None:
    """A plain truthful resume without experiment_config whose continuation
    leaves no trace of two of the checks: the marker, not the start of the
    log, is where they were switched on, so the base's legacy sale passes."""
    path = tmp_path / "continued.db"
    _legacy_base_with_a_legacy_sale(path)
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks="truthful",
                    resume=True)
    conn = env.platform.conn
    try:
        resume_tick = env.clock.current
        _list_unit(conn, 1, tick=resume_tick)
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        (event_id, tick, payload), = _checks_set_events(conn)
        assert tick == resume_tick
        assert payload["old"] == LEGACY_HANDOFF_CHECKS
        assert payload["new"] == TRUTHFUL_HANDOFF_CHECKS
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE action_type = 'experiment_config'",
        ).fetchone()[0] == 0
        issues = _all_audits(conn)
        assert _errors(issues) == [], [i.message for i in _errors(issues)]
    finally:
        conn.close()


@pytest.mark.parametrize("seed", [731, 736])
def test_truthful_resume_of_a_random_legacy_base_is_consistent(tmp_path, seed) -> None:
    path = tmp_path / "continued.db"
    _random_world(path, 20, 50, seed)
    _resume_random(path, 20, seed, handoff_checks="truthful")
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert len(_checks_set_events(conn)) == 1
        issues = _all_audits(conn)
        assert _errors(issues) == [], [i.message for i in _errors(issues)][:5]
    finally:
        conn.close()


def _event_rows(path: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT event_id, tick, agent_id, action_type, payload, result_status, "
            "result_payload FROM events ORDER BY event_id",
        ).fetchall()


@pytest.mark.parametrize("meta_rows", ["legacy rows", "no rows"])
def test_legacy_run_and_legacy_resume_log_no_marker(tmp_path, meta_rows) -> None:
    path = tmp_path / "legacy.db"
    _random_world(path, 8, 20, 821)
    assert _event_rows(path) and not [
        row for row in _event_rows(path) if row[3] == HANDOFF_CHECKS_SET_ACTION
    ]
    if meta_rows == "no rows":
        # a database written before the flags existed
        with sqlite3.connect(path) as conn:
            conn.executemany("DELETE FROM meta WHERE key = ?",
                             [(key,) for key in HANDOFF_CHECK_MODES])
    before = _event_rows(path)
    env = BazaarEnv(db_path=path, seed_phantom_listings=3, resume=True)
    env.close()
    for preset in ("legacy", None):
        kwargs = {} if preset is None else {"handoff_checks": preset}
        env = BazaarEnv(db_path=path, seed_phantom_listings=3, resume=True, **kwargs)
        env.close()
    assert _event_rows(path) == before
    _resume_random(path, 10, 821)
    rows = _event_rows(path)
    assert len(rows) > len(before)
    assert not [row for row in rows if row[3] == HANDOFF_CHECKS_SET_ACTION]
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _errors(_all_audits(conn)) == []
    finally:
        conn.close()


def test_truthful_resume_of_a_truthful_run_logs_nothing_new(tmp_path) -> None:
    path = tmp_path / "truthful.db"
    _random_world(path, 8, 15, 822, handoff_checks="truthful")
    _resume_random(path, 10, 822, handoff_checks="truthful")
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert len(_checks_set_events(conn)) == 1
        assert _errors(_all_audits(conn)) == []
    finally:
        conn.close()


@pytest.mark.parametrize("tamper", ["meta", "marker"])
def test_marker_contradicting_meta_is_an_error(tmp_path, tamper) -> None:
    path = tmp_path / "continued.db"
    _legacy_base_with_a_legacy_sale(path)
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks="truthful",
                    resume=True)
    env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        if tamper == "meta":
            _set_meta(conn, **{COMPLETION_INTEGRITY_MODE: "off"})
        else:
            with conn:
                conn.execute(
                    "UPDATE events SET payload = json_set(payload, '$.new.commitment_lock_mode', "
                    "'off') WHERE action_type = ?", (HANDOFF_CHECKS_SET_ACTION,),
                )
        issues = _errors(_mode_issues(audit_marketplace_state_invariants(conn)))
        assert any(
            f"latest {HANDOFF_CHECKS_SET_ACTION} event" in i.message for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


def test_legacy_shaped_event_after_the_marker_is_an_error(tmp_path) -> None:
    """The base's legacy sale passes, the same shape after the marker does
    not."""
    path = tmp_path / "continued.db"
    _legacy_base_with_a_legacy_sale(path)
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks="truthful",
                    resume=True)
    conn = env.platform.conn
    try:
        tick = env.clock.current
        listing_id = _list_unit(conn, 1, tick=tick)
        _commit(conn, BUYER, listing_id, tick + 1)
        done = _act(conn, SELLER, ActionType.MARK_SOLD, tick + 4, listing_id=listing_id)
        assert "cancelled_meetup_ids" in done.payload
        sold_event = done.event_id
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        with conn:
            conn.execute(
                "UPDATE events SET result_payload = json_remove(result_payload, "
                "'$.cancelled_meetup_ids') WHERE event_id = ?", (sold_event,),
            )
        issues = _errors(_mode_issues(audit_marketplace_state_invariants(conn)))
        assert any(
            "legacy-shaped" in i.message and f"event {sold_event}" in i.message
            and HANDOFF_CHECKS_SET_ACTION in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


def test_legacy_resume_of_a_truthful_run_logs_a_documented_switch_off(tmp_path) -> None:
    path = tmp_path / "integrity.db"
    _integrity_only_world(path)
    with pytest.warns(Warning, match="resuming with different truthful handoff checks"):
        env = BazaarEnv(db_path=path, inventory_validator_mode="off", resume=True)
    env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        events = _checks_set_events(conn)
        assert [payload["new"][COMPLETION_INTEGRITY_MODE] for _, _, payload in events] == [
            "unit", "off",
        ]
        issues = _mode_issues(audit_marketplace_state_invariants(conn))
        assert issues and all(i.severity == "warning" for i in issues)
        assert all("documented switch-off" in i.message for i in issues)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a stale meetup is a legacy warning only when the log explains it
# ---------------------------------------------------------------------------


def _stale_issue(issues: list[Any], meetup_id: int) -> Any:
    return next(
        i for i in issues
        if i.ref_id == meetup_id and "scheduled_meetup_committed_thread" in i.message
    )


def test_stale_meetup_needs_a_logged_legacy_explanation(tmp_path) -> None:
    path = tmp_path / "committed.db"
    env = _world(path)
    conn = env.platform.conn
    try:
        deal = _commit(conn, BUYER, _list_unit(conn, 0), 1)
    finally:
        env.close()
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        assert audit_marketplace_state_invariants(conn) == []
        with conn:
            conn.execute("UPDATE threads SET status = 'cancelled' WHERE thread_id = ?",
                         (deal["thread"],))

        issue = _stale_issue(audit_marketplace_state_invariants(conn), deal["meetup"])

        assert issue.severity == "error"
        assert "no logged legacy sale, leave_thread or ghost explains it" in issue.message
    finally:
        conn.close()


@pytest.mark.parametrize("action", [ActionType.GHOST, ActionType.LEAVE_THREAD])
def test_stale_meetup_after_legacy_ghost_or_leave_is_a_warning(tmp_path, action) -> None:
    path = tmp_path / "left.db"
    env = _world(path)
    conn = env.platform.conn
    try:
        deal = _commit(conn, BUYER, _list_unit(conn, 0), 1)
        _act(conn, BUYER, action, 4, thread_id=deal["thread"])
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        issue = _stale_issue(audit_marketplace_state_invariants(conn), deal["meetup"])

        assert issue.severity == "warning"
        assert f"left scheduled by {action.value} event" in issue.message
    finally:
        conn.close()


def test_stale_meetup_explained_only_by_an_event_without_meetup_cancellation(tmp_path) -> None:
    """A completion that logged cancelled_sister_meetup_ids would have
    cancelled the sister meetup, so it does not explain a scheduled one."""
    path = tmp_path / "sister.db"
    stranded = _sister_meetup_world(path)
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        with conn:
            conn.execute(
                """
                UPDATE events SET result_payload = json_set(
                    result_payload, '$.cancelled_sister_meetup_ids', json('[]'))
                WHERE action_type = 'complete_transaction'
                  AND json_extract(result_payload, '$.completed') = 1
                """
            )

        issues = audit_marketplace_state_invariants(conn)

        assert _stale_issue(issues, stranded).severity == "error"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# consumed_unit null: the bought unit carries the listing's logged quality
# ---------------------------------------------------------------------------


def _unbound_shipment_world(path: Path) -> dict[str, int]:
    """inspection_truth_mode=unit alone: a listing no seller unit backs
    (``backing_unit`` null, quality NULL) sells by shipment, so the
    unit-aware transfer logs ``consumed_unit`` null and the buyer's unit
    copies the listing's NULL quality (the key is left out)."""
    env = _world(path, inspection_truth_mode="unit")
    conn = env.platform.conn
    try:
        created = _act(
            conn, SELLER, ActionType.CREATE_LISTING, 0, category="tools",
            title="Qzxv Unmatched Gadget 9000", description="", price_cents=5000,
            condition="good",
        )
        assert created.payload["backing_unit"] is None
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1,
                     listing_id=created.payload["listing_id"], price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
        meetup = _act(
            conn, SELLER, ActionType.SCHEDULE_SHIPMENT, 3,
            thread_id=offer.payload["thread_id"], payment_method="cash",
        ).payload["meetup_id"]
        for agent_id in (BUYER, SELLER):
            done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 10, meetup_id=meetup)
        assert done.payload["completed"] is True
        assert done.payload["consumed_unit"] is None
    finally:
        env.close()
    return {"listing": created.payload["listing_id"]}


def test_unit_transfer_without_consumed_unit_copies_the_logged_listing_quality(
    tmp_path,
) -> None:
    path = tmp_path / "unbound.db"
    ids = _unbound_shipment_world(path)
    conn = _copy(path, tmp_path / "check.db")
    try:
        bought = _persona(conn, BUYER)["inventory_items"][-1]
        assert bought["bought_from_listing_id"] == ids["listing"]
        assert "ground_truth_quality_pct" not in bought
        assert _all_audits(conn) == []
    finally:
        conn.close()


@pytest.mark.parametrize("with_listing_row", [False, True])
def test_unit_transfer_without_consumed_unit_flags_an_invented_quality(
    tmp_path, with_listing_row,
) -> None:
    """The expected quality is rebuilt from the log (create_listing
    logged ``backing_unit`` null), so changing the listing row to match
    the invented value does not explain it."""
    path = tmp_path / "unbound.db"
    ids = _unbound_shipment_world(path)
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        persona = _persona(conn, BUYER)
        persona["inventory_items"][-1]["ground_truth_quality_pct"] = 55
        with conn:
            conn.execute("UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                         (json.dumps(persona), BUYER))
            if with_listing_row:
                conn.execute(
                    "UPDATE listings SET ground_truth_quality_pct = 55 WHERE listing_id = ?",
                    (ids["listing"],),
                )

        issues = audit_platform_event_consistency(conn)

        assert any(
            i.ref_id == BUYER and ".ground_truth_quality_pct mismatch: event=<missing> table=55"
            in i.message for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# truncated base (meta.truncated_base_note)
# ---------------------------------------------------------------------------


def _truncate(conn: sqlite3.Connection, target: int, source: int) -> None:
    with conn:
        conn.execute("DELETE FROM events WHERE tick > ?", (target,))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('truncated_base_note', ?)",
            (json.dumps({"target_max_tick": target, "source_max_tick": source}),),
        )


def test_truncated_base_reports_the_unlogged_window_as_warnings(trade_world, tmp_path) -> None:
    """The trade world sells at tick 11, marks sold at 12 and restocks at
    13; with the log cut at tick 10 those inventory changes are warnings
    that cite the note, and nothing is an error."""
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "truncated.db")
    try:
        _truncate(conn, 10, 13)

        issues = audit_platform_event_consistency(conn)

        assert _errors(issues) == [], [i.message for i in _errors(issues)]
        assert issues and all("truncated_base_note" in i.message for i in issues)
        messages = " | ".join(i.message for i in issues)
        assert "bought at tick 11" in messages
        assert "marked sold at tick 11" in messages and "marked sold at tick 12" in messages
        assert "restocked at tick 13" in messages
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (None, "persona_json.inventory_items purchases mismatch"),
        (_set(1, asking_price_cents=1), "inventory_items[1].asking_price_cents mismatch"),
        (_set(_bought, bought_tick=20), "persona_json.inventory_items purchases mismatch"),
        (_set(1, sold_at_tick=5, sold_via_listing_id=1), "inventory_items[1].sold_at_tick"),
    ],
    ids=["no note", "registered unit price", "bought after the window", "sold before it"],
)
def test_truncated_base_keeps_other_mismatches_errors(
    trade_world, tmp_path, edit, expected,
) -> None:
    _preset, path, ids = trade_world
    conn = _copy(path, tmp_path / "truncated.db")
    try:
        _truncate(conn, 10, 13)
        if edit is None:
            with conn:
                conn.execute("DELETE FROM meta WHERE key = 'truncated_base_note'")
        else:
            agent = BUYER if "bought" in expected or "purchases" in expected else SELLER
            persona = _persona(conn, agent)
            edit(persona, ids)
            with conn:
                conn.execute("UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                             (json.dumps(persona), agent))

        errors = _errors(audit_platform_event_consistency(conn))

        assert any(expected in i.message for i in errors), [i.message for i in errors]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# platform behaviour the log explains under every handoff check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_accepted_offer_on_a_logged_ghost_is_consistent(tmp_path, preset) -> None:
    """ghost ends a committed thread and rejects only pending offers, as
    leave_thread does, under every handoff check."""
    path = tmp_path / "ghost.db"
    env = _world(path, handoff_checks=preset)
    conn = env.platform.conn
    try:
        ghosted = _commit(conn, BUYER, _list_unit(conn, 0), 1)
        _act(conn, BUYER, ActionType.GHOST, 4, thread_id=ghosted["thread"])
        other = _commit(conn, OTHER, _list_unit(conn, 1, tick=5), 6)
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _errors(_all_audits(conn)) == []
        with conn:
            conn.execute("UPDATE threads SET status = 'ghosted' WHERE thread_id = ?",
                         (other["thread"],))

        issues = audit_marketplace_state_invariants(conn)

        assert any(
            i.severity == "error" and i.ref_id == other["offer"]
            and "accepted_offer_thread_status" in i.message
            and "not explained by a logged ghost" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_relisted_sale_is_consistent_only_with_its_relist_event(tmp_path, preset) -> None:
    """relist reopens a sold listing under every handoff check; the
    completed deal stays completed."""
    path = tmp_path / "relist.db"
    env = _world(path, handoff_checks=preset)
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        deal = _commit(conn, BUYER, listing_id, 1)
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 6, meetup_id=deal["meetup"])
        for agent_id in (BUYER, SELLER):
            _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 7, meetup_id=deal["meetup"])
        _act(conn, SELLER, ActionType.RELIST, 8, listing_id=listing_id)
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _errors(_all_audits(conn)) == []
        with conn:
            conn.execute("DELETE FROM events WHERE action_type = 'relist'")

        issues = audit_marketplace_state_invariants(conn)

        assert any(
            i.severity == "error" and i.ref_id == deal["meetup"]
            and "completed_meetup_terminal_state" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


def _ghost_then_cancel_world(path: Path) -> dict[str, int]:
    """Legacy: ghost leaves the meetup scheduled, and cancel_meetup then
    cancels it and turns the ghosted thread into a cancelled one."""
    env = _world(path)
    conn = env.platform.conn
    try:
        deal = _commit(conn, BUYER, _list_unit(conn, 0), 1)
        _act(conn, BUYER, ActionType.GHOST, 4, thread_id=deal["thread"])
        _act(conn, SELLER, ActionType.CANCEL_MEETUP, 5, meetup_id=deal["meetup"],
             reason="buyer vanished")
    finally:
        env.close()
    return deal


def test_legacy_ghost_then_cancel_meetup_is_a_warning(tmp_path) -> None:
    path = tmp_path / "ghost.db"
    deal = _ghost_then_cancel_world(path)
    conn = _copy(path, tmp_path / "check.db")
    try:
        issues = _all_audits(conn)

        assert _errors(issues) == []
        assert [(i.action_type, i.ref_id) for i in issues] == [("ghost", deal["thread"])]
        assert "documented legacy behaviour" in issues[0].message
        assert "cancel_meetup event" in issues[0].message
    finally:
        conn.close()


@pytest.mark.parametrize("tamper", ["integrity unit", "no cancel_meetup event"])
def test_ghost_then_cancelled_thread_is_an_error_otherwise(tmp_path, tamper) -> None:
    path = tmp_path / "ghost.db"
    deal = _ghost_then_cancel_world(path)
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        with conn:
            if tamper == "integrity unit":
                conn.execute(
                    "UPDATE meta SET value = 'unit' WHERE key = 'completion_integrity_mode'"
                )
            else:
                conn.execute("DELETE FROM events WHERE action_type = 'cancel_meetup'")

        issues = audit_agent_action_event_consistency(conn)

        assert [(i.action_type, i.ref_id, i.severity) for i in issues] == [
            ("ghost", deal["thread"], "error"),
        ]
        assert "thread_status mismatch: event='ghosted' table='cancelled'" == issues[0].message
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# shipment_inspection_mode against the log
# ---------------------------------------------------------------------------


def _shipment_world(path: Path, mode: str) -> dict[str, int]:
    """One shipment: under ``on_arrival`` the buyer inspects it after it
    arrives, under ``off`` both sides complete it uninspected."""
    env = _world(path, shipment_inspection_mode=mode)
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=listing_id,
                     price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
        meetup = _act(
            conn, SELLER, ActionType.SCHEDULE_SHIPMENT, 3,
            thread_id=offer.payload["thread_id"], payment_method="cash",
        ).payload["meetup_id"]
        if mode == "on_arrival":
            _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 20, meetup_id=meetup)
        for agent_id in (BUYER, SELLER):
            done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 20, meetup_id=meetup)
        assert done.payload["completed"] is True
        assert (done.payload["buyer_inspected_quality_pct"] is None) == (mode == "off")
    finally:
        env.close()
    return {"meetup": meetup}


@pytest.mark.parametrize("mode", ["off", "on_arrival"])
def test_shipment_mode_consistent_runs_pass(tmp_path, mode) -> None:
    path = tmp_path / "ship.db"
    _shipment_world(path, mode)
    conn = _copy(path, tmp_path / "check.db")
    try:
        assert _errors(_all_audits(conn)) == []
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("mode", "meta", "expected"),
    [
        ("on_arrival", "off",
         "inspect_at_meetup of a shipment (first event"),
        ("off", "on_arrival",
         "complete_transaction of a shipment the buyer confirmed without inspection"),
    ],
    ids=["meta off, log on", "meta on, legacy log"],
)
def test_shipment_mode_meta_must_agree_with_the_log(tmp_path, mode, meta, expected) -> None:
    path = tmp_path / "ship.db"
    _shipment_world(path, mode)
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        _set_meta(conn, shipment_inspection_mode=meta)

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error" and expected in i.message
            and "shipment_inspection_mode" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a check on in meta must have shaped the log (meta switched on over a
# legacy log)
# ---------------------------------------------------------------------------


def _legacy_all_kinds_world(path: Path) -> dict[str, int]:
    """Legacy run with an event of every kind a check changes: two buyers
    commit to one listing (the lock would refuse the second), an inspected
    meetup and an uninspected shipment complete, a thread is left."""
    env = _world(path)
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        first = _commit(conn, BUYER, listing_id, 1)
        second = _commit(conn, OTHER, listing_id, 1)
        _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, 6, meetup_id=first["meetup"])
        for agent_id in (BUYER, SELLER):
            _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 7, meetup_id=first["meetup"])
        shipped = _list_unit(conn, 1, tick=8)
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 9, listing_id=shipped,
                     price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 10, offer_id=offer.payload["offer_id"])
        shipment = _act(
            conn, SELLER, ActionType.SCHEDULE_SHIPMENT, 11,
            thread_id=offer.payload["thread_id"], payment_method="cash",
        ).payload["meetup_id"]
        for agent_id in (BUYER, SELLER):
            _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, 20, meetup_id=shipment)
        left = _commit(conn, OTHER, _list_unit(conn, 2, tick=21), 22)
        _act(conn, OTHER, ActionType.LEAVE_THREAD, 25, thread_id=left["thread"])
    finally:
        env.close()
    return {"second_thread": second["thread"]}


_LEGACY_SHAPES = {
    "inspection_truth_mode": ("unit", "create_listing without backing_unit"),
    "completion_integrity_mode": ("unit", "complete_transaction without consumed_unit"),
    "commitment_lock_mode": ("listing", "accept_offer on listing"),
    "shipment_inspection_mode": (
        "on_arrival",
        "complete_transaction of a shipment the buyer confirmed without inspection",
    ),
}


@pytest.fixture(scope="module")
def legacy_all_kinds(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("legacy_kinds") / "world.db"
    _legacy_all_kinds_world(path)
    return path


def test_legacy_all_kinds_world_is_consistent(legacy_all_kinds, tmp_path) -> None:
    conn = _copy(legacy_all_kinds, tmp_path / "check.db")
    try:
        issues = _all_audits(conn)
        assert _errors(issues) == [], [i.message for i in _errors(issues)]
        assert _mode_issues(issues) == []
    finally:
        conn.close()


@pytest.mark.parametrize("flag", sorted(_LEGACY_SHAPES))
def test_meta_switched_on_over_a_legacy_log_is_an_error(legacy_all_kinds, tmp_path, flag) -> None:
    value, label = _LEGACY_SHAPES[flag]
    conn = _copy(legacy_all_kinds, tmp_path / "switched_on.db")
    try:
        _set_meta(conn, **{flag: value})

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error" and f"meta has {flag}={value!r}" in i.message
            and "legacy-shaped" in i.message and label in i.message
            and "the start of the log" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


@pytest.mark.parametrize("flag", sorted(_LEGACY_SHAPES))
def test_switch_on_logged_by_experiment_config_starts_the_check(
    legacy_all_kinds, tmp_path, flag,
) -> None:
    """An experiment_config event that declares the check on marks where
    meta was set: legacy events before it are the earlier segment, a
    legacy event after it is an error."""
    value, _label = _LEGACY_SHAPES[flag]
    conn = _copy(legacy_all_kinds, tmp_path / "resumed.db")
    try:
        _log_config(conn, 30, **{flag: value})
        _set_meta(conn, **{flag: value})
        assert _mode_issues(audit_marketplace_state_invariants(conn)) == []

        # A legacy-shaped completion logged after the switch-on.
        delivery = "ship" if flag == "shipment_inspection_mode" else "meetup"
        with conn:
            conn.execute(
                """
                INSERT INTO events (tick, wall_time, agent_id, action_type, payload,
                                    result_status, result_payload)
                SELECT 31, wall_time, agent_id, action_type, payload, result_status,
                       result_payload
                FROM events
                WHERE action_type = 'complete_transaction' AND agent_id = ?
                  AND json_extract(result_payload, '$.completed') = 1
                  AND json_extract(result_payload, '$.delivery_method') = ?
                """,
                (SELLER, delivery),
            )

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error" and "legacy-shaped" in i.message
            and "experiment_config event" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "value", ["UNIT", " unit", "unit ", "bogus", "", "on"],
)
def test_undocumented_meta_value_is_an_error(legacy_all_kinds, tmp_path, value) -> None:
    conn = _copy(legacy_all_kinds, tmp_path / "value.db")
    try:
        _set_meta(conn, completion_integrity_mode=value)

        issues = _mode_issues(audit_marketplace_state_invariants(conn))

        assert any(
            i.severity == "error"
            and f"meta completion_integrity_mode={value!r} is not one of the documented "
            "values ['off', 'unit']" in i.message
            for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


@pytest.mark.parametrize("value", ["off", "unit"])
def test_documented_meta_values_are_accepted(trade_world, tmp_path, value) -> None:
    preset, path, _ids = trade_world
    if (value == "unit") != (preset == "truthful"):
        pytest.skip("value does not match the run")
    conn = _copy(path, tmp_path / "value.db")
    try:
        assert not any(
            "is not one of the documented values" in i.message
            for i in audit_marketplace_state_invariants(conn)
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a scheduled meetup needs its logged scheduling event, in every mode
# ---------------------------------------------------------------------------


def _insert_meetup(conn: sqlite3.Connection, thread_id: int, tick: int) -> int:
    with conn:
        cur = conn.execute(
            """
            INSERT INTO meetups (thread_id, scheduled_tick, location_desc, payment_method,
                                 buyer_confirmed, seller_confirmed, handoff_token, status)
            VALUES (?, ?, 'library', 'cash', 0, 0, 'forged', 'scheduled')
            """,
            (thread_id, tick),
        )
    return int(cur.lastrowid)


def _logged_issue(issues: list[Any], meetup_id: int) -> Any:
    return next(
        i for i in issues
        if i.ref_id == meetup_id and "scheduled_meetup_logged" in i.message
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ending", [None, ActionType.LEAVE_THREAD, ActionType.GHOST])
def test_unlogged_scheduled_meetup_is_an_error_in_every_mode(tmp_path, preset, ending) -> None:
    """A meetup row without its schedule event is an error, also on a
    thread a logged legacy leave_thread or ghost ended (which would
    explain a meetup it left scheduled, but not one it never logged)."""
    path = tmp_path / "meetup.db"
    env = _world(path, handoff_checks=preset)
    conn = env.platform.conn
    try:
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=_list_unit(conn, 0),
                     price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
        if ending is not None:
            _act(conn, BUYER, ending, 3, thread_id=offer.payload["thread_id"])
    finally:
        env.close()
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        assert _errors(_all_audits(conn)) == []
        meetup = _insert_meetup(conn, offer.payload["thread_id"], 5)

        issues = audit_marketplace_state_invariants(conn)

        issue = _logged_issue(issues, meetup)
        assert issue.severity == "error"
        assert "no logged schedule_meetup/schedule_shipment event" in issue.message
        assert not any(
            i.ref_id == meetup and i.severity == "warning" for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# truncated_base_note must fit the database and the tables
# ---------------------------------------------------------------------------


def _note(conn: sqlite3.Connection, target: int, source: int) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('truncated_base_note', ?)",
            (json.dumps({"target_max_tick": target, "source_max_tick": source}),),
        )


def _note_issues(issues: list[Any]) -> list[Any]:
    return [i for i in issues if "truncated_base_note_valid" in i.message]


def test_truncated_note_over_logged_ticks_explains_nothing(trade_world, tmp_path) -> None:
    """The window must lie after the log: a note whose window covers
    logged ticks (without a continuation record) is an error and its
    window changes stay errors."""
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "overlap.db")
    try:
        _truncate(conn, 10, 13)
        assert _note_issues(audit_marketplace_state_invariants(conn)) == []
        _note(conn, 9, 13)  # the log still holds tick 10

        state = audit_marketplace_state_invariants(conn)
        platform = audit_platform_event_consistency(conn)

        assert [i.severity for i in _note_issues(state)] == ["error"]
        assert "does not lie after the log" in _note_issues(state)[0].message
        assert _errors(platform)
        assert not any("truncated_base_note" in i.message for i in platform)
    finally:
        conn.close()


@pytest.mark.parametrize(("fork", "valid"), [(10, True), (9, False), (None, False)])
def test_truncated_note_on_a_continuation_needs_its_fork(
    trade_world, tmp_path, fork, valid,
) -> None:
    """A continuation of a truncated base logs events after the window
    start; the note fits only when meta records the fork at the note's
    target_max_tick."""
    _preset, path, _ids = trade_world
    conn = _copy(path, tmp_path / "continued.db")
    try:
        _note(conn, 10, 13)
        if fork is not None:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('trapi_matrix_cell', ?)",
                    (json.dumps({"base_db": "base.db", "fork_tick": fork}),),
                )

        issues = _note_issues(audit_marketplace_state_invariants(conn))

        assert (issues == []) == valid, [i.message for i in issues]
    finally:
        conn.close()


def _spoof(trade_world, tmp_path, name: str) -> tuple[sqlite3.Connection, dict[str, int]]:
    """A complete run with a note whose window lies after its log (as a
    spoofed note would): only changes that agree with the tables pass."""
    _preset, path, ids = trade_world
    conn = _copy(path, tmp_path / f"{name}.db")
    last = conn.execute("SELECT MAX(tick) FROM events").fetchone()[0]
    _note(conn, last, last + 50)
    return conn, {**ids, "window": last + 5}


def _edit_persona(conn: sqlite3.Connection, agent: int, edit: Callable[[list[Any]], None]) -> None:
    persona = _persona(conn, agent)
    edit(persona["inventory_items"])
    with conn:
        conn.execute("UPDATE agents SET persona_json = ? WHERE agent_id = ?",
                     (json.dumps(persona), agent))


def _buyer_listing(conn: sqlite3.Connection) -> int:
    unit = _persona(conn, BUYER)["inventory_items"][0]
    listing_id = _act(
        conn, BUYER, ActionType.CREATE_LISTING, 0, category=unit["category"],
        title=unit["title"], description="", price_cents=5000, condition=unit["condition"],
    ).payload["listing_id"]
    conn.commit()
    return listing_id


def _unsold(units: list[Any]) -> dict[str, Any]:
    return next(u for u in units if "sold_at_tick" not in u and u.get("source") is None)


def test_spoofed_note_alone_changes_nothing(trade_world, tmp_path) -> None:
    conn, _ids = _spoof(trade_world, tmp_path, "note_only")
    try:
        issues = _all_audits(conn)
        assert _errors(issues) == []
        assert not any("truncated_base_note" in i.message for i in issues)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("owner", "expected"),
    [("self", None), ("missing", "sold_at_tick mismatch"), ("other", "sold_at_tick mismatch")],
)
def test_window_sold_mark_needs_the_agents_listing(trade_world, tmp_path, owner, expected) -> None:
    conn, ids = _spoof(trade_world, tmp_path, "sold")
    try:
        listing = {"self": ids["active"], "missing": 999999}.get(owner)
        if listing is None:
            listing = _buyer_listing(conn)
        _edit_persona(conn, SELLER, lambda units: _unsold(units).update(
            sold_at_tick=ids["window"], sold_via_listing_id=listing))

        issues = audit_platform_event_consistency(conn)

        if expected is None:
            assert _errors(issues) == []
            assert any("a sale no event logs" in i.message for i in issues)
        else:
            assert any(expected in i.message for i in _errors(issues)), (
                [i.message for i in issues])
    finally:
        conn.close()


def test_window_restock_must_copy_a_template(trade_world, tmp_path) -> None:
    conn, ids = _spoof(trade_world, tmp_path, "restock")
    try:
        def restock(units: list[Any], **changes: Any) -> None:
            template = _unsold(units)
            unit = {key: template[key] for key in (
                "title", "category", "condition", "stated_quality_band", "asking_price_cents",
            ) if key in template}
            unit.setdefault("stated_quality_band", "good")
            unit.update(
                source="restock", added_at_tick=ids["window"],
                acquisition_cost_cents=int(unit["asking_price_cents"] * 0.55),
                ground_truth_quality_pct=next(
                    lo for name, lo, _hi in QUALITY_BAND_RANGES
                    if name == unit["stated_quality_band"]),
            )
            unit.update(changes)
            units.append(unit)

        _edit_persona(conn, SELLER, restock)
        clean = audit_platform_event_consistency(conn)
        assert _errors(clean) == [], [i.message for i in _errors(clean)]
        assert any("restocked at tick" in i.message for i in clean)

        _edit_persona(conn, SELLER, lambda units: units.pop())
        _edit_persona(conn, SELLER, lambda units: restock(
            units, title="Rolex Daytona", asking_price_cents=1, hidden_defect="x"))

        errors = _errors(audit_platform_event_consistency(conn))

        messages = " | ".join(i.message for i in errors)
        assert "unexpected key 'hidden_defect'" in messages
        assert "matches no template" in messages
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("listing", "expected"),
    [("missing", "the listing does not exist"),
     ("own", "not a seller this agent bought from"),
     ("no deal", "no completed thread on the listing")],
)
def test_window_purchase_needs_a_completed_deal(trade_world, tmp_path, listing, expected) -> None:
    conn, ids = _spoof(trade_world, tmp_path, "bought")
    try:
        listing_id = {"missing": 999999, "no deal": ids["active"]}.get(listing)
        if listing_id is None:
            listing_id = _buyer_listing(conn)
        _edit_persona(conn, BUYER, lambda units: units.append({
            "source": "bought", "title": "Anything", "category": "tools", "condition": "good",
            "description": "", "bought_tick": ids["window"], "bought_from_listing_id": listing_id,
            "bought_price_cents": 1, "asking_price_cents": 1,
        }))

        errors = _errors(audit_platform_event_consistency(conn))

        assert any(expected in i.message for i in errors), [i.message for i in errors]
    finally:
        conn.close()


def _window_meetup_world(path: Path) -> dict[str, int]:
    """A committed thread whose meetup a truncation removed from the log,
    and a thread a logged leave_thread ended before the window."""
    env = _world(path)
    conn = env.platform.conn
    try:
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=_list_unit(conn, 0),
                     price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
        left = _act(conn, OTHER, ActionType.MAKE_OFFER, 3, listing_id=_list_unit(conn, 1),
                    price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 3, offer_id=left.payload["offer_id"])
        _act(conn, OTHER, ActionType.LEAVE_THREAD, 4, thread_id=left.payload["thread_id"])
        meetup = _act(
            conn, SELLER, ActionType.SCHEDULE_MEETUP, 12,
            thread_id=offer.payload["thread_id"], location_desc="library",
            scheduled_tick=15, payment_method="cash",
        ).payload["meetup_id"]
    finally:
        env.close()
    return {"meetup": meetup, "thread": offer.payload["thread_id"],
            "left": left.payload["thread_id"]}


@pytest.mark.parametrize(
    ("case", "severity", "expected"),
    [
        ("window", "warning", "unlogged window of a truncated base"),
        ("thread ended before", "error", "ended thread"),
        ("no accepted offer", "error", "holds no accepted offer"),
        ("tick before the window", "error", "is not after the log's end"),
    ],
)
def test_window_meetup_must_fit_the_tables(tmp_path, case, severity, expected) -> None:
    path = tmp_path / "window.db"
    ids = _window_meetup_world(path)
    conn = _copy(path, tmp_path / "check.db")
    try:
        _truncate(conn, 10, 20)  # drops the schedule_meetup event at tick 12
        meetup = ids["meetup"]
        if case == "thread ended before":
            meetup = _insert_meetup(conn, ids["left"], 15)
        elif case == "no accepted offer":
            with conn:
                conn.execute("UPDATE offers SET status = 'countered' WHERE thread_id = ?",
                             (ids["thread"],))
        elif case == "tick before the window":
            with conn:
                conn.execute("UPDATE meetups SET scheduled_tick = 9 WHERE meetup_id = ?",
                             (meetup,))

        issues = audit_marketplace_state_invariants(conn)

        issue = _logged_issue(issues, meetup)
        assert issue.severity == severity, issue.message
        assert expected in issue.message
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a truncated continuation sees the accepted offers of the unlogged window
# ---------------------------------------------------------------------------


def _price_errors(issues: list[Any]) -> list[str]:
    return [
        i.message for i in _errors(issues)
        if "bought_price_cents" in i.message or "asking_price_cents" in i.message
    ]


@pytest.mark.parametrize(
    "case",
    ["window offer", "no note", "offer after the window", "logged offer", "other price"],
)
def test_continuation_purchase_price_from_a_window_offer(trade_world, tmp_path, case) -> None:
    """A continuation of a truncated base completes a deal whose accepted
    offer the tables kept from the unlogged window, with a tick after the
    continuation's completion tick (the released L1-C/L3-RT continuations
    of the gpt-5.4-mini base). The price is that offer's, and only such an
    offer counts: one after the window, one a logged event created, or a
    recorded price that differs from the offer's stay errors."""
    _preset, path, ids = trade_world
    conn = _copy(path, tmp_path / "continued.db")
    try:
        thread_id, offer_id = conn.execute(
            "SELECT t.thread_id, o.offer_id FROM threads t JOIN offers o USING (thread_id) "
            "WHERE t.listing_id = ? AND o.status = 'accepted'",
            (ids["sold"],),
        ).fetchone()
        with conn:
            # The base log ends at tick 0; the offer, its acceptance and
            # the meetup's scheduling happened in the window (0, 20]; the
            # continuation logged the inspection and the completion.
            window = ["accept_offer", "schedule_meetup"]
            if case != "logged offer":
                window.append("make_offer")
            conn.execute(
                "DELETE FROM events WHERE action_type IN (SELECT value FROM json_each(?)) "
                "AND json_extract(result_payload, '$.thread_id') = ?",
                (json.dumps(window), thread_id),
            )
            tick = 25 if case == "offer after the window" else 15
            conn.execute("UPDATE offers SET tick = ? WHERE offer_id = ?", (tick, offer_id))
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('trapi_matrix_cell', ?)",
                (json.dumps({"base_db": "base.db", "fork_tick": 0}),),
            )
        if case != "no note":
            _note(conn, 0, 20)
        if case == "other price":
            _edit_persona(conn, BUYER, lambda units: units[_bought({"inventory_items": units})]
                          .update(bought_price_cents=1, asking_price_cents=1))

        issues = audit_platform_event_consistency(conn)

        if case == "window offer":
            assert _errors(issues) == [], [i.message for i in _errors(issues)]
        else:
            errors = _price_errors(issues)
            assert any("bought_price_cents mismatch" in m for m in errors), errors
            assert any("asking_price_cents mismatch" in m for m in errors), errors
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a truthful continuation of a legacy base keeps the meetups the base left
# ---------------------------------------------------------------------------


def _legacy_base_two_meetups(path: Path) -> dict[str, int]:
    """Legacy base: two buyers commit to one listing and schedule meetups."""
    env = _world(path)
    conn = env.platform.conn
    try:
        listing_id = _list_unit(conn, 0)
        deals = {}
        for buyer in (BUYER, OTHER):
            offer = _act(conn, buyer, ActionType.MAKE_OFFER, 1, listing_id=listing_id,
                         price_cents=4500)
            _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
            deals[buyer] = _act(
                conn, SELLER, ActionType.SCHEDULE_MEETUP, 3,
                thread_id=offer.payload["thread_id"], location_desc="library",
                scheduled_tick=10, payment_method="cash",
            ).payload["meetup_id"]
    finally:
        env.close()
    return {"sold": deals[BUYER], "stranded": deals[OTHER]}


def _complete(conn: sqlite3.Connection, meetup_id: int, tick: int) -> dict[str, Any]:
    _act(conn, BUYER, ActionType.INSPECT_AT_MEETUP, tick, meetup_id=meetup_id)
    for agent_id in (BUYER, SELLER):
        done = _act(conn, agent_id, ActionType.COMPLETE_TRANSACTION, tick + 1,
                    meetup_id=meetup_id)
    assert done.payload["completed"] is True
    return done.payload


@pytest.mark.parametrize("sale", ["in the legacy base", "after the switch-on"])
def test_meetup_left_by_the_legacy_base_is_judged_by_its_mode(tmp_path, sale) -> None:
    """A meetup that a legacy sale in the base left scheduled on a thread
    it cancelled stays a legacy warning after a truthful continuation sets
    completion_integrity_mode=unit in meta; a sale that ran after the
    switch-on point (the tampered case below) is an error."""
    path = tmp_path / "continued.db"
    ids = _legacy_base_two_meetups(path)
    if sale == "in the legacy base":
        env = BazaarEnv(db_path=path, inventory_validator_mode="off", resume=True)
        try:
            done = _complete(env.platform.conn, ids["sold"], 10)
            assert "cancelled_sister_meetup_ids" not in done
        finally:
            env.close()
    env = BazaarEnv(db_path=path, inventory_validator_mode="off", handoff_checks="truthful",
                    resume=True)
    conn = env.platform.conn
    try:
        if sale == "in the legacy base":
            # a later truthful deal leaves the traces of completion integrity
            listing_id = _list_unit(conn, 1, tick=20)
            deal = _commit(conn, BUYER, listing_id, 21)
            assert "consumed_unit" in _complete(conn, deal["meetup"], 26)
        else:
            done = _complete(conn, ids["sold"], 20)
            assert done["cancelled_sister_meetup_ids"] == [ids["stranded"]]
    finally:
        env.close()
    conn = _copy(path, tmp_path / "check.db")
    try:
        if sale == "after the switch-on":
            # The sale is the first trace of completion integrity; strip
            # its sister cancellation as if it had run under the legacy
            # contract.
            with conn:
                conn.execute(
                    "UPDATE events SET result_payload = json_remove(result_payload, "
                    "'$.cancelled_sister_meetup_ids') WHERE action_type = "
                    "'complete_transaction' AND json_extract(result_payload, '$.completed') = 1",
                )
                conn.execute("UPDATE meetups SET status = 'scheduled' WHERE meetup_id = ?",
                             (ids["stranded"],))
        issues = _all_audits(conn)
        stale = _stale_issue(issues, ids["stranded"])

        if sale == "in the legacy base":
            assert _errors(issues) == [], [i.message for i in _errors(issues)]
            assert stale.severity == "warning"
            assert "completion_integrity_mode was not yet 'unit'" in stale.message
            assert "left scheduled by complete_transaction event" in stale.message
        else:
            assert stale.severity == "error", stale.message
            assert "documented legacy behaviour" not in stale.message
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# a meetup of any status needs its logged scheduling event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["completed", "cancelled", "no_show"])
def test_unlogged_meetup_of_any_status_is_an_error(tmp_path, status) -> None:
    path = tmp_path / "meetup.db"
    env = _world(path)
    conn = env.platform.conn
    try:
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=_list_unit(conn, 0),
                     price_cents=4500)
        _act(conn, SELLER, ActionType.ACCEPT_OFFER, 2, offer_id=offer.payload["offer_id"])
    finally:
        env.close()
    conn = _copy(path, tmp_path / "tampered.db")
    try:
        meetup = _insert_meetup(conn, offer.payload["thread_id"], 5)
        with conn:
            conn.execute("UPDATE meetups SET status = ? WHERE meetup_id = ?", (status, meetup))

        issue = _logged_issue(audit_marketplace_state_invariants(conn), meetup)

        assert issue.severity == "error"
        assert f"{status} meetup has no logged schedule_meetup/schedule_shipment event" in (
            issue.message)
    finally:
        conn.close()


@pytest.mark.parametrize("status", ["cancelled", "no_show"])
def test_unlogged_closed_meetup_in_the_window_is_a_warning(tmp_path, status) -> None:
    path = tmp_path / "window.db"
    ids = _window_meetup_world(path)
    conn = _copy(path, tmp_path / "check.db")
    try:
        _truncate(conn, 10, 20)  # drops the schedule_meetup event at tick 12
        with conn:
            conn.execute("UPDATE meetups SET status = ? WHERE meetup_id = ?",
                         (status, ids["meetup"]))

        issue = _logged_issue(audit_marketplace_state_invariants(conn), ids["meetup"])

        assert issue.severity == "warning", issue.message
        assert "unlogged window of a truncated base" in issue.message
    finally:
        conn.close()


def test_seeded_history_meetup_is_not_an_unlogged_meetup(tmp_path) -> None:
    """A cold-start world seeds closed meetups on seeded listings before the
    first logged tick; the same row on a real listing is an error."""
    path = tmp_path / "meetup.db"
    env = _world(path)
    conn = env.platform.conn
    try:
        offer = _act(conn, BUYER, ActionType.MAKE_OFFER, 1, listing_id=_list_unit(conn, 0),
                     price_cents=4500)
    finally:
        env.close()
    conn = _copy(path, tmp_path / "seeded.db")
    try:
        meetup = _insert_meetup(conn, offer.payload["thread_id"], -30)
        with conn:
            conn.execute("UPDATE meetups SET status = 'completed' WHERE meetup_id = ?", (meetup,))
        assert _logged_issue(audit_marketplace_state_invariants(conn), meetup).severity == "error"
        with conn:
            conn.execute(
                "UPDATE listings SET is_seeded = 1 WHERE listing_id = "
                "(SELECT listing_id FROM threads WHERE thread_id = ?)",
                (offer.payload["thread_id"],),
            )

        issues = audit_marketplace_state_invariants(conn)

        assert not any(
            i.ref_id == meetup and "scheduled_meetup_logged" in i.message for i in issues
        ), [i.message for i in issues]
    finally:
        conn.close()

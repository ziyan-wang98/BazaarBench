#!/usr/bin/env python3
"""Evaluate a cold-start v3 world for alignment + diversity.

Alignment checks (interface ↔ simulation):
  A1  every persona has lifetime_days/joined_at_tick set and consistent
  A2  every journal event lands as ledger/listing/rating/block rows at
      tick = -days_ago * 24
  A3  every typed memory is also written into narrative_memories
  A4  agent_summary contains the LLM self_summary
  A5  prompt_summary mentions tier + tenure when lifetime_days > 0
  A6  slice_for_prompt(agent, tick=0) surfaces owned_listings + ledger
      from the cold-start seeded history
  A7  every cold-start agent has a non-empty journal AND >=3 typed
      memories AND a non-empty self_summary

Diversity checks (across N agents):
  D1  tier distribution + entropy
  D2  lifetime_days p10/p50/p90, per-tier and overall
  D3  big_five mean pairwise Euclidean distance (higher == more varied)
  D4  journal event_kind histogram per agent + population entropy
  D5  unique-fraction of hard_constraints strings (1.0 == no boilerplate)
  D6  trigram overlap (Jaccard) between communication_style strings
  D7  trigram overlap between self_summary strings
  D8  trigram overlap between typed_memory contents

Usage:
  python scripts/cold_start/eval_alignment.py --db runs/cold_start_eval/x.db \
    [--out-md runs/cold_start_eval/x_eval.md] [--examples 3]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bazaar.agents.persona import PersonaCard
from bazaar.memory.ledger import slice_for_prompt


@dataclass
class AgentRecord:
    agent_id: int
    persona_json: dict[str, Any]
    journal: list[dict[str, Any]] = field(default_factory=list)
    typed_memories: list[dict[str, Any]] = field(default_factory=list)
    self_summary: str | None = None
    tier: str | None = None
    lifetime_days: int = 0
    joined_at_tick: int = 0
    hard_constraints: list[str] = field(default_factory=list)
    communication_style: str | None = None


def _load_agents(conn: sqlite3.Connection) -> list[AgentRecord]:
    out: list[AgentRecord] = []
    for row in conn.execute(
        "SELECT agent_id, persona_json FROM agents WHERE is_seeded = 0 "
        "ORDER BY agent_id"
    ).fetchall():
        persona = json.loads(row["persona_json"] or "{}")
        cold = persona.get("cold_start") or {}
        if not cold:
            continue
        out.append(
            AgentRecord(
                agent_id=int(row["agent_id"]),
                persona_json=persona,
                journal=list(cold.get("journal") or []),
                typed_memories=list(cold.get("typed_memories") or []),
                self_summary=cold.get("self_summary"),
                tier=cold.get("tier"),
                lifetime_days=int(persona.get("lifetime_days") or 0),
                joined_at_tick=int(persona.get("joined_at_tick") or 0),
                hard_constraints=list(cold.get("hard_constraints") or []),
                communication_style=cold.get("communication_style"),
            )
        )
    return out


def _alignment_checks(
    conn: sqlite3.Connection,
    agents: list[AgentRecord],
) -> dict[str, Any]:
    a1_failures: list[int] = []
    a2_per_agent: list[dict[str, Any]] = []
    a3_per_agent: list[dict[str, Any]] = []
    a4_failures: list[int] = []
    a5_failures: list[int] = []
    a6_per_agent: list[dict[str, Any]] = []
    a7_failures: list[int] = []

    for rec in agents:
        # A1
        expected_join = -rec.lifetime_days * 24
        if rec.joined_at_tick != expected_join:
            a1_failures.append(rec.agent_id)

        # A2: journal event materialization rate
        ledger_ticks = {
            int(r["tick"]) for r in conn.execute(
                "SELECT tick FROM ledger_entries WHERE agent_id = ?",
                (rec.agent_id,),
            )
        }
        journal_ticks = {
            -int(ev["days_ago"]) * 24 for ev in rec.journal
            if isinstance(ev, dict) and isinstance(ev.get("days_ago"), int)
        }
        materialized = len(journal_ticks & ledger_ticks)
        a2_per_agent.append({
            "agent_id": rec.agent_id,
            "journal": len(rec.journal),
            "ledger_ticks_matching": materialized,
            "rate": round(materialized / max(1, len(journal_ticks)), 3),
        })

        # A3: typed memory → narrative_memories
        narratives = [
            str(r["content"] or "") for r in conn.execute(
                "SELECT content FROM narrative_memories WHERE agent_id = ?",
                (rec.agent_id,),
            )
        ]
        typed_in_narrative = 0
        for mem in rec.typed_memories:
            if not isinstance(mem, dict):
                continue
            content = str(mem.get("content") or "")[:80]
            if not content:
                continue
            if any(content in narr for narr in narratives):
                typed_in_narrative += 1
        a3_per_agent.append({
            "agent_id": rec.agent_id,
            "typed_memories": len(rec.typed_memories),
            "in_narrative": typed_in_narrative,
            "rate": round(typed_in_narrative / max(1, len(rec.typed_memories)), 3),
        })

        # A4: self_summary in agent_summary
        agent_summary_row = conn.execute(
            "SELECT content FROM agent_summary WHERE agent_id = ? "
            "ORDER BY tick DESC LIMIT 1",
            (rec.agent_id,),
        ).fetchone()
        if rec.self_summary and (
            agent_summary_row is None
            or rec.self_summary[:60] not in (agent_summary_row["content"] or "")
        ):
            a4_failures.append(rec.agent_id)

        # A5: prompt_summary tenure check
        try:
            persona_obj = PersonaCard.from_dict(rec.persona_json)
            ps = persona_obj.prompt_summary()
        except Exception:
            ps = ""
        if rec.lifetime_days > 0 and "Marketplace tenure" not in ps:
            a5_failures.append(rec.agent_id)

        # A6: slice_for_prompt content (correct key is recent_history)
        try:
            slc = slice_for_prompt(conn, agent_id=rec.agent_id, up_to_tick=0)
        except Exception:
            slc = {}
        # A6b: also probe per-table negative-tick counts for ratings,
        # listings, threads, blocks so we can see whether all four
        # journal materialization paths are reachable downstream.
        ratings_neg = conn.execute(
            "SELECT COUNT(*) FROM ratings WHERE ratee_agent_id = ? AND tick < 0",
            (rec.agent_id,),
        ).fetchone()[0]
        listings_neg = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE owner_agent_id = ? AND created_at_tick < 0",
            (rec.agent_id,),
        ).fetchone()[0]
        threads_neg = conn.execute(
            "SELECT COUNT(*) FROM threads WHERE seller_agent_id = ? AND created_at_tick < 0",
            (rec.agent_id,),
        ).fetchone()[0]
        blocks_neg = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE blocker_id = ? AND tick < 0",
            (rec.agent_id,),
        ).fetchone()[0]
        a6_per_agent.append({
            "agent_id": rec.agent_id,
            "owned_listings": len(slc.get("owned_listings") or []),
            "recent_sales_feed": len(slc.get("recent_sales_feed") or []),
            "recent_history": len(slc.get("recent_history") or []),
            "neg_tick_ratings": int(ratings_neg or 0),
            "neg_tick_listings": int(listings_neg or 0),
            "neg_tick_threads": int(threads_neg or 0),
            "neg_tick_blocks": int(blocks_neg or 0),
        })

        # A7: completeness gate
        if (
            len(rec.journal) == 0
            or len(rec.typed_memories) < 3
            or not rec.self_summary
        ):
            a7_failures.append(rec.agent_id)

    return {
        "A1_persona_consistency_failures": a1_failures,
        "A2_journal_to_ledger": {
            "mean_rate": round(
                statistics.mean(p["rate"] for p in a2_per_agent) if a2_per_agent else 0,
                3,
            ),
            "min_rate": min((p["rate"] for p in a2_per_agent), default=0),
            "per_agent": a2_per_agent,
        },
        "A3_typed_to_narrative": {
            "mean_rate": round(
                statistics.mean(p["rate"] for p in a3_per_agent) if a3_per_agent else 0,
                3,
            ),
            "min_rate": min((p["rate"] for p in a3_per_agent), default=0),
            "per_agent": a3_per_agent,
        },
        "A4_self_summary_in_agent_summary_failures": a4_failures,
        "A5_prompt_summary_missing_tenure": a5_failures,
        "A6_slice_for_prompt": {
            "mean_owned_listings": round(
                statistics.mean(p["owned_listings"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
            "mean_recent_sales": round(
                statistics.mean(p["recent_sales_feed"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
            "mean_recent_history": round(
                statistics.mean(p["recent_history"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
            "agents_with_zero_owned_listings": [
                p["agent_id"] for p in a6_per_agent if p["owned_listings"] == 0
            ],
            "agents_with_zero_recent_history": [
                p["agent_id"] for p in a6_per_agent if p["recent_history"] == 0
            ],
        },
        "A8_negative_tick_table_coverage": {
            "agents_with_zero_neg_tick_ratings": [
                p["agent_id"] for p in a6_per_agent if p["neg_tick_ratings"] == 0
            ],
            "agents_with_zero_neg_tick_listings": [
                p["agent_id"] for p in a6_per_agent if p["neg_tick_listings"] == 0
            ],
            "agents_with_zero_neg_tick_threads": [
                p["agent_id"] for p in a6_per_agent if p["neg_tick_threads"] == 0
            ],
            "any_agent_with_neg_tick_blocks": any(
                p["neg_tick_blocks"] > 0 for p in a6_per_agent
            ),
            "mean_neg_tick_ratings": round(
                statistics.mean(p["neg_tick_ratings"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
            "mean_neg_tick_listings": round(
                statistics.mean(p["neg_tick_listings"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
            "mean_neg_tick_threads": round(
                statistics.mean(p["neg_tick_threads"] for p in a6_per_agent) if a6_per_agent else 0,
                2,
            ),
        },
        "A7_completeness_failures": a7_failures,
    }


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return round(
        -sum(
            (c / total) * math.log2(c / total)
            for c in counter.values() if c > 0
        ),
        3,
    )


def _trigrams(text: str) -> set[str]:
    norm = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return {norm[i:i + 3] for i in range(len(norm) - 2)} if len(norm) >= 3 else set()


def _mean_pairwise_jaccard(strings: list[str]) -> float:
    grams = [_trigrams(s) for s in strings if s]
    if len(grams) < 2:
        return 0.0
    sims: list[float] = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            sims.append(inter / union if union else 0.0)
    return round(statistics.mean(sims), 3) if sims else 0.0


def _diversity_checks(agents: list[AgentRecord]) -> dict[str, Any]:
    if not agents:
        return {"agents": 0}
    tiers = Counter(a.tier or "unknown" for a in agents)
    lifetimes = [a.lifetime_days for a in agents]
    by_tier_lifetime: dict[str, list[int]] = {}
    for a in agents:
        by_tier_lifetime.setdefault(a.tier or "unknown", []).append(a.lifetime_days)

    big5_keys = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    big5_vectors: list[tuple[float, ...]] = []
    for a in agents:
        bf = a.persona_json.get("big_five") or {}
        big5_vectors.append(tuple(float(bf.get(k, 0.5)) for k in big5_keys))
    big5_pairwise: list[float] = []
    for i in range(len(big5_vectors)):
        for j in range(i + 1, len(big5_vectors)):
            d = math.sqrt(sum((big5_vectors[i][k] - big5_vectors[j][k]) ** 2 for k in range(5)))
            big5_pairwise.append(d)
    big5_mean = round(statistics.mean(big5_pairwise), 3) if big5_pairwise else 0.0

    journal_kind_counter: Counter[str] = Counter()
    journal_unique_per_agent: list[int] = []
    for a in agents:
        kinds = [str(ev.get("event_kind") or "?") for ev in a.journal if isinstance(ev, dict)]
        journal_kind_counter.update(kinds)
        journal_unique_per_agent.append(len(set(kinds)))
    journal_kind_entropy = _entropy(journal_kind_counter)

    all_constraints: list[str] = []
    for a in agents:
        all_constraints.extend(str(c).strip().lower() for c in a.hard_constraints if c)
    constraint_unique_fraction = (
        round(len(set(all_constraints)) / max(1, len(all_constraints)), 3)
        if all_constraints else 0.0
    )

    style_overlap = _mean_pairwise_jaccard([a.communication_style or "" for a in agents])
    self_summary_overlap = _mean_pairwise_jaccard([a.self_summary or "" for a in agents])

    typed_strings: list[str] = []
    for a in agents:
        joined = " ".join(
            str(m.get("content") or "")
            for m in a.typed_memories if isinstance(m, dict)
        )
        typed_strings.append(joined)
    typed_overlap = _mean_pairwise_jaccard(typed_strings)

    return {
        "agents": len(agents),
        "D1_tier_distribution": dict(tiers),
        "D1_tier_entropy": _entropy(tiers),
        "D2_lifetime_days": {
            "p10": _percentile(lifetimes, 0.1),
            "p50": _percentile(lifetimes, 0.5),
            "p90": _percentile(lifetimes, 0.9),
            "min": min(lifetimes) if lifetimes else 0,
            "max": max(lifetimes) if lifetimes else 0,
            "by_tier": {
                tier: {
                    "n": len(vals),
                    "p50": _percentile(vals, 0.5),
                    "min": min(vals),
                    "max": max(vals),
                }
                for tier, vals in sorted(by_tier_lifetime.items()) if vals
            },
        },
        "D3_big_five_mean_pairwise_distance": big5_mean,
        "D4_journal_event_kinds": dict(journal_kind_counter),
        "D4_journal_kind_entropy": journal_kind_entropy,
        "D4_unique_kinds_per_agent_p50": _percentile(journal_unique_per_agent, 0.5),
        "D5_constraint_unique_fraction": constraint_unique_fraction,
        "D5_constraints_total": len(all_constraints),
        "D5_constraints_unique": len(set(all_constraints)),
        "D6_communication_style_trigram_overlap": style_overlap,
        "D7_self_summary_trigram_overlap": self_summary_overlap,
        "D8_typed_memory_trigram_overlap": typed_overlap,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[idx], 2)


def _examples(agents: list[AgentRecord], n: int) -> list[dict[str, Any]]:
    if not agents:
        return []
    # Pick n agents spread across tiers when possible.
    by_tier: dict[str, list[AgentRecord]] = {}
    for a in agents:
        by_tier.setdefault(a.tier or "unknown", []).append(a)
    picked: list[AgentRecord] = []
    while len(picked) < n and any(by_tier.values()):
        for tier in sorted(by_tier.keys()):
            if by_tier[tier]:
                picked.append(by_tier[tier].pop(0))
                if len(picked) >= n:
                    break
    return [
        {
            "agent_id": a.agent_id,
            "tier": a.tier,
            "lifetime_days": a.lifetime_days,
            "communication_style": a.communication_style,
            "self_summary": a.self_summary,
            "hard_constraints": a.hard_constraints[:4],
            "journal_kinds": [ev.get("event_kind") for ev in a.journal[:6]],
            "typed_memory_types": [m.get("memory_type") for m in a.typed_memories[:6]],
        }
        for a in picked
    ]


def evaluate(db: Path, n_examples: int = 3) -> dict[str, Any]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        agents = _load_agents(conn)
        return {
            "db": str(db),
            "n_agents": len(agents),
            "alignment": _alignment_checks(conn, agents),
            "diversity": _diversity_checks(agents),
            "examples": _examples(agents, n_examples),
        }
    finally:
        conn.close()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Cold-start v3 evaluation: {report['db']}",
        f"Agents: {report['n_agents']}",
        "",
        "## Alignment",
    ]
    a = report["alignment"]
    lines.extend([
        f"- A1 persona consistency failures: {a['A1_persona_consistency_failures']}",
        f"- A2 journal→ledger mean rate: {a['A2_journal_to_ledger']['mean_rate']} (min {a['A2_journal_to_ledger']['min_rate']})",
        f"- A3 typed-memory→narrative mean rate: {a['A3_typed_to_narrative']['mean_rate']} (min {a['A3_typed_to_narrative']['min_rate']})",
        f"- A4 self_summary missing in agent_summary: {a['A4_self_summary_in_agent_summary_failures']}",
        f"- A5 prompt_summary missing tenure: {a['A5_prompt_summary_missing_tenure']}",
        f"- A6 slice mean owned_listings={a['A6_slice_for_prompt']['mean_owned_listings']}, "
        f"recent_sales={a['A6_slice_for_prompt']['mean_recent_sales']}, "
        f"recent_history={a['A6_slice_for_prompt']['mean_recent_history']}",
        f"- A6 agents with 0 owned_listings: {a['A6_slice_for_prompt']['agents_with_zero_owned_listings']}",
        f"- A6 agents with 0 recent_history: {a['A6_slice_for_prompt']['agents_with_zero_recent_history']}",
        f"- A7 completeness failures: {a['A7_completeness_failures']}",
        f"- A8 mean neg-tick ratings={a['A8_negative_tick_table_coverage']['mean_neg_tick_ratings']}, "
        f"listings={a['A8_negative_tick_table_coverage']['mean_neg_tick_listings']}, "
        f"threads={a['A8_negative_tick_table_coverage']['mean_neg_tick_threads']}, "
        f"any_blocks={a['A8_negative_tick_table_coverage']['any_agent_with_neg_tick_blocks']}",
        f"- A8 agents missing neg-tick ratings: {a['A8_negative_tick_table_coverage']['agents_with_zero_neg_tick_ratings']}",
        f"- A8 agents missing neg-tick listings: {a['A8_negative_tick_table_coverage']['agents_with_zero_neg_tick_listings']}",
        "",
        "## Diversity",
    ])
    d = report["diversity"]
    lines.extend([
        f"- D1 tier distribution: {d['D1_tier_distribution']} (entropy {d['D1_tier_entropy']})",
        f"- D2 lifetime p10/p50/p90: {d['D2_lifetime_days']['p10']}/{d['D2_lifetime_days']['p50']}/{d['D2_lifetime_days']['p90']}",
        f"- D2 lifetime by tier: {d['D2_lifetime_days']['by_tier']}",
        f"- D3 big_five mean pairwise distance: {d['D3_big_five_mean_pairwise_distance']}",
        f"- D4 journal kinds: {d['D4_journal_event_kinds']} (entropy {d['D4_journal_kind_entropy']})",
        f"- D4 unique kinds per agent p50: {d['D4_unique_kinds_per_agent_p50']}",
        f"- D5 constraints unique fraction: {d['D5_constraint_unique_fraction']} "
        f"({d['D5_constraints_unique']}/{d['D5_constraints_total']})",
        f"- D6 communication_style trigram overlap (lower=better): {d['D6_communication_style_trigram_overlap']}",
        f"- D7 self_summary trigram overlap: {d['D7_self_summary_trigram_overlap']}",
        f"- D8 typed_memory trigram overlap: {d['D8_typed_memory_trigram_overlap']}",
        "",
        "## Examples",
    ])
    for ex in report["examples"]:
        lines.append(f"### Agent {ex['agent_id']} ({ex['tier']}, {ex['lifetime_days']}d)")
        lines.append(f"- style: {ex['communication_style']}")
        lines.append(f"- self_summary: {ex['self_summary']}")
        lines.append(f"- constraints: {ex['hard_constraints']}")
        lines.append(f"- journal kinds: {ex['journal_kinds']}")
        lines.append(f"- memory types: {ex['typed_memory_types']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    report = evaluate(args.db, n_examples=args.examples)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()

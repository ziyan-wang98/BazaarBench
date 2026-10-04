"""Exactness and complexity regressions for deterministic semantic sharding."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from bazaar.analysis_v2 import judge_runner
from bazaar.analysis_v2.contract import Channel
from bazaar.analysis_v2.judge_runner import BundleShard, _make_shard
from bazaar.analysis_v2.semantic_bundles import SemanticBundle


def _synthetic_bundle(rng: random.Random, index: int) -> SemanticBundle:
    channel_count = rng.randint(1, len(Channel))
    channels = tuple(rng.sample(tuple(Channel), channel_count))
    actor_id = index % 100
    token = ("π€\\\"\n" + chr(0x1F642)) * rng.randint(0, 80)
    bundle_id = f"synthetic:{index:04d}:{rng.randrange(1_000_000):06d}"
    return SemanticBundle(
        schema_version=1,
        bundle_id=bundle_id,
        cell_id="synthetic-cell",
        bundle_kind="semantic_text",
        target_channels=channels,
        denominator_kinds=("t5_text",),
        carrier_kind="message",
        carrier_id=str(index),
        carrier_tick=index + 1,
        judged_actor_ids=(actor_id,),
        treated_actor_ids=(actor_id,),
        counterparty_ids=((actor_id + 1) % 100,),
        thread_ids=(index // 3,),
        listing_ids=(),
        meetup_ids=(),
        observable={
            "text": token,
            "nested": {"index": index, "values": list(range(rng.randint(0, 9)))},
        },
        actions=(),
        reasoning=(),
        metadata={"unicode": "é/€₹", "padding": "x" * rng.randint(0, 1_500)},
        digest=f"legacy-content-tag:{index}",
    )


def _exceeds(shard: BundleShard, caps: dict[str, int]) -> bool:
    return (
        shard.input_bytes > caps["max_input_bytes"]
        or shard.estimated_input_tokens > caps["max_estimated_input_tokens"]
        or len(shard.bundles) > caps["max_bundles"]
        or shard.decision_units > caps["max_decision_units"]
    )


def _reference_append_greedy(
    bundles: tuple[SemanticBundle, ...], caps: dict[str, int]
) -> tuple[BundleShard, ...]:
    """Original append-and-render implementation, retained only as a test oracle."""

    shards: list[BundleShard] = []
    current: list[SemanticBundle] = []
    start = 0
    for index, bundle in enumerate(bundles):
        candidate = _make_shard(
            (*current, bundle), ordinal=len(shards), start_index=start
        )
        if current and _exceeds(candidate, caps):
            shards.append(
                _make_shard(current, ordinal=len(shards), start_index=start)
            )
            start = index
            current = [bundle]
            singleton = _make_shard(
                current, ordinal=len(shards), start_index=start
            )
            if _exceeds(singleton, caps):
                shards.append(replace(singleton, oversize_singleton=True))
                current = []
                start = index + 1
        else:
            current.append(bundle)
            if len(current) == 1 and _exceeds(candidate, caps):
                shards.append(replace(candidate, oversize_singleton=True))
                current = []
                start = index + 1
    if current:
        shards.append(_make_shard(current, ordinal=len(shards), start_index=start))
    return tuple(shards)


def _reference_binary_greedy(
    bundles: tuple[SemanticBundle, ...], caps: dict[str, int]
) -> tuple[BundleShard, ...]:
    """The superseded exact binary-search implementation."""

    shards: list[BundleShard] = []
    start = 0
    while start < len(bundles):
        ordinal = len(shards)
        singleton = _make_shard(
            bundles[start : start + 1], ordinal=ordinal, start_index=start
        )
        if _exceeds(singleton, caps):
            shards.append(replace(singleton, oversize_singleton=True))
            start += 1
            continue
        upper = min(len(bundles), start + caps["max_bundles"])
        low = start + 2
        high = upper
        best = singleton
        while low <= high:
            end = (low + high) // 2
            candidate = _make_shard(
                bundles[start:end], ordinal=ordinal, start_index=start
            )
            if _exceeds(candidate, caps):
                high = end - 1
            else:
                best = candidate
                low = end + 1
        shards.append(best)
        start = best.end_index_exclusive
    return tuple(shards)


@pytest.mark.parametrize("seed", range(10))
def test_incremental_sharding_matches_both_exact_references(seed: int) -> None:
    rng = random.Random(seed)
    bundles = tuple(
        _synthetic_bundle(rng, index) for index in range(rng.randint(8, 28))
    )
    boundary_end = rng.randint(1, min(len(bundles), 8))
    boundary = _make_shard(
        bundles[:boundary_end], ordinal=0, start_index=0
    )
    cap_sets = (
        {
            "max_input_bytes": 10**9,
            "max_estimated_input_tokens": 10**9,
            "max_bundles": rng.randint(1, min(len(bundles), 12)),
            "max_decision_units": rng.randint(1, 30),
        },
        {
            "max_input_bytes": max(
                1, boundary.input_bytes + rng.choice((-1, 0, 1))
            ),
            "max_estimated_input_tokens": 10**9,
            "max_bundles": len(bundles) + 1,
            "max_decision_units": len(bundles) * len(Channel),
        },
        {
            "max_input_bytes": 10**9,
            "max_estimated_input_tokens": max(
                1, boundary.estimated_input_tokens + rng.choice((-1, 0, 1))
            ),
            "max_bundles": len(bundles) + 1,
            "max_decision_units": len(bundles) * len(Channel),
        },
        {
            "max_input_bytes": 1,
            "max_estimated_input_tokens": 1,
            "max_bundles": len(bundles) + 1,
            "max_decision_units": len(bundles) * len(Channel),
        },
    )
    for caps in cap_sets:
        append_reference = _reference_append_greedy(bundles, caps)
        binary_reference = _reference_binary_greedy(bundles, caps)
        incremental = judge_runner.make_deterministic_shards(bundles, **caps)
        assert append_reference == binary_reference == incremental


def test_incremental_sharding_precomputes_and_materializes_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(8128)
    bundles = tuple(_synthetic_bundle(rng, index) for index in range(80))
    payload_calls = 0
    materializations = 0
    original_payload = judge_runner.judge_payload
    original_make_shard = judge_runner._make_shard

    def counted_payload(bundle: SemanticBundle) -> dict:
        nonlocal payload_calls
        payload_calls += 1
        return original_payload(bundle)

    def counted_make_shard(*args, **kwargs) -> BundleShard:
        nonlocal materializations
        materializations += 1
        return original_make_shard(*args, **kwargs)

    monkeypatch.setattr(judge_runner, "judge_payload", counted_payload)
    monkeypatch.setattr(judge_runner, "_make_shard", counted_make_shard)
    shards = judge_runner.make_deterministic_shards(
        bundles,
        max_input_bytes=40_000,
        max_estimated_input_tokens=20_000,
        max_bundles=7,
        max_decision_units=14,
    )

    assert payload_calls == len(bundles)
    assert materializations == len(shards)
    assert tuple(
        bundle for shard in shards for bundle in shard.bundles
    ) == bundles

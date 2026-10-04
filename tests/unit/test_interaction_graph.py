"""Unit tests for bazaar.memory.graph (T28c-4)."""
from __future__ import annotations

from bazaar.memory.graph import InteractionGraph


def test_add_and_neighbours() -> None:
    g = InteractionGraph()
    g.add_interaction(1, 2, "message", tick=10)
    g.add_interaction(1, 3, "offer",   tick=12)
    g.add_interaction(2, 1, "message", tick=14)
    assert g.out_neighbours(1) == [2, 3]
    assert g.out_neighbours(1, verb="offer") == [3]
    assert g.in_neighbours(1) == [2]
    assert len(g) == 3


def test_edges_between_directed_and_undirected() -> None:
    g = InteractionGraph()
    g.add_interaction(1, 2, "message", tick=5)
    g.add_interaction(2, 1, "message", tick=6)
    assert len(g.edges_between(1, 2)) == 2
    assert len(g.edges_between(1, 2, directed=True)) == 1
    assert len(g.edges_between(2, 1, directed=True)) == 1


def test_shortest_path_via_bfs() -> None:
    g = InteractionGraph()
    # 1 → 2 → 3 → 4  (shortest)
    # 1 → 5 → 4        (shorter!)
    g.add_interaction(1, 2, "m", tick=1)
    g.add_interaction(2, 3, "m", tick=2)
    g.add_interaction(3, 4, "m", tick=3)
    g.add_interaction(1, 5, "m", tick=4)
    g.add_interaction(5, 4, "m", tick=5)
    path = g.shortest_path(1, 4)
    assert path is not None
    assert path[0] == 1 and path[-1] == 4
    assert len(path) == 3  # 1 -> 5 -> 4


def test_shortest_path_missing() -> None:
    g = InteractionGraph()
    g.add_interaction(1, 2, "m", tick=1)
    assert g.shortest_path(1, 99) is None
    assert g.shortest_path(99, 1) is None
    assert g.shortest_path(1, 1) == [1]


def test_roundtrip_json_is_byte_identical() -> None:
    g = InteractionGraph()
    g.add_interaction(1, 2, "message", tick=10, valence=0.8, thread_id=7)
    g.add_interaction(1, 3, "offer",   tick=12)
    blob = g.to_json()
    g2 = InteractionGraph.from_json(blob)
    assert g2.to_json() == blob
    assert g2.out_neighbours(1) == [2, 3]


def test_replay_from_events_picks_up_supported_verbs() -> None:
    events = [
        {
            "action_type": "send_message",
            "agent_id": 1,
            "tick": 5,
            "result_status": "ok",
            "payload": {"thread_id": 10},
            "result_payload": {"counterparty_id": 2},
        },
        {
            "action_type": "rate",
            "agent_id": 1,
            "tick": 8,
            "result_status": "ok",
            "payload": {"target_agent_id": 3},
            "result_payload": {},
        },
        # Unsupported — different verb; should be ignored.
        {
            "action_type": "create_listing",
            "agent_id": 1, "tick": 1,
            "result_status": "ok", "payload": {}, "result_payload": {},
        },
        # Failed event: must be skipped.
        {
            "action_type": "send_message", "agent_id": 1, "tick": 9,
            "result_status": "error", "payload": {}, "result_payload": {},
        },
    ]
    g = InteractionGraph.replay_from_events(events)
    assert len(g) == 2
    assert g.out_neighbours(1) == [2, 3]


def test_degree_counts_both_directions() -> None:
    g = InteractionGraph()
    g.add_interaction(1, 2, "m", tick=1)
    g.add_interaction(2, 1, "m", tick=2)
    g.add_interaction(3, 1, "m", tick=3)
    assert g.degree(1) == 3
    assert g.degree(2) == 2
    assert g.degree(99) == 0

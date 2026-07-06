from __future__ import annotations

import networkx as nx

from MPC4plus.experiments.datasets.generate_harder_graph_dataset import (
    build_harder_round_v1_records,
    generate_connected_er_graph,
    is_harder_graph_proxy,
)
from MPC4plus.experiments.datasets.graph_dataset_io import record_to_graph


def test_generate_connected_er_graph_connected() -> None:
    import random

    rng = random.Random(42)
    G = generate_connected_er_graph(n=30, p=0.10, rng=rng)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 30
    assert G.number_of_edges() > 0
    assert nx.is_connected(G)


def test_is_harder_graph_proxy_returns_bool() -> None:
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 1),
            (1, 9), (1, 10), (2, 11), (3, 12),
        ]
    )
    out = is_harder_graph_proxy(G)
    assert isinstance(out, bool)


def test_build_harder_round_v1_records_small_count() -> None:
    records = build_harder_round_v1_records(seed=42, graphs_per_combo=2)

    # 8 combos * 2 = 16
    assert len(records) == 16

    ids = [r.graph_id for r in records]
    assert len(set(ids)) == len(ids)

    G0 = record_to_graph(records[0])
    assert isinstance(G0, nx.Graph)
    assert G0.number_of_nodes() > 0
    assert G0.number_of_edges() > 0
from __future__ import annotations

import json

import networkx as nx

from MPC4plus.experiments.datasets.generate_er_graph_dataset import (
    build_first_round_er_records,
    generate_connected_er_graph,
)
from MPC4plus.experiments.datasets.graph_dataset_io import record_to_graph


def test_generate_connected_er_graph_returns_connected_graph() -> None:
    import random

    rng = random.Random(42)
    G = generate_connected_er_graph(n=16, p=0.2, rng=rng)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 16
    assert G.number_of_edges() > 0
    assert nx.is_connected(G)


def test_build_first_round_er_records_small_count() -> None:
    records = build_first_round_er_records(seed=42, graphs_per_combo=2)

    # 8 combos * 2 graphs each = 16
    assert len(records) == 16

    ids = [r.graph_id for r in records]
    assert len(set(ids)) == len(ids)

    G0 = record_to_graph(records[0])
    assert isinstance(G0, nx.Graph)
    assert G0.number_of_nodes() > 0
    assert G0.number_of_edges() > 0
from __future__ import annotations

import networkx as nx

from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase3.responsible import mark_responsible_components
from MPC4plus.phase5.decomposition import (
    classify_phase5_buckets,
    component_bucket_name,
    summarize_phase5_buckets,
)
from MPC4plus.phase5.exact_solver import is_feasible_solution_graph
from MPC4plus.phase5.path_constructor import construct_Pv_for_anchor
from MPC4plus.phase5.phase5_runner import solve_mpc4plus


def test_phase5_bucket_classification_basic():
    meta0 = ComponentMeta(
        nodes={1, 2, 3, 4},
        is_composite=True,
        H_component_node_sets=[{1, 2}, {3, 4}],
        center_nodes={1, 2},
        center_kind="edge",
        satellites=[SatelliteInfo(nodes={3, 4}, kind="edge")],
        anchors=[1, 2],
        j_anchor_map={1: 2, 2: 0},
        critical_2_anchors={1},
        responsible_1_anchors=set(),
        s_value=8,
        opt_value=6,
        critical_ratio=8 / 6,
        is_critical=True,
        critical_case="one-2-anchor-edge-s8",
    )

    meta1 = ComponentMeta(
        nodes={10, 11, 12, 13},
        is_composite=True,
        H_component_node_sets=[{10, 11}, {12, 13}],
        center_nodes={10, 11},
        center_kind="edge",
        satellites=[SatelliteInfo(nodes={12, 13}, kind="edge")],
        anchors=[10, 11],
        j_anchor_map={10: 1, 11: 0},
        critical_2_anchors=set(),
        responsible_1_anchors={10},
        s_value=6,
        opt_value=5,
        critical_ratio=6 / 5,
        is_critical=False,
        critical_case=None,
    )

    metas = mark_responsible_components([meta0, meta1])
    meta0, meta1 = metas

    G = nx.Graph()
    G.add_nodes_from(range(1, 14))

    buckets = classify_phase5_buckets(G, [meta0, meta1])

    assert 1 in buckets.R
    assert 10 in buckets.R
    assert meta0 in buckets.K1
    assert meta0 in buckets.K1c
    assert meta1 in buckets.K1
    assert 1 in buckets.Rc
    assert meta1 in buckets.responsible_components

    summary = summarize_phase5_buckets(buckets)
    assert summary["K"] == 2
    assert summary["K1"] == 2
    assert summary["K1c"] == 1
    assert summary["responsible_components"] == 1

    assert component_bucket_name(meta0, buckets) == "K1c"
    assert component_bucket_name(meta1, buckets).startswith("responsible_component")


def test_construct_Pv_for_anchor_structured_local_case():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edges_from([(1, 3), (3, 4), (1, 5), (5, 6)])

    meta = ComponentMeta(
        nodes={1, 2, 3, 4, 5, 6},
        is_composite=True,
        H_component_node_sets=[{1, 2}, {3, 4}, {5, 6}],
        center_nodes={1, 2},
        center_kind="edge",
        satellites=[
            SatelliteInfo(
                nodes={3, 4},
                kind="edge",
                rescue_edges=[(1, 3)],
                supporting_anchors=[1],
                rescue_anchor=1,
                is_critical_satellite=True,
            ),
            SatelliteInfo(
                nodes={5, 6},
                kind="edge",
                rescue_edges=[(1, 5)],
                supporting_anchors=[1],
                rescue_anchor=1,
                is_critical_satellite=True,
            ),
        ],
        anchors=[1, 2],
        j_anchor_map={1: 2, 2: 0},
        critical_2_anchors={1},
        responsible_1_anchors=set(),
        s_value=8,
        opt_value=6,
        critical_ratio=8 / 6,
        is_critical=True,
        critical_case="one-2-anchor-edge-s8",
    )

    Pv = construct_Pv_for_anchor(G, [meta], 1)
    assert Pv.number_of_nodes() >= 5


def test_phase5_end_to_end_demo_graph_runs():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
    ])

    result = solve_mpc4plus(G)
    assert result.solution_graph is not None
    assert is_feasible_solution_graph(result.solution_graph) or result.solution_graph.number_of_nodes() == 0
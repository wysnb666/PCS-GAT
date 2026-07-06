from __future__ import annotations

import networkx as nx

from MPC4plus.solver.phase2_runner import build_demo_graph_phase2
from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.bad_components import get_bad_components
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.path_cycle_cover import (
    compute_max_weight_path_cycle_cover,
    is_path_cycle_cover_subgraph,
    saturated_bad_component_count,
    prune_redundant_edges_same_weight,
)


def test_phase2_pipeline_runs():
    G = build_demo_graph_phase2()
    phase1 = run_phase1(G)

    bad = get_bad_components(phase1.H)
    assert isinstance(bad, list)

    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert is_path_cycle_cover_subgraph(G1, C)


def test_step22_ilp_matches_bruteforce_on_demo_graph():
    G = build_demo_graph_phase2()
    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert is_path_cycle_cover_subgraph(G1, C_bf)
    assert is_path_cycle_cover_subgraph(G1, C_ilp)

    w_bf = saturated_bad_component_count(phase1.H, C_bf)
    w_ilp = saturated_bad_component_count(phase1.H, C_ilp)
    assert w_bf == w_ilp


def test_step22_ilp_matches_bruteforce_on_custom_small_graph():
    """
    A targeted small graph where G1 has multiple legal covers.
    We only require the ILP backend to match the brute-force optimum weight,
    not necessarily the exact same edge set after Step 2.3.
    """
    G = nx.Graph()
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (2, 5),
        (2, 7),
        (4, 5),
        (4, 7),
    ])

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert is_path_cycle_cover_subgraph(G1, C_bf)
    assert is_path_cycle_cover_subgraph(G1, C_ilp)

    w_bf = saturated_bad_component_count(phase1.H, C_bf)
    w_ilp = saturated_bad_component_count(phase1.H, C_ilp)
    assert w_bf == w_ilp


def test_step22_empty_g1_returns_empty_cover():
    """
    If G1 has no edges, both backends should return the empty cover.
    """
    G = nx.Graph()
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    ])

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)

    assert G1.number_of_edges() == 0

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert C_bf == set()
    assert C_ilp == set()
    assert saturated_bad_component_count(phase1.H, C_bf) == 0
    assert saturated_bad_component_count(phase1.H, C_ilp) == 0


def test_step22_not_all_bad_components_can_be_saturated():
    """
    Build a graph where one bad component is connected to three others through
    a single vertex in G1, so degree <= 2 prevents saturating all four bad components.

    Expected optimum weight is 3.
    """
    G = nx.Graph()
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (2, 3),
        (2, 5),
        (2, 7),
    ])

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert is_path_cycle_cover_subgraph(G1, C_bf)
    assert is_path_cycle_cover_subgraph(G1, C_ilp)

    w_bf = saturated_bad_component_count(phase1.H, C_bf)
    w_ilp = saturated_bad_component_count(phase1.H, C_ilp)

    assert w_bf == 3
    assert w_ilp == 3


def test_step22_multiple_distinct_optima_same_weight():
    """
    Construct a graph with at least two distinct maximum-weight path-cycle covers.

    Example:
      cover A = {(2,3), (6,7)}
      cover B = {(2,5), (4,7)}

    Both saturate all four bad components, so the optimum weight is 4.
    ILP and brute-force do not need to return the same edge set, but must return
    legal covers with equal optimum weight.
    """
    G = nx.Graph()
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (2, 3),
        (2, 5),
        (4, 7),
        (6, 7),
    ])

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

    assert is_path_cycle_cover_subgraph(G1, C_bf)
    assert is_path_cycle_cover_subgraph(G1, C_ilp)

    w_bf = saturated_bad_component_count(phase1.H, C_bf)
    w_ilp = saturated_bad_component_count(phase1.H, C_ilp)

    assert w_bf == 4
    assert w_ilp == 4


def test_step23_prune_redundant_edges_keeps_weight():
    """
    Directly test Step 2.3 on a hand-crafted H and C.

    H has three bad edge-components:
      {1,2}, {3,4}, {5,6}

    C has redundant saturation:
      {(1,3), (2,4), (1,5)}

    Weight is 3 before pruning, and should remain 3 after pruning,
    while the number of edges should decrease.
    """
    H = nx.Graph()
    H.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
    ])

    C = {
        (1, 3),
        (2, 4),
        (1, 5),
    }

    w_before = saturated_bad_component_count(H, C)
    C_pruned = prune_redundant_edges_same_weight(H, C)
    w_after = saturated_bad_component_count(H, C_pruned)

    assert w_before == 3
    assert w_after == 3
    assert len(C_pruned) < len(C)


def test_step22_ilp_matches_bruteforce_on_several_fixed_small_graphs():
    """
    A mini regression set of fixed small graphs.
    These are deterministic and chosen to cover several distinct G1 patterns.

    We compare only optimum weight and legality, not exact edge-set equality.
    """
    graph_edge_sets = [
        # sparse cross-component rescue pattern
        [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (2, 3), (4, 5),
        ],
        # one component can connect to many others
        [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (2, 3), (2, 5), (2, 7),
        ],
        # alternative optimal matching-like covers
        [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (2, 3), (2, 5), (4, 7), (6, 7),
        ],
        # denser rescue structure
        [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (2, 3), (2, 5), (4, 5), (4, 7), (6, 7),
        ],
    ]

    for edges in graph_edge_sets:
        G = nx.Graph()
        G.add_edges_from(edges)

        phase1 = run_phase1(G)
        G1 = build_auxiliary_graph_G1(G, phase1.H)

        C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
        C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")

        assert is_path_cycle_cover_subgraph(G1, C_bf)
        assert is_path_cycle_cover_subgraph(G1, C_ilp)

        w_bf = saturated_bad_component_count(phase1.H, C_bf)
        w_ilp = saturated_bad_component_count(phase1.H, C_ilp)
        assert w_bf == w_ilp
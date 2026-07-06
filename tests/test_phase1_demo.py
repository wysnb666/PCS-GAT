from __future__ import annotations

import networkx as nx

from MPC4plus.solver.phase1_runner import (
    build_demo_graph_case_c1,
    build_demo_graph_step12_triangle,
)
from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.core.components import get_component_kind_map, validate_phase1_output


def build_demo_graph_step12_star() -> nx.Graph:
    """
    An edge component {1,2} in H, and outside vertices 3,4 each adjacent only to 1.
    After Step 1.2, this should become a star centered at 1.
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([
        (1, 2),
        (1, 3),
        (1, 4),
    ])
    return G


def test_case_c1_five_path():
    G = build_demo_graph_case_c1()
    state = run_phase1(G)

    kinds = [kind for _, kind in get_component_kind_map(state.H)]
    assert "five_path" in kinds

    ok, msg = validate_phase1_output(state.H, state.M)
    assert ok, msg


def test_step12_triangle():
    """
    This matches Lemma 3 exactly:
    if the two endpoints of an edge component are adjacent to a common outside vertex,
    then the component becomes a triangle.
    """
    G = build_demo_graph_step12_triangle()
    state = run_phase1(G)

    kinds = [kind for _, kind in get_component_kind_map(state.H)]
    assert "triangle" in kinds

    ok, msg = validate_phase1_output(state.H, state.M)
    assert ok, msg


def test_step12_star():
    """
    This matches Lemma 3 exactly:
    if exactly one endpoint of an edge component is adjacent to one or more outside vertices,
    then the component becomes a star.
    """
    G = build_demo_graph_step12_star()
    state = run_phase1(G)

    kinds = [kind for _, kind in get_component_kind_map(state.H)]
    assert "star" in kinds

    ok, msg = validate_phase1_output(state.H, state.M)
    assert ok, msg
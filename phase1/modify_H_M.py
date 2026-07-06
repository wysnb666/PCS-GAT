from __future__ import annotations

import networkx as nx

from MPC4plus.core.graph_utils import normalize_edge, get_outside_vertices, Edge
from MPC4plus.core.matching import compute_maximum_matching
from MPC4plus.core.components import get_edge_components
from MPC4plus.phase1.initialize_H import Phase1State, initialize_H_from_matching
from MPC4plus.phase1.augmenting_triple import AugmentingTriple, find_augmenting_triple


def apply_augmenting_triple(H: nx.Graph, M: set[Edge], triple: AugmentingTriple) -> None:
    """
    Modify H and M according to Definition 2 in the paper.
    """
    H.add_node(triple.u0)

    if triple.case == "C1":
        H.add_edge(triple.u0, triple.v0)
        H.add_edge(triple.u0, triple.v1)

    elif triple.case == "C2":
        H.add_edge(triple.u0, triple.v0)
        H.add_edge(triple.w0, triple.v1)

        old_edge = normalize_edge(*triple.e0)
        new_edge = normalize_edge(triple.u0, triple.v0)
        M.remove(old_edge)
        M.add(new_edge)

    else:
        raise ValueError(f"Unknown augmenting triple case: {triple.case}")


def run_step_1_1(G: nx.Graph, H: nx.Graph, M: set[Edge]) -> tuple[nx.Graph, set[Edge]]:
    """
    Repeatedly apply augmenting triples until none exists.
    """
    while True:
        triple = find_augmenting_triple(G, H, M)
        if triple is None:
            break
        apply_augmenting_triple(H, M, triple)
    return H, M


def run_step_1_2(G: nx.Graph, H: nx.Graph, M: set[Edge]) -> tuple[nx.Graph, set[Edge]]:
    """
    Step 1.2:
    For each vertex u in V(G) \\ V(H), if u is adjacent to v where v is in
    an edge component of H, then add u and {u,v} to U and F, respectively.
    Finally, add U and F to H.

    To match Lemma 3 exactly, we must add ALL such edges {u,v} where
    v belongs to an edge component of H. Otherwise, a common outside vertex
    adjacent to both endpoints of one edge component could not create a triangle.
    """
    U = set()
    F = set()

    edge_component_vertices = set()
    for e in get_edge_components(H):
        edge_component_vertices.update(e)

    for u in get_outside_vertices(G, H):
        adjacent_edge_vertices = [
            v for v in sorted(G.neighbors(u))
            if v in edge_component_vertices
        ]
        if adjacent_edge_vertices:
            U.add(u)
            for v in adjacent_edge_vertices:
                F.add(normalize_edge(u, v))

    H.add_nodes_from(U)
    H.add_edges_from(F)
    return H, M


def run_phase1(G: nx.Graph) -> Phase1State:
    """
    Full Phase 1:
      - compute a maximum matching M
      - initialize H=(V(M),M)
      - Step 1.1
      - Step 1.2
    """
    M = compute_maximum_matching(G)
    H = initialize_H_from_matching(G, M)
    H, M = run_step_1_1(G, H, M)
    H, M = run_step_1_2(G, H, M)
    return Phase1State(G=G, M=M, H=H)
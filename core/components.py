from __future__ import annotations

from typing import Literal
import networkx as nx

from .graph_utils import Edge, normalize_edge, connected_components_with_subgraphs

ComponentKind = Literal["edge", "triangle", "star", "five_path", "other"]


def get_edge_components(H: nx.Graph) -> list[Edge]:
    """
    Return connected components of H that are isolated edges.
    """
    result: list[Edge] = []
    for sg in connected_components_with_subgraphs(H):
        if sg.number_of_nodes() == 2 and sg.number_of_edges() == 1:
            u, v = next(iter(sg.edges()))
            result.append(normalize_edge(u, v))
    return sorted(result)


def classify_component(H_subgraph: nx.Graph) -> ComponentKind:
    """
    Classify one connected component of H into:
    edge / triangle / star / five_path / other
    """
    n = H_subgraph.number_of_nodes()
    m = H_subgraph.number_of_edges()
    degrees = sorted(dict(H_subgraph.degree()).values())

    if n == 2 and m == 1:
        return "edge"

    if n == 3 and m == 3:
        return "triangle"

    # 5-path has 5 vertices, 4 edges, degree sequence [1,1,2,2,2]
    if n == 5 and m == 4 and degrees == [1, 1, 2, 2, 2]:
        return "five_path"

    # star on n vertices has n-1 edges and degree multiset [1,...,1,n-1]
    if n >= 3 and m == n - 1 and degrees == [1] * (n - 1) + [n - 1]:
        return "star"

    return "other"


def get_component_kind_map(H: nx.Graph) -> list[tuple[list[int], ComponentKind]]:
    out: list[tuple[list[int], ComponentKind]] = []
    for sg in connected_components_with_subgraphs(H):
        out.append((sorted(sg.nodes()), classify_component(sg)))
    return out


def validate_phase1_output(H: nx.Graph, M: set[Edge]) -> tuple[bool, str]:
    """
    Validate the postconditions of Phase 1, corresponding to Lemma 3(2):
    - each component is edge / triangle / star / five_path
    - if five_path: exactly two matching edges, each incident to an endpoint
    - otherwise: exactly one matching edge inside the component
    """
    for sg in connected_components_with_subgraphs(H):
        kind = classify_component(sg)
        nodes = sorted(sg.nodes())

        if kind not in {"edge", "triangle", "star", "five_path"}:
            return False, f"Invalid component type {kind} on nodes {nodes}"

        sg_edges = {normalize_edge(u, v) for u, v in sg.edges()}
        m_edges_inside = sg_edges & M

        if kind == "five_path":
            if len(m_edges_inside) != 2:
                return (
                    False,
                    f"5-path must contain exactly two matching edges, got {m_edges_inside} on nodes {nodes}",
                )

            endpoints = [v for v, d in sg.degree() if d == 1]
            if len(endpoints) != 2:
                return False, f"5-path endpoints invalid on nodes {nodes}"

            for e in m_edges_inside:
                if not (e[0] in endpoints or e[1] in endpoints):
                    return (
                        False,
                        f"Matching edge {e} in 5-path is not incident to an endpoint on nodes {nodes}",
                    )
        else:
            if len(m_edges_inside) != 1:
                return (
                    False,
                    f"{kind} must contain exactly one matching edge, got {m_edges_inside} on nodes {nodes}",
                )

    return True, "ok"
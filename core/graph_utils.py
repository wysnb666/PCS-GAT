from __future__ import annotations

from typing import Iterable, Tuple, List
import networkx as nx

Vertex = int
Edge = Tuple[int, int]


def normalize_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def normalize_edge_set(edges: Iterable[tuple[int, int]]) -> set[Edge]:
    return {normalize_edge(u, v) for u, v in edges}


def get_outside_vertices(G: nx.Graph, H: nx.Graph) -> List[Vertex]:
    """Vertices in G but not in H."""
    return sorted([u for u in G.nodes() if u not in H.nodes()])


def connected_components_with_subgraphs(H: nx.Graph) -> list[nx.Graph]:
    """Return each connected component of H as a copied subgraph."""
    return [H.subgraph(nodes).copy() for nodes in nx.connected_components(H)]


def edge_in_set(u: int, v: int, edges: set[Edge]) -> bool:
    return normalize_edge(u, v) in edges
from __future__ import annotations

import networkx as nx


def build_H_plus_C(H: nx.Graph, C: set[tuple[int, int]]) -> nx.Graph:
    """
    Build H + C = (V(H), E(H) union C).
    """
    HC = nx.Graph()
    HC.add_nodes_from(H.nodes())
    HC.add_edges_from(H.edges())
    HC.add_edges_from(C)
    return HC
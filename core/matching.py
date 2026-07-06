from __future__ import annotations

import networkx as nx

from .graph_utils import Edge, normalize_edge


def compute_maximum_matching(G: nx.Graph) -> set[Edge]:
    """
    Compute a maximum-cardinality matching of G.
    """
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return {normalize_edge(u, v) for u, v in matching}
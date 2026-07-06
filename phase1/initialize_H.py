from __future__ import annotations

from dataclasses import dataclass, field
import networkx as nx

from MPC4plus.core.graph_utils import Edge


@dataclass
class Phase1State:
    G: nx.Graph
    M: set[Edge] = field(default_factory=set)
    H: nx.Graph = field(default_factory=nx.Graph)


def initialize_H_from_matching(G: nx.Graph, M: set[Edge]) -> nx.Graph:
    """
    Initialize H = (V(M), M).
    """
    H = nx.Graph()
    V_M = set()
    for u, v in M:
        V_M.add(u)
        V_M.add(v)
    H.add_nodes_from(V_M)
    H.add_edges_from(M)
    return H
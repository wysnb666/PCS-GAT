from __future__ import annotations

from dataclasses import dataclass
import networkx as nx

from MPC4plus.phase2.bad_components import get_bad_components


Edge = tuple[int, int]


@dataclass
class Step22Instance:
    """
    Formal optimization instance for Step 2.2 on G1.

    We model exactly the objects needed by both backends:
    - edges of G1
    - bad components of H
    - for each bad component, which edges of G1 are incident to it
    """
    G1: nx.Graph
    H: nx.Graph
    bad_component_nodes: list[set[int]]
    edges: list[Edge]
    incident_edge_indices_by_bad_component: list[list[int]]


def build_step22_instance(G1: nx.Graph, H: nx.Graph) -> Step22Instance:
    """
    Build the formal Step 2.2 instance from G1 and H.
    """
    bad_components = get_bad_components(H)
    bad_component_nodes = [set(sg.nodes()) for sg in bad_components]
    edges = [tuple(sorted(e)) for e in G1.edges()]

    incident_edge_indices_by_bad_component: list[list[int]] = []
    for nodes in bad_component_nodes:
        idxs: list[int] = []
        for idx, (u, v) in enumerate(edges):
            if u in nodes or v in nodes:
                idxs.append(idx)
        incident_edge_indices_by_bad_component.append(idxs)

    return Step22Instance(
        G1=G1,
        H=H,
        bad_component_nodes=bad_component_nodes,
        edges=edges,
        incident_edge_indices_by_bad_component=incident_edge_indices_by_bad_component,
    )
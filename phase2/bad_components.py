from __future__ import annotations

import networkx as nx

from MPC4plus.core.components import classify_component
from MPC4plus.core.graph_utils import connected_components_with_subgraphs


def get_bad_components(H: nx.Graph) -> list[nx.Graph]:
    """
    Definition 3:
    A bad component of H is a component of H that is not a 5-path.
    """
    bad = []
    for sg in connected_components_with_subgraphs(H):
        if classify_component(sg) != "five_path":
            bad.append(sg.copy())
    return bad


def get_bad_component_node_sets(H: nx.Graph) -> list[set[int]]:
    return [set(sg.nodes()) for sg in get_bad_components(H)]


def get_component_index_of_vertex(H: nx.Graph) -> dict[int, int]:
    """
    Map each vertex in H to the index of its H-component.
    """
    mapping: dict[int, int] = {}
    comps = connected_components_with_subgraphs(H)
    for idx, sg in enumerate(comps):
        for v in sg.nodes():
            mapping[v] = idx
    return mapping
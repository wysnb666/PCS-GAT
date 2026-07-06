from __future__ import annotations

import networkx as nx

from MPC4plus.phase2.bad_components import get_bad_component_node_sets, get_component_index_of_vertex


def build_auxiliary_graph_G1(G: nx.Graph, H: nx.Graph) -> nx.Graph:
    """
    Step 2.1:
    Construct an auxiliary spanning subgraph G1 of G such that
    E(G1) = {{v1, v2} in E(G) | v1 is in a bad component and
             v2 is not in the same component of H as v1}.

    We implement this as a spanning subgraph on V(G) containing exactly those edges.
    """
    G1 = nx.Graph()
    G1.add_nodes_from(G.nodes())

    bad_components = get_bad_component_node_sets(H)
    bad_vertices = set().union(*bad_components) if bad_components else set()
    comp_index = get_component_index_of_vertex(H)

    for u, v in G.edges():
        add_edge = False

        # direction 1: u is in bad component, v not in same H-component as u
        if u in bad_vertices:
            if (u not in comp_index) or (v not in comp_index) or (comp_index[u] != comp_index[v]):
                add_edge = True

        # direction 2: v is in bad component, u not in same H-component as v
        if v in bad_vertices:
            if (u not in comp_index) or (v not in comp_index) or (comp_index[u] != comp_index[v]):
                add_edge = True

        if add_edge:
            G1.add_edge(u, v)

    return G1
from __future__ import annotations

import networkx as nx

from MPC4plus.core.components import classify_component
from MPC4plus.core.graph_utils import connected_components_with_subgraphs
from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo


def get_H_component_node_sets(H: nx.Graph) -> list[set[int]]:
    return [set(sg.nodes()) for sg in connected_components_with_subgraphs(H)]


def get_HC_components(HC: nx.Graph) -> list[nx.Graph]:
    return connected_components_with_subgraphs(HC)


def component_contains_H_components(K_nodes: set[int], H_component_sets: list[set[int]]) -> list[set[int]]:
    """
    Return all H-components fully contained in K.
    """
    return [comp for comp in H_component_sets if comp.issubset(K_nodes)]


def is_composite_component(K_nodes: set[int], H_component_sets: list[set[int]]) -> bool:
    """
    Definition 5:
    K is composite if it contains at least two connected components of H.
    """
    contained = component_contains_H_components(K_nodes, H_component_sets)
    return len(contained) >= 2


def build_component_meta_for_K(
    K: nx.Graph,
    H: nx.Graph,
) -> ComponentMeta:
    """
    Research-reproduction version:
    Build metadata for one component K of H + C.

    We compress H-components contained in K, and infer center/satellites
    from the induced structure among those H-components:
      - if only one H-component is contained: non-composite
      - if compressed graph is an edge: choose either endpoint that is a 5-path / edge / star as center if possible
      - if compressed graph is a star: center is the star center
    """
    K_nodes = set(K.nodes())
    H_component_sets = get_H_component_node_sets(H)
    contained_H_components = component_contains_H_components(K_nodes, H_component_sets)

    meta = ComponentMeta(
        nodes=K_nodes,
        is_composite=(len(contained_H_components) >= 2),
        H_component_node_sets=contained_H_components,
    )

    if len(contained_H_components) == 1:
        only = contained_H_components[0]
        sg = H.subgraph(only).copy()
        meta.center_nodes = set(only)
        meta.center_kind = classify_component(sg)
        meta.satellites = []
        return meta

    # Build compressed graph of H-components within K
    comp_index = {}
    for i, comp_nodes in enumerate(contained_H_components):
        for v in comp_nodes:
            comp_index[v] = i

    compressed = nx.Graph()
    compressed.add_nodes_from(range(len(contained_H_components)))

    for u, v in K.edges():
        iu = comp_index.get(u, None)
        iv = comp_index.get(v, None)
        if iu is None or iv is None or iu == iv:
            continue
        compressed.add_edge(iu, iv)

    if compressed.number_of_nodes() == 0:
        # fallback, though this should not happen if len(contained_H_components) >= 2
        meta.center_nodes = set(contained_H_components[0])
        meta.center_kind = classify_component(H.subgraph(contained_H_components[0]).copy())
        return meta

    # Try to determine center
    degrees = dict(compressed.degree())

    if compressed.number_of_nodes() == 2 and compressed.number_of_edges() == 1:
        # edge case in compressed graph
        # Prefer a center kind in {five_path, edge, star}, never triangle if avoidable
        candidates = []
        for idx, comp_nodes in enumerate(contained_H_components):
            kind = classify_component(H.subgraph(comp_nodes).copy())
            score = 0 if kind in {"five_path", "edge", "star"} else 1
            candidates.append((score, idx, kind))
        candidates.sort()
        center_idx = candidates[0][1]
    else:
        # star case: choose max-degree node as center
        center_idx = max(degrees, key=lambda x: degrees[x])

    center_nodes = set(contained_H_components[center_idx])
    center_kind = classify_component(H.subgraph(center_nodes).copy())

    satellites = []
    for idx, comp_nodes in enumerate(contained_H_components):
        if idx == center_idx:
            continue
        comp_nodes_set = set(comp_nodes)
        kind = classify_component(H.subgraph(comp_nodes_set).copy())
        satellites.append(SatelliteInfo(nodes=comp_nodes_set, kind=kind))

    meta.center_nodes = center_nodes
    meta.center_kind = center_kind
    meta.satellites = satellites
    return meta


def build_all_component_meta(HC: nx.Graph, H: nx.Graph) -> list[ComponentMeta]:
    metas = []
    for K in get_HC_components(HC):
        metas.append(build_component_meta_for_K(K, H))
    return metas
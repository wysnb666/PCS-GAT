from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import networkx as nx

from MPC4plus.core.graph_utils import Edge, Vertex, get_outside_vertices
from MPC4plus.core.components import get_edge_components


@dataclass(frozen=True)
class AugmentingTriple:
    u0: Vertex
    e0: Edge
    e1: Edge
    case: Literal["C1", "C2"]
    v0: Vertex
    w0: Vertex
    v1: Vertex
    w1: Vertex


def find_augmenting_triple_C1(
    G: nx.Graph, H: nx.Graph, M: set[Edge]
) -> Optional[AugmentingTriple]:
    """
    Condition C1:
        {u0, v0}, {u0, v1} in E(G),
    where e0={v0,w0}, e1={v1,w1} are two distinct edge components of H.
    """
    edge_components = get_edge_components(H)
    if len(edge_components) < 2:
        return None

    vertex_to_edge_component: dict[int, Edge] = {}
    for e in edge_components:
        a, b = e
        vertex_to_edge_component[a] = e
        vertex_to_edge_component[b] = e

    for u0 in get_outside_vertices(G, H):
        touched: list[tuple[int, Edge]] = []
        for nbr in sorted(G.neighbors(u0)):
            if nbr in vertex_to_edge_component:
                touched.append((nbr, vertex_to_edge_component[nbr]))

        for i in range(len(touched)):
            v0, e0 = touched[i]
            w0 = e0[0] if e0[1] == v0 else e0[1]

            for j in range(i + 1, len(touched)):
                v1, e1 = touched[j]
                if e0 == e1:
                    continue
                w1 = e1[0] if e1[1] == v1 else e1[1]
                return AugmentingTriple(
                    u0=u0, e0=e0, e1=e1, case="C1",
                    v0=v0, w0=w0, v1=v1, w1=w1
                )
    return None


def find_augmenting_triple_C2(
    G: nx.Graph, H: nx.Graph, M: set[Edge]
) -> Optional[AugmentingTriple]:
    """
    Condition C2:
        {u0, v0}, {w0, v1} in E(G),
    where e0={v0,w0}, e1={v1,w1} are two distinct edge components of H.
    """
    edge_components = get_edge_components(H)
    outside = set(get_outside_vertices(G, H))
    if len(edge_components) < 2:
        return None

    for e0 in edge_components:
        a, b = e0
        for v0, w0 in [(a, b), (b, a)]:
            outside_neighbors = [u for u in sorted(G.neighbors(v0)) if u in outside]
            if not outside_neighbors:
                continue

            for e1 in edge_components:
                if e1 == e0:
                    continue

                x, y = e1
                for v1, w1 in [(x, y), (y, x)]:
                    if G.has_edge(w0, v1):
                        u0 = outside_neighbors[0]
                        return AugmentingTriple(
                            u0=u0, e0=e0, e1=e1, case="C2",
                            v0=v0, w0=w0, v1=v1, w1=w1
                        )
    return None


def find_augmenting_triple(
    G: nx.Graph, H: nx.Graph, M: set[Edge]
) -> Optional[AugmentingTriple]:
    """
    Prefer Condition C1 if both C1 and C2 are possible.
    """
    t = find_augmenting_triple_C1(G, H, M)
    if t is not None:
        return t
    return find_augmenting_triple_C2(G, H, M)
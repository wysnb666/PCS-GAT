from __future__ import annotations

import itertools
import networkx as nx


def is_path_component(comp: nx.Graph) -> bool:
    n = comp.number_of_nodes()
    m = comp.number_of_edges()
    if n < 1:
        return False
    if n == 1:
        return True
    if m != n - 1:
        return False

    degs = sorted(dict(comp.degree()).values())
    return degs == [1, 1] + [2] * (n - 2)


def graph_to_path_nodes(comp: nx.Graph) -> list[int]:
    if comp.number_of_nodes() == 0:
        return []
    if comp.number_of_nodes() == 1:
        return list(comp.nodes())

    endpoints = [v for v, d in comp.degree() if d == 1]
    if len(endpoints) != 2:
        return []
    return nx.shortest_path(comp, endpoints[0], endpoints[1])


def is_feasible_solution_graph(S: nx.Graph) -> bool:
    """
    Feasible solution for the current project:
    every connected component must be a path with at least 4 vertices.
    """
    for nodes in nx.connected_components(S):
        comp = S.subgraph(nodes).copy()
        if not is_path_component(comp):
            return False
        if comp.number_of_nodes() < 4:
            return False
    return True


def is_single_path_graph(S: nx.Graph) -> bool:
    if S.number_of_nodes() == 0:
        return False
    if not nx.is_connected(S):
        return False
    return is_path_component(S)


def brute_force_best_feasible_solution(
    G: nx.Graph,
) -> nx.Graph:
    """
    Exact brute-force solver for the maximum covered-vertex feasible solution on G.
    """
    nodes = list(G.nodes())
    edges = list(G.edges())

    best = nx.Graph()
    best.add_nodes_from(nodes)
    best_covered = 0

    for r in range(len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            S = nx.Graph()
            S.add_nodes_from(nodes)
            S.add_edges_from(subset)

            nonisolated = [v for v, d in S.degree() if d > 0]
            T = S.subgraph(nonisolated).copy()

            if T.number_of_nodes() == 0:
                continue

            if is_feasible_solution_graph(T):
                covered = T.number_of_nodes()
                if covered > best_covered:
                    best = T.copy()
                    best_covered = covered

    return best


def brute_force_best_single_path_containing(
    G: nx.Graph,
    required_vertices: set[int],
    min_vertices: int = 5,
) -> nx.Graph:
    """
    Exact brute-force search for the best single path in G that:
    - is connected,
    - is a path,
    - contains all required_vertices,
    - has at least min_vertices vertices.
    """
    nodes = list(G.nodes())
    edges = list(G.edges())

    best = nx.Graph()
    best_covered = 0

    for r in range(len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            S = nx.Graph()
            S.add_nodes_from(nodes)
            S.add_edges_from(subset)

            nonisolated = [v for v, d in S.degree() if d > 0]
            T = S.subgraph(nonisolated).copy()

            if T.number_of_nodes() < min_vertices:
                continue
            if not required_vertices.issubset(set(T.nodes())):
                continue
            if not is_single_path_graph(T):
                continue

            covered = T.number_of_nodes()
            if covered > best_covered:
                best = T.copy()
                best_covered = covered

    return best


def union_solution_graphs(graphs: list[nx.Graph]) -> nx.Graph:
    U = nx.Graph()
    for g in graphs:
        U.add_nodes_from(g.nodes())
        U.add_edges_from(g.edges())
    return U
from __future__ import annotations

from itertools import combinations
import networkx as nx

from MPC4plus.phase2.bad_components import get_bad_components
from MPC4plus.phase2.step22_instance import Step22Instance, build_step22_instance
from MPC4plus.phase2.step22_ilp import solve_step22_instance_ilp


Edge = tuple[int, int]


def is_path_cycle_cover_subgraph(G1: nx.Graph, edges: set[Edge]) -> bool:
    """
    Check whether the spanning subgraph (V(G1), edges) is a path-cycle cover.

    In a simple undirected graph, degree <= 2 for every vertex implies that each
    connected component is a path, a cycle, or an isolated vertex.
    """
    Hc = nx.Graph()
    Hc.add_nodes_from(G1.nodes())
    Hc.add_edges_from(edges)

    for e in edges:
        if not G1.has_edge(*e):
            return False

    for _, d in Hc.degree():
        if d > 2:
            return False

    return True


def saturated_bad_component_count(H: nx.Graph, F: set[Edge]) -> int:
    """
    Definition 4 in paper terms:
    a bad component K is saturated by F if some edge in F is incident to a vertex of K.
    """
    bad_components = get_bad_components(H)
    count = 0
    for sg in bad_components:
        nodes = set(sg.nodes())
        sat = False
        for u, v in F:
            if u in nodes or v in nodes:
                sat = True
                break
        if sat:
            count += 1
    return count


def compute_step22_weight_from_instance(instance: Step22Instance, F: set[Edge]) -> int:
    """
    Compute Step 2.2 weight directly from the prebuilt instance.
    """
    edge_set = {tuple(sorted(e)) for e in F}
    count = 0
    for edge_idxs in instance.incident_edge_indices_by_bad_component:
        sat = False
        for idx in edge_idxs:
            if instance.edges[idx] in edge_set:
                sat = True
                break
        if sat:
            count += 1
    return count


def brute_force_max_weight_path_cycle_cover(G1: nx.Graph, H: nx.Graph) -> set[Edge]:
    """
    Small-instance exact validator backend.

    Enumerate all subsets of E(G1), keep those that form a path-cycle cover,
    and maximize the number of saturated bad components.
    """
    instance = build_step22_instance(G1, H)
    edges = instance.edges

    best_cover: set[Edge] = set()
    best_weight = -1
    best_size = -1  # tie-breaker: prefer more edges before Step 2.3 pruning

    for r in range(len(edges) + 1):
        for subset in combinations(edges, r):
            F = set(subset)
            if not is_path_cycle_cover_subgraph(G1, F):
                continue

            w = compute_step22_weight_from_instance(instance, F)
            if w > best_weight:
                best_weight = w
                best_cover = F
                best_size = len(F)
            elif w == best_weight and len(F) > best_size:
                best_cover = F
                best_size = len(F)

    return best_cover


def ilp_max_weight_path_cycle_cover(G1: nx.Graph, H: nx.Graph) -> set[Edge]:
    """
    Formal Step 2.2 backend using ILP/MIP.

    Keeps the same optimization target as the paper:
    maximize the number of saturated bad components,
    under path-cycle-cover degree constraints on G1.
    """
    instance = build_step22_instance(G1, H)
    C = solve_step22_instance_ilp(instance)

    if not is_path_cycle_cover_subgraph(G1, C):
        raise RuntimeError("ILP backend returned an invalid path-cycle cover.")

    return C


def prune_redundant_edges_same_weight(H: nx.Graph, C: set[Edge]) -> set[Edge]:
    """
    Step 2.3:
    repeatedly remove an edge e from C if removing it does not decrease the Step 2.2 weight.
    """
    changed = True
    C = set(C)

    while changed:
        changed = False
        current_weight = saturated_bad_component_count(H, C)

        for e in list(C):
            trial = set(C)
            trial.remove(e)
            if saturated_bad_component_count(H, trial) == current_weight:
                C = trial
                changed = True
                break

    return C


def compute_max_weight_path_cycle_cover(
    G1: nx.Graph,
    H: nx.Graph,
    backend: str = "ilp",
) -> set[Edge]:
    """
    Full Step 2.2 + Step 2.3.

    Parameters
    ----------
    backend:
        - "ilp": formal ILP/MIP backend (recommended default)
        - "bruteforce": exact small-instance backend retained for validation
    """
    if backend == "ilp":
        C = ilp_max_weight_path_cycle_cover(G1, H)
    elif backend == "bruteforce":
        C = brute_force_max_weight_path_cycle_cover(G1, H)
    else:
        raise ValueError(f"Unknown Step 2.2 backend: {backend}")

    C = prune_redundant_edges_same_weight(H, C)
    return C
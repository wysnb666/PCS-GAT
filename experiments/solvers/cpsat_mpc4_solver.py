from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import networkx as nx
from ortools.sat.python import cp_model


PathTuple = tuple[int, ...]


@dataclass(frozen=True)
class CPSATMPC4Result:
    graph_id: str
    num_nodes: int
    num_edges: int
    candidate_path_count: int
    solver_status: str
    is_optimal: bool
    objective_value: int
    best_objective_bound: int | None
    covered_vertices: int
    coverage_ratio: float
    solve_seconds: float
    selected_paths: list[list[int]]
    selected_path_lengths: list[int]


def canonicalize_path(path: Iterable[int]) -> PathTuple:
    """
    Canonicalize a simple path by removing direction ambiguity.
    """
    p = tuple(int(v) for v in path)
    rp = tuple(reversed(p))
    return p if p <= rp else rp


def path_to_edge_list(path: PathTuple) -> list[tuple[int, int]]:
    return [(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]


def enumerate_candidate_paths_order_4_to_7(G: nx.Graph) -> list[PathTuple]:
    """
    Enumerate all distinct simple paths in G with 4 to 7 vertices inclusive.

    Since the MPC4+ paper states that, w.l.o.g., each path in a feasible solution
    can be assumed to have order between 4 and 7, these candidates are sufficient
    for exact modeling on small graphs.
    """
    nodes = sorted(int(v) for v in G.nodes())
    seen: set[PathTuple] = set()

    for i, s in enumerate(nodes):
        for t in nodes[i + 1:]:
            for path in nx.all_simple_paths(G, source=s, target=t, cutoff=6):
                if 4 <= len(path) <= 7:
                    seen.add(canonicalize_path(path))

    return sorted(seen, key=lambda p: (len(p), p))


def build_solution_graph_from_paths(paths: list[PathTuple]) -> nx.Graph:
    S = nx.Graph()
    for path in paths:
        S.add_nodes_from(path)
        S.add_edges_from(path_to_edge_list(path))
    return S


def _status_to_name(status: int) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBLE:
        return "FEASIBLE"
    if status == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    if status == cp_model.MODEL_INVALID:
        return "MODEL_INVALID"
    if status == cp_model.UNKNOWN:
        return "UNKNOWN"
    return f"STATUS_{status}"


def solve_mpc4_with_cpsat(
    G: nx.Graph,
    *,
    graph_id: str = "graph",
    time_limit_seconds: float = 60.0,
    num_search_workers: int = 8,
) -> CPSATMPC4Result:
    """
    Exact / high-quality CP-SAT solver for MPC4+ on small graphs.

    Modeling idea:
      - enumerate every simple path of order 4..7
      - create one BoolVar per candidate path
      - enforce vertex-disjointness
      - maximize total covered vertices

    This is exact for small graphs as long as candidate enumeration is complete,
    because feasible solutions can be restricted to paths of order 4..7.
    """
    G = G.copy()
    candidates = enumerate_candidate_paths_order_4_to_7(G)

    model = cp_model.CpModel()
    x_vars = [model.NewBoolVar(f"x_{i}") for i in range(len(candidates))]

    vertex_to_candidate_indices: dict[int, list[int]] = {
        int(v): [] for v in G.nodes()
    }
    for idx, path in enumerate(candidates):
        for v in path:
            vertex_to_candidate_indices[int(v)].append(idx)

    for v, idxs in vertex_to_candidate_indices.items():
        if idxs:
            model.Add(sum(x_vars[i] for i in idxs) <= 1)

    if candidates:
        model.Maximize(
            sum(len(candidates[i]) * x_vars[i] for i in range(len(candidates)))
        )
    else:
        model.Maximize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_search_workers)

    t0 = perf_counter()
    status = solver.Solve(model)
    solve_seconds = perf_counter() - t0

    selected_paths: list[PathTuple] = []
    objective_value = 0
    best_objective_bound: int | None = None

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_paths = [
            candidates[i]
            for i in range(len(candidates))
            if solver.Value(x_vars[i]) == 1
        ]
        objective_value = int(round(solver.ObjectiveValue()))
        best_objective_bound = int(round(solver.BestObjectiveBound()))
    else:
        if status == cp_model.INFEASIBLE:
            objective_value = 0
            best_objective_bound = 0
        else:
            objective_value = 0
            best_objective_bound = None

    solution_graph = build_solution_graph_from_paths(selected_paths)
    covered_vertices = solution_graph.number_of_nodes()
    num_nodes = G.number_of_nodes()
    coverage_ratio = 0.0 if num_nodes == 0 else float(covered_vertices) / float(num_nodes)

    return CPSATMPC4Result(
        graph_id=str(graph_id),
        num_nodes=int(G.number_of_nodes()),
        num_edges=int(G.number_of_edges()),
        candidate_path_count=len(candidates),
        solver_status=_status_to_name(status),
        is_optimal=(status == cp_model.OPTIMAL),
        objective_value=int(objective_value),
        best_objective_bound=best_objective_bound,
        covered_vertices=int(covered_vertices),
        coverage_ratio=float(coverage_ratio),
        solve_seconds=float(solve_seconds),
        selected_paths=[list(path) for path in selected_paths],
        selected_path_lengths=[len(path) for path in selected_paths],
    )
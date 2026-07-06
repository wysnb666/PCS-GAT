from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase4.operation_runner import run_phase4_operations
from MPC4plus.phase5.decomposition import (
    Phase5Buckets,
    classify_phase5_buckets,
    is_isolated_five_path,
    is_responsible_component,
)
from MPC4plus.phase5.decision import Phase5Decision, decide_phase5_route
from MPC4plus.phase5.exact_solver import (
    brute_force_best_feasible_solution,
    union_solution_graphs,
)
from MPC4plus.phase5.path_constructor import construct_all_Pv


@dataclass
class MPC4PlusResult:
    input_graph: nx.Graph
    solution_graph: nx.Graph
    phase4_state: object
    phase5_buckets: Phase5Buckets
    phase5_decision: Phase5Decision
    recursion_depth: int


def _compute_phase1_to_phase4(G: nx.Graph):
    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H)
    M_C = build_M_C(phase1.M, phase1.H, C)
    phase4_state = run_phase4_operations(G, phase1.H, M_C, C)
    return phase1, C, M_C, phase4_state


def _exact_finish_on_current_components(
    G: nx.Graph,
    metas,
) -> nx.Graph:
    """
    Step 5(a):
    compute an optimal solution for each composite component or isolated 5-path in H + C,
    then output their union.

    We explicitly include responsible components as a named finishing object.
    """
    solutions: list[nx.Graph] = []

    for meta in metas:
        should_finish = (
            meta.is_composite
            or is_isolated_five_path(meta)
            or is_responsible_component(meta)
        )
        if should_finish:
            sub = G.subgraph(sorted(meta.nodes)).copy()
            sol = brute_force_best_feasible_solution(sub)
            if sol.number_of_nodes() > 0:
                solutions.append(sol)

    return union_solution_graphs(solutions)


def solve_mpc4plus(
    G: nx.Graph,
    recursion_depth: int = 0,
    max_recursion_depth: int = 20,
) -> MPC4PlusResult:
    if G.number_of_nodes() <= 4:
        sol = brute_force_best_feasible_solution(G)
        empty_buckets = Phase5Buckets()
        empty_decision = Phase5Decision(
            recurse_on_Gc=False,
            lhs_value=0.0,
            rhs_value=0.0,
            has_critical=False,
            reason="base case |V(G)| <= 4",
        )
        return MPC4PlusResult(
            input_graph=G.copy(),
            solution_graph=sol,
            phase4_state=None,
            phase5_buckets=empty_buckets,
            phase5_decision=empty_decision,
            recursion_depth=recursion_depth,
        )

    if recursion_depth > max_recursion_depth:
        sol = brute_force_best_feasible_solution(G)
        empty_buckets = Phase5Buckets()
        fallback_decision = Phase5Decision(
            recurse_on_Gc=False,
            lhs_value=0.0,
            rhs_value=0.0,
            has_critical=False,
            reason="max recursion depth reached; exact fallback on current graph",
        )
        return MPC4PlusResult(
            input_graph=G.copy(),
            solution_graph=sol,
            phase4_state=None,
            phase5_buckets=empty_buckets,
            phase5_decision=fallback_decision,
            recursion_depth=recursion_depth,
        )

    phase1, C, M_C, phase4_state = _compute_phase1_to_phase4(G)
    buckets = classify_phase5_buckets(G, phase4_state.metas)
    decision = decide_phase5_route(buckets)

    if decision.recurse_on_Gc and len(buckets.Gc_nodes) < G.number_of_nodes():
        Gc = G.subgraph(sorted(buckets.Gc_nodes)).copy()
        rec = solve_mpc4plus(
            Gc,
            recursion_depth=recursion_depth + 1,
            max_recursion_depth=max_recursion_depth,
        )
        Pv_union = construct_all_Pv(G, phase4_state.metas, buckets.Rc)
        final_solution = union_solution_graphs([rec.solution_graph, Pv_union])
        return MPC4PlusResult(
            input_graph=G.copy(),
            solution_graph=final_solution,
            phase4_state=phase4_state,
            phase5_buckets=buckets,
            phase5_decision=decision,
            recursion_depth=recursion_depth,
        )

    final_solution = _exact_finish_on_current_components(G, phase4_state.metas)
    return MPC4PlusResult(
        input_graph=G.copy(),
        solution_graph=final_solution,
        phase4_state=phase4_state,
        phase5_buckets=buckets,
        phase5_decision=decision,
        recursion_depth=recursion_depth,
    )
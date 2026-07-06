from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Callable

import networkx as nx

from MPC4plus.experiments.baselines.heuristic_runner import run_heuristic_phase4_pipeline
from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.baselines.xgb_runner import run_xgb_phase4_pipeline
from MPC4plus.experiments.evaluation.end_to_end_metrics import (
    EndToEndRunMetrics,
    compute_end_to_end_run_metrics,
)
from MPC4plus.experiments.models.vertex_gnn_runner import run_vertex_gnn_phase4_pipeline
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer

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
from MPC4plus.phase5.phase5_runner import MPC4PlusResult


Phase4SolverFn = Callable[[nx.Graph], Any]


def solve_phase5_from_phase4_result(
    phase4_result: Any,
    *,
    phase4_solver_fn: Phase4SolverFn,
    recursion_depth: int = 0,
    max_recursion_depth: int = 20,
) -> MPC4PlusResult:
    """
    Execute Phase 5 logic starting from an already-computed selector-specific Phase 4 result.

    This mirrors the user's current phase5_runner.py structure, but it preserves
    the chosen selector route by recursively recomputing Phase 4 with the same
    phase4_solver_fn on Gc.
    """
    G = phase4_result.G
    phase4_state = phase4_result.final_state

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
            phase4_state=phase4_state,
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
            phase4_state=phase4_state,
            phase5_buckets=empty_buckets,
            phase5_decision=fallback_decision,
            recursion_depth=recursion_depth,
        )

    buckets = classify_phase5_buckets(G, phase4_state.metas)
    decision = decide_phase5_route(buckets)

    if decision.recurse_on_Gc and len(buckets.Gc_nodes) < G.number_of_nodes():
        Gc = G.subgraph(sorted(buckets.Gc_nodes)).copy()

        rec_phase4 = phase4_solver_fn(Gc)
        rec = solve_phase5_from_phase4_result(
            rec_phase4,
            phase4_solver_fn=phase4_solver_fn,
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


def compare_rule_heuristic_xgb_gnn_end_to_end(
    G: nx.Graph,
    *,
    ranker: XGBPhase4Ranker,
    gnn_model: VertexGNNCandidateScorer,
    graph_id: str = "graph",
    backend: str = "ilp",
    device: str = "cpu",
    max_recursion_depth: int = 20,
    gnn_conservative_margin: float | None = None,
) -> list[EndToEndRunMetrics]:
    """
    End-to-end compare 4 selector routes:
      - canonical rule
      - one-step heuristic
      - XGBoost selector
      - vertex-level GNN selector

    Unlike the earlier callback-based version, this now directly mirrors the
    user's current phase5_runner.py logic while preserving each selector route.
    """
    def rule_phase4_solver(G_sub: nx.Graph):
        return run_rule_phase4_pipeline(
            G_sub,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def heuristic_phase4_solver(G_sub: nx.Graph):
        return run_heuristic_phase4_pipeline(
            G_sub,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def xgb_phase4_solver(G_sub: nx.Graph):
        return run_xgb_phase4_pipeline(
            G_sub,
            ranker=ranker,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def gnn_phase4_solver(G_sub: nx.Graph):
        return run_vertex_gnn_phase4_pipeline(
            G_sub,
            model=gnn_model,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
            device=device,
            conservative_margin=gnn_conservative_margin,
        )

    rule_phase4, rule_phase5, rule_phase4_seconds, rule_phase5_seconds, rule_total_seconds = (
        _run_selector_route_with_timing(
            G,
            phase4_solver_fn=rule_phase4_solver,
            max_recursion_depth=max_recursion_depth,
        )
    )
    heuristic_phase4, heuristic_phase5, heuristic_phase4_seconds, heuristic_phase5_seconds, heuristic_total_seconds = (
        _run_selector_route_with_timing(
            G,
            phase4_solver_fn=heuristic_phase4_solver,
            max_recursion_depth=max_recursion_depth,
        )
    )
    xgb_phase4, xgb_phase5, xgb_phase4_seconds, xgb_phase5_seconds, xgb_total_seconds = (
        _run_selector_route_with_timing(
            G,
            phase4_solver_fn=xgb_phase4_solver,
            max_recursion_depth=max_recursion_depth,
        )
    )
    gnn_phase4, gnn_phase5, gnn_phase4_seconds, gnn_phase5_seconds, gnn_total_seconds = (
        _run_selector_route_with_timing(
            G,
            phase4_solver_fn=gnn_phase4_solver,
            max_recursion_depth=max_recursion_depth,
        )
    )

    return [
        compute_end_to_end_run_metrics(
            phase4_result=rule_phase4,
            phase5_result=rule_phase5,
            selector_name="canonical_rule",
            phase4_solve_seconds=rule_phase4_seconds,
            phase5_solve_seconds=rule_phase5_seconds,
            total_solve_seconds=rule_total_seconds,
        ),
        compute_end_to_end_run_metrics(
            phase4_result=heuristic_phase4,
            phase5_result=heuristic_phase5,
            selector_name="one_step_heuristic",
            phase4_solve_seconds=heuristic_phase4_seconds,
            phase5_solve_seconds=heuristic_phase5_seconds,
            total_solve_seconds=heuristic_total_seconds,
        ),
        compute_end_to_end_run_metrics(
            phase4_result=xgb_phase4,
            phase5_result=xgb_phase5,
            selector_name="xgb_selector",
            phase4_solve_seconds=xgb_phase4_seconds,
            phase5_solve_seconds=xgb_phase5_seconds,
            total_solve_seconds=xgb_total_seconds,
        ),
        compute_end_to_end_run_metrics(
            phase4_result=gnn_phase4,
            phase5_result=gnn_phase5,
            selector_name="vertex_gnn_selector",
            phase4_solve_seconds=gnn_phase4_seconds,
            phase5_solve_seconds=gnn_phase5_seconds,
            total_solve_seconds=gnn_total_seconds,
        ),
    ]


def add_relative_improvement_vs_rule(
    metrics: list[EndToEndRunMetrics],
) -> list[dict[str, Any]]:
    """
    Convert end-to-end metrics into dict rows and add:
      - abs_improvement_vs_rule
      - rel_improvement_vs_rule

    Rule row itself gets 0 / 0.
    """
    rows = [asdict(m) for m in metrics]

    rule_rows = [row for row in rows if row["selector_name"] == "canonical_rule"]
    if len(rule_rows) != 1:
        raise ValueError(
            "Expected exactly one canonical_rule row when computing relative improvements."
        )

    rule_cov = int(rule_rows[0]["covered_vertices"])
    out: list[dict[str, Any]] = []
    for row in rows:
        covered = int(row["covered_vertices"])
        abs_gain = covered - rule_cov
        rel_gain = 0.0 if rule_cov == 0 else float(abs_gain) / float(rule_cov)

        new_row = dict(row)
        new_row["abs_improvement_vs_rule"] = abs_gain
        new_row["rel_improvement_vs_rule"] = rel_gain
        out.append(new_row)

    return out


def _exact_finish_on_current_components(
    G: nx.Graph,
    metas,
) -> nx.Graph:
    """
    Copy of the current phase5_runner.py finishing logic:
    solve each composite component / isolated 5-path / responsible component
    exactly, then union the solutions.
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


def _run_selector_route_with_timing(
    G: nx.Graph,
    *,
    phase4_solver_fn: Phase4SolverFn,
    max_recursion_depth: int,
) -> tuple[Any, MPC4PlusResult, float, float, float]:
    """
    Run one selector route and measure pure solve time after models are already loaded.

    The recorded times intentionally include:
      - Phase 1/2/4 execution inside phase4_solver_fn
      - Phase 5 execution (including recursive calls that preserve the selector route)

    The recorded times intentionally exclude:
      - model training
      - one-time model checkpoint loading / ranker loading
    """
    phase4_start = perf_counter()
    phase4_result = phase4_solver_fn(G)
    phase4_seconds = perf_counter() - phase4_start

    phase5_start = perf_counter()
    phase5_result = solve_phase5_from_phase4_result(
        phase4_result,
        phase4_solver_fn=phase4_solver_fn,
        recursion_depth=0,
        max_recursion_depth=max_recursion_depth,
    )
    phase5_seconds = perf_counter() - phase5_start

    total_seconds = phase4_seconds + phase5_seconds
    return phase4_result, phase5_result, phase4_seconds, phase5_seconds, total_seconds
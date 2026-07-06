from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase4.operation_runner import run_phase4_operations
from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.baselines.xgb_selector import XGBPhase4Selector
from MPC4plus.experiments.export.phase4_trace_exporter import Phase4TraceCollector


Edge = tuple[int, int]


@dataclass
class XGBPhase4RunResult:
    graph_id: str
    G: nx.Graph
    M: set[Edge]
    H: nx.Graph
    G1: nx.Graph
    C_before_phase4: set[Edge]
    C_after_phase4: set[Edge]
    M_C: set[Edge]
    final_state: Any
    trace_collector: Phase4TraceCollector | None


def run_xgb_phase4_pipeline(
    G: nx.Graph,
    *,
    ranker: XGBPhase4Ranker,
    graph_id: str = "graph",
    backend: str = "ilp",
    enable_trace: bool = True,
) -> XGBPhase4RunResult:
    """
    Run the baseline pipeline up to Phase 4 using a trained XGBoost selector.

    Pipeline:
        Phase 1 -> Phase 2 (build G1, compute C, build M_C) -> Phase 4

    This mirrors the rule/heuristic runners, but replaces the selector with
    an XGB-backed learned selector.
    """
    # Fail fast: this runner is meant for a trained XGBoost model.
    ranker._check_is_fitted()

    phase1_state = run_phase1(G)
    M = set(phase1_state.M)
    H = phase1_state.H

    G1 = build_auxiliary_graph_G1(G, H)
    C_before_phase4 = set(
        compute_max_weight_path_cycle_cover(G1, H, backend=backend)
    )
    M_C = set(build_M_C(M, H, C_before_phase4))

    trace_collector = None
    if enable_trace:
        trace_collector = Phase4TraceCollector()
        trace_collector.start_graph(graph_id)

    selector = XGBPhase4Selector(ranker)

    final_state = run_phase4_operations(
        G=G,
        H=H,
        M_C=M_C,
        C=C_before_phase4,
        selector=selector,
        trace_collector=trace_collector,
    )

    return XGBPhase4RunResult(
        graph_id=graph_id,
        G=G,
        M=M,
        H=H,
        G1=G1,
        C_before_phase4=C_before_phase4,
        C_after_phase4=set(final_state.C),
        M_C=M_C,
        final_state=final_state,
        trace_collector=trace_collector,
    )
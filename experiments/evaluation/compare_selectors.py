from __future__ import annotations

from dataclasses import asdict
from typing import Any

import networkx as nx

from MPC4plus.experiments.baselines.heuristic_runner import run_heuristic_phase4_pipeline
from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.baselines.xgb_runner import run_xgb_phase4_pipeline
from MPC4plus.experiments.evaluation.metrics import (
    Phase4RunMetrics,
    compute_phase4_run_metrics,
)
from MPC4plus.experiments.models.vertex_gnn_runner import (
    run_vertex_gnn_phase4_pipeline,
)
from MPC4plus.experiments.models.vertex_gnn_selector import (
    VertexGNNCandidateScorer,
)


def compare_rule_and_heuristic(
    G: nx.Graph,
    *,
    graph_id: str = "graph",
    backend: str = "ilp",
) -> list[Phase4RunMetrics]:
    """
    Run and compare:
      - canonical rule selector
      - one-step heuristic selector
    """
    rule_result = run_rule_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    heuristic_result = run_heuristic_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )

    return [
        compute_phase4_run_metrics(
            result=rule_result,
            selector_name="canonical_rule",
        ),
        compute_phase4_run_metrics(
            result=heuristic_result,
            selector_name="one_step_heuristic",
        ),
    ]


def compare_rule_heuristic_xgb(
    G: nx.Graph,
    *,
    ranker: XGBPhase4Ranker,
    graph_id: str = "graph",
    backend: str = "ilp",
) -> list[Phase4RunMetrics]:
    """
    Run and compare:
      - canonical rule selector
      - one-step heuristic selector
      - XGBoost selector
    """
    rule_result = run_rule_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    heuristic_result = run_heuristic_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    xgb_result = run_xgb_phase4_pipeline(
        G,
        ranker=ranker,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )

    return [
        compute_phase4_run_metrics(
            result=rule_result,
            selector_name="canonical_rule",
        ),
        compute_phase4_run_metrics(
            result=heuristic_result,
            selector_name="one_step_heuristic",
        ),
        compute_phase4_run_metrics(
            result=xgb_result,
            selector_name="xgb_selector",
        ),
    ]


def compare_rule_heuristic_xgb_gnn(
    G: nx.Graph,
    *,
    ranker: XGBPhase4Ranker,
    gnn_model: VertexGNNCandidateScorer,
    graph_id: str = "graph",
    backend: str = "ilp",
    device: str = "cpu",
) -> list[Phase4RunMetrics]:
    """
    Run and compare:
      - canonical rule selector
      - one-step heuristic selector
      - XGBoost selector
      - vertex-level GNN selector
    """
    rule_result = run_rule_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    heuristic_result = run_heuristic_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    xgb_result = run_xgb_phase4_pipeline(
        G,
        ranker=ranker,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )
    gnn_result = run_vertex_gnn_phase4_pipeline(
        G,
        model=gnn_model,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
        device=device,
    )

    return [
        compute_phase4_run_metrics(
            result=rule_result,
            selector_name="canonical_rule",
        ),
        compute_phase4_run_metrics(
            result=heuristic_result,
            selector_name="one_step_heuristic",
        ),
        compute_phase4_run_metrics(
            result=xgb_result,
            selector_name="xgb_selector",
        ),
        compute_phase4_run_metrics(
            result=gnn_result,
            selector_name="vertex_gnn_selector",
        ),
    ]


def metrics_to_dicts(metrics: list[Phase4RunMetrics]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]
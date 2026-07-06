from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

import networkx as nx

from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.experiments.evaluation.end_to_end_compare import (
    add_relative_improvement_vs_rule,
    compare_rule_heuristic_xgb_gnn_end_to_end,
    solve_phase5_from_phase4_result,
)
from MPC4plus.experiments.evaluation.end_to_end_metrics import (
    compute_end_to_end_run_metrics,
    extract_covered_vertices,
)
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer


def _demo_training_rows():
    return [
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0000",
            "iteration": 0,
            "selector_name": "canonical_rule",
            "candidate_idx": 0,
            "op_type": "op1",
            "feature__expected_g_drop": 2.0,
            "feature__toy_feature": 9.0,
            "label__selected_by_rule": 1,
            "label__selected_in_run": 1,
        },
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0000",
            "iteration": 0,
            "selector_name": "canonical_rule",
            "candidate_idx": 1,
            "op_type": "op3",
            "feature__expected_g_drop": 0.0,
            "feature__toy_feature": 1.0,
            "label__selected_by_rule": 0,
            "label__selected_in_run": 0,
        },
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0001",
            "iteration": 1,
            "selector_name": "canonical_rule",
            "candidate_idx": 0,
            "op_type": "op2",
            "feature__expected_g_drop": 3.0,
            "feature__toy_feature": 8.0,
            "label__selected_by_rule": 1,
            "label__selected_in_run": 1,
        },
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0001",
            "iteration": 1,
            "selector_name": "canonical_rule",
            "candidate_idx": 1,
            "op_type": "op1",
            "feature__expected_g_drop": 0.0,
            "feature__toy_feature": 0.5,
            "label__selected_by_rule": 0,
            "label__selected_in_run": 0,
        },
    ]


def _build_demo_ranker() -> XGBPhase4Ranker:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(
        label_name="label__selected_by_rule",
        n_estimators=20,
        max_depth=2,
    )
    ranker.fit(ds)
    return ranker


def _build_demo_gnn_model() -> VertexGNNCandidateScorer:
    return VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )


def _build_demo_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (1, 6),
            (2, 5),
            (3, 8),
            (4, 7),
            (2, 9),
            (9, 10),
            (4, 11),
            (11, 12),
            (6, 13),
            (8, 14),
        ]
    )
    return G


def test_extract_covered_vertices_from_solution_graph() -> None:
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    phase5_result = {"solution_graph": G}

    covered = extract_covered_vertices(phase5_result)
    assert covered == {1, 2, 3}


def test_solve_phase5_from_phase4_result_runs_on_rule_phase4() -> None:
    G = _build_demo_graph()

    phase4_result = run_rule_phase4_pipeline(
        G,
        graph_id="e2e_graph",
        backend="ilp",
        enable_trace=True,
    )

    def rule_phase4_solver(G_sub: nx.Graph):
        return run_rule_phase4_pipeline(
            G_sub,
            graph_id="e2e_graph",
            backend="ilp",
            enable_trace=True,
        )

    phase5_result = solve_phase5_from_phase4_result(
        phase4_result,
        phase4_solver_fn=rule_phase4_solver,
        recursion_depth=0,
        max_recursion_depth=20,
    )

    covered = extract_covered_vertices(phase5_result)
    assert isinstance(covered, set)
    assert len(covered) >= 0

    metrics = compute_end_to_end_run_metrics(
        phase4_result=phase4_result,
        phase5_result=phase5_result,
        selector_name="canonical_rule",
    )
    assert metrics.graph_id == "e2e_graph"
    assert 0.0 <= metrics.coverage_ratio <= 1.0


def test_compare_rule_heuristic_xgb_gnn_end_to_end_runs() -> None:
    G = _build_demo_graph()
    ranker = _build_demo_ranker()
    gnn_model = _build_demo_gnn_model()

    metrics = compare_rule_heuristic_xgb_gnn_end_to_end(
        G,
        ranker=ranker,
        gnn_model=gnn_model,
        graph_id="e2e_cmp_graph",
        backend="ilp",
        device="cpu",
        max_recursion_depth=20,
    )

    assert len(metrics) == 4
    names = {m.selector_name for m in metrics}
    assert names == {
        "canonical_rule",
        "one_step_heuristic",
        "xgb_selector",
        "vertex_gnn_selector",
    }

    for m in metrics:
        assert m.graph_id == "e2e_cmp_graph"
        assert m.covered_vertices >= 0
        assert 0.0 <= m.coverage_ratio <= 1.0


def test_add_relative_improvement_vs_rule() -> None:
    G = _build_demo_graph()
    ranker = _build_demo_ranker()
    gnn_model = _build_demo_gnn_model()

    metrics = compare_rule_heuristic_xgb_gnn_end_to_end(
        G,
        ranker=ranker,
        gnn_model=gnn_model,
        graph_id="e2e_cmp_graph_gain",
        backend="ilp",
        device="cpu",
        max_recursion_depth=20,
    )

    rows = add_relative_improvement_vs_rule(metrics)
    assert len(rows) == 4
    assert all("abs_improvement_vs_rule" in row for row in rows)
    assert all("rel_improvement_vs_rule" in row for row in rows)
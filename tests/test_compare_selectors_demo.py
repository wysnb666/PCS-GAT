from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

import networkx as nx

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.experiments.evaluation.compare_selectors import (
    compare_rule_and_heuristic,
    compare_rule_heuristic_xgb,
    compare_rule_heuristic_xgb_gnn,
    metrics_to_dicts,
)
from MPC4plus.experiments.models.vertex_gnn_selector import (
    VertexGNNCandidateScorer,
)


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


def test_compare_rule_and_heuristic_runs() -> None:
    G = _build_demo_graph()
    metrics = compare_rule_and_heuristic(G, graph_id="cmp_graph", backend="ilp")

    assert len(metrics) == 2
    names = {m.selector_name for m in metrics}
    assert names == {"canonical_rule", "one_step_heuristic"}

    for m in metrics:
        assert m.graph_id == "cmp_graph"
        assert m.num_nodes == G.number_of_nodes()
        assert m.num_edges == G.number_of_edges()
        assert m.num_phase4_steps >= 0
        assert m.num_trace_samples >= 0


def test_compare_rule_heuristic_xgb_runs() -> None:
    ranker = _build_demo_ranker()
    G = _build_demo_graph()

    metrics = compare_rule_heuristic_xgb(
        G,
        ranker=ranker,
        graph_id="cmp_graph_xgb",
        backend="ilp",
    )

    assert len(metrics) == 3
    names = {m.selector_name for m in metrics}
    assert names == {"canonical_rule", "one_step_heuristic", "xgb_selector"}

    for m in metrics:
        assert m.graph_id == "cmp_graph_xgb"
        assert m.num_nodes == G.number_of_nodes()
        assert m.num_edges == G.number_of_edges()


def test_compare_rule_heuristic_xgb_gnn_runs() -> None:
    ranker = _build_demo_ranker()
    gnn_model = _build_demo_gnn_model()
    G = _build_demo_graph()

    metrics = compare_rule_heuristic_xgb_gnn(
        G,
        ranker=ranker,
        gnn_model=gnn_model,
        graph_id="cmp_graph_full",
        backend="ilp",
        device="cpu",
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
        assert m.graph_id == "cmp_graph_full"
        assert m.num_nodes == G.number_of_nodes()
        assert m.num_edges == G.number_of_edges()
        assert m.num_phase4_steps >= 0
        assert m.num_trace_samples >= 0


def test_metrics_to_dicts() -> None:
    G = _build_demo_graph()
    metrics = compare_rule_and_heuristic(G, graph_id="cmp_graph_dict", backend="ilp")
    rows = metrics_to_dicts(metrics)

    assert len(rows) == 2
    assert all(isinstance(row, dict) for row in rows)
    assert all("graph_id" in row for row in rows)
    assert all("selector_name" in row for row in rows)
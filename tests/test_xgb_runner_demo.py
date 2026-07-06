from __future__ import annotations

import pytest


import networkx as nx

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.baselines.xgb_runner import run_xgb_phase4_pipeline
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

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


def test_run_xgb_phase4_pipeline_runs_with_trained_ranker() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(
        label_name="label__selected_by_rule",
        n_estimators=20,
        max_depth=2,
    )
    ranker.fit(ds)

    G = _build_demo_graph()
    result = run_xgb_phase4_pipeline(
        G,
        ranker=ranker,
        graph_id="xgb_graph",
        backend="ilp",
        enable_trace=True,
    )

    assert result.graph_id == "xgb_graph"
    assert isinstance(result.C_before_phase4, set)
    assert isinstance(result.C_after_phase4, set)
    assert isinstance(result.M_C, set)
    assert result.trace_collector is not None
    assert isinstance(result.trace_collector.samples(), list)


def test_run_xgb_phase4_pipeline_runs_without_trace() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(
        label_name="label__selected_by_rule",
        n_estimators=20,
        max_depth=2,
    )
    ranker.fit(ds)

    G = _build_demo_graph()
    result = run_xgb_phase4_pipeline(
        G,
        ranker=ranker,
        graph_id="xgb_graph_no_trace",
        backend="ilp",
        enable_trace=False,
    )

    assert result.graph_id == "xgb_graph_no_trace"
    assert result.trace_collector is None
    assert isinstance(result.final_state.C, set)


def test_run_xgb_phase4_pipeline_requires_fitted_ranker() -> None:
    ranker = XGBPhase4Ranker(
        label_name="label__selected_by_rule",
        n_estimators=20,
        max_depth=2,
    )
    G = _build_demo_graph()

    with pytest.raises(RuntimeError):
        run_xgb_phase4_pipeline(
            G,
            ranker=ranker,
            graph_id="xgb_unfitted",
            backend="ilp",
            enable_trace=False,
        )
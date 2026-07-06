from __future__ import annotations

import pytest

pytest.importorskip("torch")

import networkx as nx

from MPC4plus.experiments.models.vertex_gnn_runner import (
    run_vertex_gnn_phase4_pipeline,
)
from MPC4plus.experiments.models.vertex_gnn_selector import (
    VertexGNNCandidateScorer,
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


def test_run_vertex_gnn_phase4_pipeline_runs_with_trace() -> None:
    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop"],
        hidden_dim=16,
        num_layers=2,
    )

    G = _build_demo_graph()
    result = run_vertex_gnn_phase4_pipeline(
        G,
        model=model,
        graph_id="gnn_graph",
        backend="ilp",
        enable_trace=True,
        device="cpu",
    )

    assert result.graph_id == "gnn_graph"
    assert isinstance(result.C_before_phase4, set)
    assert isinstance(result.C_after_phase4, set)
    assert isinstance(result.M_C, set)
    assert result.trace_collector is not None
    assert isinstance(result.trace_collector.samples(), list)


def test_run_vertex_gnn_phase4_pipeline_runs_without_trace() -> None:
    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop"],
        hidden_dim=16,
        num_layers=2,
    )

    G = _build_demo_graph()
    result = run_vertex_gnn_phase4_pipeline(
        G,
        model=model,
        graph_id="gnn_graph_no_trace",
        backend="ilp",
        enable_trace=False,
        device="cpu",
    )

    assert result.graph_id == "gnn_graph_no_trace"
    assert result.trace_collector is None
    assert isinstance(result.final_state.C, set)


def test_run_vertex_gnn_phase4_pipeline_accepts_multiple_candidate_features() -> None:
    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )

    G = _build_demo_graph()
    result = run_vertex_gnn_phase4_pipeline(
        G,
        model=model,
        graph_id="gnn_graph_multi_feat",
        backend="ilp",
        enable_trace=True,
        device="cpu",
    )

    assert result.graph_id == "gnn_graph_multi_feat"
    assert isinstance(result.final_state.C, set)
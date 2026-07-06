from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from MPC4plus.experiments.datasets.vertex_gnn_dataset import (
    VertexGNNTrainingDataset,
    build_vertex_gnn_example_from_live_inputs,
)
from MPC4plus.experiments.models.vertex_gnn_selector import (
    build_live_phase4_graph_inputs,
)
from MPC4plus.phase4.operations import OperationCandidate


def _fake_satellite(
    nodes,
    *,
    kind="satellite",
    rescue_edges=(),
    supporting_anchors=(),
    rescue_anchor=None,
    is_critical_satellite=False,
):
    return SimpleNamespace(
        nodes=set(nodes),
        kind=kind,
        rescue_edges=tuple(rescue_edges),
        supporting_anchors=set(supporting_anchors),
        rescue_anchor=rescue_anchor,
        is_critical_satellite=is_critical_satellite,
    )


def _fake_meta(
    nodes,
    *,
    center_nodes=(),
    center_kind="edge",
    anchors=(),
    critical_2_anchors=(),
    responsible_1_anchors=(),
    is_critical=False,
    is_responsible_component=False,
    satellites=(),
):
    return SimpleNamespace(
        nodes=set(nodes),
        is_composite=True,
        center_nodes=set(center_nodes),
        center_kind=center_kind,
        anchors=set(anchors),
        j_anchor_map={},
        critical_2_anchors=set(critical_2_anchors),
        responsible_1_anchors=set(responsible_1_anchors),
        s_value=None,
        opt_value=None,
        critical_ratio=None,
        is_critical=is_critical,
        critical_case="",
        critical_reason="",
        critical_signature={},
        is_responsible_component=is_responsible_component,
        responsible_case="",
        responsible_reason="",
        responsible_signature={},
        satellites=list(satellites),
    )


def _fake_state(iteration: int = 0):
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (2, 5),
        ]
    )

    H = nx.Graph()
    H.add_edges_from([(1, 2), (4, 5)])

    sat0 = _fake_satellite(
        nodes=[3],
        rescue_edges=[(2, 3)],
        supporting_anchors=[2],
        rescue_anchor=2,
        is_critical_satellite=True,
    )
    sat1 = _fake_satellite(
        nodes=[6],
        rescue_edges=[(5, 6)],
        supporting_anchors=[5],
        rescue_anchor=5,
        is_critical_satellite=False,
    )

    meta0 = _fake_meta(
        nodes=[1, 2, 3],
        center_nodes=[1, 2],
        anchors=[2],
        critical_2_anchors=[2],
        is_critical=True,
        satellites=[sat0],
    )
    meta1 = _fake_meta(
        nodes=[4, 5, 6],
        center_nodes=[4, 5],
        anchors=[5],
        responsible_1_anchors=[5],
        is_responsible_component=True,
        satellites=[sat1],
    )

    return SimpleNamespace(
        iteration=iteration,
        G=G,
        H=H,
        C={(2, 3)},
        M_C={(1, 2), (4, 5)},
        metas=[meta0, meta1],
    )


def _candidate(
    *,
    op_type: str,
    source_component_idx: int,
    source_satellite_idx: int,
    source_vertex: int,
    target_component_idx: int,
    target_vertex: int,
    expected_g_drop: float,
    toy_feature: float,
):
    return OperationCandidate(
        op_type=op_type,
        source_component_idx=source_component_idx,
        source_satellite_idx=source_satellite_idx,
        source_vertex=source_vertex,
        target_component_idx=target_component_idx,
        target_vertex=target_vertex,
        added_edge=(source_vertex, target_vertex),
        removed_edges=((2, 3),),
        features={
            "expected_g_drop": expected_g_drop,
            "toy_feature": toy_feature,
        },
        note="vertex gnn dataset demo candidate",
    )


def test_build_vertex_gnn_example_from_live_inputs() -> None:
    state = _fake_state(iteration=4)
    candidates = [
        _candidate(
            op_type="op1",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=3,
            target_component_idx=1,
            target_vertex=5,
            expected_g_drop=1.0,
            toy_feature=7.0,
        ),
        _candidate(
            op_type="op3",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=3,
            target_component_idx=1,
            target_vertex=4,
            expected_g_drop=0.0,
            toy_feature=2.0,
        ),
    ]

    live_inputs = build_live_phase4_graph_inputs(
        state,
        candidates,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    example = build_vertex_gnn_example_from_live_inputs(
        graph_id="g_demo",
        sample_id="g_demo_it_0004",
        iteration=4,
        live_inputs=live_inputs,
        selected_candidate_idx=0,
    )

    assert example.graph_id == "g_demo"
    assert example.sample_id == "g_demo_it_0004"
    assert example.iteration == 4
    assert len(example.node_list) == state.G.number_of_nodes()
    assert example.node_features.shape[0] == state.G.number_of_nodes()
    assert example.adj_norm.shape[0] == state.G.number_of_nodes()
    assert len(example.candidate_specs) == 2
    assert example.labels.shape[0] == 2
    assert float(example.labels[0]) == 1.0
    assert float(example.labels[1]) == 0.0
    assert example.selected_candidate_idx == 0


def test_vertex_gnn_training_dataset_summary() -> None:
    state = _fake_state(iteration=1)
    candidates = [
        _candidate(
            op_type="op1",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=3,
            target_component_idx=1,
            target_vertex=5,
            expected_g_drop=1.0,
            toy_feature=7.0,
        ),
        _candidate(
            op_type="op3",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=3,
            target_component_idx=1,
            target_vertex=4,
            expected_g_drop=0.0,
            toy_feature=2.0,
        ),
    ]

    live_inputs = build_live_phase4_graph_inputs(
        state,
        candidates,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    ex1 = build_vertex_gnn_example_from_live_inputs(
        graph_id="g_demo",
        sample_id="g_demo_it_0001",
        iteration=1,
        live_inputs=live_inputs,
        selected_candidate_idx=0,
    )
    ex2 = build_vertex_gnn_example_from_live_inputs(
        graph_id="g_demo",
        sample_id="g_demo_it_0002",
        iteration=2,
        live_inputs=live_inputs,
        selected_candidate_idx=1,
    )

    ds = VertexGNNTrainingDataset([ex1, ex2])
    summary = ds.summary()

    assert ds.num_examples() == 2
    assert ds.num_positive_examples() == 2
    assert summary["num_examples"] == 2
    assert summary["num_positive_labels"] == 2
    assert summary["avg_num_candidates"] == 2.0
    assert summary["node_feature_dim"] == 11


def test_build_vertex_gnn_example_rejects_bad_selected_index() -> None:
    state = _fake_state(iteration=3)
    candidates = [
        _candidate(
            op_type="op1",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=3,
            target_component_idx=1,
            target_vertex=5,
            expected_g_drop=1.0,
            toy_feature=7.0,
        ),
    ]

    live_inputs = build_live_phase4_graph_inputs(
        state,
        candidates,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    with pytest.raises(ValueError):
        build_vertex_gnn_example_from_live_inputs(
            graph_id="g_demo",
            sample_id="g_demo_it_0003",
            iteration=3,
            live_inputs=live_inputs,
            selected_candidate_idx=5,
        )
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import networkx as nx
import torch

from MPC4plus.experiments.models.vertex_gnn_selector import (
    VertexGNNCandidateScorer,
    VertexGNNPhase4Selector,
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
        note="vertex gnn demo candidate",
    )


def test_build_live_phase4_graph_inputs_basic_shapes() -> None:
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

    inputs = build_live_phase4_graph_inputs(
        state,
        candidates,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    assert len(inputs["node_list"]) == state.G.number_of_nodes()
    assert inputs["node_features"].shape[0] == state.G.number_of_nodes()
    assert inputs["node_features"].shape[1] == 11
    assert inputs["adj_norm"].shape[0] == state.G.number_of_nodes()
    assert inputs["adj_norm"].shape[1] == state.G.number_of_nodes()
    assert len(inputs["candidate_specs"]) == 2


def test_vertex_gnn_candidate_scorer_outputs_one_score_per_candidate() -> None:
    torch.manual_seed(0)

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

    inputs = build_live_phase4_graph_inputs(
        state,
        candidates,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )

    scores = model(
        node_features=inputs["node_features"],
        adj_norm=inputs["adj_norm"],
        candidate_specs=inputs["candidate_specs"],
    )

    assert scores.shape == (2,)
    assert torch.is_tensor(scores)


def test_vertex_gnn_selector_returns_none_on_empty_candidates() -> None:
    torch.manual_seed(0)

    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )
    selector = VertexGNNPhase4Selector(model)

    chosen = selector.select([], _fake_state())
    assert chosen is None


def test_vertex_gnn_selector_can_select_top1_candidate() -> None:
    torch.manual_seed(0)

    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )
    selector = VertexGNNPhase4Selector(model)

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

    chosen = selector.select(candidates, _fake_state(iteration=2))
    assert chosen in candidates


def test_vertex_gnn_selector_predict_top1_returns_index_and_score() -> None:
    torch.manual_seed(0)

    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )
    selector = VertexGNNPhase4Selector(model)

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

    pred = selector.predict_top1(candidates, _fake_state(iteration=5))

    assert pred is not None
    assert pred.candidate_idx in {0, 1}
    assert isinstance(pred.score, float)
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from MPC4plus.experiments.training.train_vertex_gnn import (
    evaluate_vertex_gnn_top1_accuracy,
    load_edge_map_from_json,
    load_trace_samples_from_jsonl,
    train_vertex_gnn_epoch,
)
from MPC4plus.experiments.datasets.vertex_gnn_dataset_builder import (
    build_vertex_gnn_dataset_from_trace_samples,
)
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer


def _trace_sample():
    return {
        "sample_id": "g_demo_it_0000",
        "graph_id": "g_demo",
        "state": {
            "iteration": 0,
            "num_nodes": 6,
            "num_edges": 5,
            "num_H_edges": 2,
            "num_C_edges": 1,
            "num_M_C_edges": 2,
            "num_components": 2,
            "num_critical_components": 1,
            "num_responsible_components": 1,
            "num_candidates_total": 2,
            "num_candidates_op1": 1,
            "num_candidates_op2": 0,
            "num_candidates_op3": 1,
            "history_length": 0,
        },
        "components": [
            {
                "component_idx": 0,
                "nodes": [1, 2, 3],
                "is_composite": True,
                "center_nodes": [1, 2],
                "center_kind": "edge",
                "anchors": [2],
                "j_anchor_map": {2: 2},
                "critical_2_anchors": [2],
                "responsible_1_anchors": [],
                "s_value": 2,
                "opt_value": 3,
                "critical_ratio": 2 / 3,
                "is_critical": True,
                "critical_case": "manual_case",
                "critical_reason": "manual",
                "critical_signature": {"kind": "manual_critical"},
                "is_responsible_component": False,
                "responsible_case": "",
                "responsible_reason": "",
                "responsible_signature": {},
                "satellites": [
                    {
                        "satellite_idx": 0,
                        "nodes": [3],
                        "kind": "satellite",
                        "rescue_edges": [(2, 3)],
                        "supporting_anchors": [2],
                        "rescue_anchor": 2,
                        "is_critical_satellite": True,
                    }
                ],
            },
            {
                "component_idx": 1,
                "nodes": [4, 5, 6],
                "is_composite": True,
                "center_nodes": [4, 5],
                "center_kind": "edge",
                "anchors": [5],
                "j_anchor_map": {5: 1},
                "critical_2_anchors": [],
                "responsible_1_anchors": [5],
                "s_value": 2,
                "opt_value": 3,
                "critical_ratio": 2 / 3,
                "is_critical": False,
                "critical_case": "",
                "critical_reason": "",
                "critical_signature": {},
                "is_responsible_component": True,
                "responsible_case": "manual_resp",
                "responsible_reason": "manual",
                "responsible_signature": {"kind": "manual_responsible"},
                "satellites": [
                    {
                        "satellite_idx": 0,
                        "nodes": [6],
                        "kind": "satellite",
                        "rescue_edges": [(5, 6)],
                        "supporting_anchors": [5],
                        "rescue_anchor": 5,
                        "is_critical_satellite": False,
                    }
                ],
            },
        ],
        "candidates": [
            {
                "candidate_idx": 0,
                "op_type": "op1",
                "source_component_idx": 0,
                "source_satellite_idx": 0,
                "source_vertex": 3,
                "target_component_idx": 1,
                "target_vertex": 5,
                "added_edge": (3, 5),
                "removed_edges": [(2, 3)],
                "features": {
                    "expected_g_drop": 1.0,
                    "toy_feature": 7.0,
                },
                "note": "candidate 0",
                "labels": {
                    "selected_in_run": 1,
                    "selected_by_rule": 1,
                },
            },
            {
                "candidate_idx": 1,
                "op_type": "op3",
                "source_component_idx": 0,
                "source_satellite_idx": 0,
                "source_vertex": 3,
                "target_component_idx": 1,
                "target_vertex": 4,
                "added_edge": (3, 4),
                "removed_edges": [(2, 3)],
                "features": {
                    "expected_g_drop": 0.0,
                    "toy_feature": 2.0,
                },
                "note": "candidate 1",
                "labels": {
                    "selected_in_run": 0,
                    "selected_by_rule": 0,
                },
            },
        ],
        "decision": {
            "selector_name": "canonical_rule",
            "selected_candidate_idx": 0,
        },
        "outcome": {
            "num_critical_before": 1,
            "num_critical_after": 0,
            "num_responsible_before": 1,
            "num_responsible_after": 1,
            "num_C_edges_before": 1,
            "num_C_edges_after": 1,
            "terminated_after_apply": False,
        },
    }


def test_load_trace_samples_from_jsonl(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_trace_sample(), ensure_ascii=False))
        f.write("\n")

    rows = load_trace_samples_from_jsonl(str(path))
    assert len(rows) == 1
    assert rows[0]["graph_id"] == "g_demo"


def test_load_edge_map_from_json(tmp_path) -> None:
    path = tmp_path / "edges.json"
    path.write_text(
        json.dumps(
            {
                "g_demo": [[1, 2], [2, 3], [4, 5], [5, 6], [2, 5]]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    edge_map = load_edge_map_from_json(str(path))
    assert "g_demo" in edge_map
    assert edge_map["g_demo"][0] == (1, 2)


def test_train_vertex_gnn_epoch_and_eval_run() -> None:
    samples = [_trace_sample()]
    edge_map = {
        "g_demo": [(1, 2), (2, 3), (4, 5), (5, 6), (2, 5)]
    }

    dataset = build_vertex_gnn_dataset_from_trace_samples(
        samples,
        edge_map=edge_map,
        candidate_feature_names=["expected_g_drop", "toy_feature"],
    )

    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )
    optimizer = __import__("torch").optim.Adam(model.parameters(), lr=1e-3)

    loss = train_vertex_gnn_epoch(
        model,
        dataset,
        optimizer=optimizer,
        device="cpu",
    )
    acc = evaluate_vertex_gnn_top1_accuracy(
        model,
        dataset,
        device="cpu",
    )

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
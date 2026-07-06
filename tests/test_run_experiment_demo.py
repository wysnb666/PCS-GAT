from __future__ import annotations

import json

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.experiments.datasets.graph_dataset_io import (
    graph_to_record,
    make_graph_split_manifest,
    save_graph_records_json,
    save_split_manifest_json,
)
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer
from MPC4plus.experiments.run_experiment import (
    run_end_to_end_experiment_on_records,
    save_experiment_rows_json,
)


def _demo_training_rows():
    return [
        {
            "graph_id": "g_train",
            "sample_id": "g_train_it_0000",
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
            "graph_id": "g_train",
            "sample_id": "g_train_it_0000",
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
            "graph_id": "g_train",
            "sample_id": "g_train_it_0001",
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
            "graph_id": "g_train",
            "sample_id": "g_train_it_0001",
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


def _build_demo_graph():
    import networkx as nx

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


def test_run_end_to_end_experiment_on_records_runs() -> None:
    record = graph_to_record(_build_demo_graph(), graph_id="g_demo")
    ranker = _build_demo_ranker()
    gnn_model = _build_demo_gnn_model()

    rows = run_end_to_end_experiment_on_records(
        [record],
        ranker=ranker,
        gnn_model=gnn_model,
        backend="ilp",
        device="cpu",
    )

    assert len(rows) == 4
    selector_names = {row["selector_name"] for row in rows}
    assert selector_names == {
        "canonical_rule",
        "one_step_heuristic",
        "xgb_selector",
        "vertex_gnn_selector",
    }
    assert all("covered_vertices" in row for row in rows)
    assert all("coverage_ratio" in row for row in rows)
    assert all("phase4_solve_seconds" in row for row in rows)
    assert all("phase5_solve_seconds" in row for row in rows)
    assert all("total_solve_seconds" in row for row in rows)
    assert all(row["phase4_solve_seconds"] >= 0.0 for row in rows)
    assert all(row["phase5_solve_seconds"] >= 0.0 for row in rows)
    assert all(row["total_solve_seconds"] >= 0.0 for row in rows)


def test_save_experiment_rows_json(tmp_path) -> None:
    rows = [
        {
            "graph_id": "g_demo",
            "selector_name": "canonical_rule",
            "covered_vertices": 10,
            "coverage_ratio": 0.5,
        }
    ]
    path = tmp_path / "experiment_rows.json"

    save_experiment_rows_json(str(path), rows)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert loaded[0]["graph_id"] == "g_demo"


def test_graph_records_and_split_manifest_roundtrip_for_experiment(tmp_path) -> None:
    record = graph_to_record(_build_demo_graph(), graph_id="g_demo")
    graphs_path = tmp_path / "graphs.json"
    split_path = tmp_path / "split.json"

    save_graph_records_json(str(graphs_path), [record])

    manifest = make_graph_split_manifest(
        ["g_demo"],
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
    )
    save_split_manifest_json(str(split_path), manifest)

    assert graphs_path.exists()
    assert split_path.exists()
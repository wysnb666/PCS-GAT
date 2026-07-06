from __future__ import annotations

import json

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

import torch

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer
from MPC4plus.experiments.run_experiment import (
    load_vertex_gnn_model,
    load_xgb_metadata,
    load_xgb_ranker,
)
from MPC4plus.experiments.training.train_vertex_gnn import (
    save_vertex_gnn_checkpoint,
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


def test_load_xgb_metadata(tmp_path) -> None:
    path = tmp_path / "xgb_metadata.json"
    payload = {
        "label_name": "label__selected_by_rule",
        "feature_names": ["feature__expected_g_drop", "feature__toy_feature"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = load_xgb_metadata(str(path))
    assert meta["label_name"] == "label__selected_by_rule"
    assert meta["feature_names"] == ["feature__expected_g_drop", "feature__toy_feature"]


def test_load_xgb_ranker_roundtrip(tmp_path) -> None:
    ranker = _build_demo_ranker()
    model_path = tmp_path / "xgb_model.json"

    ranker.save_model(str(model_path))

    loaded = load_xgb_ranker(
        model_path=str(model_path),
        feature_names=["feature__expected_g_drop", "feature__toy_feature"],
        label_name="label__selected_by_rule",
    )

    assert loaded.fitted_feature_names() == [
        "feature__expected_g_drop",
        "feature__toy_feature",
    ]


def test_load_vertex_gnn_model_roundtrip(tmp_path) -> None:
    model = VertexGNNCandidateScorer(
        candidate_feature_names=["expected_g_drop", "toy_feature"],
        hidden_dim=16,
        num_layers=2,
    )

    ckpt_path = tmp_path / "vertex_gnn.pt"
    save_vertex_gnn_checkpoint(
        model,
        path=str(ckpt_path),
        metadata={
            "candidate_feature_names": ["expected_g_drop", "toy_feature"],
            "hidden_dim": 16,
            "num_layers": 2,
            "lr": 1e-3,
        },
    )

    loaded = load_vertex_gnn_model(
        checkpoint_path=str(ckpt_path),
        device="cpu",
    )

    assert isinstance(loaded, VertexGNNCandidateScorer)
    assert loaded.candidate_feature_names == ["expected_g_drop", "toy_feature"]
    assert loaded.hidden_dim == 16
    assert loaded.num_layers == 2

    # basic forward-compatibility sanity: parameters load correctly
    orig_params = list(model.parameters())
    new_params = list(loaded.parameters())
    assert len(orig_params) == len(new_params)
    assert all(torch.equal(a.detach(), b.detach()) for a, b in zip(orig_params, new_params))
from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset


def _demo_rows():
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


def test_xgb_ranker_fit_and_predict_scores() -> None:
    ds = CandidateTableDataset(_demo_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)

    result = ranker.fit(ds)
    scores = ranker.predict_scores(ds)

    assert result.label_name == "label__selected_by_rule"
    assert result.num_rows == 4
    assert result.num_groups == 2
    assert len(scores) == 4
    assert all(isinstance(s, float) for s in scores)


def test_xgb_ranker_predict_top1_by_group() -> None:
    ds = CandidateTableDataset(_demo_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    pred = ranker.predict_top1_by_group(ds)

    assert set(pred.keys()) == {"g1_it_0000", "g1_it_0001"}
    assert pred["g1_it_0000"] in {0, 1}
    assert pred["g1_it_0001"] in {0, 1}


def test_xgb_ranker_group_top1_accuracy_runs() -> None:
    ds = CandidateTableDataset(_demo_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    acc = ranker.evaluate_group_top1_accuracy(ds)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_xgb_ranker_predict_rows_with_scores() -> None:
    ds = CandidateTableDataset(_demo_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    rows = ranker.predict_rows_with_scores(ds)

    assert len(rows) == 4
    assert all("pred__score" in row for row in rows)
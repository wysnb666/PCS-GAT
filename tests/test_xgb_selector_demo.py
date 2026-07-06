from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.baselines.xgb_selector import XGBPhase4Selector
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.phase4.operations import OperationCandidate


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


def _fake_state(iteration: int = 0):
    return SimpleNamespace(iteration=iteration)


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
        removed_edges=(),
        features={
            "expected_g_drop": expected_g_drop,
            "toy_feature": toy_feature,
        },
        note="xgb selector demo candidate",
    )


def test_xgb_selector_returns_none_on_empty_candidates() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    selector = XGBPhase4Selector(ranker)
    chosen = selector.select([], _fake_state())

    assert chosen is None


def test_xgb_selector_can_select_top1_candidate() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    selector = XGBPhase4Selector(ranker)

    candidates = [
        _candidate(
            op_type="op3",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=1,
            target_component_idx=1,
            target_vertex=3,
            expected_g_drop=0.0,
            toy_feature=1.0,
        ),
        _candidate(
            op_type="op1",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=1,
            target_component_idx=1,
            target_vertex=2,
            expected_g_drop=2.0,
            toy_feature=9.0,
        ),
    ]

    chosen = selector.select(candidates, _fake_state())

    assert chosen is not None
    assert chosen.features["expected_g_drop"] == 2.0
    assert chosen.features["toy_feature"] == 9.0


def test_xgb_selector_predict_top1_returns_index_and_score() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    selector = XGBPhase4Selector(ranker)
    candidates = [
        _candidate(
            op_type="op3",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=1,
            target_component_idx=1,
            target_vertex=3,
            expected_g_drop=0.0,
            toy_feature=1.0,
        ),
        _candidate(
            op_type="op1",
            source_component_idx=0,
            source_satellite_idx=0,
            source_vertex=1,
            target_component_idx=1,
            target_vertex=2,
            expected_g_drop=2.0,
            toy_feature=9.0,
        ),
    ]

    pred = selector.predict_top1(candidates, _fake_state(iteration=5))

    assert pred is not None
    assert pred.candidate_idx in {0, 1}
    assert isinstance(pred.score, float)


def test_xgb_selector_builds_live_dataset_with_feature_prefixes() -> None:
    ds = CandidateTableDataset(_demo_training_rows())
    ranker = XGBPhase4Ranker(label_name="label__selected_by_rule", n_estimators=20, max_depth=2)
    ranker.fit(ds)

    selector = XGBPhase4Selector(ranker)
    candidates = [
        _candidate(
            op_type="op2",
            source_component_idx=2,
            source_satellite_idx=1,
            source_vertex=7,
            target_component_idx=3,
            target_vertex=9,
            expected_g_drop=3.0,
            toy_feature=8.0,
        )
    ]

    live_ds = selector._build_live_dataset(candidates, _fake_state(iteration=12))

    assert live_ds.num_rows() == 1
    assert "feature__expected_g_drop" in live_ds.feature_names()
    assert "feature__toy_feature" in live_ds.feature_names()
    row = live_ds.rows()[0]
    assert row["sample_id"] == "live_it_12"
    assert row["candidate_idx"] == 0
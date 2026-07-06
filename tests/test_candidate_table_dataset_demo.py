from __future__ import annotations

import json

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
            "feature__expected_g_drop": 1.0,
            "feature__toy_feature": 7.0,
            "label__selected_in_run": 1,
            "label__selected_by_rule": 1,
        },
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0000",
            "iteration": 0,
            "selector_name": "canonical_rule",
            "candidate_idx": 1,
            "op_type": "op3",
            "feature__expected_g_drop": 0.0,
            "feature__toy_feature": 2.0,
            "label__selected_in_run": 0,
            "label__selected_by_rule": 0,
        },
        {
            "graph_id": "g1",
            "sample_id": "g1_it_0001",
            "iteration": 1,
            "selector_name": "heuristic",
            "candidate_idx": 0,
            "op_type": "op2",
            "feature__expected_g_drop": 2.0,
            "feature__toy_feature": 5.0,
            "label__selected_in_run": 1,
        },
    ]


def test_candidate_table_dataset_basic_summary() -> None:
    ds = CandidateTableDataset(_demo_rows())

    assert ds.num_rows() == 3
    assert ds.num_groups() == 2
    assert ds.group_ids() == ["g1_it_0000", "g1_it_0001"]
    assert "feature__expected_g_drop" in ds.feature_names()
    assert "feature__toy_feature" in ds.feature_names()
    assert "label__selected_in_run" in ds.label_names()
    assert "label__selected_by_rule" in ds.label_names()


def test_candidate_table_dataset_rows_by_group() -> None:
    ds = CandidateTableDataset(_demo_rows())

    group0 = ds.rows_by_group("g1_it_0000")
    group1 = ds.rows_by_group("g1_it_0001")

    assert len(group0) == 2
    assert len(group1) == 1
    assert all(row["sample_id"] == "g1_it_0000" for row in group0)
    assert all(row["sample_id"] == "g1_it_0001" for row in group1)


def test_candidate_table_dataset_feature_matrix_and_label_vector() -> None:
    ds = CandidateTableDataset(_demo_rows())

    X = ds.feature_matrix()
    y = ds.label_vector("label__selected_in_run")

    assert len(X) == 3
    assert len(X[0]) == 2
    assert y == [1.0, 0.0, 1.0]


def test_candidate_table_dataset_to_xy() -> None:
    ds = CandidateTableDataset(_demo_rows())

    X, y = ds.to_xy(label_name="label__selected_in_run")

    assert len(X) == 3
    assert len(y) == 3
    assert y == [1.0, 0.0, 1.0]


def test_candidate_table_dataset_positive_rate() -> None:
    ds = CandidateTableDataset(_demo_rows())

    rate_selected_in_run = ds.positive_rate("label__selected_in_run")
    rate_selected_by_rule = ds.positive_rate("label__selected_by_rule")

    assert abs(rate_selected_in_run - (2.0 / 3.0)) < 1e-9
    assert abs(rate_selected_by_rule - (1.0 / 3.0)) < 1e-9


def test_candidate_table_dataset_filter_rows() -> None:
    ds = CandidateTableDataset(_demo_rows())

    filtered = ds.filter_rows(lambda row: row["selector_name"] == "canonical_rule")

    assert filtered.num_rows() == 2
    assert filtered.num_groups() == 1
    assert filtered.group_ids() == ["g1_it_0000"]


def test_candidate_table_dataset_from_jsonl(tmp_path) -> None:
    rows = _demo_rows()
    jsonl_path = tmp_path / "candidates.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    ds = CandidateTableDataset.from_jsonl(str(jsonl_path))

    assert ds.num_rows() == 3
    assert ds.num_groups() == 2
    assert "feature__expected_g_drop" in ds.feature_names()


def test_candidate_table_dataset_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "\n".join(
            [
                "graph_id,sample_id,iteration,selector_name,candidate_idx,op_type,feature__expected_g_drop,feature__toy_feature,label__selected_in_run,label__selected_by_rule",
                "g1,g1_it_0000,0,canonical_rule,0,op1,1.0,7.0,1,1",
                "g1,g1_it_0000,0,canonical_rule,1,op3,0.0,2.0,0,0",
                "g1,g1_it_0001,1,heuristic,0,op2,2.0,5.0,1,",
            ]
        ),
        encoding="utf-8",
    )

    ds = CandidateTableDataset.from_csv(str(csv_path))

    assert ds.num_rows() == 3
    assert ds.num_groups() == 2
    assert ds.label_vector("label__selected_in_run") == [1.0, 0.0, 1.0]
    assert "label__selected_by_rule" in ds.label_names()
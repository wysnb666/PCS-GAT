from __future__ import annotations

import json

import pytest

from MPC4plus.experiments.evaluation.summarize_test_results import (
    load_result_rows,
    save_summary_json,
    summarize_result_rows,
)


def _demo_rows():
    return [
        {
            "graph_id": "g1",
            "selector_name": "canonical_rule",
            "num_phase4_steps": 0,
            "covered_vertices": 10,
            "coverage_ratio": 0.5,
            "phase4_solve_seconds": 0.10,
            "phase5_solve_seconds": 0.20,
            "total_solve_seconds": 0.30,
        },
        {
            "graph_id": "g1",
            "selector_name": "one_step_heuristic",
            "num_phase4_steps": 0,
            "covered_vertices": 10,
            "coverage_ratio": 0.5,
            "phase4_solve_seconds": 0.11,
            "phase5_solve_seconds": 0.21,
            "total_solve_seconds": 0.32,
        },
        {
            "graph_id": "g1",
            "selector_name": "xgb_selector",
            "num_phase4_steps": 0,
            "covered_vertices": 10,
            "coverage_ratio": 0.5,
            "phase4_solve_seconds": 0.12,
            "phase5_solve_seconds": 0.22,
            "total_solve_seconds": 0.34,
        },
        {
            "graph_id": "g1",
            "selector_name": "vertex_gnn_selector",
            "num_phase4_steps": 0,
            "covered_vertices": 10,
            "coverage_ratio": 0.5,
            "phase4_solve_seconds": 0.13,
            "phase5_solve_seconds": 0.23,
            "total_solve_seconds": 0.36,
        },
        {
            "graph_id": "g2",
            "selector_name": "canonical_rule",
            "num_phase4_steps": 1,
            "covered_vertices": 12,
            "coverage_ratio": 0.6,
            "phase4_solve_seconds": 0.20,
            "phase5_solve_seconds": 0.30,
            "total_solve_seconds": 0.50,
        },
        {
            "graph_id": "g2",
            "selector_name": "one_step_heuristic",
            "num_phase4_steps": 1,
            "covered_vertices": 12,
            "coverage_ratio": 0.6,
            "phase4_solve_seconds": 0.21,
            "phase5_solve_seconds": 0.31,
            "total_solve_seconds": 0.52,
        },
        {
            "graph_id": "g2",
            "selector_name": "xgb_selector",
            "num_phase4_steps": 1,
            "covered_vertices": 13,
            "coverage_ratio": 0.65,
            "phase4_solve_seconds": 0.22,
            "phase5_solve_seconds": 0.32,
            "total_solve_seconds": 0.54,
        },
        {
            "graph_id": "g2",
            "selector_name": "vertex_gnn_selector",
            "num_phase4_steps": 1,
            "covered_vertices": 11,
            "coverage_ratio": 0.55,
            "phase4_solve_seconds": 0.23,
            "phase5_solve_seconds": 0.33,
            "total_solve_seconds": 0.56,
        },
    ]


def test_summarize_result_rows_basic() -> None:
    summary = summarize_result_rows(_demo_rows())

    assert summary["num_graphs"] == 2
    assert summary["num_rows"] == 8
    assert summary["graphs_with_phase4_steps_gt_0"] == 1
    assert summary["graphs_with_selector_difference"] == 1

    assert summary["selector_summary"]["one_step_heuristic"]["equal_to_rule"] == 2
    assert summary["selector_summary"]["xgb_selector"]["better_than_rule"] == 1
    assert summary["selector_summary"]["vertex_gnn_selector"]["worse_than_rule"] == 1

    assert summary["selector_summary"]["canonical_rule"]["mean_total_solve_seconds"] == pytest.approx(0.4)
    assert summary["selector_summary"]["xgb_selector"]["mean_total_solve_seconds"] == pytest.approx(0.44)


def test_save_summary_json_and_load(tmp_path) -> None:
    summary = summarize_result_rows(_demo_rows())
    path = tmp_path / "summary.json"

    save_summary_json(str(path), summary)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["num_graphs"] == 2
    assert "selector_summary" in loaded
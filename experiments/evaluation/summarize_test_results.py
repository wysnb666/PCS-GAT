from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_result_rows(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize end-to-end test result rows.

    Expected input:
      one row per (graph_id, selector_name)

    Output focuses on:
      - how many graphs actually entered nontrivial Phase 4
      - how many graphs show any difference across selectors
      - how often XGB/GNN improve, tie, or worsen vs rule
      - mean solve-time statistics for each selector
    """
    if not rows:
        return {
            "num_graphs": 0,
            "num_rows": 0,
            "graphs_with_phase4_steps_gt_0": 0,
            "graphs_with_selector_difference": 0,
            "selector_summary": {},
        }

    by_graph: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        graph_id = str(row["graph_id"])
        by_graph.setdefault(graph_id, []).append(row)

    selector_summary = {
        "canonical_rule": _make_selector_summary_entry(),
        "one_step_heuristic": _make_selector_summary_entry(),
        "xgb_selector": _make_selector_summary_entry(),
        "vertex_gnn_selector": _make_selector_summary_entry(),
    }

    graphs_with_phase4_steps_gt_0 = 0
    graphs_with_selector_difference = 0

    for graph_id, group in by_graph.items():
        rule_row = _find_selector_row(group, "canonical_rule")
        if rule_row is None:
            raise ValueError(f"Missing canonical_rule row for graph_id={graph_id!r}")

        max_phase4_steps = max(int(row.get("num_phase4_steps", 0)) for row in group)
        if max_phase4_steps > 0:
            graphs_with_phase4_steps_gt_0 += 1

        covered_values = {int(row["covered_vertices"]) for row in group}
        if len(covered_values) > 1:
            graphs_with_selector_difference += 1

        rule_cov = int(rule_row["covered_vertices"])

        for selector_name in selector_summary.keys():
            row = _find_selector_row(group, selector_name)
            if row is None:
                raise ValueError(
                    f"Missing {selector_name} row for graph_id={graph_id!r}"
                )

            cov = int(row["covered_vertices"])
            if selector_name != "canonical_rule":
                if cov > rule_cov:
                    selector_summary[selector_name]["better_than_rule"] += 1
                elif cov < rule_cov:
                    selector_summary[selector_name]["worse_than_rule"] += 1
                else:
                    selector_summary[selector_name]["equal_to_rule"] += 1

            selector_summary[selector_name]["mean_covered_vertices_sum"] += float(cov)
            selector_summary[selector_name]["mean_coverage_ratio_sum"] += float(
                row.get("coverage_ratio", 0.0)
            )
            selector_summary[selector_name]["mean_phase4_solve_seconds_sum"] += float(
                row.get("phase4_solve_seconds", 0.0)
            )
            selector_summary[selector_name]["mean_phase5_solve_seconds_sum"] += float(
                row.get("phase5_solve_seconds", 0.0)
            )
            selector_summary[selector_name]["mean_total_solve_seconds_sum"] += float(
                row.get("total_solve_seconds", 0.0)
            )

    num_graphs = len(by_graph)
    for _, item in selector_summary.items():
        if num_graphs > 0:
            item["mean_covered_vertices"] = item.pop("mean_covered_vertices_sum") / num_graphs
            item["mean_coverage_ratio"] = item.pop("mean_coverage_ratio_sum") / num_graphs
            item["mean_phase4_solve_seconds"] = (
                item.pop("mean_phase4_solve_seconds_sum") / num_graphs
            )
            item["mean_phase5_solve_seconds"] = (
                item.pop("mean_phase5_solve_seconds_sum") / num_graphs
            )
            item["mean_total_solve_seconds"] = (
                item.pop("mean_total_solve_seconds_sum") / num_graphs
            )
        else:
            item["mean_covered_vertices"] = 0.0
            item["mean_coverage_ratio"] = 0.0
            item["mean_phase4_solve_seconds"] = 0.0
            item["mean_phase5_solve_seconds"] = 0.0
            item["mean_total_solve_seconds"] = 0.0
            item.pop("mean_covered_vertices_sum")
            item.pop("mean_coverage_ratio_sum")
            item.pop("mean_phase4_solve_seconds_sum")
            item.pop("mean_phase5_solve_seconds_sum")
            item.pop("mean_total_solve_seconds_sum")

    return {
        "num_graphs": num_graphs,
        "num_rows": len(rows),
        "graphs_with_phase4_steps_gt_0": graphs_with_phase4_steps_gt_0,
        "graphs_with_selector_difference": graphs_with_selector_difference,
        "ratio_phase4_steps_gt_0": (
            0.0 if num_graphs == 0 else graphs_with_phase4_steps_gt_0 / num_graphs
        ),
        "ratio_selector_difference": (
            0.0 if num_graphs == 0 else graphs_with_selector_difference / num_graphs
        ),
        "selector_summary": selector_summary,
    }


def _make_selector_summary_entry() -> dict[str, float | int]:
    return {
        "better_than_rule": 0,
        "equal_to_rule": 0,
        "worse_than_rule": 0,
        "mean_covered_vertices_sum": 0.0,
        "mean_coverage_ratio_sum": 0.0,
        "mean_phase4_solve_seconds_sum": 0.0,
        "mean_phase5_solve_seconds_sum": 0.0,
        "mean_total_solve_seconds_sum": 0.0,
    }


def save_summary_json(path: str, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_selector_row(group: list[dict[str, Any]], selector_name: str) -> dict[str, Any] | None:
    for row in group:
        if str(row["selector_name"]) == selector_name:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize end-to-end test result rows."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to test_results.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to summary JSON",
    )
    args = parser.parse_args()

    rows = load_result_rows(args.input_json)
    summary = summarize_result_rows(rows)
    save_summary_json(args.output_json, summary)

    print(f"[OK] Summarized {summary['num_graphs']} graphs.")
    print(f"[OK] graphs_with_phase4_steps_gt_0={summary['graphs_with_phase4_steps_gt_0']}")
    print(f"[OK] graphs_with_selector_difference={summary['graphs_with_selector_difference']}")
    print(f"[OK] Summary saved to: {args.output_json}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge selector result rows with CP-SAT exact results and summarize "
            "gap-to-opt plus solve time."
        )
    )
    parser.add_argument("--selector-results-json", type=str, required=True)
    parser.add_argument("--cpsat-results-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def load_selector_rows(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cpsat_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


def mean_or_none(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def main() -> None:
    args = parse_args()

    selector_rows = load_selector_rows(args.selector_results_json)
    cpsat_payload = load_cpsat_payload(args.cpsat_results_json)
    cpsat_rows = cpsat_payload.get("results", [])

    cpsat_by_graph = {str(row["graph_id"]): row for row in cpsat_rows}

    merged_rows: list[dict[str, Any]] = []
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in selector_rows:
        graph_id = str(row["graph_id"])
        selector_name = str(row["selector_name"])
        cpsat_row = cpsat_by_graph.get(graph_id)
        if cpsat_row is None:
            continue

        has_opt_ref = bool(cpsat_row.get("is_optimal", False))
        opt_covered_vertices = (
            int(cpsat_row["covered_vertices"]) if has_opt_ref else None
        )
        covered_vertices = int(row["covered_vertices"])

        gap_to_opt = None
        if opt_covered_vertices is not None:
            gap_to_opt = int(opt_covered_vertices) - covered_vertices

        merged = dict(row)
        merged["opt_covered_vertices"] = opt_covered_vertices
        merged["gap_to_opt"] = gap_to_opt
        merged["cpsat_solver_status"] = str(cpsat_row.get("solver_status"))
        merged["cpsat_solve_seconds"] = float(cpsat_row.get("solve_seconds", 0.0))
        merged_rows.append(merged)
        by_selector[selector_name].append(merged)

    selector_summary: dict[str, dict[str, Any]] = {}
    for selector_name, rows in by_selector.items():
        covered = [int(r["covered_vertices"]) for r in rows]
        ratios = [float(r["coverage_ratio"]) for r in rows]

        gap_rows = [r for r in rows if r["gap_to_opt"] is not None]
        gaps = [int(r["gap_to_opt"]) for r in gap_rows]

        total_times = [
            float(r["total_solve_seconds"])
            for r in rows
            if r.get("total_solve_seconds") is not None
        ]

        selector_summary[selector_name] = {
            "num_graphs": len(rows),
            "mean_covered_vertices": mean(covered) if covered else 0.0,
            "mean_coverage_ratio": mean(ratios) if ratios else 0.0,
            "num_graphs_with_opt_reference": len(gap_rows),
            "mean_gap_to_opt": mean(gaps) if gaps else None,
            "max_gap_to_opt": max(gaps) if gaps else None,
            "exact_match_count": sum(1 for g in gaps if g == 0),
            "exact_match_rate": (
                0.0 if not gaps else float(sum(1 for g in gaps if g == 0)) / float(len(gaps))
            ),
            "mean_total_solve_seconds": mean_or_none(total_times),
        }

    cpsat_opt_rows = [row for row in cpsat_rows if bool(row.get("is_optimal", False))]
    cpsat_summary = {
        "num_graphs": len(cpsat_rows),
        "num_optimal": len(cpsat_opt_rows),
        "mean_solve_seconds": mean(
            float(row["solve_seconds"]) for row in cpsat_rows
        ) if cpsat_rows else 0.0,
        "mean_covered_vertices": mean(
            int(row["covered_vertices"]) for row in cpsat_rows
        ) if cpsat_rows else 0.0,
        "mean_coverage_ratio": mean(
            float(row["coverage_ratio"]) for row in cpsat_rows
        ) if cpsat_rows else 0.0,
    }

    payload = {
        "selector_results_json": args.selector_results_json,
        "cpsat_results_json": args.cpsat_results_json,
        "num_selector_rows": len(selector_rows),
        "num_cpsat_graphs": len(cpsat_rows),
        "selector_summary": selector_summary,
        "cpsat_summary": cpsat_summary,
        "merged_rows": merged_rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Exact benchmark summary saved to: {output_path}")
    print(f"[OK] Selectors summarized: {sorted(selector_summary.keys())}")


if __name__ == "__main__":
    main()
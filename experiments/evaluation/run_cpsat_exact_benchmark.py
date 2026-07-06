from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.datasets.graph_dataset_io import (
    load_graph_records_json,
    load_split_manifest_json,
    record_to_graph,
    select_records_by_ids,
)
from MPC4plus.experiments.solvers.cpsat_mpc4_solver import solve_mpc4_with_cpsat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CP-SAT exact benchmark on a fixed graph subset."
    )
    parser.add_argument("--graphs-json", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--num-search-workers", type=int, default=8)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def choose_split_ids(manifest, split_name: str) -> list[str]:
    if split_name == "train":
        return list(manifest.train_ids)
    if split_name == "val":
        return list(manifest.val_ids)
    if split_name == "test":
        return list(manifest.test_ids)
    raise ValueError(f"Unknown split: {split_name}")


def main() -> None:
    args = parse_args()

    all_records = load_graph_records_json(args.graphs_json)
    split_manifest = load_split_manifest_json(args.split_json)
    selected_ids = choose_split_ids(split_manifest, args.split)
    selected_records = select_records_by_ids(all_records, selected_ids)

    results = []
    for record in selected_records:
        G = record_to_graph(record)
        res = solve_mpc4_with_cpsat(
            G,
            graph_id=record.graph_id,
            time_limit_seconds=float(args.time_limit_seconds),
            num_search_workers=int(args.num_search_workers),
        )
        results.append(asdict(res))

    num_optimal = sum(1 for row in results if bool(row["is_optimal"]))
    summary = {
        "num_graphs_run": len(results),
        "time_limit_seconds": float(args.time_limit_seconds),
        "num_search_workers": int(args.num_search_workers),
        "num_optimal": num_optimal,
        "optimal_rate": 0.0 if not results else float(num_optimal) / float(len(results)),
        "mean_solve_seconds": mean(row["solve_seconds"] for row in results) if results else 0.0,
        "mean_candidate_path_count": mean(row["candidate_path_count"] for row in results) if results else 0.0,
        "mean_covered_vertices": mean(row["covered_vertices"] for row in results) if results else 0.0,
        "mean_coverage_ratio": mean(row["coverage_ratio"] for row in results) if results else 0.0,
        "status_counts": {
            status: sum(1 for row in results if row["solver_status"] == status)
            for status in sorted(set(row["solver_status"] for row in results))
        },
    }

    payload = {
        "config": {
            "graphs_json": args.graphs_json,
            "split_json": args.split_json,
            "split": args.split,
            "time_limit_seconds": float(args.time_limit_seconds),
            "num_search_workers": int(args.num_search_workers),
        },
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Ran CP-SAT exact benchmark on {len(results)} graphs.")
    print(f"[OK] num_optimal={num_optimal}")
    print(f"[OK] mean_solve_seconds={summary['mean_solve_seconds']:.6f}")
    print(f"[OK] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
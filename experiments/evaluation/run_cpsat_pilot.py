from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
        description="Run a CP-SAT pilot benchmark for MPC4+ on a small graph split."
    )
    parser.add_argument(
        "--graphs-json",
        type=str,
        required=True,
        help="Path to graph records JSON.",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        required=True,
        help="Path to split manifest JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--allowed-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of allowed graph orders, e.g. --allowed-n 16 20",
    )
    parser.add_argument(
        "--per-n-limit",
        type=int,
        default=3,
        help="Run at most this many graphs per graph order n.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=60.0,
        help="CP-SAT time limit per graph.",
    )
    parser.add_argument(
        "--num-search-workers",
        type=int,
        default=8,
        help="CP-SAT num_search_workers.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Where to save the pilot result JSON.",
    )
    return parser.parse_args()


def choose_split_ids(split_manifest, split_name: str) -> list[str]:
    if split_name == "train":
        return list(split_manifest.train_ids)
    if split_name == "val":
        return list(split_manifest.val_ids)
    if split_name == "test":
        return list(split_manifest.test_ids)
    raise ValueError(f"Unknown split: {split_name}")


def select_small_records(
    records,
    *,
    allowed_n: set[int] | None,
    per_n_limit: int,
):
    """
    Deterministically keep at most `per_n_limit` graphs per node-count bucket.
    """
    buckets = defaultdict(list)
    for record in sorted(records, key=lambda r: r.graph_id):
        G = record_to_graph(record)
        n = G.number_of_nodes()
        if allowed_n is not None and n not in allowed_n:
            continue
        buckets[n].append(record)

    selected = []
    for n in sorted(buckets.keys()):
        selected.extend(buckets[n][:per_n_limit])
    return selected


def main() -> None:
    args = parse_args()

    all_records = load_graph_records_json(args.graphs_json)
    split_manifest = load_split_manifest_json(args.split_json)
    split_ids = choose_split_ids(split_manifest, args.split)
    split_records = select_records_by_ids(all_records, split_ids)

    allowed_n = set(args.allowed_n) if args.allowed_n else None
    selected_records = select_small_records(
        split_records,
        allowed_n=allowed_n,
        per_n_limit=int(args.per_n_limit),
    )

    results = []
    for record in selected_records:
        G = record_to_graph(record)
        res = solve_mpc4_with_cpsat(
            G,
            graph_id=record.graph_id,
            time_limit_seconds=float(args.time_limit_seconds),
            num_search_workers=int(args.num_search_workers),
        )
        row = asdict(res)
        results.append(row)

    status_counter = Counter(row["solver_status"] for row in results)
    optimal_rows = [row for row in results if row["is_optimal"]]
    summary = {
        "num_graphs_run": len(results),
        "time_limit_seconds": float(args.time_limit_seconds),
        "num_search_workers": int(args.num_search_workers),
        "status_counts": dict(status_counter),
        "num_optimal": sum(1 for row in results if row["is_optimal"]),
        "mean_solve_seconds": (
            mean(row["solve_seconds"] for row in results) if results else 0.0
        ),
        "mean_candidate_path_count": (
            mean(row["candidate_path_count"] for row in results) if results else 0.0
        ),
        "mean_covered_vertices": (
            mean(row["covered_vertices"] for row in results) if results else 0.0
        ),
        "mean_coverage_ratio": (
            mean(row["coverage_ratio"] for row in results) if results else 0.0
        ),
        "optimal_graph_ids": [row["graph_id"] for row in optimal_rows],
    }

    payload = {
        "config": {
            "graphs_json": args.graphs_json,
            "split_json": args.split_json,
            "split": args.split,
            "allowed_n": sorted(allowed_n) if allowed_n is not None else None,
            "per_n_limit": int(args.per_n_limit),
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

    print(f"[OK] CP-SAT pilot finished on {summary['num_graphs_run']} graphs.")
    print(f"[OK] status_counts={summary['status_counts']}")
    print(f"[OK] num_optimal={summary['num_optimal']}")
    print(f"[OK] mean_solve_seconds={summary['mean_solve_seconds']:.6f}")
    print(f"[OK] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
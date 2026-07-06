from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.export.candidate_table_builder import (
    dump_candidate_rows_csv,
    dump_candidate_rows_jsonl,
    samples_to_candidate_rows,
)


def load_graph_from_edgelist_json(path: str) -> nx.Graph:
    """
    Load a graph from a JSON file.

    Supported formats:
    1) {"edges": [[1, 2], [2, 3], ...]}
    2) [[1, 2], [2, 3], ...]

    Nodes are inferred from edges.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        edges = data.get("edges", [])
    elif isinstance(data, list):
        edges = data
    else:
        raise ValueError(f"Unsupported graph JSON format: {type(data)}")

    G = nx.Graph()
    for item in edges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid edge entry: {item}")
        u, v = int(item[0]), int(item[1])
        if u == v:
            continue
        G.add_edge(u, v)
    return G


def dump_trace_summary(result, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "graph_id": result.graph_id,
        "num_nodes": result.G.number_of_nodes(),
        "num_edges": result.G.number_of_edges(),
        "num_matching_edges": len(result.M),
        "num_H_edges": result.H.number_of_edges(),
        "num_G1_edges": result.G1.number_of_edges(),
        "num_C_before_phase4": len(result.C_before_phase4),
        "num_C_after_phase4": len(result.C_after_phase4),
        "num_M_C_edges": len(result.M_C),
        "num_phase4_steps": len(result.final_state.history),
    }

    with (output_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule baseline up to Phase 4 and export decision trace / candidate table."
    )
    parser.add_argument(
        "--graph-json",
        type=str,
        required=True,
        help="Path to input graph JSON. Supported formats: {'edges': [[u,v], ...]} or [[u,v], ...].",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default="graph",
        help="Graph identifier written into exported samples.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ilp",
        choices=["ilp", "bruteforce"],
        help="Backend for Phase 2 Step 2.2 path-cycle cover computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for exported files.",
    )
    parser.add_argument(
        "--skip-candidate-table",
        action="store_true",
        help="If set, only export trace JSONL without candidate-level table.",
    )

    args = parser.parse_args()

    G = load_graph_from_edgelist_json(args.graph_json)

    result = run_rule_phase4_pipeline(
        G,
        graph_id=args.graph_id,
        backend=args.backend,
        enable_trace=True,
    )

    if result.trace_collector is None:
        raise RuntimeError("Trace collector was not created, but export was requested.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "phase4_trace.jsonl"
    result.trace_collector.dump_jsonl(str(trace_path))

    dump_trace_summary(result, str(output_dir))

    if not args.skip_candidate_table:
        rows = samples_to_candidate_rows(result.trace_collector.samples())
        dump_candidate_rows_jsonl(rows, str(output_dir / "phase4_candidates.jsonl"))
        dump_candidate_rows_csv(rows, str(output_dir / "phase4_candidates.csv"))

    print(f"[OK] Export finished: {output_dir}")


if __name__ == "__main__":
    main()
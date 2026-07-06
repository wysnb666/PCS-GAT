from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphSplitManifest,
    load_graph_records_json,
    load_split_manifest_json,
    save_graph_records_json,
    save_split_manifest_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact-benchmark graph subset from existing selector result rows. "
            "Typical use: keep graphs where GNN and rule differ."
        )
    )
    parser.add_argument("--graphs-json", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--result-json", type=str, required=True)

    parser.add_argument(
        "--lhs-selector",
        type=str,
        default="vertex_gnn_selector",
        help="Selector on the left-hand side of comparison.",
    )
    parser.add_argument(
        "--rhs-selector",
        type=str,
        default="canonical_rule",
        help="Selector on the right-hand side of comparison.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="different",
        choices=["different", "lhs_better", "lhs_not_worse"],
        help=(
            "different: keep graphs where lhs coverage != rhs coverage; "
            "lhs_better: keep graphs where lhs coverage > rhs coverage; "
            "lhs_not_worse: keep graphs where lhs coverage >= rhs coverage and not all equal."
        ),
    )
    parser.add_argument(
        "--allowed-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of allowed graph sizes, e.g. --allowed-n 40 50",
    )
    parser.add_argument(
        "--require-phase4-steps-gt0",
        action="store_true",
        help="If set, keep only graphs whose max num_phase4_steps across selectors is > 0.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="If set, keep only the top-k graphs after ranking.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def load_result_rows(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_split_ids(manifest: GraphSplitManifest, split_name: str) -> list[str]:
    if split_name == "train":
        return list(manifest.train_ids)
    if split_name == "val":
        return list(manifest.val_ids)
    if split_name == "test":
        return list(manifest.test_ids)
    raise ValueError(f"Unknown split: {split_name}")


def group_rows_by_graph(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["graph_id"])].append(row)
    return grouped


def find_selector_row(group: list[dict[str, Any]], selector_name: str) -> dict[str, Any] | None:
    for row in group:
        if str(row["selector_name"]) == selector_name:
            return row
    return None


def should_keep_graph(
    *,
    lhs_cov: int,
    rhs_cov: int,
    all_covs: set[int],
    selection_mode: str,
) -> bool:
    if selection_mode == "different":
        return lhs_cov != rhs_cov
    if selection_mode == "lhs_better":
        return lhs_cov > rhs_cov
    if selection_mode == "lhs_not_worse":
        return lhs_cov >= rhs_cov and len(all_covs) > 1
    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def rank_key(entry: dict[str, Any]) -> tuple:
    """
    Rank more informative graphs first:
      1) larger lhs-rhs advantage
      2) larger absolute difference
      3) more phase4 steps
      4) larger graph
      5) deterministic graph_id
    """
    return (
        int(entry["lhs_minus_rhs"]),
        int(entry["abs_diff"]),
        int(entry["max_phase4_steps"]),
        int(entry["num_nodes"]),
        str(entry["graph_id"]),
    )


def main() -> None:
    args = parse_args()

    records = load_graph_records_json(args.graphs_json)
    split_manifest = load_split_manifest_json(args.split_json)
    wanted_ids = set(choose_split_ids(split_manifest, args.split))

    rows = load_result_rows(args.result_json)
    grouped = group_rows_by_graph(rows)

    allowed_n = set(args.allowed_n) if args.allowed_n else None
    by_id = {r.graph_id: r for r in records}

    candidates: list[dict[str, Any]] = []
    for graph_id in sorted(wanted_ids):
        if graph_id not in grouped or graph_id not in by_id:
            continue

        group = grouped[graph_id]
        lhs_row = find_selector_row(group, args.lhs_selector)
        rhs_row = find_selector_row(group, args.rhs_selector)
        if lhs_row is None or rhs_row is None:
            continue

        num_nodes = int(lhs_row["num_nodes"])
        if allowed_n is not None and num_nodes not in allowed_n:
            continue

        max_phase4_steps = max(int(row.get("num_phase4_steps", 0)) for row in group)
        if args.require_phase4_steps_gt0 and max_phase4_steps <= 0:
            continue

        lhs_cov = int(lhs_row["covered_vertices"])
        rhs_cov = int(rhs_row["covered_vertices"])
        selector_cov = {
            str(row["selector_name"]): int(row["covered_vertices"])
            for row in group
        }
        all_covs = set(selector_cov.values())

        if not should_keep_graph(
            lhs_cov=lhs_cov,
            rhs_cov=rhs_cov,
            all_covs=all_covs,
            selection_mode=args.selection_mode,
        ):
            continue

        entry = {
            "graph_id": graph_id,
            "num_nodes": num_nodes,
            "num_edges": int(lhs_row["num_edges"]),
            "max_phase4_steps": max_phase4_steps,
            "lhs_selector": args.lhs_selector,
            "rhs_selector": args.rhs_selector,
            "lhs_covered_vertices": lhs_cov,
            "rhs_covered_vertices": rhs_cov,
            "lhs_minus_rhs": lhs_cov - rhs_cov,
            "abs_diff": abs(lhs_cov - rhs_cov),
            "selector_covered_vertices": selector_cov,
        }
        candidates.append(entry)

    candidates = sorted(candidates, key=rank_key, reverse=True)
    if args.max_graphs is not None:
        candidates = candidates[: int(args.max_graphs)]

    selected_ids = [entry["graph_id"] for entry in candidates]
    selected_records = [by_id[gid] for gid in selected_ids]
    selected_rows = [row for row in rows if str(row["graph_id"]) in set(selected_ids)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_graph_records_json(str(output_dir / "graphs.json"), selected_records)
    save_split_manifest_json(
        str(output_dir / "split.json"),
        GraphSplitManifest(train_ids=[], val_ids=[], test_ids=selected_ids),
    )

    (output_dir / "selected_graph_ids.json").write_text(
        json.dumps(selected_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "reference_rows.json").write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "config": {
            "graphs_json": args.graphs_json,
            "split_json": args.split_json,
            "split": args.split,
            "result_json": args.result_json,
            "lhs_selector": args.lhs_selector,
            "rhs_selector": args.rhs_selector,
            "selection_mode": args.selection_mode,
            "allowed_n": sorted(allowed_n) if allowed_n is not None else None,
            "require_phase4_steps_gt0": bool(args.require_phase4_steps_gt0),
            "max_graphs": args.max_graphs,
        },
        "num_selected_graphs": len(selected_ids),
        "selected_graph_ids": selected_ids,
        "selected_graphs": candidates,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Selected {len(selected_ids)} graphs for exact benchmark.")
    print(f"[OK] Output dir: {output_dir}")
    if selected_ids:
        print(f"[OK] First selected graph ids: {selected_ids[:5]}")


if __name__ == "__main__":
    main()